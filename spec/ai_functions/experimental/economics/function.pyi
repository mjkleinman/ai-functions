"""EconomicFunction: an AI function with candidates, a value, and a budget.

The class both decorators (:func:`~.decorators.routed`,
:func:`~.decorators.economic`) construct. It owns the standing configuration
— candidates, beliefs, value, budget, policy — and mirrors the calling
surface of ``AIFunction``: ``await fn(...)``, ``run_sync``, ``trace``,
``spawn``, plus ``plan()`` — decide without executing. One call runs one
search: estimate the candidates, loop :class:`~.search.Search`, spawn each
attempt as a child thread, book an :class:`~.types.AttemptRecord` per
attempt (measured in dollars from the event log), and update the beliefs
online.

Invariants:
    E1 — dollars everywhere; ``value`` is the only place task success is
    priced, and post-conditions are never a reward definition, only the
    local booking signal (E2 settles them later).

    E5 — the function never knowingly spends more than expected reward:
    when no candidate has positive net value it abstains rather than
    running the least-bad one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, final

from ...protocols import Spawnable
from ...types import InputShape, ThreadContext
from .beliefs import Beliefs
from .search import Policy
from .types import AttemptRecord, Candidate, Ranking

if TYPE_CHECKING:
    from ...handle import ThreadHandle
    from ...types.graph import GradFeedback, Result


Value = float | Callable[[Any], float]
"""Dollar worth of a passing result. The two arms belong to the two doors:
a constant under ``@routed`` (a pass books the full value), a callable
under ``@economic`` (it prices the *merged* total; the search books the
differences, so each attempt's reward is its marginal gain)."""


def keep_best[T](value: Callable[[T], float]) -> Callable[[T, T], T]:
    """The keep-the-best fold, ``@economic``'s default merge.

    A new result replaces the incumbent only by scoring strictly higher
    under ``value``.

    Args:
        value: The same dollar-worth function passed to the decorator.

    Returns:
        A ``(best, new) -> best`` merge function.
    """
    ...


DECISION_EVENT: str
"""``CustomEvent.kind`` emitted once per estimation round. Payload: the
ranked ``(label, reservation_price)`` entries, the per-candidate estimates,
and the record ids booked so far — telemetry read by :func:`decisions`."""

ATTEMPT_EVENT: str
"""``CustomEvent.kind`` emitted after each attempt. Payload: the serialized
``AttemptRecord`` plus the id of the child thread that ran the attempt —
the durable per-candidate spend attribution read by :func:`attempts` and
:func:`spend`."""

DECISION_PARAMETER: str
"""Name of the grad-enabled parameter the economic run emits so its routing
decision becomes an optimization target. The backward pass scores it; the
function's :class:`~ai_functions.types.graph.ParameterHost` methods settle the
run's records from that score."""


@dataclass(frozen=True)
class Decision:
    """One planning round: what the search would do for a task, without doing it.

    Returned by :meth:`EconomicFunction.plan`. Costs one estimation round
    and books no records; a caller that executes the decision itself closes
    the learning loop with :meth:`report`.
    """

    candidate: Candidate[..., Any] | None
    """The candidate the search would try first, or ``None`` when no
    candidate's expected reward covers its cost (E5)."""
    ranking: list[Ranking]
    """Every eligible candidate with its reservation price, ranked."""

    _report_hook: Callable[[Any, float | None], None] | None = field(default=None, init=False, compare=False)
    """Reporting closure attached by ``plan`` after construction; not an
    ``__init__`` parameter."""

    def report(self, result: Any | None, cost: float | None = None) -> None:
        """Close the learning loop for an externally executed decision.

        Books an :class:`~.types.AttemptRecord` for ``candidate`` (valued
        under the function's ``value``) and feeds it to ``beliefs.update``
        — exactly what ``__call__`` does for attempts it runs itself.
        Without this call, a plan-then-execute-yourself pattern never
        teaches the beliefs anything.

        Args:
            result: The typed result the caller obtained, or ``None`` for a
                failed attempt.
            cost: Dollars the caller actually spent. Pass the measured spend
                when you have it; ``None`` books the *estimated* cost from
                the plan — fine for learning success rates, biased for costs.

        Raises:
            ValueError: ``candidate`` is ``None`` (nothing was decided).
        """
        ...


class EconomicThread[**P, T]:
    """Live thread that runs one search per ``run`` call.

    Produced by ``EconomicFunction.to_thread``. Satisfies the ``Thread``
    protocol; ``execute`` receives the call arguments directly and manages
    estimation, attempt spawning, booking, and stopping internally. Each
    attempt is a separate child thread, warm-seeded from the prior attempt
    when ``carry_context`` is set. Per-attempt usage is measured from the
    attempt's event log against a post-spawn baseline, so warm-seeded copies
    of earlier usage events are not double-counted.
    """

    def __init__(self, fn: EconomicFunction[P, T]) -> None: ...

    @property
    def name(self) -> str:
        """Thread name derived from the wrapped function."""
        ...

    async def execute(self, ctx: ThreadContext, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run the full search loop for one task.

        Returns:
            The typed result of the highest-reward passing attempt (the
            merged result under ``merge``).

        Raises:
            Abstained: No candidate had positive net value at decision time (E5).
            CandidatesExhausted: Every profitable candidate ran, none passed.
            BudgetExceeded: The next profitable attempt did not fit the budget
                and no passing result exists yet.
        """
        ...

    async def notify(self, text: str) -> None:
        """No-op — the economic thread ignores injected messages."""
        ...

    def serialize_result(self, result: T) -> str:
        """Serialize via the first candidate's thread."""
        ...

    def deserialize_result(self, payload: str) -> T:
        """Deserialize via the first candidate's thread."""
        ...

    async def fork(self) -> EconomicFunction[P, T]:
        """Return the wrapped ``EconomicFunction`` as a spawnable."""
        ...

    async def teardown(self) -> None:
        """No-op teardown."""
        ...


@final
class EconomicFunction[**P, T](Spawnable[P, T]):
    """A set of candidates managed under one value, one budget, and shared beliefs.

    Callable with the candidates' task arguments; construct via
    :func:`~.decorators.routed` / :func:`~.decorators.economic` or directly.
    Statefulness lives entirely in ``beliefs`` (shared and persistent by
    design); the function itself is a template like every other spawnable.

    Args:
        candidates: The alternatives, keyed by label.
        value: Dollar worth of success (see :data:`Value`): a constant
            without ``merge`` (routing), a callable pricing the merged
            result under ``merge`` (repeated sampling).
        beliefs: Estimate/learn provider consulted per call.
        budget: Hard dollar cap per call; ``None`` = no cap.
        policy: Search policy; ``None`` uses the ``Search`` default
            (``Greedy``). A merged search should pass a continuing policy
            such as ``ReservationPricePolicy`` — a one-shot policy stops at
            the first banked reward. The decorators set this per door.
        max_tries: Attempts per candidate per call; ``None`` = unbounded
            (requires ``budget``). Under ``merge`` this caps draws per
            candidate: the default 1 is Weitzman's classic open-each-box-
            at-most-once; ``None`` is open-ended repeated sampling.
        reestimate: Re-run ``beliefs.estimate`` after each attempt. Needed
            whenever the beliefs project from the search's ``history``
            (e.g. ``DiminishingReturns``); the ``@economic`` door turns it on.
        carry_context: Seed each attempt from the previous attempt's transcript.
        merge: Fold each passing attempt's result into a running result
            (repeated sampling); rewards are booked as marginal gains.
            Requires ``budget``.

    Raises:
        ValueError: Empty or duplicate labels; ``max_tries=None`` without
            ``budget``; or a callable ``value`` without ``merge``.
    """

    def __init__(
        self,
        candidates: Mapping[str, Candidate[P, T]],
        value: Value,
        beliefs: Beliefs,
        budget: float | None = None,
        policy: Policy | None = None,
        max_tries: int | None = 1,
        reestimate: bool = False,
        carry_context: bool = False,
        merge: Callable[[T, T], T] | None = None,
    ) -> None: ...

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run one search over the candidates and return the best passing result.

        Raises:
            Abstained: No candidate had positive net value at decision time (E5).
            CandidatesExhausted: Every profitable candidate ran and none passed.
            BudgetExceeded: The next attempt did not fit the remaining budget.
        """
        ...

    def run_sync(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Blocking wrapper around ``__call__``; mirrors ``AIFunction.run_sync``."""
        ...

    async def plan(self, *args: P.args, **kwargs: P.kwargs) -> Decision:
        """Decide without executing: one estimation round, no attempts.

        For callers that want the routing decision but own the execution
        (report back via :meth:`Decision.report`), or want to anticipate
        abstention without an exception.
        """
        ...

    async def trace(self, *args: Any, **kwargs: Any) -> Result[T]:
        """Run like ``__call__`` and return a traced ``Result`` node.

        The traced thread's event log (decisions and attempts) survives for
        post-hoc inspection and graph reconstruction. Arguments may be
        ``Result`` / ``ParameterView`` handles, as with ``AIFunction.trace``.
        """
        ...

    # ── Spawnable ──

    def to_thread(self) -> EconomicThread[P, T]:
        """Produce a fresh live thread that runs one search per ``run`` call."""
        ...

    async def spawn(self) -> ThreadHandle[P, T]:
        """Spawn on a private in-process worker and return the handle."""
        ...

    @property
    def input_shape(self) -> InputShape:
        """Input shape of the candidates' shared task signature."""
        ...

    @property
    def name(self) -> str:
        """The wrapped function's name; for telemetry, errors, and graph nodes.

        Deliberately identical to the plain function's: a traced economic run
        reconstructs to a node named after the function it routes, so a
        backward pass reads ``research-a1b2``, not an implementation detail.
        """
        ...

    # ── Introspection ──

    @property
    def candidates(self) -> Mapping[str, Candidate[P, T]]:
        """The configured candidates, keyed by label."""
        ...

    @property
    def beliefs(self) -> Beliefs:
        """The live beliefs provider (shared, stateful)."""
        ...

    # ── ParameterHost (optimizer bridge) ──
    # The function is its own parameter host: each run emits its routing
    # decision as a grad-enabled parameter, and the backward-pass score settles
    # the run's records (E2). Pass the function into
    # ``TextGradOptimizer.step(backends=[...])`` alongside any memory backend.

    @property
    def backend_id(self) -> str:
        """Stable id matched against this function's decision recall events.

        Derived from the function name, so it is stable across processes: a
        fresh function over beliefs reloaded from persisted stats settles the
        same record ids.
        """
        ...

    def deserialize_value(self, name: str, raw: Any) -> Any:
        """Return the decision context verbatim — it is already display text."""
        ...

    def _is_procedural(self, name: str) -> bool:
        """The decision parameter is never code."""
        ...

    def consolidate(self, name: str, feedback: list[GradFeedback], retrieved: dict[str, str] | None = None) -> None:
        """Settle the run's records to the averaged downstream score (E2).

        Args:
            name: The decision parameter name (opaque here).
            feedback: Gradients routed to the decision parameter; their scores
                are averaged into the settled value.
            retrieved: ``{record_id: candidate_label}`` for the run's records,
                carried from the recall event's ``meta["results"]``. Without
                it there is nothing to settle.
        """
        ...

    def replace(self, **changes: Any) -> EconomicFunction[P, T]:
        """Return a new economic function with constructor fields overridden.

        ``self`` is unchanged; ``beliefs`` is shared, not copied, unless
        explicitly replaced.
        """
        ...


async def attempts(run: ThreadHandle[..., Any] | Result[Any]) -> list[AttemptRecord]:
    """Read the booked records from an economic function run's event log.

    Args:
        run: A handle to a spawned economic function thread, or the
            ``Result`` of ``EconomicFunction.trace``.

    Returns:
        The run's ``AttemptRecord``s in booking order.
    """
    ...


async def decisions(run: ThreadHandle[..., Any] | Result[Any]) -> list[list[Ranking]]:
    """Read the ranked estimation rounds from the run's event log.

    Args:
        run: A handle to a spawned economic function thread, or the
            ``Result`` of ``EconomicFunction.trace``.

    Returns:
        One ranking per estimation round, in chronological order.
    """
    ...


async def spend(run: ThreadHandle[..., Any] | Result[Any]) -> float:
    """Total dollars booked by a run and its spawned subtree.

    Sums the cost of every ``ATTEMPT_EVENT`` record across the subtree
    (attempt costs already reflect each attempt's own sub-thread usage),
    giving workflow-level cost attribution from the event log alone.

    Args:
        run: A handle to a spawned economic function thread, or the
            ``Result`` of ``EconomicFunction.trace``.

    Returns:
        Total cost in dollars.
    """
    ...
