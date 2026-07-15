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

import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn, cast, final

from ai_functions.ai_thread.errors import AIFunctionError
from ai_functions.runtime.usage import last_event_id, subtree_usage
from ai_functions.types import (
    CustomEvent,
    EventKind,
    GradFeedback,
    InputShape,
    MessageUserEvent,
    ParameterRecalledEvent,
    TokenUsage,
    TraceDelegationEvent,
)
from .beliefs import Beliefs
from .search import Estimate, Policy, Search
from .types import (
    Abstained,
    AttemptRecord,
    BudgetExceeded,
    Candidate,
    CandidatesExhausted,
    Ranking,
    RecordId,
    TaskView,
)

if TYPE_CHECKING:
    from ai_functions.handle import ThreadHandle
    from ai_functions.protocols import Coordinator, ThreadContext
    from ai_functions.types.graph import Result

logger = logging.getLogger(__name__)


DECISION_EVENT = "economics_decision"
"""``CustomEvent.kind`` emitted once per estimation round. Payload: the
ranked ``(label, reservation_price)`` entries, the per-candidate estimates,
and the record ids booked so far — telemetry read by :func:`decisions`."""

ATTEMPT_EVENT = "economics_attempt"
"""``CustomEvent.kind`` emitted after each attempt. Payload: the serialized
``AttemptRecord`` plus the id of the child thread that ran the attempt —
the durable per-candidate spend attribution read by :func:`attempts` and
:func:`spend`."""

DECISION_PARAMETER = "routing_decision"
"""Name of the grad-enabled parameter the economic run emits so its routing
decision becomes an optimization target. The backward pass scores it; the
function's :class:`~ai_functions.types.graph.ParameterHost` methods settle the
run's records from that score."""

_DECISION_SCORE_NOTE = (
    "the score you assign settles the routing decision that produced this input, "
    "revising the routed model's success statistics — score the VALUE's actual "
    "usefulness, not the feedback text"
)


Value = float | Callable[[Any], float]
"""Dollar worth of a passing result. The two arms belong to the two doors:
a constant under ``@routed`` (a pass books the full value), a callable
under ``@economic`` (it prices the *merged* total; the search books the
differences, so each attempt's reward is its marginal gain)."""


def _value_of(value: Value, result: Any) -> float:
    """Dollar worth of ``result`` under ``value``."""
    return value(result) if callable(value) else float(value)


def keep_best[T](value: Callable[[T], float]) -> Callable[[T, T], T]:
    """The keep-the-best fold: a new result replaces the incumbent only by
    scoring strictly higher under ``value``. ``@economic``'s default merge.

    Args:
        value: The same dollar-worth function passed to the decorator.

    Returns:
        A ``(best, new) -> best`` merge function.
    """

    def _keep(best: T, new: T) -> T:
        return new if value(new) > value(best) else best

    return _keep


# ── Decision (plan output) ────────────────────────────────────────


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

    # ``plan`` attaches the reporting closure after construction (via
    # ``object.__setattr__``, as frozen dataclasses allow). ``init=False``
    # keeps it out of ``__init__`` and the public contract.
    _report_hook: Callable[[Any, float | None], None] | None = field(
        default=None, init=False, repr=False, compare=False
    )

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
        if self.candidate is None:
            raise ValueError("Decision.report: nothing was decided (candidate is None)")
        assert self._report_hook is not None  # always set by EconomicFunction.plan
        self._report_hook(result, cost)


# ── EconomicThread (live search loop) ─────────────────────────────


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

    def __init__(self, fn: EconomicFunction[P, T]) -> None:
        self._fn = fn

    @property
    def name(self) -> str:
        """Thread name derived from the wrapped function."""
        return self._fn.name

    async def _run_attempt(
        self,
        ctx: ThreadContext,
        candidate: Candidate[P, T],
        seed_from: Any,
        notify_text: str | None,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[ThreadHandle[P, T], Any, bool, TokenUsage, int]:
        """Spawn and run one attempt; return its handle, result, outcome, usage, and turns."""
        info = await ctx.coordinator.get_thread_info(ctx.thread_id)
        # max_attempts=0 disables the attempt thread's own post-condition
        # retries: the search owns all retry logic.
        handle = await ctx.coordinator.spawn(
            candidate.fn.replace(max_attempts=0),
            seed_from=seed_from,
            worker_id=info.worker_id,
            parent_id=ctx.thread_id,
        )
        baseline = await last_event_id(ctx.coordinator, handle.id)

        if notify_text is not None:
            await handle.notify(notify_text)

        passed = False
        result: Any = None
        try:
            result = await handle.run(*args, **kwargs)
            passed = True
        except AIFunctionError:
            # Post-condition or validation failure — the ordinary "this
            # candidate didn't produce an acceptable result" path.
            pass
        except Exception:  # noqa: BLE001
            # Any other attempt failure — a model max-tokens/throttling error,
            # a tool crash — is still just a failed attempt for a fallback
            # layer: book it and let the search escalate. ``CancelledError`` is
            # a ``BaseException`` and is intentionally not caught, so teardown
            # and cooperative cancel still propagate.
            logger.info("economics: attempt by %s raised; treating as failed", candidate.label, exc_info=True)

        usage, turns = await subtree_usage(ctx.coordinator, handle.id, since_id=baseline)
        return handle, result, passed, usage, turns

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
        fn = self._fn
        candidates = list(fn._candidates.values())
        task = await self._task_view(candidates[0], args, kwargs)

        records: list[AttemptRecord] = []
        attempt_ids: list[str] = []  # parallel to records: thread id that ran each
        decision_rounds: list[dict[str, Any]] = []  # one payload per estimation round
        by_label = fn._candidates
        merged: Any = None  # running result under merge
        last_attempt_id: Any = None
        record_seq = 0

        # Estimate once up front to seed the search; re-estimate in-loop when
        # configured (the @economic door turns it on, since its beliefs
        # project the next gain from the attempts observed so far).
        estimates = await self._estimate(fn, task, candidates, records)
        search = Search(estimates, budget=fn._budget, policy=fn._policy, max_tries=fn._max_tries)
        self._emit_decision(ctx, search, records, estimates, decision_rounds)

        attempt_handles: list[ThreadHandle[P, T]] = []
        try:
            first = True
            while True:
                if not first and fn._reestimate:
                    estimates = await self._estimate(fn, task, candidates, records)
                    search.update_estimates(estimates)
                    self._emit_decision(ctx, search, records, estimates, decision_rounds)
                first = False

                label = search.next()
                if label is None:
                    break
                candidate = by_label[label]

                # Warm-seed and announce prior outcome only when carrying context.
                seed_from = last_attempt_id if fn._carry_context else None
                notify_text: str | None = None
                if fn._carry_context and records:
                    prior = records[-1]
                    status = "passed" if prior.local_score > 0 else "failed"
                    notify_text = f"[Prior attempt by {prior.candidate} {status}. Continue from there.]"

                handle, result, passed, usage, turns = await self._run_attempt(
                    ctx, candidate, seed_from, notify_text, args, kwargs
                )
                attempt_handles.append(handle)
                last_attempt_id = handle.id
                cost = candidate.prices.cost_of(usage)

                # Reward: dollars this attempt added. Under merge, fold into the
                # running result and book the marginal gain; otherwise the raw
                # result's value.
                if passed and fn._merge is not None:
                    prev_value = _value_of(fn._value, merged) if merged is not None else 0.0
                    merged = result if merged is None else fn._merge(merged, result)
                    reward = _value_of(fn._value, merged) - prev_value
                elif passed:
                    reward = _value_of(fn._value, result)
                else:
                    reward = 0.0

                record = AttemptRecord(
                    id=RecordId(f"{ctx.thread_id}-{record_seq}"),
                    task=task,
                    candidate=candidate.label,
                    cost=cost,
                    usage=usage,
                    turns=turns,
                    reward=reward,
                    local_score=1.0 if passed else 0.0,
                )
                record_seq += 1
                records.append(record)
                attempt_ids.append(str(handle.id))
                fn._beliefs.update(record)
                self._emit_attempt(ctx, record, handle.id)

                # The reward in hand is the worth of the running result — the
                # bar the stopping rule compares reservation prices against.
                # Under merge that is the merged result's value, and beliefs
                # project the worth *after* one more attempt, so ``g > bar``
                # holds exactly while the projected gain covers the next
                # attempt's cost. Without merge it is the attempt's reward.
                observed = (
                    (_value_of(fn._value, merged) if merged is not None else 0.0)
                    if fn._merge is not None
                    else reward
                )
                search.observe(label, reward=observed, cost=cost)

                # Plain routing returns on the first pass; merge continues,
                # banking the running result, and lets the stopping rule decide.
                if passed and fn._merge is None:
                    return cast("T", result)

            if merged is not None:
                return cast("T", merged)

            # Nothing passed. Distinguish "never tried anything" (abstained)
            # from "tried and all failed" (exhausted) from "ran out of budget".
            self._raise_empty(fn, records, search)
        finally:
            # Turn this run's telemetry into optimization structure, so a later
            # backward pass over the traced graph can revise the beliefs. Done
            # even on failure — a failed run's records are still learnable
            # signal. Two emissions, both consumed at graph-build time:
            #   1. a routing-summary user message + a trace-delegation marker,
            #      so the reconstructed node adopts the responsible attempt's
            #      conversation (the backward model needs the real trace);
            #   2. a grad-enabled decision parameter recall carrying the run's
            #      record ids, so settlement flows through the ordinary
            #      parameter-consolidation path (the function is its own host).
            if records:
                self._emit_trace_delegation(ctx, decision_rounds, records, attempt_ids)
                # Merged searches are not currently supported for textual
                # optimization: what a score should settle depends on the
                # particular merge function.
                if fn._merge is None:
                    await self._emit_decision_parameter(ctx, fn, decision_rounds, records)
            for h in attempt_handles:
                try:
                    await h.terminate_now()
                except Exception:  # noqa: BLE001 — teardown is best-effort
                    logger.debug("economics: failed to terminate attempt %s", h.id, exc_info=True)

    async def _task_view(self, candidate: Candidate[P, T], args: tuple[Any, ...], kwargs: dict[str, Any]) -> TaskView:
        """Build the ``TaskView`` estimators see: rendered prompt plus bound kwargs."""
        prompt = await candidate.fn.render_prompt(*args, **kwargs)
        return TaskView(prompt=prompt, arguments=dict(kwargs))

    async def _estimate(
        self,
        fn: EconomicFunction[P, T],
        task: TaskView,
        candidates: list[Candidate[P, T]],
        history: list[AttemptRecord],
    ) -> dict[str, Estimate]:
        """Call the beliefs estimator; guarantee an estimate for every candidate (E4).

        The scale passed is the declared constant value. Under ``merge`` the
        value is a callable pricing merged totals, so no constant scale
        exists; ``None`` is passed, and providers that price at the constant
        scale (e.g. :class:`~.beliefs.EmpiricalBeliefs`) raise rather than
        estimate at a fictitious one.
        """
        scale = None if callable(fn._value) else float(fn._value)
        estimates = await fn._beliefs.estimate(task, cast("list[Candidate]", candidates), scale, history)
        missing = [c.label for c in candidates if c.label not in estimates]
        if missing:
            raise AIFunctionError(f"beliefs returned no estimate for {missing}", function_name=fn.name)
        return estimates

    def _emit_decision(
        self,
        ctx: ThreadContext,
        search: Search,
        records: list[AttemptRecord],
        estimates: dict[str, Estimate],
        rounds: list[dict[str, Any]],
    ) -> None:
        """Emit a ``DECISION_EVENT`` and append its payload to ``rounds``.

        Alongside the search-internal ranking, the payload carries what the
        beliefs actually predicted per candidate (expected reward, estimated
        cost) and a snapshot of their learned statistics — the decision
        context the run later renders into the routing summary so a backward
        pass can compare prediction against outcome.
        """
        ranking = search.explain()
        stats = self._fn._beliefs.stats()
        payload: dict[str, Any] = {
            "ranking": [
                {"label": r.label, "reservation_price": r.reservation_price, "net_value": r.net_value}
                for r in ranking
            ],
            "estimates": {
                label: {"expected_reward": e.dist.mean(), "cost": e.cost} for label, e in estimates.items()
            },
            "stats": dict(stats),
            "record_ids": [r.id for r in records],
        }
        rounds.append(payload)
        ctx.on_event(CustomEvent(kind=DECISION_EVENT, payload=payload))

    def _emit_trace_delegation(
        self,
        ctx: ThreadContext,
        decision_rounds: list[dict[str, Any]],
        records: list[AttemptRecord],
        attempt_ids: list[str],
    ) -> None:
        """Emit the routing summary and a delegation marker for the responsible attempt.

        The summary (a ``MessageUserEvent``) names the decision rounds and every
        attempt; the marker (a ``TraceDelegationEvent``) points at the attempt
        whose result the run returned — the last locally passing attempt, or the
        last attempt when none passed. At graph-build time the node adopts that
        attempt's conversation after the summary.
        """
        responsible_idx = next(
            (i for i in range(len(records) - 1, -1, -1) if records[i].local_score > 0),
            len(records) - 1,
        )
        summary = _render_routing_summary(self._fn.name, decision_rounds, records, records[responsible_idx].candidate)
        ctx.on_event(MessageUserEvent(thread_id=ctx.thread_id, text=summary))
        ctx.on_event(
            TraceDelegationEvent(thread_id=ctx.thread_id, child_thread_id=attempt_ids[responsible_idx])
        )

    async def _emit_decision_parameter(
        self,
        ctx: ThreadContext,
        fn: EconomicFunction[P, T],
        decision_rounds: list[dict[str, Any]],
        records: list[AttemptRecord],
    ) -> None:
        """Emit the grad-enabled routing-decision parameter for this run.

        The value is the rendered routing summary (what the backward model
        scores); ``meta["results"]`` maps each record id to its candidate, so the
        function's ``consolidate`` knows which records the assigned score
        settles. Emitted through the memory-backend event path so ``build_graph``
        matches it to the function by ``backend_id`` exactly as for a memory
        parameter.
        """
        responsible = next(
            (r.candidate for r in reversed(records) if r.local_score > 0),
            records[-1].candidate,
        )
        summary = _render_routing_summary(fn.name, decision_rounds, records, responsible)
        event = ParameterRecalledEvent(
            thread_id=ctx.thread_id,
            name=DECISION_PARAMETER,
            value=summary,
            derivation="full",
            requires_grad=True,
            backend_id=fn.backend_id,
            description=_DECISION_SCORE_NOTE,
            meta={"results": {str(r.id): r.candidate for r in records}},
        )
        ctx.on_event(event)

    def _emit_attempt(self, ctx: ThreadContext, record: AttemptRecord, attempt_thread_id: Any) -> None:
        """Emit an ``ATTEMPT_EVENT`` carrying the serialized record and attempt thread id."""
        ctx.on_event(
            CustomEvent(
                kind=ATTEMPT_EVENT,
                payload={"record": record.model_dump(), "attempt_thread_id": str(attempt_thread_id)},
            )
        )

    def _raise_empty(self, fn: EconomicFunction[P, T], records: list[AttemptRecord], search: Search | None) -> NoReturn:
        """Raise the failure that fits: abstained, budget-exceeded, or exhausted."""
        if not records:
            raise Abstained(
                f"{fn.name}: no candidate's expected reward covered its cost",
                function_name=fn.name,
                records=records,
            )
        if search is not None and search.blocked_by_budget():
            raise BudgetExceeded(
                f"{fn.name}: the next profitable candidate did not fit the remaining budget",
                function_name=fn.name,
                records=records,
            )
        raise CandidatesExhausted(
            f"{fn.name}: every profitable candidate ran and none passed",
            function_name=fn.name,
            records=records,
        )

    # ── Thread protocol (rest) ──

    async def notify(self, text: str) -> None:
        """No-op — the economic thread ignores injected messages."""
        del text

    def serialize_result(self, result: T) -> str:
        """Serialize via the first candidate's thread."""
        return next(iter(self._fn._candidates.values())).fn.to_thread().serialize_result(result)

    def deserialize_result(self, payload: str) -> T:
        """Deserialize via the first candidate's thread."""
        return next(iter(self._fn._candidates.values())).fn.to_thread().deserialize_result(payload)

    async def fork(self) -> EconomicFunction[P, T]:
        """Return the wrapped ``EconomicFunction`` as a spawnable."""
        return self._fn

    async def teardown(self) -> None:
        """No-op teardown."""


# ── EconomicFunction ──────────────────────────────────────────────


@final
class EconomicFunction[**P, T]:
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
    ) -> None:
        if not candidates:
            raise ValueError("EconomicFunction requires at least one candidate")
        for label, c in candidates.items():
            if not label:
                raise ValueError("candidate labels must be non-empty")
            if c.label != label:
                raise ValueError(f"candidate keyed {label!r} carries mismatched label {c.label!r}")
        self._candidates: dict[str, Candidate[P, T]] = dict(candidates)
        self._value = value
        self._beliefs = beliefs
        self._budget = budget
        self._policy = policy
        self._carry_context = carry_context
        self._merge = merge
        self._max_tries = max_tries
        self._reestimate = reestimate

        if merge is not None and budget is None:
            raise ValueError("merge requires a budget (the stopping backstop)")
        if merge is None and callable(value):
            raise ValueError(
                "a callable value requires merge (repeated sampling); routing takes a constant dollar value"
            )
        if max_tries is None and budget is None:
            raise ValueError("max_tries=None requires a budget")

    # ── Spawnable ──

    def to_thread(self) -> EconomicThread[P, T]:
        """Produce a fresh live thread that runs one search per ``run`` call."""
        return EconomicThread(self)

    @property
    def input_shape(self) -> InputShape:
        """Input shape of the candidates' shared task signature."""
        return next(iter(self._candidates.values())).fn.input_shape

    @property
    def name(self) -> str:
        """The wrapped function's name; for telemetry, errors, and graph nodes.

        Deliberately identical to the plain function's: a traced economic run
        reconstructs to a node named after the function it routes, so a
        backward pass reads ``research-a1b2``, not an implementation detail.
        """
        return next(iter(self._candidates.values())).fn.name

    # ── ParameterHost (optimizer bridge) ──
    #
    # An economic function is its own parameter host: each run emits its routing
    # decision as a grad-enabled parameter, and the score the backward pass
    # assigns it settles the run's records (E2). Pass the function itself into
    # ``TextGradOptimizer.step(backends=[...])`` alongside any memory backend.
    # The four methods below are the whole ``ParameterHost`` contract; they are
    # optimizer machinery, not part of the calling surface.

    @property
    def backend_id(self) -> str:
        """Stable id matched against this function's decision recall events.

        Derived from the function name, so it is stable across processes: a
        fresh function over beliefs reloaded from persisted stats settles the
        same record ids.
        """
        return f"economics:{self.name}"

    def deserialize_value(self, name: str, raw: Any) -> Any:
        """Return the decision context verbatim — it is already display text."""
        del name
        return raw

    def _is_procedural(self, name: str) -> bool:
        """The decision parameter is never code."""
        del name
        return False

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
        del name
        scores = [g.score for g in feedback if g.score is not None]
        if not scores or not retrieved:
            return
        settled = min(max(sum(scores) / len(scores), 0.0), 1.0)
        for record_id in retrieved:
            self._beliefs.settle(RecordId(record_id), settled)

    # ── Standalone execution ──

    async def spawn(self) -> ThreadHandle[P, T]:
        """Spawn on a private in-process worker and return the handle."""
        from ...runtime import InMemoryCoordinator
        from ...runtime.worker import LocalWorker

        coord = InMemoryCoordinator()
        worker = LocalWorker(coord)
        await worker.register()
        return await coord.spawn(self)

    async def _spawn_in_context(self) -> ThreadHandle[P, T]:
        """Spawn for a one-shot call, reusing the ambient thread scope if set."""
        from ...types import current_thread_scope

        scope = current_thread_scope()
        if scope is None:
            return await self.spawn()
        caller = await scope.coordinator.get_thread_info(scope.thread_id)
        return await scope.coordinator.spawn(self, worker_id=caller.worker_id, parent_id=scope.thread_id)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run one search over the candidates and return the best passing result.

        Raises:
            Abstained: No candidate had positive net value at decision time (E5).
            CandidatesExhausted: Every profitable candidate ran and none passed.
            BudgetExceeded: The next attempt did not fit the remaining budget.
        """
        handle = await self._spawn_in_context()
        try:
            return await handle.run(*args, **kwargs)
        finally:
            await handle.terminate_now()

    def run_sync(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Blocking wrapper around ``__call__``; mirrors ``AIFunction.run_sync``."""
        from ...utils import run_blocking

        return run_blocking(lambda: self(*args, **kwargs))

    async def trace(self, *args: Any, **kwargs: Any) -> Result[T]:
        """Run like ``__call__`` and return a traced ``Result`` node.

        The traced thread's event log (decisions and attempts) survives for
        post-hoc inspection and graph reconstruction. Arguments may be
        ``Result`` / ``ParameterView`` handles, as with ``AIFunction.trace``.
        """
        from ...types.graph import ParameterView, Result, collect_nodes

        inputs = collect_nodes((args, kwargs))
        handle = await self._spawn_in_context()
        try:
            for node in inputs:
                if isinstance(node, ParameterView):
                    await node.backend.emit_recall(node, handle.coordinator, handle.id)
            value = await handle.run(*args, **kwargs)
        finally:
            await handle.terminate_now()
        return Result(value=value, coordinator=handle.coordinator, thread_id=handle.id, inputs=inputs)

    async def plan(self, *args: P.args, **kwargs: P.kwargs) -> Decision:
        """Decide without executing: one estimation round, no attempts.

        For callers that want the routing decision but own the execution
        (report back via :meth:`Decision.report`), or want to anticipate
        abstention without an exception.
        """
        candidates = list(self._candidates.values())
        thread = self.to_thread()
        task = await thread._task_view(candidates[0], args, kwargs)
        estimates = await thread._estimate(self, task, candidates, history=[])

        search = Search(estimates, budget=self._budget, policy=self._policy, max_tries=self._max_tries)
        label = search.next()
        ranking = search.explain()
        chosen = self._candidates[label] if label is not None else None
        chosen_estimate = estimates[label] if label is not None else None
        decision = Decision(candidate=chosen, ranking=ranking)
        object.__setattr__(decision, "_report_hook", self._make_reporter(chosen, task, chosen_estimate))
        return decision

    def _make_reporter(
        self, candidate: Candidate[P, T] | None, task: TaskView, estimate: Estimate | None
    ) -> Callable[[Any, float | None], None]:
        """Build the closure ``Decision.report`` calls to book an external attempt."""
        # Unique per plan() call, like the run path's thread-id prefix:
        plan_id = uuid.uuid4().hex[:12]
        counter = [0]

        def _report(result: Any, cost: float | None) -> None:
            assert candidate is not None and estimate is not None  # guarded by Decision.report
            passed = result is not None
            reward = _value_of(self._value, result) if passed else 0.0
            counter[0] += 1
            record = AttemptRecord(
                id=RecordId(f"plan-{plan_id}-{counter[0]}"),
                task=task,
                candidate=candidate.label,
                cost=cost if cost is not None else estimate.cost,
                reward=reward,
                local_score=1.0 if passed else 0.0,
            )
            self._beliefs.update(record)

        return _report

    # ── Introspection ──

    @property
    def candidates(self) -> Mapping[str, Candidate[P, T]]:
        """The configured candidates, keyed by label."""
        return dict(self._candidates)

    @property
    def beliefs(self) -> Beliefs:
        """The live beliefs provider (shared, stateful)."""
        return self._beliefs

    def replace(self, **changes: Any) -> EconomicFunction[P, T]:
        """Return a new economic function with constructor fields overridden.

        ``self`` is unchanged; ``beliefs`` is shared, not copied, unless
        explicitly replaced.
        """
        kwargs: dict[str, Any] = {
            "candidates": self._candidates,
            "value": self._value,
            "beliefs": self._beliefs,
            "budget": self._budget,
            "policy": self._policy,
            "max_tries": self._max_tries,
            "reestimate": self._reestimate,
            "carry_context": self._carry_context,
            "merge": self._merge,
        }
        kwargs.update(changes)
        return EconomicFunction(**kwargs)


# ── Routing-summary rendering ─────────────────────────────────────


def _render_routing_summary(
    func_name: str,
    decisions: list[dict[str, Any]],
    records: list[AttemptRecord],
    responsible: str,
) -> str:
    """Render the decision rounds and attempt outcomes as a backward-pass preamble.

    The text the reconstructed economic node carries ahead of the responsible
    attempt's conversation: what was estimated per candidate, which candidates
    ran and how they fared, and which one produced the returned output. Kept
    self-contained (no trailing "conversation follows") so it reads correctly
    whether or not a delegated conversation is spliced after it.
    """
    lines = [
        f"[Cost-aware routing for {func_name or 'this function'}]",
        "A router chose which candidate model would attempt this task, based on "
        "per-candidate predictions and past statistics:",
    ]
    for i, payload in enumerate(decisions, 1):
        prefix = f"Estimation round {i}, per candidate" if len(decisions) > 1 else "Per candidate"
        lines.append(f"{prefix}:")
        estimates = payload.get("estimates")
        estimates = estimates if isinstance(estimates, dict) else {}
        stats = payload.get("stats")
        stats = stats if isinstance(stats, dict) else {}
        ranking = payload.get("ranking")
        ranking = ranking if isinstance(ranking, list) else []
        net_values = {r.get("label"): r.get("net_value") for r in ranking if isinstance(r, dict)}
        order = list(net_values)
        for label in [*order, *(lb for lb in estimates if lb not in order)]:
            est = estimates.get(label)
            est = est if isinstance(est, dict) else {}
            parts: list[str] = []
            if "expected_reward" in est:
                parts.append(f"predicted expected reward ${float(est['expected_reward']):.4f}")
            if "cost" in est:
                parts.append(f"estimated cost ${float(est['cost']):.4f}")
            parts.append(f"track record: {stats.get(label) or 'no attempts yet'}")
            net = net_values.get(label)
            if isinstance(net, (int, float)) and net < 0:
                parts.append("judged not worth attempting (expected cost exceeds expected reward)")
            lines.append(f"  {label}: {'; '.join(parts)}")
    for i, record in enumerate(records, 1):
        passed = record.local_score > 0
        line = (
            f"Attempt {i}: {record.candidate} — "
            f"{'passed local checks' if passed else 'failed local checks'}, "
            f"cost ${record.cost:.4f}"
        )
        if record.turns > 0:
            line += f", {record.turns} turns, {record.usage.output_tokens} output tokens"
        lines.append(line)
    lines.append(f"The returned output was produced by: {responsible}")
    return "\n".join(lines)


# ── Event-log readers ─────────────────────────────────────────────


def _events_source(run: ThreadHandle[..., Any] | Result[Any]) -> tuple[Coordinator, Any]:
    """Extract ``(coordinator, thread_id)`` from a handle or a traced ``Result``."""
    from ...handle import ThreadHandle
    from ...types.graph import Result

    if isinstance(run, ThreadHandle):
        return run.coordinator, run.id
    if isinstance(run, Result):
        return run.coordinator, run.thread_id
    raise TypeError(f"expected a ThreadHandle or Result, got {type(run).__name__}")


async def attempts(run: ThreadHandle[..., Any] | Result[Any]) -> list[AttemptRecord]:
    """Read the booked records from an economic function run's event log.

    Args:
        run: A handle to a spawned economic function thread, or the
            ``Result`` of ``EconomicFunction.trace``.

    Returns:
        The run's ``AttemptRecord``s in booking order.
    """
    coordinator, thread_id = _events_source(run)
    events = await coordinator.get_events(thread_id)
    out: list[AttemptRecord] = []
    for e in events:
        if isinstance(e, CustomEvent) and e.kind == ATTEMPT_EVENT:
            raw = e.payload.get("record")
            if isinstance(raw, dict):
                out.append(AttemptRecord.model_validate(raw))
    return out


async def decisions(run: ThreadHandle[..., Any] | Result[Any]) -> list[list[Ranking]]:
    """Read the ranked estimation rounds from the run's event log.

    Args:
        run: A handle to a spawned economic function thread, or the
            ``Result`` of ``EconomicFunction.trace``.

    Returns:
        One ranking per estimation round, in chronological order.
    """
    coordinator, thread_id = _events_source(run)
    events = await coordinator.get_events(thread_id)
    out: list[list[Ranking]] = []
    for e in events:
        if isinstance(e, CustomEvent) and e.kind == DECISION_EVENT:
            raw = e.payload.get("ranking", [])
            rows = raw if isinstance(raw, list) else []
            out.append(
                [
                    Ranking(
                        label=str(r["label"]),
                        reservation_price=float(r["reservation_price"]),
                        net_value=float(r.get("net_value", 0.0)),
                    )
                    for r in rows
                    if isinstance(r, dict)
                ]
            )
    return out


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
    coordinator, thread_id = _events_source(run)
    total = 0.0
    visited: set[Any] = set()

    async def _fold(tid: Any) -> None:
        nonlocal total
        if tid in visited:
            return
        visited.add(tid)
        events = await coordinator.get_events(tid)
        for e in events:
            if isinstance(e, CustomEvent) and e.kind == ATTEMPT_EVENT:
                raw = e.payload.get("record")
                if isinstance(raw, dict):
                    total += float(raw.get("cost", 0.0))
            elif getattr(e, "kind", None) == EventKind.THREAD_SPAWNED:
                await _fold(e.child_thread_id)

    await _fold(thread_id)
    return total
