"""``@routed`` and ``@economic`` — the two entry points to the economics module.

Both stack on ``@ai_function`` and return an
:class:`~.function.EconomicFunction` that is called exactly like the
function it wraps. They are two doors to one machine: ``@routed`` decides
*which* candidate attempts the task (model routing with fallback and
abstention); ``@economic`` decides *how many times* the candidates should
try (repeated sampling with automatic stopping — one arm or several over
the same merged pool).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from .beliefs import Beliefs
from .function import EconomicFunction
from .search import Policy
from .types import Candidate, PricedModel

if TYPE_CHECKING:
    from ...ai_thread.ai_function import AIFunction


def routed[**P, T](
    *,
    value: float,
    models: list[PricedModel] | None = None,
    candidates: list[Candidate[P, T]] | None = None,
    beliefs: Beliefs | None = None,
    budget: float | None = None,
    policy: Policy | None = None,
    max_tries: int | None = 1,
    carry_context: bool = False,
) -> Callable[[AIFunction[P, T]], EconomicFunction[P, T]]:
    """Route each call to the candidate worth running; escalate on failure.

    One call = one search across the candidates: try the highest net value
    first, escalate while failure leaves a profitable option, decline
    (:class:`~.types.Abstained`) when no candidate's expected reward covers
    its cost.

    Args:
        value: Dollars a passing result is worth — a positive constant. It is
            the scale estimates are built at and the reward a pass books;
            routing has no per-result grading (a result either passes the
            post-conditions or does not). To value results by content, use
            :func:`economic`, whose callable ``value`` prices merged results.
        models: Priced models to build candidates from, one per entry.
            Exactly one of ``models`` and ``candidates`` must be given.
        candidates: Explicit candidates, for variants beyond model swaps.
        beliefs: Estimate/learn provider. Defaults to a fresh
            :class:`~.beliefs.EmpiricalBeliefs`.
        budget: Hard dollar cap per call.
        policy: Search policy; defaults to ``Greedy`` (highest net value,
            stop once a reward is in hand). Pass ``ReservationPricePolicy()``
            for Weitzman-style escalation by reservation price.
        max_tries: Attempts per candidate per call; ``None`` = unbounded
            (requires ``budget``).
        carry_context: Seed each escalation from the prior attempt's
            transcript, so the next candidate sees what was tried and why it
            was rejected instead of starting fresh.

    Returns:
        A decorator producing the configured ``EconomicFunction``.

    Raises:
        ValueError: A callable or non-positive ``value``; both or neither of
            ``models``/``candidates`` given; duplicate labels; or
            ``max_tries=None`` without ``budget``.
    """
    ...


def economic[**P, T](
    *,
    value: Callable[[T], float],
    budget: float,
    models: list[PricedModel] | None = None,
    candidates: list[Candidate[P, T]] | None = None,
    merge: Callable[[T, T], T] | None = None,
    beliefs: Beliefs | None = None,
    max_tries: int | None = None,
    reestimate: bool = True,
) -> Callable[[AIFunction[P, T]], EconomicFunction[P, T]]:
    """Sample the candidates repeatedly, folding results, while it keeps paying.

    One call = a sequence of attempts whose passing results fold into a
    running result via ``merge``. Each attempt's reward is its *marginal*
    dollar gain — ``value(merged_after) - value(merged_before)`` — and the
    search runs under ``ReservationPricePolicy``: draw the arm with the
    highest reservation price while one prices above the worth in hand,
    i.e. while some arm's projected gain covers its cost. With one model
    this is repeated sampling with automatic stopping; with several it is
    the multi-arm (Pandora's box) search — when the beliefs price each arm
    separately. The default beliefs project one shared gain curve, so
    supply per-arm ``beliefs`` to differentiate the arms.

    Args:
        value: Dollar worth of the *running* result — a callable pricing it
            (e.g. ``lambda r: 0.02 * len(r.defects)``); the search books the
            differences. A constant is rejected: its marginals are ``$0``
            after the first pass, which would stop the search regardless of
            what remains to be found.
        budget: Hard dollar cap per call. Required: with unbounded tries, the
            budget is the backstop when the stopping projection is wrong.
        models: Priced models to build candidates from, one arm per entry.
            Required even for the wrapped function's own model, because only
            the caller knows its prices. Exactly one of ``models`` and
            ``candidates`` must be given.
        candidates: Explicit candidates, for variants beyond model swaps.
        merge: Fold a passing attempt's result into the running result:
            ``(running, new) -> running``. ``value`` must price whatever
            ``merge`` produces. Defaults to :func:`~.function.keep_best`
            over ``value`` — keep the highest-worth result.
        beliefs: Worth-after-one-more-attempt estimator, answering in
            dollars from the search's history. Defaults to
            :class:`~.beliefs.DiminishingReturns`. Fixed-scale providers
            (:class:`~.beliefs.EmpiricalBeliefs`,
            :class:`~.beliefs.LLMForecaster`) are rejected: they price
            answer correctness at a constant value, which a callable
            ``value`` does not have.
        max_tries: Attempts per candidate per call; ``None`` (the default)
            lets the stopping rule and budget decide. A finite cap is the
            classic Pandora bound: each box opened at most that many times.
        reestimate: Re-run ``beliefs.estimate`` after each attempt. On by
            default because the default beliefs project the next gain from
            the search's history; turn off for task-fixed estimates that
            return the same boxes every round.

    Returns:
        A decorator producing the configured ``EconomicFunction``.

    Raises:
        ValueError: A non-callable ``value``; ``budget`` non-positive; a
            fixed-scale ``beliefs`` provider; both or neither of
            ``models``/``candidates`` given; duplicate labels.
    """
    ...
