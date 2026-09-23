"""Pure sequential-search core: reservation prices over labeled estimates.

This layer knows nothing about ``AIFunction``, threads, or money sources —
it is a decision calculator over ``{label: RewardCostEstimate}``. The
:class:`~.function.EconomicFunction` (and the thread it spawns) binds labels to
executable candidates; tests bind them to closed-form optima. Power users
import from here; the top-level package exports only the decorator path.

The rule implemented by :class:`Search` under :class:`ReservationPricePolicy`
is Weitzman's Pandora's box rule, optimal for independent alternatives:

- Each estimate has a *reservation price* ``g``, the solution of
  ``E[(R - g)_+] = cost``: the reward in hand at which trying this
  candidate is exactly break-even.
- Try candidates in descending ``g``; stop as soon as the best remaining
  ``g`` does not exceed the best reward already realized.

``R`` is a priced score: a :class:`ScoreDistribution` on ``[0, 1]`` becomes a
:class:`RewardDistribution` once multiplied by a ``value``.

Invariants:
    E1 — rewards, costs, budgets, and reservation prices are all dollars.

    E3 — ``Search`` is deterministic and synchronous: identical construction
    and an identical observe-sequence yield identical decisions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .types import Ranking


# ── Score distributions ──

SUPPORT_TOL: float
"""Tolerance for the ``[0, 1]`` support check on a score distribution."""


def bisect_reservation_index(
    score_dist: ScoreDistribution,
    normalized_cost: float,
    *,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> float:
    """Solve ``E[(S - k)_+] = normalized_cost`` for the reservation index ``k``.

    The generic solver behind ``ScoreDistribution.reservation_index`` for every
    distribution without a closed form. ``expected_improvement`` is
    non-increasing in ``k``, so the root is unique. All quantities, including
    ``normalized_cost``, are in score units.

    Args:
        score_dist: The score distribution to solve for.
        normalized_cost: One attempt's cost divided by ``value``, in score units.
        tol: Convergence tolerance on ``E[(S - k)_+] - normalized_cost``.
        max_iter: Maximum bisection iterations.

    Returns:
        The reservation index ``k``, in ``(-inf, 1]``; ``+inf`` when
        ``normalized_cost <= 0`` (a free attempt is always worth making).
    """
    ...


class ScoreDistribution(ABC):
    """Estimated distribution of an attempt's score, before running it.

    Support is ``[0, 1]``.
    """

    @abstractmethod
    def expected_improvement(self, current: float) -> float:
        """Return ``E[(S - current)_+]``, the expected gain over a score in hand.

        Args:
            current: The best score already realized, in ``[0, 1]``.

        Returns:
            Expected improvement in score units; non-negative and
            non-increasing in ``current``.
        """
        ...

    @abstractmethod
    def mean(self) -> float:
        """Return ``E[S]``, in ``[0, 1]``."""
        ...

    def reservation_index(self, normalized_cost: float) -> float:
        """Solve ``E[(S - k)_+] = normalized_cost``; see :func:`bisect_reservation_index`.

        Subclasses with a closed form override this (:class:`Bernoulli`).

        Args:
            normalized_cost: One attempt's cost divided by ``value``, in score units.

        Returns:
            The reservation index ``k``; ``+inf`` when ``normalized_cost <= 0``.
        """
        ...


@dataclass(frozen=True)
class Bernoulli(ScoreDistribution):
    """Two-point score: ``1.0`` with probability ``p``, else ``0.0``.

    Args:
        p: Probability of success, in ``[0, 1]``.

    Raises:
        ValueError: ``p`` outside ``[0, 1]``.
    """

    p: float

    def __post_init__(self) -> None:
        """Validate the ``p`` range."""
        ...

    def reservation_index(self, normalized_cost: float) -> float:
        """Closed form: ``k = 1 - c / p`` when ``c < p``, else ``k = p - c``.

        ``c`` is the normalized cost. The second branch is the ``k <= 0`` case
        and also covers ``p == 0``.
        """
        ...


@dataclass(frozen=True)
class Categorical(ScoreDistribution):
    """Discrete score over ``scores`` with probabilities ``probs``.

    Args:
        scores: Score outcomes; each must lie in ``[0, 1]``.
        probs: Probability of each outcome; same length as ``scores``,
            non-negative, summing to 1.

    Raises:
        ValueError: Length mismatch, a score outside ``[0, 1]``, a negative
            probability, or probs sum != 1.
    """

    scores: tuple[float, ...]
    probs: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate lengths, the ``[0, 1]`` score range, and sum-to-one."""
        ...


# ── Score estimate (what beliefs return) ──

@dataclass(frozen=True)
class ScoreCostEstimate:
    """What a :class:`~.beliefs.Beliefs` provider returns, before pricing.

    The economic function prices it at its ``value`` to get a
    :class:`RewardCostEstimate`, the form :class:`Search` consumes (E1).

    Args:
        score_dist: Estimated distribution of one attempt's score, on
            ``[0, 1]``.
        cost: Expected dollar cost of one attempt.

    Raises:
        ValueError: ``cost`` negative, or ``score_dist`` puts mass outside
            ``[0, 1]``.
    """

    score_dist: ScoreDistribution
    cost: float

    def __post_init__(self) -> None:
        """Validate ``cost`` and the score support."""
        ...


def check_score_support(score_dist: ScoreDistribution) -> None:
    """Raise if ``score_dist`` puts mass outside ``[0, 1]``.

    Read off the two methods every score distribution already has, so custom
    subclasses are covered without support introspection:

    - ``E[(S - 1)_+] == 0`` iff no mass above 1.
    - ``E[(S - 0)_+] == E[S]`` iff no mass below 0, since the left side is
      ``E[max(S, 0)]``.

    Raises:
        ValueError: Mass above 1, or mass below 0.
    """
    ...


# ── Reward distribution (the dollar boundary) ──

@dataclass(frozen=True)
class RewardDistribution:
    """A score distribution priced in dollars: ``R = value * S``.

    The module's unit boundary (E1). Rescaling is exact for a positive
    ``value``: ``E[(vS - g)_+] == v * E[(S - g/v)_+]``.

    Args:
        score_dist: Estimated distribution of one attempt's score, on ``[0, 1]``.
        value: Dollars a fully-successful (score 1.0) result is worth;
            positive.

    Raises:
        ValueError: ``value`` non-positive, or ``score_dist`` puts mass outside
            ``[0, 1]``.
    """

    score_dist: ScoreDistribution
    value: float

    def __post_init__(self) -> None:
        """Validate ``value`` and the score support."""
        ...

    def expected_improvement(self, current: float) -> float:
        """Return ``E[(R - current)_+]`` in dollars, the gain over a reward in hand.

        Args:
            current: The best reward already realized, in dollars.

        Returns:
            Expected improvement in dollars; non-negative and non-increasing
            in ``current``.
        """
        ...

    def mean(self) -> float:
        """Return ``E[R] = value * E[S]``, in dollars."""
        ...

    def reservation_price(self, cost: float) -> float:
        """Solve ``E[(R - g)_+] = cost`` for ``g``, in dollars.

        Normalizes the cost by ``value``, solves in score units via
        :meth:`~ScoreDistribution.reservation_index`, and scales back.

        Args:
            cost: Expected dollar cost of one attempt.

        Returns:
            The reservation price ``g`` in dollars; ``+inf`` when ``cost <= 0``.
        """
        ...


# ── RewardCostEstimate ──

@dataclass(frozen=True)
class RewardCostEstimate:
    """One candidate's estimated economics for one task: reward distribution plus cost.

    A :class:`ScoreCostEstimate` once the economic function has priced it.

    Args:
        reward_dist: Estimated distribution of the dollar reward of one attempt.
        cost: Expected dollar cost of one attempt.

    Raises:
        ValueError: ``cost`` negative.
    """

    reward_dist: RewardDistribution
    cost: float

    def __post_init__(self) -> None:
        """Validate ``cost`` is non-negative."""
        ...

    def reservation_price(self) -> float:
        """Solve ``E[(R - g)_+] = cost`` for ``g``.

        Returns:
            The reservation price in dollars: ``+inf`` when ``cost == 0``,
            below ``reward_dist.mean()`` when the cost is high. Exact when the
            underlying score distribution has a closed-form
            :meth:`~ScoreDistribution.reservation_index`
            (:class:`Bernoulli`), bisected otherwise.
        """
        ...

    def net_value(self) -> float:
        """Return ``E[R] - cost``: the myopic value of a single committed attempt.

        The correct metric for one-shot routing, where no option to continue
        exists; :meth:`reservation_price` additionally prices that option in.
        """
        ...


# ── Policy ──

class Policy(ABC):
    """Ordering-and-stopping rule consulted by :class:`Search`."""

    @abstractmethod
    def next(
        self,
        estimates: dict[str, RewardCostEstimate],
        best: float,
        remaining_budget: float | None,
    ) -> str | None:
        """Pick the next label to try, or ``None`` to stop.

        Args:
            estimates: Current estimate per not-yet-exhausted label.
            best: Best dollar reward realized so far (0.0 before any success).
            remaining_budget: Dollars left to spend, or ``None`` for no cap.

        Returns:
            The chosen label, or ``None`` when no candidate is worth its cost.

        Requires:
            ``estimates`` contains only labels still eligible to run.
        """
        ...


class ReservationPricePolicy(Policy):
    """Weitzman's rule: highest reservation price above ``best``, else stop.

    Skips candidates whose expected cost exceeds the remaining budget.
    Optimal for independent candidates; the default policy.
    """


class Greedy(Policy):
    """Highest net value above zero; stop as soon as anything succeeds.

    Ranks candidates by ``net_value()``, the expected reward of one attempt
    minus its cost. On failure it tries the next-best candidate.
    """


class Cheapest(Policy):
    """Lowest cost first, escalating on failure until one succeeds."""


# ── Search ──

class Search:
    """Mutable state of one sequential search over labeled estimates.

    The caller owns the loop: ask :meth:`next` which label to try, run the
    attempt however it likes, report the outcome with :meth:`observe`, and
    repeat. Estimates may be replaced between rounds via
    :meth:`update_estimates` (re-estimation).

    Args:
        estimates: Initial estimate per label. Labels are opaque to the search.
        budget: Optional hard cap on total observed cost, in dollars.
        policy: Ordering-and-stopping rule; defaults to
            :class:`ReservationPricePolicy`, Weitzman's rule and the optimal
            one for independent candidates.
        max_tries: Attempts allowed per label; ``None`` = unbounded (the
            policy's stopping rule is the only limit).

    Raises:
        ValueError: Empty ``estimates``, or a negative ``budget``.
    """

    def __init__(
        self,
        estimates: dict[str, RewardCostEstimate],
        budget: float | None = None,
        policy: Policy | None = None,
        max_tries: int | None = 1,
    ) -> None: ...

    def next(self) -> str | None:
        """Return the label to try next, or ``None`` when the search should stop.

        Delegates to the policy over the labels still eligible (tries
        remaining, expected cost within budget).

        Ensures:
            Repeated calls without an intervening :meth:`observe` or
            :meth:`update_estimates` return the same label (E3).
        """
        ...

    def blocked_by_budget(self) -> bool:
        """Whether :meth:`next` stopped only because the budget is too small.

        ``True`` when a candidate would still be tried on unlimited budget but
        every such candidate's expected cost exceeds the remaining budget —
        i.e. the search is not done on its own terms, it merely ran out of
        money. Lets the economic function distinguish ``BudgetExceeded`` from a genuine
        stop or exhaustion.
        """
        ...

    def observe(self, label: str, reward: float, cost: float) -> None:
        """Record the outcome of one attempt.

        Args:
            label: The label returned by :meth:`next`.
            reward: Realized dollar reward (0.0 for a failed attempt).
            cost: Dollars actually spent on the attempt.

        Ensures:
            - ``spent`` grows by ``cost``; ``best`` is ``max(best, reward)``.
            - The label's remaining tries decrease by one.

        Raises:
            KeyError: ``label`` is not one of the search's labels.
        """
        ...

    def update_estimates(self, estimates: dict[str, RewardCostEstimate]) -> None:
        """Replace the estimates consulted by subsequent :meth:`next` calls.

        Args:
            estimates: New estimate per label; labels must be a subset of
                the construction-time labels.

        Raises:
            KeyError: An estimate names a label not present at construction.
        """
        ...

    @property
    def best(self) -> float:
        """Best dollar reward observed so far; 0.0 before any success."""
        ...

    @property
    def spent(self) -> float:
        """Total dollars observed as cost so far."""
        ...

    @property
    def remaining_budget(self) -> float | None:
        """``budget - spent``, or ``None`` when constructed without a budget."""
        ...

    def explain(self) -> list[Ranking]:
        """Return the eligible labels with their reservation prices, ranked.

        The transparency hook: what the search believes right now, in the
        order it would try things. Intended for logging, event payloads,
        and ``Decision.ranking``.
        """
        ...
