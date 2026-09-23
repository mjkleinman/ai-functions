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

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .types import Ranking

# ── Score distributions ───────────────────────────────────────────

SUPPORT_TOL = 1e-9
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
    if normalized_cost <= 0:
        return math.inf
    mean = score_dist.mean()
    if normalized_cost >= mean:
        # k <= 0 <= S, so (S - k)_+ never truncates: E[(S - k)_+] = E[S] - k.
        return mean - normalized_cost
    lo, hi = 0.0, 1.0  # scores are between [0, 1]
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        ei = score_dist.expected_improvement(mid)
        if abs(ei - normalized_cost) < tol:
            return mid
        if ei > normalized_cost:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


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
        return bisect_reservation_index(self, normalized_cost)


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
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")

    def expected_improvement(self, current: float) -> float:
        """E[(S - current)_+] for a two-point score on ``{0, 1}``."""
        if current >= 1.0:
            return 0.0
        if current <= 0:
            # Both outcomes clear ``current``: E[S] - current.
            return self.p - current
        # Only the success outcome clears ``current``.
        return self.p * (1.0 - current)

    def mean(self) -> float:
        """E[S] = p."""
        return self.p

    def reservation_index(self, normalized_cost: float) -> float:
        """Closed form: ``k = 1 - c / p`` when ``c < p``, else ``k = p - c``.

        ``c`` is the normalized cost. The second branch is the ``k <= 0`` case
        and also covers ``p == 0``.
        """
        if normalized_cost <= 0:
            return math.inf
        if normalized_cost >= self.p:
            return self.p - normalized_cost
        return 1.0 - normalized_cost / self.p


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
        if len(self.scores) != len(self.probs):
            raise ValueError(
                f"scores and probs must have the same length, got {len(self.scores)} and {len(self.probs)}"
            )
        if any(not 0.0 <= s <= 1.0 for s in self.scores):
            raise ValueError(f"all scores must be in [0, 1], got {self.scores}")
        if any(p < 0 for p in self.probs):
            raise ValueError("all probs must be non-negative")
        if abs(sum(self.probs) - 1.0) > 1e-9:
            raise ValueError(f"probs must sum to 1, got {sum(self.probs)}")

    def expected_improvement(self, current: float) -> float:
        """E[(S - current)_+] for a discrete score."""
        return sum(p * (s - current) for s, p in zip(self.scores, self.probs, strict=True) if s > current)

    def mean(self) -> float:
        """E[S] = sum(score * probability)."""
        return sum(s * p for s, p in zip(self.scores, self.probs, strict=True))


# ── Score estimate (what beliefs return) ──────────────────────────


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
        if self.cost < 0:
            raise ValueError(f"cost must be non-negative, got {self.cost}")
        check_score_support(self.score_dist)


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
    if score_dist.expected_improvement(1.0) > SUPPORT_TOL:
        raise ValueError(f"score distribution {score_dist!r} puts mass above 1; scores live in [0, 1]")
    if abs(score_dist.expected_improvement(0.0) - score_dist.mean()) > SUPPORT_TOL:
        raise ValueError(f"score distribution {score_dist!r} puts mass below 0; scores live in [0, 1]")


# ── Reward distribution (the dollar boundary) ─────────────────────


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
        if self.value <= 0:
            raise ValueError(f"value must be positive dollars, got {self.value}")
        check_score_support(self.score_dist)

    def expected_improvement(self, current: float) -> float:
        """Return ``E[(R - current)_+]`` in dollars, the gain over a reward in hand.

        Args:
            current: The best reward already realized, in dollars.

        Returns:
            Expected improvement in dollars; non-negative and non-increasing
            in ``current``.
        """
        return self.value * self.score_dist.expected_improvement(current / self.value)

    def mean(self) -> float:
        """Return ``E[R] = value * E[S]``, in dollars."""
        return self.value * self.score_dist.mean()

    def reservation_price(self, cost: float) -> float:
        """Solve ``E[(R - g)_+] = cost`` for ``g``, in dollars.

        Normalizes the cost by ``value``, solves in score units via
        :meth:`~ScoreDistribution.reservation_index`, and scales back.

        Args:
            cost: Expected dollar cost of one attempt.

        Returns:
            The reservation price ``g`` in dollars; ``+inf`` when ``cost <= 0``.
        """
        if cost <= 0:
            return math.inf
        return self.value * self.score_dist.reservation_index(cost / self.value)


# ── RewardCostEstimate ──────────────────────────────────────────────────────


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
        if self.cost < 0:
            raise ValueError(f"cost must be non-negative, got {self.cost}")

    def reservation_price(self) -> float:
        """Solve ``E[(R - g)_+] = cost`` for ``g``.

        Returns:
            The reservation price in dollars: ``+inf`` when ``cost == 0``,
            below ``reward_dist.mean()`` when the cost is high. Exact when the
            underlying score distribution has a closed-form
            :meth:`~ScoreDistribution.reservation_index`
            (:class:`Bernoulli`), bisected otherwise.
        """
        return self.reward_dist.reservation_price(self.cost)

    def net_value(self) -> float:
        """Return ``E[R] - cost``: the myopic value of a single committed attempt.

        The correct metric for one-shot routing, where no option to continue
        exists; :meth:`reservation_price` additionally prices that option in.
        """
        return self.reward_dist.mean() - self.cost


# ── Policy ────────────────────────────────────────────────────────


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


def _affordable(
    estimates: dict[str, RewardCostEstimate], remaining_budget: float | None
) -> dict[str, RewardCostEstimate]:
    """Drop candidates whose expected cost exceeds the remaining budget."""
    if remaining_budget is None:
        return estimates
    return {label: e for label, e in estimates.items() if e.cost <= remaining_budget}


class ReservationPricePolicy(Policy):
    """Weitzman's rule: highest reservation price above ``best``, else stop.

    Skips candidates whose expected cost exceeds the remaining budget.
    Optimal for independent candidates; the default policy.
    """

    def next(self, estimates: dict[str, RewardCostEstimate], best: float, remaining_budget: float | None) -> str | None:
        """Return the candidate within budget with the highest reservation price above ``best``, else ``None``."""
        affordable = _affordable(estimates, remaining_budget)
        if not affordable:
            return None
        label = max(affordable, key=lambda k: affordable[k].reservation_price())
        return label if affordable[label].reservation_price() > best else None


class Greedy(Policy):
    """Highest net value above zero; stop as soon as anything succeeds.

    Ranks candidates by ``net_value()``, the expected reward of one attempt
    minus its cost. On failure it tries the next-best candidate.
    """

    def next(self, estimates: dict[str, RewardCostEstimate], best: float, remaining_budget: float | None) -> str | None:
        """Return the candidate within budget with the highest net value if it is positive, else ``None``."""
        if best > 0:
            return None
        affordable = _affordable(estimates, remaining_budget)
        if not affordable:
            return None
        label = max(affordable, key=lambda k: affordable[k].net_value())
        return label if affordable[label].net_value() > 0 else None


class Cheapest(Policy):
    """Lowest cost first, escalating on failure until one succeeds."""

    def next(self, estimates: dict[str, RewardCostEstimate], best: float, remaining_budget: float | None) -> str | None:
        """Return the cheapest candidate within budget, or ``None`` once anything has succeeded."""
        if best > 0:
            return None
        affordable = _affordable(estimates, remaining_budget)
        if not affordable:
            return None
        return min(affordable, key=lambda k: affordable[k].cost)


# ── Search ────────────────────────────────────────────────────────


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
    ) -> None:
        if not estimates:
            raise ValueError("Search requires at least one labeled estimate")
        if budget is not None and budget < 0:
            raise ValueError(f"budget must be non-negative, got {budget}")
        self._estimates: dict[str, RewardCostEstimate] = dict(estimates)
        self._labels: frozenset[str] = frozenset(estimates)
        self._budget: float | None = budget
        self._policy: Policy = policy if policy is not None else ReservationPricePolicy()
        self._max_tries: int | None = max_tries
        self._tries: dict[str, int] = {}
        self._best: float = 0.0
        self._spent: float = 0.0

    def _eligible(self) -> dict[str, RewardCostEstimate]:
        """Estimates for labels that still have tries left."""
        if self._max_tries is None:
            return dict(self._estimates)
        return {label: e for label, e in self._estimates.items() if self._tries.get(label, 0) < self._max_tries}

    def next(self) -> str | None:
        """Return the label to try next, or ``None`` when the search should stop.

        Delegates to the policy over the labels still eligible (tries
        remaining, expected cost within budget).

        Ensures:
            Repeated calls without an intervening :meth:`observe` or
            :meth:`update_estimates` return the same label (E3).
        """
        eligible = self._eligible()
        if not eligible:
            return None
        return self._policy.next(eligible, self._best, self.remaining_budget)

    def blocked_by_budget(self) -> bool:
        """Whether :meth:`next` stopped only because the budget is too small.

        ``True`` when a candidate would still be tried on unlimited budget but
        every such candidate's expected cost exceeds the remaining budget —
        i.e. the search is not done on its own terms, it merely ran out of
        money. Lets the economic function distinguish ``BudgetExceeded`` from a genuine
        stop or exhaustion.
        """
        eligible = self._eligible()
        if not eligible or self.remaining_budget is None:
            return False
        # Would the unbudgeted policy still pick something here?
        wanted = self._policy.next(eligible, self._best, None)
        return wanted is not None and self._policy.next(eligible, self._best, self.remaining_budget) is None

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
        if label not in self._labels:
            raise KeyError(f"unknown label {label!r}; search labels are {sorted(self._labels)}")
        self._tries[label] = self._tries.get(label, 0) + 1
        self._spent += cost
        self._best = max(self._best, reward)

    def update_estimates(self, estimates: dict[str, RewardCostEstimate]) -> None:
        """Replace the estimates consulted by subsequent :meth:`next` calls.

        Args:
            estimates: New estimate per label; labels must be a subset of
                the construction-time labels.

        Raises:
            KeyError: An estimate names a label not present at construction.
        """
        unknown = set(estimates) - self._labels
        if unknown:
            raise KeyError(f"unknown label(s) {sorted(unknown)}; search labels are {sorted(self._labels)}")
        self._estimates = dict(estimates)

    @property
    def best(self) -> float:
        """Best dollar reward observed so far; 0.0 before any success."""
        return self._best

    @property
    def spent(self) -> float:
        """Total dollars observed as cost so far."""
        return self._spent

    @property
    def remaining_budget(self) -> float | None:
        """``budget - spent``, or ``None`` when constructed without a budget."""
        if self._budget is None:
            return None
        return self._budget - self._spent

    def explain(self) -> list[Ranking]:
        """Return the eligible labels with their reservation prices, ranked.

        The transparency hook: what the search believes right now, in the
        order it would try things. Intended for logging, event payloads,
        and ``Decision.ranking``.
        """
        eligible = self._eligible()
        rankings = [
            Ranking(label=label, reservation_price=e.reservation_price(), net_value=e.net_value())
            for label, e in eligible.items()
        ]
        rankings.sort(key=lambda r: r.reservation_price, reverse=True)
        return rankings
