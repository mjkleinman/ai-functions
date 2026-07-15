"""Pure sequential-search core: reservation prices over labeled estimates.

This layer knows nothing about ``AIFunction``, threads, or money sources —
it is a decision calculator over ``{label: Estimate}``. The runner layer
binds labels to executable candidates; tests bind them to closed-form
optima. Power users import from here; the top-level package exports only
the decorator path.

The rule implemented by :class:`Search` under :class:`ReservationPricePolicy`
is Weitzman's Pandora's box rule, optimal for independent alternatives:

- Each estimate has a *reservation price* ``g``, the solution of
  ``E[(R - g)_+] = cost``: the reward in hand at which trying this
  candidate is exactly break-even.
- Try candidates in descending ``g``; stop as soon as the best remaining
  ``g`` does not exceed the best reward already realized.

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

# ── Reward distributions ──────────────────────────────────────────


def bisect_reservation_price(
    dist: RewardDistribution,
    cost: float,
    *,
    bound: float = 1e3,
    tol: float = 1e-9,
    max_iter: int = 100,
) -> float:
    """Solve ``E[(R - g)_+] = cost`` for ``g`` by bisection.

    The generic solver behind ``RewardDistribution.reservation_price`` for
    every distribution without a closed form. ``expected_improvement`` is
    non-increasing in ``g``, so the root is unique and bisection converges.

    Args:
        dist: The reward distribution to solve for.
        cost: Dollar cost of one attempt.
        bound: Search bracket ``[-bound, bound]`` in dollars. A root outside
            the bracket is clipped to the nearest endpoint, so distributions
            whose rewards approach this scale need a larger bound for an
            exact price.
        tol: Convergence tolerance on ``E[(R - g)_+] - cost``, in dollars.
        max_iter: Maximum bisection iterations.

    Returns:
        The reservation price ``g`` in dollars, clipped to ``[-bound, bound]``;
        ``+inf`` when ``cost <= 0`` (a free attempt is always worth making).
    """
    if cost <= 0:
        return math.inf
    lo, hi = -bound, bound
    if dist.expected_improvement(lo) < cost:
        return lo
    if dist.expected_improvement(hi) > cost:
        return hi
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        ei = dist.expected_improvement(mid)
        if abs(ei - cost) < tol:
            return mid
        if ei > cost:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class RewardDistribution(ABC):
    """Estimated distribution of an attempt's dollar reward, before running it."""

    @abstractmethod
    def expected_improvement(self, current: float) -> float:
        """Return ``E[(R - current)_+]``, the expected gain over a reward in hand.

        Args:
            current: The best reward already realized, in dollars.

        Returns:
            Expected improvement in dollars; non-negative and non-increasing
            in ``current``.
        """
        ...

    @abstractmethod
    def mean(self) -> float:
        """Return ``E[R]`` in dollars."""
        ...

    def reservation_price(self, cost: float) -> float:
        """Solve ``E[(R - g)_+] = cost`` for ``g``; see :func:`bisect_reservation_price`.

        Delegates to the generic bisection solver with its default controls.
        Subclasses with a closed form override this (:class:`Bernoulli`);
        callers needing non-default solver controls use
        :func:`bisect_reservation_price` directly or override this method
        to bake them into the distribution.

        Args:
            cost: Dollar cost of one attempt.

        Returns:
            The reservation price ``g`` in dollars; ``+inf`` when
            ``cost <= 0`` (a free attempt is always worth making).
        """
        return bisect_reservation_price(self, cost)


@dataclass(frozen=True)
class Bernoulli(RewardDistribution):
    """Two-point reward: ``value`` with probability ``p``, else 0.

    Args:
        p: Probability of success, in ``[0, 1]``.
        value: Dollar reward on success.

    Raises:
        ValueError: ``p`` outside ``[0, 1]`` or ``value`` negative.
    """

    p: float
    value: float

    def __post_init__(self) -> None:
        """Validate ``p`` and ``value`` ranges."""
        if not 0.0 <= self.p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {self.p}")
        if self.value < 0:
            raise ValueError(f"value must be non-negative, got {self.value}")

    def expected_improvement(self, current: float) -> float:
        """E[(R - current)_+] for a two-point reward on ``{0, value}``."""
        if current >= self.value:
            return 0.0
        if current <= 0:
            # Both outcomes clear ``current``: E[R] - current.
            return self.p * self.value - current
        # Only the ``value`` outcome clears ``current``.
        return self.p * (self.value - current)

    def mean(self) -> float:
        """E[R] = p * value."""
        return self.p * self.value

    def reservation_price(self, cost: float) -> float:
        """Closed form for the two-point reward.

        With ``R in {0, value}`` and ``P(R = value) = p``:
        ``E[(R - g)_+] = p * (value - g)`` for ``0 <= g <= value``. Solving
        ``= cost`` gives ``g = value - cost / p``. Outside that band the
        equation degenerates, so fall back to the generic solver.
        """
        if cost <= 0:
            return math.inf
        if self.p <= 0 or self.value <= 0:
            return -cost
        g = self.value - cost / self.p
        if 0.0 <= g <= self.value:
            return g
        return super().reservation_price(cost)


@dataclass(frozen=True)
class Gaussian(RewardDistribution):
    """Normal reward ``R ~ N(mu, sigma^2)``, in dollars.

    Args:
        mu: Mean reward.
        sigma: Standard deviation; must be non-negative.

    Raises:
        ValueError: ``sigma`` negative.
    """

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        """Validate ``sigma`` is non-negative."""
        if self.sigma < 0:
            raise ValueError(f"sigma must be non-negative, got {self.sigma}")

    def expected_improvement(self, current: float) -> float:
        """E[(R - current)_+] for a Gaussian reward."""
        if self.sigma == 0:
            return max(self.mu - current, 0.0)
        z = (self.mu - current) / self.sigma
        return self.sigma * (_phi(z) + z * _Phi(z))

    def mean(self) -> float:
        """E[R] = mu."""
        return self.mu


@dataclass(frozen=True)
class Categorical(RewardDistribution):
    """Discrete reward over ``values`` with probabilities ``probs``.

    Args:
        values: Dollar outcomes.
        probs: Probability of each outcome; same length as ``values``,
            non-negative, summing to 1.

    Raises:
        ValueError: Length mismatch, negative probability, or sum != 1.
    """

    values: tuple[float, ...]
    probs: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate lengths, non-negativity, and sum-to-one."""
        if len(self.values) != len(self.probs):
            raise ValueError(
                f"values and probs must have the same length, got {len(self.values)} and {len(self.probs)}"
            )
        if any(p < 0 for p in self.probs):
            raise ValueError("all probs must be non-negative")
        if abs(sum(self.probs) - 1.0) > 1e-9:
            raise ValueError(f"probs must sum to 1, got {sum(self.probs)}")

    def expected_improvement(self, current: float) -> float:
        """E[(R - current)_+] for a discrete reward."""
        return sum(p * (v - current) for v, p in zip(self.values, self.probs, strict=True) if v > current)

    def mean(self) -> float:
        """E[R] = sum(value * probability)."""
        return sum(v * p for v, p in zip(self.values, self.probs, strict=True))


# ── Estimate ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Estimate:
    """One candidate's estimated economics for one task: reward distribution plus cost.

    Args:
        dist: Estimated distribution of the dollar reward of one attempt.
        cost: Expected dollar cost of one attempt.

    Raises:
        ValueError: ``cost`` negative.
    """

    dist: RewardDistribution
    cost: float

    def __post_init__(self) -> None:
        """Validate ``cost`` is non-negative."""
        if self.cost < 0:
            raise ValueError(f"cost must be non-negative, got {self.cost}")

    def reservation_price(self) -> float:
        """Solve ``E[(R - g)_+] = cost`` for ``g``.

        Returns:
            The reservation price in dollars: ``+inf`` when ``cost == 0``,
            below ``dist.mean()`` when the cost is high. Uses the Bernoulli
            closed form when ``dist`` is :class:`Bernoulli`, bisection
            otherwise.
        """
        return self.dist.reservation_price(self.cost)

    def net_value(self) -> float:
        """Return ``E[R] - cost``: the myopic value of a single committed attempt.

        The correct metric for one-shot routing, where no option to continue
        exists; :meth:`reservation_price` additionally prices that option in.
        """
        return self.dist.mean() - self.cost


# ── Policy ────────────────────────────────────────────────────────


class Policy(ABC):
    """Ordering-and-stopping rule consulted by :class:`Search`."""

    @abstractmethod
    def next(
        self,
        estimates: dict[str, Estimate],
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


def _affordable(estimates: dict[str, Estimate], remaining_budget: float | None) -> dict[str, Estimate]:
    """Drop candidates whose expected cost exceeds the remaining budget."""
    if remaining_budget is None:
        return estimates
    return {label: e for label, e in estimates.items() if e.cost <= remaining_budget}


class ReservationPricePolicy(Policy):
    """Weitzman's rule: highest reservation price above ``best``, else stop.

    Skips candidates whose expected cost exceeds the remaining budget.
    Optimal for independent candidates; the default policy.
    """

    def next(self, estimates: dict[str, Estimate], best: float, remaining_budget: float | None) -> str | None:
        """Return the affordable label with the highest reservation price above ``best``."""
        affordable = _affordable(estimates, remaining_budget)
        if not affordable:
            return None
        label = max(affordable, key=lambda k: affordable[k].reservation_price())
        return label if affordable[label].reservation_price() > best else None


class Greedy(Policy):
    """Highest net value above zero, then commit: try at most one candidate."""

    def next(self, estimates: dict[str, Estimate], best: float, remaining_budget: float | None) -> str | None:
        """Return the affordable label of highest positive net value, else stop.

        ``best > 0`` means a result is already in hand, so a one-shot policy
        stops rather than committing a second attempt.
        """
        if best > 0:
            return None
        affordable = _affordable(estimates, remaining_budget)
        if not affordable:
            return None
        label = max(affordable, key=lambda k: affordable[k].net_value())
        return label if affordable[label].net_value() > 0 else None


class Exhaustive(Policy):
    """Cheapest-first, no early stopping: try every candidate the budget allows."""

    def next(self, estimates: dict[str, Estimate], best: float, remaining_budget: float | None) -> str | None:
        """Return the cheapest affordable label, or ``None`` when none remain."""
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
            :class:`Greedy`.
        max_tries: Attempts allowed per label; ``None`` = unbounded (the
            policy's stopping rule is the only limit).

    Raises:
        ValueError: Empty ``estimates``, or a negative ``budget``.
    """

    def __init__(
        self,
        estimates: dict[str, Estimate],
        budget: float | None = None,
        policy: Policy | None = None,
        max_tries: int | None = 1,
    ) -> None:
        if not estimates:
            raise ValueError("Search requires at least one labeled estimate")
        if budget is not None and budget < 0:
            raise ValueError(f"budget must be non-negative, got {budget}")
        self._estimates: dict[str, Estimate] = dict(estimates)
        self._labels: frozenset[str] = frozenset(estimates)
        self._budget: float | None = budget
        self._policy: Policy = policy if policy is not None else Greedy()
        self._max_tries: int | None = max_tries
        self._tries: dict[str, int] = {}
        self._best: float = 0.0
        self._spent: float = 0.0

    def _eligible(self) -> dict[str, Estimate]:
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
        money. Lets the runner distinguish ``BudgetExceeded`` from a genuine
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

    def update_estimates(self, estimates: dict[str, Estimate]) -> None:
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


# ── Standard normal helpers ───────────────────────────────────────


def _phi(z: float) -> float:
    """Standard normal PDF."""
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
