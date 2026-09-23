"""Tests for the economics module: search core, beliefs, and end-to-end routing."""

from __future__ import annotations

import math

import pytest

from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult
from ai_functions.experimental.economics import (
    Abstained,
    AttemptRecord,
    Beliefs,
    BudgetExceeded,
    Candidate,
    CandidatesExhausted,
    EconomicFunction,
    EmpiricalBeliefs,
    PricedModel,
    Prices,
    RecordId,
    TaskView,
    attempts,
    decisions,
    routed,
    spend,
)
from ai_functions.experimental.economics.search import (
    Bernoulli,
    Categorical,
    Cheapest,
    Greedy,
    ReservationPricePolicy,
    RewardCostEstimate,
    RewardDistribution,
    ScoreCostEstimate,
    ScoreDistribution,
    Search,
    bisect_reservation_index,
)
from ai_functions.testing import RuntimeHarness, ScriptedModel, Turn

# ── Fixtures ──────────────────────────────────────────────────────

CHEAP_PRICES = Prices(input=1.0, output=1.0)
STRONG_PRICES = Prices(input=1.0, output=10.0)


def _task(prompt: str = "p", **args: object) -> TaskView:
    return TaskView(prompt=prompt, arguments=dict(args))


# ══════════════════════════════════════════════════════════════════
# Distributions and reservation prices
# ══════════════════════════════════════════════════════════════════


class TestReservationIndex:
    """The score layer: dimensionless, no dollars anywhere."""

    @pytest.mark.parametrize("p", [0.0, 0.05, 0.5, 0.95, 1.0])
    @pytest.mark.parametrize("cost", [0.001, 0.05, 0.5, 2.0])
    def test_bernoulli_closed_form_matches_bisection(self, p, cost):
        """Closed form vs generic solver; ``cost`` straddles ``p`` to hit both branches."""
        b = Bernoulli(p=p)
        assert b.reservation_index(cost) == pytest.approx(bisect_reservation_index(b, cost), abs=1e-6)

    def test_bernoulli_closed_form_is_exact(self):
        """Where bisection only converges, the closed form lands exactly."""
        assert Bernoulli(p=0.5).reservation_index(0.1) == 0.8  # 1 - 0.1/0.5
        assert Bernoulli(p=0.5).reservation_index(0.5) == 0.0  # branches meet at cost == p
        assert Bernoulli(p=0.05).reservation_index(0.5) == pytest.approx(-0.45)  # p - cost
        assert Bernoulli(p=0.0).reservation_index(0.5) == pytest.approx(-0.5)  # degenerate p

    def test_negative_branch_is_exact_for_any_distribution(self):
        """``cost >= E[S]`` gives exactly ``E[S] - cost`` for any distribution on ``[0, 1]``."""
        for dist in (
            Bernoulli(p=0.6),
            Bernoulli(p=0.0),
            Categorical(scores=(0.0, 0.5, 1.0), probs=(0.2, 0.5, 0.3)),
        ):
            mean = dist.mean()
            for cost in (mean, mean + 0.25, 5.0):
                if cost <= 0:
                    # A free attempt is always worth making, whatever the mean:
                    # that guard precedes the negative branch.
                    assert dist.reservation_index(cost) == math.inf
                    continue
                assert dist.reservation_index(cost) == pytest.approx(mean - cost), f"{dist!r} at {cost}"

    def test_index_never_exceeds_one(self):
        # The index is at most 1 for any positive cost, even with all mass at 1.
        for dist in (Bernoulli(p=1.0), Categorical(scores=(0.0, 1.0), probs=(0.01, 0.99))):
            for cost in (1e-9, 1e-4, 0.5):
                assert dist.reservation_index(cost) <= 1.0

    @pytest.mark.parametrize("cost", [0.01, 0.1, 0.2])
    def test_categorical_self_consistency(self, cost):
        """E[(S - k)_+] = cost must hold at the returned index (generic bisection)."""
        cat = Categorical(scores=(0.0, 0.5, 1.0), probs=(0.2, 0.5, 0.3))
        s = cat.reservation_index(cost)
        assert abs(cat.expected_improvement(s) - cost) < 1e-5

    def test_free_attempt_index_is_infinite(self):
        assert Bernoulli(p=0.5).reservation_index(0.0) == math.inf
        assert Categorical(scores=(0.0, 1.0), probs=(0.5, 0.5)).reservation_index(0.0) == math.inf

    def test_categorical_mean_and_improvement(self):
        cat = Categorical(scores=(0.0, 1.0), probs=(0.25, 0.75))
        assert cat.mean() == pytest.approx(0.75)
        # E[(S - 0)_+] = 0.75 * 1.0
        assert cat.expected_improvement(0.0) == pytest.approx(0.75)

    def test_bernoulli_rejects_bad_p(self):
        with pytest.raises(ValueError, match="p must be"):
            Bernoulli(p=1.5)

    @pytest.mark.parametrize("scores", [(-0.05, 0.5), (0.5, 1.5)])
    def test_categorical_rejects_scores_outside_unit_range(self, scores):
        # Scores must be in [0, 1]; Categorical's own constructor check, before check_score_support.
        with pytest.raises(ValueError, match=r"scores must be in \[0, 1\]"):
            Categorical(scores=scores, probs=(0.5, 0.5))


class TestReservationPrice:
    """The dollar layer: ``RewardDistribution`` prices a score distribution."""

    def test_price_is_the_index_scaled_by_value(self):
        # Pricing must reproduce the familiar g = value - cost / p.
        value, cost, p = 0.10, 0.002, 0.6
        d = RewardDistribution(score_dist=Bernoulli(p=p), value=value)
        assert d.reservation_price(cost) == pytest.approx(value - cost / p)
        assert d.reservation_price(cost) == pytest.approx(value * Bernoulli(p=p).reservation_index(cost / value))

    def test_price_is_homogeneous_in_money(self):
        # E[(vS - g)_+] = c and E[(2vS - 2g)_+] = 2c are the same equation, so
        # doubling value and cost together must double the price exactly.
        score_dist = Categorical(scores=(0.0, 0.4, 1.0), probs=(0.3, 0.4, 0.3))
        base = RewardDistribution(score_dist=score_dist, value=0.10).reservation_price(0.002)
        doubled = RewardDistribution(score_dist=score_dist, value=0.20).reservation_price(0.004)
        assert doubled == pytest.approx(2 * base, rel=1e-6)

    def test_mean_and_improvement_are_dollars(self):
        d = RewardDistribution(score_dist=Bernoulli(p=0.4), value=0.10)
        assert d.mean() == pytest.approx(0.4 * 0.10)
        assert d.expected_improvement(0.0) == pytest.approx(0.4 * 0.10)

    def test_zero_cost_is_infinite(self):
        assert RewardDistribution(score_dist=Bernoulli(p=0.5), value=1.0).reservation_price(0.0) == math.inf

    def test_rejects_non_positive_value(self):
        # value must be strictly positive.
        for value in (0.0, -1.0):
            with pytest.raises(ValueError, match="value must be positive"):
                RewardDistribution(score_dist=Bernoulli(p=0.5), value=value)

    def test_rejects_a_score_distribution_with_impossible_support(self):
        # The net for custom subclasses, which self-validate nothing.
        class _ShiftedLow(ScoreDistribution):
            """Mass at -0.1 and 1.0: legal arithmetic, impossible score."""

            def expected_improvement(self, current: float) -> float:
                return sum(0.5 * (s - current) for s in (-0.1, 1.0) if s > current)

            def mean(self) -> float:
                return 0.5 * (-0.1) + 0.5 * 1.0

        class _ShiftedHigh(ScoreDistribution):
            """Mass at 0.0 and 1.5: above the unit range."""

            def expected_improvement(self, current: float) -> float:
                return sum(0.5 * (s - current) for s in (0.0, 1.5) if s > current)

            def mean(self) -> float:
                return 0.5 * 1.5

        with pytest.raises(ValueError, match="puts mass below 0"):
            RewardDistribution(score_dist=_ShiftedLow(), value=0.10)
        with pytest.raises(ValueError, match="puts mass above 1"):
            RewardDistribution(score_dist=_ShiftedHigh(), value=0.10)


class TestEstimate:
    def test_net_value_prices_the_score(self):
        e = RewardCostEstimate(RewardDistribution(Bernoulli(p=0.4), 0.10), 0.01)
        assert e.net_value() == pytest.approx(0.4 * 0.10 - 0.01)

    def test_rejects_negative_cost(self):
        with pytest.raises(ValueError, match="cost must be"):
            ScoreCostEstimate(Bernoulli(p=0.5), cost=-0.01)


# ══════════════════════════════════════════════════════════════════
# Search loop and policies
# ══════════════════════════════════════════════════════════════════


class TestSearch:
    def _estimates(self):
        return {
            "cheap": RewardCostEstimate(RewardDistribution(Bernoulli(0.6), 0.10), 0.002),
            "strong": RewardCostEstimate(RewardDistribution(Bernoulli(0.95), 0.10), 0.02),
        }

    def test_escalation_order_and_stop(self):
        s = Search(self._estimates(), budget=0.25, policy=ReservationPricePolicy())
        assert s.next() == "cheap"  # higher reservation price
        s.observe("cheap", reward=0.0, cost=0.002)
        assert s.next() == "strong"
        s.observe("strong", reward=0.10, cost=0.02)
        assert s.next() is None  # best (0.10) tops every reservation price
        assert s.best == pytest.approx(0.10)
        assert s.spent == pytest.approx(0.022)

    def test_max_tries_exhausts_labels(self):
        s = Search({"only": RewardCostEstimate(RewardDistribution(Bernoulli(0.5), 1.0), 0.01)}, budget=1.0, max_tries=1)
        assert s.next() == "only"
        s.observe("only", reward=0.0, cost=0.01)
        assert s.next() is None  # single try used up

    def test_budget_blocks_unaffordable(self):
        s = Search(self._estimates(), budget=0.005)
        # only cheap (cost 0.002) fits; strong (0.02) is filtered out
        assert s.next() == "cheap"
        s.observe("cheap", reward=0.0, cost=0.002)
        assert s.next() is None  # strong unaffordable, cheap exhausted

    def test_default_policy_is_reservation_price(self):
        # cheap has the higher reservation price, strong the higher net value:
        # the default is Weitzman's rule, so it must pick cheap; Greedy would
        # pick strong.
        est = self._estimates()
        assert est["cheap"].reservation_price() > est["strong"].reservation_price()
        assert est["strong"].net_value() > est["cheap"].net_value()
        s = Search(est)
        assert s.next() == "cheap"

    def test_default_matches_the_function_layer(self):
        # Search, EconomicFunction and @routed share one default: ReservationPricePolicy.
        assert isinstance(Search(self._estimates())._policy, ReservationPricePolicy)
        fn = EconomicFunction({"c": Candidate("c", _dummy_fn(), CHEAP_PRICES)}, value=0.10, beliefs=EmpiricalBeliefs())
        assert isinstance(fn._policy, ReservationPricePolicy)
        assert isinstance(
            routed(models=[PricedModel("m", CHEAP_PRICES, label="c")], value=0.10)(_dummy_fn())._policy,
            ReservationPricePolicy,
        )

    def test_greedy_stops_on_any_success(self):
        s = Search(self._estimates(), policy=Greedy())
        assert s.next() == "strong"  # highest net value
        s.observe("strong", reward=0.10, cost=0.02)
        assert s.next() is None  # best > 0

    def test_greedy_tries_next_best_on_failure(self):
        # Greedy stops once a positive reward is in hand. When "strong" fails
        # and reaches max_tries, the search moves on to the next-best candidate.
        s = Search(self._estimates(), policy=Greedy())
        assert s.next() == "strong"
        s.observe("strong", reward=0.0, cost=0.02)
        assert s.next() == "cheap"  # a second candidate, despite Greedy being myopic
        s.observe("cheap", reward=0.0, cost=0.002)
        assert s.next() is None  # both labels exhausted
        assert s.spent == pytest.approx(0.022)

    def test_cheapest_escalates_on_failure(self):
        s = Search(self._estimates(), policy=Cheapest(), budget=1.0, max_tries=1)
        assert s.next() == "cheap"
        s.observe("cheap", reward=0.0, cost=0.002)
        assert s.next() == "strong"

    def test_cheapest_stops_on_success(self):
        # Cheapest tries the lowest-cost candidate first and stops at the first positive reward.
        s = Search(self._estimates(), policy=Cheapest(), budget=1.0, max_tries=1)
        assert s.next() == "cheap"
        s.observe("cheap", reward=0.10, cost=0.002)
        assert s.next() is None

    def test_update_estimates_rejects_unknown_label(self):
        s = Search(self._estimates())
        with pytest.raises(KeyError):
            s.update_estimates({"nope": RewardCostEstimate(RewardDistribution(Bernoulli(0.5), 1.0), 0.01)})

    def test_explain_ranked(self):
        s = Search(self._estimates())
        ranking = s.explain()
        prices = [r.reservation_price for r in ranking]
        assert prices == sorted(prices, reverse=True)
        assert {r.label for r in ranking} == {"cheap", "strong"}

    def test_empty_estimates_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            Search({})


# ══════════════════════════════════════════════════════════════════
# Beliefs
# ══════════════════════════════════════════════════════════════════


def _dummy_fn():
    @ai_function[str](structured_output=False)
    def fn(x: str) -> str:
        """{x}"""

    return fn


class TestEmpiricalBeliefs:
    def _candidates(self):
        fn = _dummy_fn()
        return [
            Candidate("cheap", fn, CHEAP_PRICES),
            Candidate("strong", fn, STRONG_PRICES),
        ]

    @pytest.mark.asyncio
    async def test_prior_then_learns(self):
        b = EmpiricalBeliefs()
        cands = self._candidates()
        est0 = await b.estimate(_task(), cands, history=[])
        assert est0["cheap"].score_dist.mean() == pytest.approx(0.5)  # 1/(1+1)

        b.update(
            AttemptRecord(id=RecordId("r1"), task=_task(), candidate="cheap", cost=0.002, reward=0.0, local_score=0.0)
        )
        b.update(
            AttemptRecord(id=RecordId("r2"), task=_task(), candidate="cheap", cost=0.002, reward=0.10, local_score=1.0)
        )
        est1 = await b.estimate(_task(), cands, history=[])
        # alpha=1+1, beta=1+1 -> p=0.5
        assert est1["cheap"].score_dist.mean() == pytest.approx(0.5)
        # cost now reflects observed mean, not the token prior
        assert est1["cheap"].cost == pytest.approx(0.002)

    @pytest.mark.asyncio
    async def test_estimates_are_independent_of_value(self):
        # A provider never sees ``value``, so one estimate serves any price —
        # and pricing scales it linearly.
        b = EmpiricalBeliefs()
        est = (await b.estimate(_task(), self._candidates(), history=[]))["cheap"]
        cheap = RewardDistribution(est.score_dist, 0.10)
        dear = RewardDistribution(est.score_dist, 50.0)
        assert cheap.score_dist == dear.score_dist
        assert dear.mean() == pytest.approx(500 * cheap.mean())

    @pytest.mark.asyncio
    async def test_settlement_revises_exactly(self):
        b = EmpiricalBeliefs()
        rec = AttemptRecord(
            id=RecordId("r1"), task=_task(), candidate="cheap", cost=0.002, reward=0.10, local_score=1.0
        )
        b.update(rec)
        assert "100% pass" in b.stats()["cheap"]  # displayed rate is evidence only
        b.settle(RecordId("r1"), 0.0)  # downstream says it was useless
        assert "0% pass" in b.stats()["cheap"]  # settlement replaces the verdict
        # The routing estimate smooths with the prior: alpha=1+0, beta=1+1 -> p=1/3
        est = await b.estimate(_task(), self._candidates(), history=[])
        assert est["cheap"].score_dist.mean() == pytest.approx(1 / 3)

    @pytest.mark.asyncio
    async def test_fixed_beliefs_constant(self):
        b = Beliefs.fixed({"cheap": ScoreCostEstimate(Bernoulli(0.7), 0.001)})
        cand = [Candidate("cheap", _dummy_fn(), CHEAP_PRICES)]
        est = await b.estimate(_task(), cand, history=[])
        assert est["cheap"].score_dist.mean() == pytest.approx(0.7)

    @pytest.mark.asyncio
    async def test_fixed_beliefs_missing_label_raises(self):
        b = Beliefs.fixed({"cheap": ScoreCostEstimate(Bernoulli(0.7), 0.001)})
        with pytest.raises(KeyError):
            await b.estimate(_task(), self._candidates(), history=[])

    def test_decay_rejects_out_of_range(self):
        with pytest.raises(ValueError, match="decay"):
            EmpiricalBeliefs(decay=1.5)

    def test_turn_stats_tracked_in_track_record(self):
        from ai_functions.types import TokenUsage

        b = EmpiricalBeliefs()
        b.update(
            AttemptRecord(
                id=RecordId("r1"),
                task=_task(),
                candidate="cheap",
                cost=0.002,
                usage=TokenUsage(output_tokens=300),
                turns=3,
                reward=0.10,
                local_score=1.0,
            )
        )
        b.update(
            AttemptRecord(
                id=RecordId("r2"),
                task=_task(),
                candidate="cheap",
                cost=0.002,
                usage=TokenUsage(output_tokens=100),
                turns=1,
                reward=0.10,
                local_score=1.0,
            )
        )
        # mean turns (3+1)/2 = 2; tokens/turn (100 + 100)/2 = 100
        assert "avg 2.0 turns at 100 output tokens/turn" in b.stats()["cheap"]

    def test_turn_stats_skip_unmeasured_records(self):
        b = EmpiricalBeliefs()
        # A Decision.report booking carries no turn data (turns=0) and must
        # not drag the average toward zero.
        b.update(
            AttemptRecord(id=RecordId("r1"), task=_task(), candidate="cheap", cost=0.002, reward=0.10, local_score=1.0)
        )
        assert "turns" not in b.stats()["cheap"]


class TestForecastCostModel:
    def test_single_turn_is_prompt_write_plus_output(self):
        from ai_functions.experimental.economics.beliefs import _approx_cost

        # 1 turn: prompt written once at output/4, no reads, one turn of output.
        cost = _approx_cost(output_price=10.0, prompt_tokens=1000, turns=1, output_tokens_per_turn=200)
        expected = (1000 * 2.5 + 200 * 10.0) / 1e6
        assert cost == pytest.approx(expected)

    def test_multi_turn_reads_growing_context(self):
        from ai_functions.experimental.economics.beliefs import _approx_cost

        # 3 turns, prompt P=1000, o=100/turn, output price 10:
        #   written: P + 2o = 1200 @ 2.5
        #   read:    2P + o*(2*3/2) = 2300 @ 0.2
        #   output:  3o = 300 @ 10
        cost = _approx_cost(output_price=10.0, prompt_tokens=1000, turns=3, output_tokens_per_turn=100)
        expected = (1200 * 2.5 + 2300 * 0.2 + 300 * 10.0) / 1e6
        assert cost == pytest.approx(expected)

    def test_more_turns_cost_more(self):
        from ai_functions.experimental.economics.beliefs import _approx_cost

        costs = [
            _approx_cost(output_price=10.0, prompt_tokens=1000, turns=t, output_tokens_per_turn=100)
            for t in (1, 2, 5, 10)
        ]
        assert costs == sorted(costs)


# ══════════════════════════════════════════════════════════════════
# EconomicFunction — construction guards
# ══════════════════════════════════════════════════════════════════


class TestConstruction:
    def _cand(self, label="c"):
        return {label: Candidate(label, _dummy_fn(), CHEAP_PRICES)}

    def test_empty_candidates_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            EconomicFunction({}, value=0.10, beliefs=EmpiricalBeliefs())

    def test_unbounded_without_budget_rejected(self):
        with pytest.raises(ValueError, match="budget"):
            EconomicFunction(self._cand(), value=0.10, beliefs=EmpiricalBeliefs(), max_tries=None)

    def test_callable_value_rejected(self):
        with pytest.raises(ValueError, match="constant dollar amount"):
            EconomicFunction(self._cand(), value=lambda r: 0.10, beliefs=EmpiricalBeliefs())

    def test_nonpositive_value_rejected(self):
        with pytest.raises(ValueError, match="positive dollars"):
            EconomicFunction(self._cand(), value=0.0, beliefs=EmpiricalBeliefs())

    def test_score_accepted(self):
        fn = EconomicFunction(self._cand(), value=0.10, beliefs=EmpiricalBeliefs(), scorer=lambda r: 0.5)
        assert fn._scorer is not None

    def test_mismatched_label_rejected(self):
        cand = {"wrong": Candidate("actual", _dummy_fn(), CHEAP_PRICES)}
        with pytest.raises(ValueError, match="mismatched"):
            EconomicFunction(cand, value=0.10, beliefs=EmpiricalBeliefs())


class TestPricedModel:
    def test_label_derived_from_string(self):
        m = PricedModel(model="claude-x", prices=CHEAP_PRICES)
        assert m.label == "claude-x"

    def test_explicit_label_wins(self):
        m = PricedModel(model="claude-x", prices=CHEAP_PRICES, label="mine")
        assert m.label == "mine"


# ══════════════════════════════════════════════════════════════════
# End-to-end over scripted models
# ══════════════════════════════════════════════════════════════════


def _require_yes(result, **kwargs):
    if "yes" not in result:
        return PostConditionResult(passed=False, message="must contain yes")
    return None


@ai_function[str](structured_output=False, post_conditions=[_require_yes])
def _solve(task: str) -> str:
    """Solve: {task}"""


def _fixed_beliefs():
    return Beliefs.fixed(
        {
            "cheap": ScoreCostEstimate(Bernoulli(0.6), 0.00001),
            "strong": ScoreCostEstimate(Bernoulli(0.95), 0.0002),
        }
    )


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_escalates_and_attributes_cost(self):
        async with RuntimeHarness() as h:
            cheap = ScriptedModel([Turn(text="cheap says no", input_tokens=5, output_tokens=10)])
            strong = ScriptedModel([Turn(text="strong says yes", input_tokens=8, output_tokens=20)])
            cands = {
                "cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0)),
                "strong": Candidate("strong", _solve.replace(model=strong), Prices(input=1.0, output=10.0)),
            }
            fn = EconomicFunction(
                cands, value=0.10, beliefs=_fixed_beliefs(), budget=0.25, policy=ReservationPricePolicy()
            )
            handle = await h.spawn(fn)
            result = await handle.run(task="t")
            assert "yes" in result

            recs = await attempts(handle)
            assert [(r.candidate, r.local_score > 0) for r in recs] == [("cheap", False), ("strong", True)]
            # per-attempt usage measured independently
            assert recs[0].usage.output_tokens == 10
            assert recs[1].usage.output_tokens == 20
            # each attempt was a single model call
            assert recs[0].turns == 1
            assert recs[1].turns == 1
            # cost prices both input and output: cheap (5 in + 10 out) @ $1/M
            assert recs[0].cost == pytest.approx((5 * 1.0 + 10 * 1.0) / 1e6)
            # strong (8 in @ $1/M + 20 out @ $10/M)
            assert recs[1].cost == pytest.approx((8 * 1.0 + 20 * 10.0) / 1e6)

            # spend rolls the attempt costs; decisions records the ranking round
            assert await spend(handle) == pytest.approx(recs[0].cost + recs[1].cost)
            rounds = await decisions(handle)
            assert rounds and {r.label for r in rounds[0]} == {"cheap", "strong"}

    @pytest.mark.asyncio
    async def test_non_aifunction_error_is_a_failed_attempt(self):
        # An attempt that raises a non-AIFunctionError (here: a model/runtime
        # error surfaced as the script running dry) must count as a failed
        # attempt and escalate — a fallback layer swallows operational errors.
        async with RuntimeHarness() as h:
            cheap = ScriptedModel([])  # raises ScriptExhausted on invocation
            strong = ScriptedModel([Turn(text="strong says yes", input_tokens=8, output_tokens=20)])
            cands = {
                "cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0)),
                "strong": Candidate("strong", _solve.replace(model=strong), Prices(input=1.0, output=10.0)),
            }
            fn = EconomicFunction(
                cands, value=0.10, beliefs=_fixed_beliefs(), budget=0.25, policy=ReservationPricePolicy()
            )
            handle = await h.spawn(fn)
            result = await handle.run(task="t")
            assert "yes" in result
            recs = await attempts(handle)
            assert [(r.candidate, r.local_score > 0) for r in recs] == [("cheap", False), ("strong", True)]

    @pytest.mark.asyncio
    async def test_first_pass_no_escalation(self):
        async with RuntimeHarness() as h:
            cheap = ScriptedModel([Turn(text="cheap says yes", input_tokens=5, output_tokens=10)])
            strong = ScriptedModel([])
            cands = {
                "cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0)),
                "strong": Candidate("strong", _solve.replace(model=strong), Prices(input=1.0, output=10.0)),
            }
            fn = EconomicFunction(cands, value=0.10, beliefs=_fixed_beliefs(), budget=0.25)
            handle = await h.spawn(fn)
            result = await handle.run(task="t")
            assert "yes" in result
            assert strong.remaining_turns == 0  # never invoked

    @pytest.mark.asyncio
    async def test_all_fail_raises_exhausted_with_records(self):
        async with RuntimeHarness() as h:
            cheap = ScriptedModel([Turn(text="no", output_tokens=1)])
            cands = {"cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0))}
            fn = EconomicFunction(cands, value=0.10, beliefs=_fixed_beliefs(), budget=0.25)
            handle = await h.spawn(fn)
            with pytest.raises(CandidatesExhausted) as exc:
                await handle.run(task="t")
            assert len(exc.value.records) == 1
            assert exc.value.records[0].candidate == "cheap"

    @pytest.mark.asyncio
    async def test_abstains_when_nothing_worth_it(self):
        async with RuntimeHarness() as h:
            cheap = ScriptedModel([])
            cands = {"cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0))}
            # value 0.10 but cost 1.0 -> net value negative -> abstain, never runs
            beliefs = Beliefs.fixed({"cheap": ScoreCostEstimate(Bernoulli(0.5), 1.0)})
            fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=2.0)
            handle = await h.spawn(fn)
            with pytest.raises(Abstained):
                await handle.run(task="t")
            assert cheap.remaining_turns == 0  # never invoked

    @pytest.mark.asyncio
    async def test_graded_best_of_n_keeps_highest_score(self):
        # Independent graded best-of-N: each attempt is scored in [0, 1] and
        # priced at value; the search keeps the best result by reward and stops
        # once a draw beats every reservation price. FIXED estimates stop on
        # their own as the best in hand rises.
        @ai_function[str](structured_output=False)
        def _draft(src: str) -> str:
            """{src}"""

        def score(s: str) -> float:
            # Fraction of the 5-item target found -> quality in [0, 1].
            return len([x for x in s.split(",") if x]) / 5.0

        async with RuntimeHarness() as h:
            # Draws of improving quality: 1 item (0.2), then 3 (0.6); a fourth
            # turn exists but must never run — the bar stops the search first.
            model = ScriptedModel(
                [
                    Turn(text="a", output_tokens=10),
                    Turn(text="a,b,c", output_tokens=10),
                    Turn(text="a,b,c,d,e", output_tokens=10),
                ]
            )
            cand = {"m": Candidate("m", _draft.replace(model=model), Prices(input=1.0, output=1.0))}
            # reward = 0.10 * score: draw 1 banks $0.02, draw 2 banks $0.06. The fixed
            # belief {0.2, 0.6} at even odds and cost 0.005 gives reservation price
            # 0.5 * (0.06 - g) = 0.005 -> g = $0.05: continue after draw 1, stop after draw 2.
            fn = EconomicFunction(
                cand,
                value=0.10,
                scorer=score,
                beliefs=Beliefs.fixed(
                    {"m": ScoreCostEstimate(Categorical(scores=(0.2, 0.6), probs=(0.5, 0.5)), 0.005)}
                ),
                budget=1.0,
                policy=ReservationPricePolicy(),
                max_tries=None,
            )
            handle = await h.spawn(fn)
            result = await handle.run(src="doc")

            # Kept the best result (draw 2), stopped before the third turn.
            assert result.strip() == "a,b,c"
            recs = await attempts(handle)
            assert len(recs) == 2
            # reward = value * score: 0.10*0.2, then 0.10*0.6.
            assert recs[0].reward == pytest.approx(0.02)
            assert recs[1].reward == pytest.approx(0.06)
            # local_score carries the graded quality now, not just 0/1.
            assert recs[0].local_score == pytest.approx(0.2)
            assert recs[1].local_score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_out_of_range_score_raises_clearly(self):
        # A score outside [0, 1] must fail clearly at ingestion, not obscurely
        # later at Bernoulli construction.
        @ai_function[str](structured_output=False)
        def _draft(src: str) -> str:
            """{src}"""

        async with RuntimeHarness() as h:
            model = ScriptedModel([Turn(text="x", output_tokens=10)])
            cand = {"m": Candidate("m", _draft.replace(model=model), Prices(input=1.0, output=1.0))}
            fn = EconomicFunction(
                cand,
                value=0.10,
                scorer=lambda s: 5.0,  # bug: returns a dollar-like amount, not [0, 1]
                beliefs=EmpiricalBeliefs(),
                budget=1.0,
            )
            handle = await h.spawn(fn)
            with pytest.raises(ValueError, match=r"score must be in \[0, 1\]"):
                await handle.run(src="doc")

    @pytest.mark.asyncio
    async def test_budget_exceeded_raises_with_records(self):
        async with RuntimeHarness() as h:
            # cheap fails; strong would pass but its cost exceeds the tiny budget
            cheap = ScriptedModel([Turn(text="no", input_tokens=1, output_tokens=1)])
            strong = ScriptedModel([Turn(text="yes", input_tokens=1, output_tokens=1)])
            cands = {
                "cheap": Candidate("cheap", _solve.replace(model=cheap), Prices(input=1.0, output=1.0)),
                "strong": Candidate("strong", _solve.replace(model=strong), Prices(input=1.0, output=1.0)),
            }
            beliefs = Beliefs.fixed(
                {
                    "cheap": ScoreCostEstimate(Bernoulli(0.5), 0.000002),
                    "strong": ScoreCostEstimate(Bernoulli(0.99), 0.01),  # exceeds budget below
                }
            )
            fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=0.000005)
            handle = await h.spawn(fn)
            with pytest.raises(BudgetExceeded) as exc:
                await handle.run(task="t")
            assert exc.value.records  # cheap attempt is booked before the block

    @pytest.mark.asyncio
    async def test_plan_reports_and_learns(self):
        # plan() estimates without executing, so no runtime is needed.
        cands = {"cheap": Candidate("cheap", _solve, Prices(input=1.0, output=1.0))}
        beliefs = EmpiricalBeliefs()
        fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=1.0)
        decision = await fn.plan(task="t")
        assert decision.candidate is not None
        assert decision.candidate.label == "cheap"
        assert decision.ranking[0].label == "cheap"
        # report an externally executed success -> beliefs learn
        decision.report("yes", cost=0.001)
        assert "pass" in beliefs.stats()["cheap"]

    @pytest.mark.asyncio
    async def test_plan_reports_accumulate_across_calls(self):
        # Each plan()'s report must book a distinct record id: beliefs key
        # contributions by id, so a repeated id would overwrite the earlier
        # task's evidence instead of adding to it.
        cands = {"cheap": Candidate("cheap", _solve, Prices(input=1.0, output=1.0))}
        beliefs = EmpiricalBeliefs()
        fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=1.0)
        for _ in range(3):
            decision = await fn.plan(task="t")
            decision.report("yes", cost=0.001)
        assert "over 3 attempts" in beliefs.stats()["cheap"]

    @pytest.mark.asyncio
    async def test_plan_declines_when_unprofitable(self):
        cands = {"cheap": Candidate("cheap", _solve, Prices(input=1.0, output=1.0))}
        beliefs = Beliefs.fixed({"cheap": ScoreCostEstimate(Bernoulli(0.5), 1.0)})
        fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=2.0)
        decision = await fn.plan(task="t")
        assert decision.candidate is None

    @pytest.mark.asyncio
    async def test_plan_works_over_any_distribution(self):
        # The routing decision (candidate, ranking, report) needs only
        # reservation prices, which every ScoreDistribution supports.
        cands = {"cheap": Candidate("cheap", _solve, Prices(input=1.0, output=1.0))}
        score = Categorical(scores=(0.0, 0.5, 1.0), probs=(0.25, 0.5, 0.25))
        beliefs = Beliefs.fixed({"cheap": ScoreCostEstimate(score, 0.002)})
        fn = EconomicFunction(cands, value=0.10, beliefs=beliefs)
        decision = await fn.plan(task="t")
        assert decision.candidate is not None
        assert decision.ranking[0].label == "cheap"


# ══════════════════════════════════════════════════════════════════
# Decorators
# ══════════════════════════════════════════════════════════════════


class TestDecorators:
    def test_routed_builds_candidates_from_models(self):
        models = [
            PricedModel(model="haiku", prices=CHEAP_PRICES),
            PricedModel(model="sonnet", prices=STRONG_PRICES),
        ]

        @routed(models=models, value=0.10)
        @ai_function[str](structured_output=False)
        def solve(task: str) -> str:
            """Solve: {task}"""

        assert set(solve.candidates) == {"haiku", "sonnet"}
        assert isinstance(solve.beliefs, EmpiricalBeliefs)

    def test_routed_rejects_both_sources(self):
        with pytest.raises(ValueError, match="exactly one"):

            @routed(models=[PricedModel(model="m", prices=CHEAP_PRICES)], candidates=[], value=0.10)
            @ai_function[str](structured_output=False)
            def solve(task: str) -> str:
                """{task}"""

    def test_routed_rejects_callable_value(self):
        with pytest.raises(ValueError, match="constant dollar amount"):

            @routed(models=[PricedModel(model="m", prices=CHEAP_PRICES)], value=lambda r: 0.10)  # type: ignore[arg-type]
            @ai_function[str](structured_output=False)
            def solve(task: str) -> str:
                """{task}"""

    def test_routed_rejects_nonpositive_value(self):
        with pytest.raises(ValueError, match="positive dollars"):

            @routed(models=[PricedModel(model="m", prices=CHEAP_PRICES)], value=0.0)
            @ai_function[str](structured_output=False)
            def solve(task: str) -> str:
                """{task}"""

    def test_routed_default_policy_is_reservation_price(self):
        @routed(models=[PricedModel(model="haiku", prices=CHEAP_PRICES)], value=0.10)
        @ai_function[str](structured_output=False)
        def solve(task: str) -> str:
            """{task}"""

        assert isinstance(solve._policy, ReservationPricePolicy)

    def test_routed_passes_score(self):
        def grade(r: str) -> float:
            return 0.5

        @routed(models=[PricedModel(model="haiku", prices=CHEAP_PRICES)], value=0.10, scorer=grade)
        @ai_function[str](structured_output=False)
        def solve(task: str) -> str:
            """{task}"""

        assert solve._scorer is grade

    def test_routed_defaults_beliefs_to_empirical(self):
        @routed(models=[PricedModel(model="haiku", prices=CHEAP_PRICES)], value=0.10)
        @ai_function[str](structured_output=False)
        def review(src: str) -> str:
            """{src}"""

        assert set(review.candidates) == {"haiku"}
        assert isinstance(review.beliefs, EmpiricalBeliefs)


# ══════════════════════════════════════════════════════════════════
# Backward-pass learning: settlement via the optimizer
# ══════════════════════════════════════════════════════════════════


class _RecordingBeliefs(Beliefs):
    """Beliefs that record what the backward pass drives into them."""

    def __init__(self) -> None:
        self.settled: list[tuple[str, float]] = []

    async def estimate(self, task, candidates, history):  # noqa: ANN001, ARG002
        return {c.label: ScoreCostEstimate(Bernoulli(0.5), 0.001) for c in candidates}

    def settle(self, record_id, score):  # noqa: ANN001
        self.settled.append((str(record_id), score))


def _rec(rid: str, candidate: str, *, passed: bool) -> AttemptRecord:
    """A minimal AttemptRecord for driving beliefs directly."""
    return AttemptRecord(
        id=RecordId(rid),
        task=_task(),
        candidate=candidate,
        cost=0.001,
        reward=0.10 if passed else 0.0,
        local_score=1.0 if passed else 0.0,
    )


def _host_fn(beliefs):  # noqa: ANN001, ANN202
    """An EconomicFunction over ``beliefs`` — it is its own ParameterHost."""
    return EconomicFunction({"cheap": Candidate("cheap", _dummy_fn(), CHEAP_PRICES)}, value=0.10, beliefs=beliefs)


class TestFunctionAsParameterHost:
    def test_consolidate_settles_every_record_to_averaged_score(self):
        from ai_functions.types.graph import GradFeedback

        beliefs = _RecordingBeliefs()
        fn = _host_fn(beliefs)
        assert fn.backend_id == "economics:fn"  # derived from the wrapped function name
        # ``retrieved`` maps record id -> candidate; two gradients averaged.
        fn.consolidate(
            "routing_decision",
            [GradFeedback(text="too shallow", score=0.1), GradFeedback(text="also thin", score=0.3)],
            retrieved={"r1": "cheap", "r2": "strong"},
        )
        assert beliefs.settled == [("r1", 0.2), ("r2", 0.2)]

    def test_consolidate_clamps_and_ignores_missing_pieces(self):
        from ai_functions.types.graph import GradFeedback

        beliefs = _RecordingBeliefs()
        fn = _host_fn(beliefs)
        fn.consolidate("d", [GradFeedback(text="great", score=1.5)], retrieved={"r1": "cheap"})  # clamp
        assert beliefs.settled == [("r1", 1.0)]
        # No score, or no records: nothing to settle.
        fn.consolidate("d", [GradFeedback(text="x", score=None)], retrieved={"r1": "cheap"})
        fn.consolidate("d", [GradFeedback(text="y", score=0.5)], retrieved=None)
        assert beliefs.settled == [("r1", 1.0)]  # unchanged


class TestBackwardWiring:
    """The graph-level contract: the decision parameter's host settles on consolidate."""

    def test_decision_parameter_settles_records_on_consolidate(self):
        from ai_functions.experimental.economics.function import DECISION_PARAMETER
        from ai_functions.optimizer import TextGradOptimizer
        from ai_functions.types.graph import ParameterNode, ThreadNode

        beliefs = _RecordingBeliefs()
        host = _host_fn(beliefs)  # the economic function is its own ParameterHost
        # The economic child owns a grad-enabled decision parameter hosted by
        # the function; its meta carries the run's record ids.
        decision = ParameterNode(
            node_id="research-1-decision",
            value="routing summary",
            requires_grad=True,
            name=DECISION_PARAMETER,
            backend=host,
            description="score settles the routing decision",
            meta={"results": {"r1": "cheap"}},
        )
        child = ThreadNode(node_id="research-1", thread_id="research-1", value="some sources", parameters=[decision])
        root = ThreadNode(node_id="write-1", thread_id="write-1", value="the report", child_threads=[child])

        # Stubbed backward: at the root, score + text the economic child; at the
        # child, the score routes onto its decision parameter.
        opt = TextGradOptimizer()
        opt._backward_fn = _RouteWithScore(  # noqa: SLF001
            {"research-1": ("too shallow", 0.2), "research-1-decision": ("shallow", 0.2)}
        )
        opt.backward(root, "the report was too shallow")
        opt.consolidate(root)

        assert beliefs.settled == [("r1", 0.2)]

    def test_node_kept_when_it_owns_grad_decision_parameter(self):
        """An economic node stays in the walk because it owns a grad parameter."""
        from ai_functions.experimental.economics.function import DECISION_PARAMETER
        from ai_functions.optimizer._graph import leads_to_grad_parameter
        from ai_functions.types.graph import ParameterNode, ThreadNode

        decision = ParameterNode(node_id="c-decision", requires_grad=True, name=DECISION_PARAMETER)
        child = ThreadNode(node_id="c", thread_id="c", parameters=[decision])
        root = ThreadNode(node_id="r", thread_id="r", child_threads=[child])
        assert leads_to_grad_parameter(root) is True


class _RouteWithScore:
    """Offline backward stand-in that emits (feedback, score) per matching node id."""

    def __init__(self, responses: dict[str, tuple[str, float]]) -> None:
        self.responses = responses

    def replace(self, **kwargs: object) -> _RouteWithScore:
        del kwargs
        return self

    def run_sync(self, *, inputs: str, **kwargs: object) -> object:  # noqa: ARG002
        import yaml

        from ai_functions.optimizer.textgrad import Feedback, Feedbacks

        rendered = yaml.safe_load(inputs) or {}
        return Feedbacks(
            feedbacks=[
                Feedback(node_id=nid, feedback=text, score=score)
                for nid, (text, score) in self.responses.items()
                if nid in rendered
            ]
        )


class TestGraphIntegration:
    """The reconstructed economic node: naming, adopted trace, grad-enabled notes."""

    @pytest.mark.asyncio
    async def test_node_named_after_wrapped_function(self):
        from ai_functions.optimizer import build_graph_from_result

        async with RuntimeHarness() as h:

            @ai_function[str](structured_output=False)
            def research(task: str) -> str:
                """Research: {task}"""

            model = ScriptedModel([Turn(text="sources: a, b", output_tokens=10)])
            cands = {"cheap": Candidate("cheap", research.replace(model=model), CHEAP_PRICES)}
            fn = EconomicFunction(cands, value=0.10, beliefs=_fixed_beliefs(), budget=1.0)
            assert fn.name == "research"  # not economic(research)

            run = await _trace_on(h, fn, task="X")
            graph = await build_graph_from_result(run, [])
            assert graph.func_name == "research"
            assert graph.node_id.startswith("research-")

    @pytest.mark.asyncio
    async def test_node_adopts_responsible_attempt_trace(self):
        from ai_functions.optimizer import build_graph_from_result

        async with RuntimeHarness() as h:

            @ai_function[str](structured_output=False, post_conditions=[lambda r: _must_say_yes(r)])
            def research(task: str) -> str:
                """Research: {task}"""

            cheap = ScriptedModel([Turn(text="no", output_tokens=1)])
            strong = ScriptedModel([Turn(text="yes indeed", output_tokens=5)])
            cands = {
                "cheap": Candidate("cheap", research.replace(model=cheap), CHEAP_PRICES),
                "strong": Candidate("strong", research.replace(model=strong), STRONG_PRICES),
            }
            beliefs = Beliefs.fixed(
                {
                    "cheap": ScoreCostEstimate(Bernoulli(0.9), 0.000001),
                    "strong": ScoreCostEstimate(Bernoulli(0.99), 0.00001),
                }
            )
            fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=1.0, policy=ReservationPricePolicy())
            run = await _trace_on(h, fn, task="X")

            graph = await build_graph_from_result(run, [])
            text = "\n".join(str(m) for m in graph.messages)
            # The routing summary names both attempts and the responsible one...
            assert "Attempt 1: cheap — failed local checks" in text
            assert "Attempt 2: strong — passed local checks" in text
            assert "The returned output was produced by: strong" in text
            # ...and the adopted conversation is the responsible attempt's, not the loser's.
            assert "yes indeed" in text
            assert "role': 'user" in text or "'role': 'user'" in text

    @pytest.mark.asyncio
    async def test_forecaster_notes_surface_as_grad_parameter(self, tmp_path):
        from pydantic import BaseModel, Field

        from ai_functions.experimental.economics.beliefs import LLMForecaster, RoutingMemory, forecast
        from ai_functions.memory import JSONMemoryBackend
        from ai_functions.optimizer import build_graph_from_result

        class Mem(BaseModel):
            routing: RoutingMemory = Field(default_factory=RoutingMemory)

        async with RuntimeHarness() as h:

            @ai_function[str](structured_output=False)
            def research(task: str) -> str:
                """Research: {task}"""

            memory = JSONMemoryBackend(Mem, actor_id="t", path=tmp_path / "m.json")
            scripted_forecast = forecast.replace(
                model=ScriptedModel(
                    [
                        Turn(
                            tool_calls=(
                                (
                                    "ForecastResult",
                                    {
                                        "estimates": {
                                            "cheap": {
                                                "pass_percentage": 90,
                                                "turns": 1,
                                                "output_tokens_per_turn": 10,
                                            }
                                        }
                                    },
                                ),
                            )
                        )
                    ]
                )
            )
            beliefs = LLMForecaster(forecast_fn=scripted_forecast, memory=memory, memory_key="routing")
            model = ScriptedModel([Turn(text="sources", output_tokens=10)])
            cands = {"cheap": Candidate("cheap", research.replace(model=model), CHEAP_PRICES)}
            fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=1.0)
            run = await _trace_on(h, fn, task="X")

            graph = await build_graph_from_result(run, [memory])
            params = {p.name: p for p in graph.parameters}
            # The notes recall landed in the economic run's log, grad-enabled —
            # the backward pass has a routable target at this node — while the
            # machine-managed stats never surface as a parameter.
            assert "routing/notes" in params
            assert params["routing/notes"].requires_grad is True
            assert params["routing/notes"].backend is memory
            assert "routing/stats" not in params
            # The attempt's statistics were persisted through the backend.
            stats = memory.fetch("routing/stats")
            assert len(stats) == 1 and stats[0].candidate == "cheap"

    @pytest.mark.asyncio
    async def test_empirical_stats_persist_across_instances(self, tmp_path):
        from pydantic import BaseModel, Field

        from ai_functions.experimental.economics.beliefs import RoutingMemory
        from ai_functions.memory import JSONMemoryBackend

        class Mem(BaseModel):
            routing: RoutingMemory = Field(default_factory=RoutingMemory)

        memory = JSONMemoryBackend(Mem, actor_id="t", path=tmp_path / "m.json")
        b1 = EmpiricalBeliefs(memory=memory, stats_key="routing/stats")
        b1.update(_rec("r1", "cheap", passed=True))
        b1.settle(RecordId("r1"), 0.25)

        # A fresh instance (as after a process restart) reloads the settled record.
        b2 = EmpiricalBeliefs(memory=memory, stats_key="routing/stats")
        assert "pass over 1 attempts" in b2.stats()["cheap"]
        # The settled 0.25 survived the round trip, not the local 1.0.
        assert "25% pass" in b2.stats()["cheap"]

    def test_empirical_memory_requires_key(self):
        from ai_functions.memory import JSONMemoryBackend  # noqa: F401 — import guard only

        with pytest.raises(ValueError, match="stats_key"):
            EmpiricalBeliefs(memory=object())  # type: ignore[arg-type]


def _must_say_yes(result: str) -> PostConditionResult | None:
    if "yes" not in result:
        return PostConditionResult(passed=False, message="must say yes")
    return None


class TestLearningEndToEnd:
    """trace() -> build_graph -> backward -> consolidate, over a live run."""

    @pytest.mark.asyncio
    async def test_settlement_moves_posterior_after_backward(self):
        from ai_functions.optimizer import TextGradOptimizer, build_graph_from_result

        async with RuntimeHarness() as h:
            # A routed function with no post-condition, so its attempt passes
            # locally — a success that downstream feedback will then overturn.
            @ai_function[str](structured_output=False)
            def research(task: str) -> str:
                """Research: {task}"""

            model = ScriptedModel([Turn(text="sources: a, b", output_tokens=10)])
            cands = {"cheap": Candidate("cheap", research.replace(model=model), Prices(input=1.0, output=1.0))}
            beliefs = EmpiricalBeliefs()
            fn = EconomicFunction(cands, value=0.10, beliefs=beliefs, budget=1.0)

            run = await _trace_on(h, fn, task="research X")

            recs = await attempts(run)
            assert recs and recs[0].local_score == 1.0  # locally a success
            assert "100% pass" in beliefs.stats()["cheap"]  # displayed rate is evidence only

            # Reconstruct the graph with the function among the backends (it is
            # its own host), so the run's decision parameter matches and settles.
            from ai_functions.experimental.economics.function import DECISION_PARAMETER
            from ai_functions.types.graph import GradFeedback

            opt = TextGradOptimizer()
            graph = await build_graph_from_result(run, [fn])
            decision = next(p for p in graph.parameters if p.name == DECISION_PARAMETER)
            assert decision.requires_grad and decision.backend is fn

            # The downstream consumer judges the routing worthless (score 0.0).
            # Seed the decision parameter as the backward pass would, then
            # consolidate — the host settles the run's records.
            decision.gradients.append(GradFeedback(text="paraphrased news, needed primary sources", score=0.0))
            opt.consolidate(graph)

            # settle overrode the local success -> the displayed rate drops to 0
            assert "0% pass" in beliefs.stats()["cheap"]


async def _trace_on(h, fn, **kwargs):  # noqa: ANN001, ANN201
    """Trace ``fn`` on the harness coordinator so its event log is reachable."""
    handle = await h.spawn(fn)
    from ai_functions.types.graph import Result

    value = await handle.run(**kwargs)
    return Result(value=value, coordinator=h.coordinator, thread_id=handle.id, inputs=[])
