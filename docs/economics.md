# Economics-aware Execution

Post-conditions give an AI function correctness semantics: a result either passes verification or it does not. The `ai_functions.experimental.economics` module adds the economics on top — what a result is worth in dollars, what each candidate model's tokens cost — and uses them to decide which model to try, whether to switch after a failure, and when to stop. Everything shares one currency, so one rule governs every attempt — **an attempt is worth making only when it is expected to yield more than it costs** — and the search policy picks among the worthwhile attempts.

This enables:

- **Cost-aware model routing** — each call goes to the model expected to maximize profit, optionally switches models when verification fails, and declines tasks that are not worth attempting at all. Routing sharpens automatically as per-model pass rates and costs are learned from the call stream.
- **Task-aware routing with an LLM forecaster** — a lightweight AI function reads each task and predicts, per candidate, its chance of success and its cost, so easy tasks route to cheap models and hard tasks route to strong models within the same function.
- **Routing that can improve with feedback** — routing decisions plug into the library's [optimization loop](tutorial.md#memory-and-optimization): feedback on a workflow's final output propagates back to the routing decisions that produced it, corrects their statistics, and distills task-routing notes for future calls.
- **Adaptive stopping for graded tasks** — when results are worth different amounts (a review that finds four defects beats one that finds two), `@economic` keeps sampling while another attempt is expected to add more value than it costs, and stops on its own.

Note: the module is experimental — the API may change in future releases.

## Contents

- [Routing with `@routed`](#routing-with-routed)
- [Task-aware routing that learns: `LLMForecaster`](#task-aware-routing-that-learns-llmforecaster)
- [How a search ends](#how-a-search-ends)
- [Graded tasks: `@economic`](#graded-tasks-economic)
- [Customizing the search](#customizing-the-search)
- [Going further](#going-further)
- [Examples](#examples)

## Routing with `@routed`

`@routed` stacks on `@ai_function` and routes each call to the model that maximizes the expected gain, given *your* dollar value for a successful outcome and the prices of the candidate models:

```python
from ai_functions import ai_function
from ai_functions.experimental.economics import PricedModel, Prices, routed

# The candidate models, priced at what *you* pay (dollars per million tokens).
HAIKU = PricedModel(model="global.anthropic.claude-haiku-4-5-20251001-v1:0",
                    prices=Prices(input=1.00, output=5.00))
SONNET = PricedModel(model="global.anthropic.claude-sonnet-4-6",
                     prices=Prices(input=3.00, output=15.00))


# A solved instance is worth 50 cents to us; check_sat is an ordinary
# post-condition that verifies the assignment against the clauses.
@routed(models=[HAIKU, SONNET], value=0.50)
@ai_function(post_conditions=[check_sat])
def solve(clauses: str, n_vars: int) -> Assignment:
    """Find a satisfying assignment for this 3-SAT formula over variables x1..x{n_vars}.

    {clauses}"""


result = await solve(clauses=clauses, n_vars=8)   # called like any AI Function
```

Three pieces define the economics, and each is doing a specific job:

- **`value`** is what a verified success is worth to you, in dollars (a useful anchor: what would you pay a human to do this reliably?). It is the reward side of every decision — a model is only worth trying if `value` times its chance of passing exceeds its expected cost, so `value` must exceed a candidate's per-attempt cost or the function will rightly decline to run it.
- **`models`** are the candidates, each a `PricedModel` pairing a model with the per-token prices you pay for it (`Prices` also accepts `cache_read`/`cache_write` rates, and an optional `description` that estimators can read).
- **The post-conditions define success.** They are the verifier that decides whether an attempt passed; without them, the cheap model would always "succeed" and there would be nothing to escalate on.

A call then works as follows: the probability of passing and the expected cost are estimated per candidate; the search tries the candidate with the best expected profit; returns its result if it passes; switches to the next candidate if it fails; and abstains — raising `Abstained` rather than knowingly wasting money — when no candidate is worth its cost. The cost of every attempt is *measured* from its event log at the candidate's token prices, not estimated.

Routing learns by default. Out of the box, estimates come from `EmpiricalBeliefs`: per-candidate pass rates and average costs, starting from a uniform prior (so untried candidates get explored) and updated after every attempt. Over a batch of calls the cheap model keeps the tasks it handles and the strong model inherits the ones it doesn't. You can watch this happen:

```python
print(solve.beliefs.stats())
# {'haiku': '20% pass over 5 attempts, avg cost $0.0101, ...',
#  'sonnet': '79% pass over 14 attempts, avg cost $0.0487, ...'}
```

Three optional knobs bound and shape the search: `budget` is a hard dollar cap per call (distinct from `value`: `value` drives choices, `budget` bounds spend); `max_tries` (default 1) caps attempts per candidate; `carry_context=True` seeds each escalation with the failed attempt's transcript, so the stronger model sees what was tried and why it was rejected instead of starting fresh.

See `examples/economics_escalate.py` for a runnable comparison — a batch of SAT instances routed cheap-first versus sent straight to the strong model, with the dollar savings printed — and `examples/economics_learning.py` for the beliefs converging over a batch.

## Task-aware routing that learns: `LLMForecaster`

Population statistics treat every call the same. When tasks vary — "quick factual query" next to "compare regulatory regimes" — pass `beliefs=LLMForecaster(...)`: a lightweight AI Function reads each task and predicts, per candidate, its chance of passing and its expected cost, anchored on the learned statistics. Its state lives in your memory schema, one `RoutingMemory` field per routed function, so everything it learns persists across processes:

```python
from pydantic import BaseModel, Field

from ai_functions import ai_function
from ai_functions.experimental.economics import LLMForecaster, RoutingMemory, routed
from ai_functions.memory import JSONMemoryBackend
from ai_functions.optimizer import TextGradOptimizer


class Memory(BaseModel):
    research_routing: RoutingMemory = Field(default_factory=RoutingMemory)


memory = JSONMemoryBackend(schema=Memory, actor_id="demo", path="memory.json")


@routed(
    models=[HAIKU, SONNET],
    value=0.05,  # good sources are worth 5 cents
    beliefs=LLMForecaster(memory=memory, memory_key="research_routing"),
    budget=0.10,
)
@ai_function(tools=[web_search], post_conditions=[cited])
def research(query: str) -> Sources:
    """Research this topic on the web and return the key findings with sources:

    {query}"""
```

This is also where routing connects to the library's optimization loop. The post-conditions are only *local* checks (here: findings carry source URLs); whether the research was actually *useful* is decided downstream, by whoever consumes it. Run the function under `trace()` and that judgment can flow back:

```python
run = await research.trace(query="What changed in the EU AI Act's GPAI obligations in 2025?")

optimizer = TextGradOptimizer()
await optimizer.step(
    run,
    "Too shallow: this needed primary sources (the Act's text, Commission guidance).",
    backends=[memory, research],   # an economic function hosts its own routing parameters
)
```

One `optimizer.step` teaches the router two things at once. The *numeric* channel corrects the statistics: the routed model's attempts in that run are re-scored by the downstream feedback, so a model whose results pass local checks but don't hold up downstream sees its pass rate sink anyway. The *text* channel distills the feedback into the forecaster's casebook (`research_routing/notes` in your memory), steering future task-dependent routing — "regulatory comparisons: haiku's sources too thin, route strong." Feedback given on a *downstream* output propagates to the routed stages that fed it, exactly as in [Memory and optimization](tutorial.md#memory-and-optimization); `examples/economics_workflow.py` runs the full loop on a two-stage pipeline, both stages routed, settled by one line of feedback on the final report.

## How a search ends

A search stops when no remaining attempt is expected to pay for itself. If an attempt passed by then, the call returns its result. When none did, the failure is one of three typed exceptions, each carrying the attempt trail in `.records`:

- **`Abstained`** — nothing was tried: no candidate's expected reward covered its cost. This is a feature, not a failure mode: a router that cannot say "not worth it" silently overpays on hopeless tasks. It tells you the task looks unprofitable *before* money is spent — raise `value`, or stop sending this class of task here.
- **`CandidatesExhausted`** — every profitable candidate ran (up to its `max_tries`) and no attempt passed: the task defeated the models you gave it.
- **`BudgetExceeded`** — a candidate would still be worth trying, but its expected cost no longer fits the remaining budget: the cap, not the economics, stopped the search.

## Graded tasks: `@economic`

`@routed` covers pass/fail tasks. Some tasks are *graded*: a review that finds four real defects is worth more than one that finds two, and a second attempt can add value even after the first one "passed". For these, `value` becomes a function of the result, and the question changes from *which candidate* to *how many attempts*:

```python
from ai_functions.experimental.economics import economic


def merge_reports(running: Report, new: Report) -> Report:
    """Union of defects across passes, deduplicated by function name."""
    seen = {d.function: d for d in running.defects}
    for d in new.defects:
        seen.setdefault(d.function, d)
    return Report(defects=list(seen.values()))


# Each distinct defect found is worth 2 cents. value prices the MERGED
# report, so each pass's booked reward is the marginal value it added.
@economic(
    models=[HAIKU],
    value=lambda report: 0.02 * len(report.defects),
    merge=merge_reports,
    budget=0.10,
)
@ai_function(post_conditions=[at_least_one])
def review(source: str) -> Report:
    """You are reviewing this C module. Report any real bugs you find ..."""
```

Each passing result folds into a running result via `merge`, and each attempt books the marginal gain it added: a pass that finds three new defects books $0.06; a pass that only rediscovers known ones books 0.00. The search continues while the projected next gain covers the next attempt's cost, and stops on its own when it no longer does — no attempt count to tune, no quality threshold to pick. The `budget` is mandatory: the gain projection is a heuristic (by default, each gain is expected to be a fraction of the last), and the budget bounds the damage when it is wrong. A constant `value` is rejected at decoration time and should be used with `@routed` decorator.

The `@economic` decorator covers several familiar search shapes:

- **Keep the best.** Omit `merge` and it defaults to `keep_best(value)`: the search keeps sampling, a new result replaces the incumbent only by scoring strictly higher, and the best is returned — best-of-n, except n is not chosen: each redraw happens only while it is expected to be worth its cost. Note that a draw that fails to beat the incumbent books $0, which ends the search under the default belief.
- **Accumulation.** Pass a `merge` that folds results together, as in the review above: overlap with earlier draws books $0, so gains shrink as the pool depletes and the search stops itself.
- **Multiple models.** Pass several `models` and each round draws whichever currently promises the most. With the default beliefs the arms share one projected gain curve, so they are not meaningfully differentiated — supply per-arm `beliefs` if the arms differ in character. `examples/economics_graded_multi_model.py` shows one way: calibrate each model with a few graded attempts to build an empirical reward distribution and average cost per model, and the search runs the Pandora's box rule over distinct arms.

See `examples/economics_stopping.py` for the review search run against a C module with planted defects, printing each pass's marginal gain against its cost.

## Customizing the search

**Policies.** How candidates are ordered and when the search stops is a pluggable `policy=`. `@routed` defaults to `Greedy` (highest expected profit, stop once a reward is in hand); `ReservationPricePolicy` — `@economic`'s policy — instead orders candidates by their *reservation price*, following the [Pandora's box rule (Weitzman, 1979)](https://www.jstor.org/stable/1910412), which prices in the option to escalate and can prefer a cheap long-shot that `Greedy` would skip. `Exhaustive` tries everything the budget allows, cheapest first.

**Custom beliefs.** When you know the feature that governs task difficulty, you can skip learning it: subclass `Beliefs` and compute the estimates directly. `estimate` receives a `TaskView` carrying both the rendered prompt and the structured call `arguments`:

```python
from ai_functions.experimental.economics import Beliefs
from ai_functions.experimental.economics.search import Bernoulli, Estimate


class RatioBeliefs(Beliefs):
    """3-SAT hardness is governed by the clause/variable ratio; read it
    from the call's own arguments instead of learning it from outcomes."""

    async def estimate(self, task, candidates, value, history):
        ratio = (task.arguments["clauses"].count("\n") + 1) / task.arguments["n_vars"]
        return {
            c.label: Estimate(
                dist=Bernoulli(p=self._pass_probability(c.label, ratio), value=value),
                cost=self._expected_cost(c, ratio),
            )
            for c in candidates
        }
```

`estimate` must return an estimate for every candidate it is given. For known workloads and tests, `Beliefs.fixed({label: Estimate(...)})` returns constant estimates with no learning. Note that each decorator's default beliefs are matched to its search — pass-rate statistics for `@routed`, gain projection for `@economic` — care should be used when matching beliefs to the value function/scalar, merge operation, and search policy. See `examples/economics_route.py` for the complete `RatioBeliefs`.

**Custom candidates.** When model swaps are not enough, pass `candidates=[Candidate(label=..., fn=..., prices=...)]` instead of `models=`: a candidate is *any* `AIFunction` plus its prices — a different thinking budget, a different prompt, or a non-LLM heuristic wrapped as a function. Build variants with `fn.replace(...)`.

## Going further

**Inspecting a run.** `await fn.trace(...)` runs like a plain call but keeps the event log, so the per-attempt economics can be read back afterwards:

```python
from ai_functions.experimental.economics import attempts, decisions, spend

run = await review.trace(source=BUGGY_C)

for r in await attempts(run):        # one AttemptRecord per attempt, in order
    print(r.candidate, r.reward, r.cost, r.local_score)

await decisions(run)                 # the ranked estimation rounds, in order
await spend(run)                     # total dollars booked by the run and its subtree
```

Every attempt runs as a child thread and emits durable events, so `spend` gives dollar accounting across a whole tree of economic calls from the event log alone. A failed run carries the same records on the exception's `.records`.

**Deciding without executing.** `await fn.plan(...)` runs one estimation round and returns a `Decision` without attempting anything: the candidate the search would try first (`None` means it would abstain — the no-exception way to anticipate `Abstained`) and the full ranking, for dashboards and debugging. A caller that takes the decision and executes it *itself* closes the learning loop with `decision.report(result, cost)`; without it, the beliefs never see the outcome. See `examples/economics_route.py`.

**Persisting the plain statistics.** `LLMForecaster` persists everything it learns through its `RoutingMemory` field. To persist the default statistics without a forecaster, construct them with a backend directly: `EmpiricalBeliefs(memory=backend, stats_key="research_routing/stats")` reloads at construction and rewrites after every update, so a new process resumes routing where the last one left off.

## Examples

| Example | Shows |
|---|---|
| `economics_escalate.py` | `@routed` basics: cheap-first escalation on a SAT batch, dollar savings vs. a straight-to-strong baseline |
| `economics_learning.py` | `EmpiricalBeliefs` converging over a batch: exploration from a uniform prior, routing sharpening with evidence |
| `economics_route.py` | A custom task-dependent `Beliefs`; `plan()` previews each decision |
| `economics_stopping.py` | `@economic`: marginal-gain bookkeeping and adaptive stopping on a code review with planted defects |
| `economics_workflow.py` | Two routed stages in a pipeline; `LLMForecaster`, persistence, and feedback settling both stages via `optimizer.step` |
| `economics_graded_multi_model.py` | `@economic` over two models with per-arm beliefs: calibrate each model's reward distribution and average cost, then search for the best graded result using the Pandora's box rule |

Run any of them from the `examples/` folder with `uv run economics_<name>.py`.
