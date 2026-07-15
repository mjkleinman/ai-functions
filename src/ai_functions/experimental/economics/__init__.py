"""Economic semantics for AI functions: value, cost, and optimal effort.

Post-conditions give a function correctness semantics; this module adds the
economics — what success is worth, what attempts cost, and therefore which
model to route to, when to escalate to a fallback, and when to stop.
Everything is denominated in dollars (E1), so the stopping rule is one
sentence: stop when no remaining attempt is expected to pay for itself.

Two decorators are the entry points — ``@routed`` decides *which* candidate
attempts a task (model routing, cascade/fallback, abstention), ``@economic``
decides *how many times* one candidate should try (repeated sampling with
automatic stopping). Both construct an :class:`EconomicFunction`, which
mirrors the calling surface of ``AIFunction`` and adds ``plan()``.

The top level exports the decorator path. The pure search core —
:class:`~ai_functions.experimental.economics.search.Search`,
:class:`~ai_functions.experimental.economics.search.Estimate`, the reward distributions,
and the :class:`~ai_functions.experimental.economics.search.Policy` implementations —
lives in :mod:`ai_functions.experimental.economics.search` for power users.
"""

from __future__ import annotations

from .beliefs import (
    Beliefs,
    DiminishingReturns,
    EmpiricalBeliefs,
    LLMForecaster,
    ObservedAttempt,
    RoutingMemory,
)
from .decorators import economic, routed
from .function import (
    ATTEMPT_EVENT,
    DECISION_EVENT,
    Decision,
    EconomicFunction,
    Value,
    attempts,
    decisions,
    keep_best,
    spend,
)
from .types import (
    Abstained,
    AttemptRecord,
    BudgetExceeded,
    Candidate,
    CandidatesExhausted,
    EconomicsError,
    PricedModel,
    Prices,
    Ranking,
    RecordId,
    TaskView,
)

__all__ = [
    "ATTEMPT_EVENT",
    "Abstained",
    "AttemptRecord",
    "Beliefs",
    "BudgetExceeded",
    "Candidate",
    "CandidatesExhausted",
    "DECISION_EVENT",
    "Decision",
    "DiminishingReturns",
    "EconomicFunction",
    "EconomicsError",
    "EmpiricalBeliefs",
    "LLMForecaster",
    "ObservedAttempt",
    "PricedModel",
    "Prices",
    "Ranking",
    "RecordId",
    "RoutingMemory",
    "TaskView",
    "Value",
    "attempts",
    "decisions",
    "economic",
    "keep_best",
    "routed",
    "spend",
]
