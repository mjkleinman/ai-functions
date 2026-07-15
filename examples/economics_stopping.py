"""Stop when the next attempt isn't worth its cost.

Reviewing tricky code has diminishing returns: each independent pass over a
dense lock-free C module finds a couple of defects, increasingly overlapping
with what earlier passes already found. ``@economic`` is the door for exactly
this shape of work: passing attempts fold into a running report via
``merge``, each attempt's reward is booked as the *marginal* dollar gain it
added, and the search stops on its own once another pass is no longer
expected to pay for itself.

Everything is declared in dollars: each unique confirmed defect is worth 2
cents. No estimator, no stopping threshold, no attempt count to tune — the
default ``DiminishingReturns`` beliefs project the next pass's yield from
the gains observed so far (a heuristic geometric projection; the budget is
the backstop).
"""

import asyncio
import logging

from _buggy_c import BUGGY_C, PLANTED
from _economics_models import HAIKU
from _utils import display, rule
from pydantic import BaseModel, Field

from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult
from ai_functions.experimental.economics import attempts, economic

VALUE_PER_DEFECT = 0.02  # a unique confirmed defect is worth 2 cents
NUM_PLANTED = len(PLANTED)


class Defect(BaseModel):
    function: str = Field(description="Name of the function containing the defect")
    kind: str = Field(description="e.g. data race, use-after-free, integer overflow, OOB access")
    description: str = Field(description="What the defect is and how it manifests")


class Report(BaseModel):
    defects: list[Defect]


def merge_reports(running: Report, new: Report) -> Report:
    """Union defects across passes, deduplicating by function name."""
    seen = {d.function: d for d in running.defects}
    for d in new.defects:
        seen.setdefault(d.function, d)
    return Report(defects=list(seen.values()))


def at_least_one(result: Report) -> PostConditionResult | None:
    if not result.defects:
        return PostConditionResult(passed=False, message="Report at least one defect")
    return None


# value scores the MERGED report, so each pass's booked reward is the
# marginal value of the defects it added. The search stops when the
# projected next gain no longer covers a pass; the budget is the backstop.
@economic(
    models=[HAIKU],
    value=lambda report: VALUE_PER_DEFECT * len(report.defects),
    merge=merge_reports,
    budget=0.10,
)
@ai_function[Report](post_conditions=[at_least_one])
def review(source: str):
    """You are reviewing this C module. Report any real bugs you find: for each,
    name the function and explain what goes wrong. Report only bugs you are
    confident are real, not style issues or hypothetical concerns.

    ```c
    {source}
    ```"""


async def main():
    logging.basicConfig(level=logging.WARNING)

    rule("Adaptive stopping: review until it's not worth continuing")
    display("Target", f"Lock-free C: MPSC ring buffer + open-addressing cache\n{NUM_PLANTED} planted defects", "text")

    # trace() runs like a plain call but keeps the run inspectable, so the
    # per-pass economics can be read back afterwards.
    run = await review.trace(source=BUGGY_C)
    report = run.value
    records = await attempts(run)

    hits = sorted(d.function for d in report.defects if d.function in PLANTED)
    misses = sorted(set(PLANTED) - set(hits))
    false_positives = [d for d in report.defects if d.function not in PLANTED]

    lines = []
    for i, r in enumerate(records, 1):
        # Under @economic, each record's reward is the marginal dollar value
        # the pass added to the running report.
        lines.append(f"pass {i}  marginal gain ${r.reward:.3f} vs cost ${r.cost:.4f}")
    lines.append("")
    lines.append("stopping rule: stopped when one more pass was no longer expected to cover its cost")
    lines.append("")
    lines.append(
        f"planted {NUM_PLANTED} · found {len(hits)} · "
        f"false positives {len(false_positives)} · spent ${sum(r.cost for r in records):.4f}"
    )
    lines.append("")
    for fn in hits:
        lines.append(f"  ✓ {fn}: {PLANTED[fn]}")
    for fn in misses:
        lines.append(f"  ✗ missed {fn}: {PLANTED[fn]}")
    for d in false_positives:
        lines.append(f"  ? {d.function} (not planted): {d.description[:70]}")

    display("Search decisions", "\n".join(lines), lang="text")


if __name__ == "__main__":
    asyncio.run(main())
