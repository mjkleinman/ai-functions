"""Priced models shared by the economics examples.

The library deliberately ships no price table — what a token costs depends on
your contract, region, and provisioning. The prices below are ILLUSTRATIVE
EXAMPLES ONLY and do not reflect any real price list: define your own
``PricedModel`` collection with the rates you actually pay.

The models are built with an explicit ``max_tokens`` cap. Beyond bounding
cost and latency, the cap is what makes escalation *happen* in these examples:
a cheap model that would eventually solve a hard instance by reasoning for
64k tokens instead hits the ceiling, fails, and the search escalates to the
stronger model — exactly the behaviour a cost-aware layer exists to automate.
"""

from strands.models import BedrockModel

from ai_functions.experimental.economics import PricedModel, Prices

# Cap generation so a rambling cheap model fails fast (and cheap) rather than
# burning tens of thousands of tokens on one hard instance.
MAX_TOKENS = 8_000
REGION = "us-west-2"

HAIKU = PricedModel(
    model=BedrockModel(
        model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0",
        max_tokens=MAX_TOKENS,
        region_name=REGION,
    ),
    prices=Prices(input=1.00, output=5.00, cache_read=0.10, cache_write=1.25),
    label="haiku",
    description="Fast and cheap; handles straightforward tasks well",
)

SONNET = PricedModel(
    model=BedrockModel(
        model_id="global.anthropic.claude-sonnet-4-6",
        max_tokens=MAX_TOKENS,
        region_name=REGION,
    ),
    prices=Prices(input=3.00, output=15.00, cache_read=0.30, cache_write=3.75),
    label="sonnet",
    description="Balanced; strong at multi-step reasoning and constraints",
)
