# Strands AI Functions

> ⚠️ **Work in progress.** This branch (`wip/v2`) targets v2 of AI Functions.
> APIs are unstable and there is no PyPI release yet. For the published
> version, use [`main`](https://github.com/strands-labs/ai-functions)
> (`pip install strands-ai-functions`).

Strands AI Functions is a Python library for building reliable AI-powered
applications through a new abstraction: functions that behave like standard
Python functions, but whose body is written in natural language and executed
by a reasoning LLM rather than a CPU. AI Functions extend what's expressible in
standard programming by offering developers a computational model that can
solve tasks not easily written as traditional code. They leverage text
generation capabilities (e.g., to write summaries or retrieve information), and
they can also generate and execute code on the fly to process inputs and return
native Python objects. For example, an AI Function can load a user-uploaded
file in an arbitrary format and convert it to a normalized `DataFrame` for use
in the rest of the workflow.

Direct integration of AI agents into standard workflows is often avoided
because of their non-deterministic nature. AI Functions address this through
*post-conditions*: instead of "prompt-and-pray", you specify explicit
conditions the output must satisfy, and the library runs a self-correcting
loop until they pass — preventing cascading errors in larger workflows.

AI Functions are also composable and distributable. A single function is
stateless, but it can be spawned as a stateful **AI Thread** that accumulates
history, several threads can run side by side on a **coordinator** and talk to
each other, and the whole runtime can be split across machines without
changing the calling code.

## Getting Started

### Prerequisites

- Python 3.13 or higher (Python 3.14+ recommended for native
  [t-string](https://peps.python.org/pep-0750/) template literal support)
- Valid credentials for a supported model provider (AWS Bedrock, OpenAI, etc.)
- (Recommended) [uv](https://docs.astral.sh/uv/getting-started/installation/)
  to run the provided examples

### Configure Model Provider

Strands AI Functions supports various
[model providers](https://strandsagents.com/latest/documentation/docs/user-guide/concepts/model-providers/).
AI Functions use the same default provider as Strands (Amazon Bedrock). Change
the `model` option to use a different provider, model, or authentication
options (see also
[Configuring Credentials](https://strandsagents.com/latest/documentation/docs/user-guide/quickstart/python/#configuring-credentials)):

```python
from ai_functions import ai_function
from strands.models.bedrock import BedrockModel
from strands.models.openai import OpenAIModel

# Use Claude Sonnet on Amazon Bedrock (default if `model` is not specified)
model = BedrockModel(model_id="anthropic.claude-sonnet-4-20250514-v1:0")

# Or use a different provider and model
model = OpenAIModel(client_args={"api_key": "<KEY>"}, model_id="gpt-4o")

@ai_function(str, model=model)
def my_function() -> None:
    """[...]"""
```

## AI Function Basics

An AI Function is defined with the `@ai_function` decorator. The desired output type is
the first positional argument of the decorator, and the task is described
in the docstring (which is interpreted as a template and filled in with the
call arguments). AI Functions can return any data type: primitives (`str`,
`int`, ...), pydantic models, and native Python objects — the library takes
care of conversion and validation.

The example below builds a meeting-summarization function with validation. The
result is checked against the provided post-conditions: if any fail, the model
is automatically prompted to correct the errors and try again, up to
`max_attempts` times. The function only returns once all post-conditions pass.

```python
import asyncio

from pydantic import BaseModel

from ai_functions import ai_function
from ai_functions.ai_thread import PostConditionResult


# The structured output type for our meeting-summarization function.
class MeetingSummary(BaseModel):
    attendees: list[str]
    summary: str
    action_items: list[str]


# Post-conditions can be any Python function that validates the output...
def check_length(response: MeetingSummary) -> None:
    """Post-condition: summary must be less than 50 words."""
    length = len(response.summary.split())
    assert length < 50, f"Summary must be less than 50 words long, but is {length}."


# ... or they can be AI functions, since AI functions *are* just functions.
@ai_function(PostConditionResult)
def check_style(response: MeetingSummary):
    """
    Check if the summary below satisfies the following criteria:
    - It must use bullet points
    - It must provide the reader with the necessary context

    <summary>
    {response.summary}
    </summary>
    """


# The main AI function: behavior is specified both through the prompt
# (generated from the docstring) and the post-conditions. The library ensures
# the result passes every requirement before returning it.
@ai_function(MeetingSummary, post_conditions=[check_length, check_style], max_attempts=5)
def summarize_meeting(transcripts: str):
    """
    Write a summary of the following meeting in less than 50 words.
    <transcripts>
    {transcripts}
    </transcripts>
    """


# `summarize_meeting` can now be awaited like any other async function.
async def main() -> None:
    transcripts = "..."
    meeting_summary = await summarize_meeting(transcripts)  # a MeetingSummary instance
    print(meeting_summary)


if __name__ == "__main__":
    asyncio.run(main())
```

AI Functions are `async` by default; use `fn.run_sync(...)` to call one from synchronous
code. Each direct call is a one-shot: behind the scenes, a private coordinator
and worker are created, one cycle runs on a fresh thread, and everything is torn down before
the result is returned — no history is kept between calls.

## Stateful AI Threads

When the same conversation should be reused across several calls, a function
can be `spawn`ed into a stateful **AI Thread**. The handle returned by
`spawn()` refers to a live thread on which every `run` accumulates history.

```python
import asyncio

from ai_functions import ai_function


@ai_function(str)
def assistant(message: str):
    """Answer the following question: {message}"""


async def main() -> None:
    handle = await assistant.spawn()

    r1 = await handle.run(message="What is the capital of France?")
    print(f"Turn 1: {r1}")

    # The agent sees the full conversation history from turn 1.
    r2 = await handle.run(message="What about Germany?")
    print(f"Turn 2: {r2}")


if __name__ == "__main__":
    asyncio.run(main())
```

A handle also supports `notify` (inject out-of-band context without starting a
cycle), `fork` (branch a conversation, sharing the past but diverging from the
fork point), and explicit lifecycle control (`pause`, `resume`, `cancel`,
`terminate`). See the [tutorial](docs/tutorial.md) for details.

## A Team of AI Threads

Several threads can run side by side on the same **coordinator** and talk to
each other. An `InMemoryCoordinator` is the registry and router; a
`LocalWorker` is the execution engine that hosts threads and drives their
cycles. Every `AIThread` is automatically given two tools — `list_threads` (to
discover its peers) and `send_message` (to delegate work to them) — so no
manual wiring is needed.

```python
import asyncio

from ai_functions import ai_function
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from strands_tools import exa


# `researcher` knows how to look things up on the web.
@ai_function(str, tools=[exa])
def researcher(topic: str):
    """
    Research the following topic on the web and return a concise factual
    summary, citing the sources you used:

    {topic}
    """


# `writer` produces short reports and can delegate fact-finding to its teammate.
@ai_function(str)
def writer(brief: str):
    """
    Write a short report based on the following brief: {brief}

    Work with a teammate named `researcher` (who has access to web search).
    Send them messages on what to search, or follow-up messages to request
    missing information.
    """


async def main() -> None:
    # The coordinator is the registry and router; the worker runs the threads.
    coord = InMemoryCoordinator()
    worker = await LocalWorker(coord).register()

    # Spawn one thread of each kind, on the same coordinator.
    _ = await coord.spawn(researcher, thread_name="researcher")
    writer_handle = await coord.spawn(writer, thread_name="writer")

    # Kick off the writer. It reaches out to the researcher on its own, via the
    # `send_message` tool, whenever it needs a fact.
    report = await writer_handle.run(
        brief="recent progress on room-temperature superconductors",
    )
    print(report)

    await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
```

`send_message` supports three modes — `"wait"` (block on the peer's reply),
`"fire_and_forget"` (schedule and return immediately), and
`"continue_then_receive"` (dispatch, end the current cycle, and resume
automatically when the reply arrives). Children spawned with `parent_id` have
their token usage roll up to the parent, and the entire event log is available
for observability via `coordinator.on(...)`. Workflows that are not naturally
expressed as a single prompt can be written as custom **Spawnables** in plain
Python. See the [tutorial](docs/tutorial.md) for all of these.

## Distributed Operation

The coordinator and workers do not have to live in the same process. A
`CoordinatorEndpoint` is a WebSocket server that fronts a coordinator; a
`CoordinatorClient` connects to it and behaves like a local `Coordinator`. From
the application's perspective the only change is swapping
`InMemoryCoordinator` for `CoordinatorClient.connect(url)` — `coord.spawn`,
`handle.run`, `send_message`, and event subscriptions all work the same. This
lets threads with local state stay in the caller's process while still
dispatching work to remote workers. See the
[tutorial](docs/tutorial.md#distributed-operation) for the full treatment.

## Examples

This repository includes several complete, runnable examples, each building on
the previous one:

| Example | Demonstrates |
|---|---|
| `01_one_shot.py` | Direct one-shot calls, structured output, post-conditions |
| `02_multi_turn.py` | A stateful thread accumulating conversation history |
| `03_runtime_basics.py` | Coordinator + worker, parent/child threads, events |
| `04_spawnable_workflow.py` | A plain-Python workflow as a custom Spawnable |
| `05_two_workers_local.py` | Two workers on one coordinator, cross-thread messaging |
| `06_two_workers_remote.py` | The same team over a remote WebSocket endpoint |

To run them, first configure credentials for one of the supported model
providers (see [Configure Model Provider](#configure-model-provider)), then:

```bash
# Clone the repository
git clone -b wip/v2 https://github.com/strands-labs/ai-functions.git
cd ai-functions/examples

# Optional: enable rich tool visualization in the terminal
export STRANDS_TOOL_CONSOLE_MODE="enabled"

# Run an example using uv (recommended)
uv run 01_one_shot.py
```

**Note**: You may need to change the examples to use a different model provider.

## Tutorial

For a full walkthrough — AI Functions, stateful threads, teams, distributed
operation, custom spawnables, observability, and the internals — see the
[tutorial](docs/tutorial.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
