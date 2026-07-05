"""Round-trip test: ``reconstruct_messages(events) ≈ agent.messages``.

This is the single most load-bearing property in the runtime: if a
thread's in-memory ``agent.messages`` diverges from what
``reconstruct_messages`` produces for the same event log, then
rehydration (resume from history, remote runtime, fork) will feed the
model a prefix that doesn't match what
the previous cycle actually saw — and I7's cache-prefix-stability claim
is broken.

Strands itself does NOT merge consecutive same-role messages (see
``event_loop.streaming._normalize_messages`` — it only touches
assistant-side blank-text handling; every provider formatter maps input
messages 1:1 to provider-shaped messages). So the round-trip property is
byte-for-byte equality after stripping the ``metadata`` key Strands
attaches as post-hoc telemetry — nothing more.

Each scenario runs a cycle end-to-end with ``ScriptedModel``, captures
the live ``agent.messages`` via the harness, reconstructs the history
from the emitted events, and asserts the two are equivalent.
"""

from __future__ import annotations

from strands.tools import tool

from ai_functions import ai_function
from ai_functions.ai_thread.reconstruction import reconstruct_messages
from ai_functions.testing import (
    AwaitBarrier,
    RuntimeHarness,
    ScriptedModel,
    Turn,
    assert_messages_equivalent,
    normalize_messages,
)
from ai_functions.types import EventKind


@ai_function[str](structured_output=False)
def _text_only(prompt: str) -> str:
    return prompt


@tool
def _add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def _mul(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@ai_function[str](structured_output=False, tools=[_add, _mul])
def _with_tools(prompt: str) -> str:
    return prompt


# ── Plain text ────────────────────────────────────────────────────────────


async def test_single_text_turn_roundtrip() -> None:
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="hello there")])
        handle = await h.spawn(_text_only.replace(model=model))
        await handle.run("hi")
        assert_messages_equivalent(
            h.agent_messages(handle.id),
            reconstruct_messages(await h.events(handle.id)),
        )


async def test_multi_turn_conversation_roundtrip() -> None:
    """Three prompts, three replies — the full history round-trips.

    Each prompt arrives on a different cycle, so the bridge appends one
    user message per turn (never two in a row); this is the happy path
    where ``reconstruct_messages`` and ``agent.messages`` agree.
    """
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="first reply"), Turn(text="second reply"), Turn(text="third reply")])
        handle = await h.spawn(_text_only.replace(model=model))
        await handle.run("first")
        await handle.run("second")
        await handle.run("third")
        live = h.agent_messages(handle.id)
        recon = reconstruct_messages(await h.events(handle.id))
        assert len(live) == 6
        assert_messages_equivalent(live, recon)


# ── Tool use ──────────────────────────────────────────────────────────────


async def test_single_tool_call_roundtrip() -> None:
    """Assistant emits one tool call, the tool runs, assistant replies."""
    async with RuntimeHarness() as h:
        model = ScriptedModel(
            [
                Turn(text="let me add", tool_calls=(("_add", {"a": 2, "b": 3}),)),
                Turn(text="the answer is 5"),
            ]
        )
        handle = await h.spawn(_with_tools.replace(model=model))
        await handle.run("what is 2+3")
        assert_messages_equivalent(
            h.agent_messages(handle.id),
            reconstruct_messages(await h.events(handle.id)),
        )


async def test_tool_use_no_text_turn_roundtrip() -> None:
    """A tool-only assistant turn (no preceding text) must also round-trip."""
    async with RuntimeHarness() as h:
        model = ScriptedModel(
            [
                Turn(tool_calls=(("_add", {"a": 10, "b": 20}),)),
                Turn(text="30"),
            ]
        )
        handle = await h.spawn(_with_tools.replace(model=model))
        await handle.run("10+20")
        assert_messages_equivalent(
            h.agent_messages(handle.id),
            reconstruct_messages(await h.events(handle.id)),
        )


# ── Multi-tool and tool-error ─────────────────────────────────────────────


async def test_multiple_tool_calls_in_one_turn_roundtrip() -> None:
    """Two tool calls in one assistant turn must round-trip.

    Strands appends a single user message with two ``toolResult`` blocks;
    ``reconstruct_messages`` collapses the matching pair of
    ``ToolResultEvent`` s into the same shape.
    """
    async with RuntimeHarness() as h:
        model = ScriptedModel(
            [
                Turn(
                    text="two tools",
                    tool_calls=(
                        ("_add", {"a": 1, "b": 2}),
                        ("_mul", {"a": 3, "b": 4}),
                    ),
                ),
                Turn(text="done"),
            ]
        )
        handle = await h.spawn(_with_tools.replace(model=model))
        await handle.run("hi")
        assert_messages_equivalent(
            h.agent_messages(handle.id),
            reconstruct_messages(await h.events(handle.id)),
        )


@tool
def _boom(x: int) -> int:
    """Always raises."""
    raise ValueError("kaboom")


@ai_function[str](structured_output=False, tools=[_boom])
def _with_failing_tool(prompt: str) -> str:
    return prompt


async def test_tool_error_roundtrip() -> None:
    """Tool errors captured via ``AfterToolCallEvent.result`` round-trip."""
    async with RuntimeHarness() as h:
        model = ScriptedModel(
            [
                Turn(text="calling", tool_calls=(("_boom", {"x": 1}),)),
                Turn(text="recovered"),
            ]
        )
        handle = await h.spawn(_with_failing_tool.replace(model=model))
        await handle.run("please")
        assert_messages_equivalent(
            h.agent_messages(handle.id),
            reconstruct_messages(await h.events(handle.id)),
        )


# ── Concurrency: messages arriving mid-cycle ──────────────────────────────


async def test_multiple_user_messages_drain_together_roundtrip() -> None:
    """Two user messages drained in a single BeforeModelCallEvent.

    Block the first model call with ``await_before``; while the dispatcher
    is suspended at that boundary, deliver two ``notify`` texts.
    When the barrier releases, the bridge drains both from the thread's
    inject buffer in one go, appending two ``Message`` s to
    ``agent.messages`` and emitting two ``MESSAGE_USER`` events.
    ``reconstruct_messages`` must produce the same shape.
    """
    async with RuntimeHarness() as h:
        model = ScriptedModel(
            [
                Turn(text="first reply", await_before="let_cycle_proceed"),
                Turn(text="second reply"),
            ]
        )
        handle = await h.spawn(_text_only.replace(model=model))
        # Kick off the first cycle; it will suspend before the model call.
        fut = handle.run("prompt one")
        await h.wait_for(handle.id, EventKind.STARTED)
        # Thread is now alive; deliver two extras to the message queue.
        await handle.notify("extra A")
        await handle.notify("extra B")
        # Release the barrier — the next BeforeModelCallEvent drains all
        # three (prompt + two extras) at once.
        h.release("let_cycle_proceed")
        await fut
        # Run one more cycle so the scripted second turn is consumed.
        await handle.run("prompt two")
        assert_messages_equivalent(
            h.agent_messages(handle.id),
            reconstruct_messages(await h.events(handle.id)),
        )


async def test_notify_during_tool_call_defers_to_next_boundary() -> None:
    """An notify arriving while the agent is mid-tool-call must not
    interleave with the tool_use → tool_result pair.

    I7 requires ``MESSAGE_USER`` events to appear only at
    ``BeforeModelCallEvent`` boundaries — never between a ``TOOL_CALL``
    and its matching ``TOOL_RESULT``/``TOOL_ERROR``. If it did, the
    reconstructed message list would slot a bare user message inside an
    assistant turn's tool-use span, which is invalid for every provider
    and corrupts the cache prefix.

    Verifies both event ordering and round-trip equivalence.
    """
    async with RuntimeHarness() as h:
        # Turn 1: assistant decides to call a tool; the tool's text chunk
        # sequence has a barrier so we can suspend between tool_use emission
        # and tool execution. Actually the barrier fires during streaming,
        # which is fine — the tool has not yet run by the time we release,
        # and the cycle continues through tool execution and the next
        # model call.
        model = ScriptedModel(
            [
                Turn(
                    text_chunks=("thinking ", AwaitBarrier("mid_tool")),
                    tool_calls=(("_add", {"a": 5, "b": 7}),),
                ),
                Turn(text="12"),
            ]
        )
        handle = await h.spawn(_with_tools.replace(model=model))
        fut = handle.run("what is 5+7")
        # Wait until the first assistant turn is under way and the barrier
        # has been hit (at least one token emitted).
        await h.wait_for(handle.id, EventKind.MESSAGE_ASSISTANT_TOKEN)
        # Deliver a message while the agent is stopped mid-stream.
        await handle.notify("by the way, hello")
        h.release("mid_tool")
        await fut

        events = await h.events(handle.id)
        kinds = [e.kind for e in events]

        # Find the tool call/result indices and the late MESSAGE_USER.
        tool_result_idx = kinds.index(EventKind.TOOL_RESULT)
        # The prompt's MESSAGE_USER fires before STARTED's work boundary;
        # the interjected "by the way, hello" is the LAST MESSAGE_USER.
        user_indices = [i for i, k in enumerate(kinds) if k == EventKind.MESSAGE_USER]
        assert len(user_indices) >= 2
        late_user_idx = user_indices[-1]
        # The interjected message must land AFTER the tool result — never
        # inside the tool_use → tool_result pair.
        assert late_user_idx > tool_result_idx, (
            f"MESSAGE_USER at {late_user_idx} should come after TOOL_RESULT at {tool_result_idx}; event order: {kinds}"
        )

        # And the full history must round-trip.
        assert_messages_equivalent(
            h.agent_messages(handle.id),
            reconstruct_messages(events),
        )


# ── Normalizer itself ─────────────────────────────────────────────────────


def test_normalizer_strips_metadata_from_assistant_messages() -> None:
    """Strands attaches ``metadata`` as post-hoc telemetry; strip it."""
    normalized = normalize_messages(
        [
            {
                "role": "assistant",
                "content": [{"text": "hi"}],
                "metadata": {  # type: ignore[typeddict-unknown-key]
                    "usage": {"inputTokens": 0, "outputTokens": 1, "totalTokens": 1},
                    "metrics": {"latencyMs": 0},
                },
            }
        ]
    )
    assert "metadata" not in normalized[0]  # type: ignore[operator]
    assert normalized[0]["content"] == [{"text": "hi"}]


def test_normalizer_does_not_merge_adjacent_user_messages() -> None:
    """Consecutive same-role messages are a real shape — don't hide them.

    Strands itself does not merge these; a normalizer that did would mask
    divergences the round-trip test exists to catch.
    """
    result = normalize_messages(
        [
            {"role": "user", "content": [{"text": "first"}]},
            {"role": "user", "content": [{"text": "second"}]},
        ]
    )
    assert len(result) == 2


def test_normalizer_is_deep_copy() -> None:
    """Mutating the returned list must not affect the input."""
    original: list[object] = [{"role": "user", "content": [{"text": "x"}]}]
    result = normalize_messages(original)  # type: ignore[arg-type]
    result[0]["content"].append({"text": "y"})  # type: ignore[arg-type, index]
    assert len(original[0]["content"]) == 1  # type: ignore[arg-type, index]
