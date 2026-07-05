"""Tests for I10: no dangling tool calls in reconstructed history.

When a cycle is cancelled (or a thread is forked) between the assistant
emitting ``toolUse`` blocks and the matching ``toolResult`` events landing,
the event log contains an orphaned assistant turn. ``reconstruct_messages``
must heal that state by synthesizing the missing ``toolResult`` blocks in
exactly the same format Strands itself uses — ``"Tool was interrupted."``
via ``generate_missing_tool_result_content``. The healing must cover both:

- the *last* message (matches Strands'
  ``Agent._convert_prompt_to_messages``);
- *internal* dangling turns (matches Strands'
  ``RepositorySessionManager._fix_dangling_tool_uses``).

Together these are what Strands' own code would produce when it detects
the broken history, so the cache prefix stays stable (I9) regardless of
which path re-runs the reconstruction.
"""

from __future__ import annotations

from ai_functions import ai_function
from ai_functions.ai_thread.reconstruction import reconstruct_messages
from ai_functions.testing import AwaitBarrier, RuntimeHarness, ScriptedModel, Turn
from ai_functions.types import ThreadId
from ai_functions.types.events import (
    MessageAssistantCompleteEvent,
    MessageUserEvent,
    ToolResultEvent,
)


def _toolresult_ids_after(messages: list[dict], index: int) -> list[str]:  # type: ignore[type-arg]
    """Return toolResult ids in ``messages[index+1]``, or ``[]`` if absent."""
    if index + 1 >= len(messages):
        return []
    return [b["toolResult"]["toolUseId"] for b in messages[index + 1].get("content", []) if "toolResult" in b]


def _tooluse_ids(message: dict) -> list[str]:  # type: ignore[type-arg]
    """Return toolUse ids in a message."""
    return [b["toolUse"]["toolUseId"] for b in message.get("content", []) if "toolUse" in b]


# ── Unit-level healer: pure-function inputs ──────────────────────────────


def test_heal_last_message_with_unmatched_toolUse() -> None:
    """Dangling toolUse as the last message gets a fresh user message appended."""
    from ai_functions.types.events import EventId
    from ai_functions.types.ids import MessageId

    events = [
        MessageUserEvent(thread_id=ThreadId("t1"), text="hi"),
        MessageAssistantCompleteEvent(
            thread_id=ThreadId("t1"),
            message_id=MessageId("m1"),
            id=EventId("e1"),
            content=[
                {"text": "let me check"},
                {"toolUse": {"toolUseId": "tu-1", "name": "add", "input": {"a": 1, "b": 2}}},
            ],
        ),
    ]
    messages = reconstruct_messages(events)
    # Original two messages + one synthesized healer.
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"
    healer = messages[2]
    assert len(healer["content"]) == 1
    healer_block = healer["content"][0]
    assert "toolResult" in healer_block
    tr = healer_block["toolResult"]
    assert tr["toolUseId"] == "tu-1"
    assert tr["status"] == "error"
    assert tr["content"] == [{"text": "Tool was interrupted."}]


def test_heal_partial_tool_completion() -> None:
    """Some toolResults present, others missing → extend the existing user message."""
    from ai_functions.types.events import EventId
    from ai_functions.types.ids import MessageId

    events = [
        MessageUserEvent(thread_id=ThreadId("t1"), text="compute"),
        MessageAssistantCompleteEvent(
            thread_id=ThreadId("t1"),
            message_id=MessageId("m1"),
            id=EventId("e1"),
            content=[
                {"toolUse": {"toolUseId": "tu-A", "name": "add", "input": {"a": 1, "b": 2}}},
                {"toolUse": {"toolUseId": "tu-B", "name": "mul", "input": {"a": 3, "b": 4}}},
            ],
        ),
        ToolResultEvent(
            thread_id=ThreadId("t1"),
            message_id=MessageId("m1"),
            id=EventId("e2"),
            tool_use_id="tu-A",
            status="success",
            content=[{"text": "3"}],
        ),
    ]
    messages = reconstruct_messages(events)
    assert len(messages) == 3
    user_msg = messages[2]
    assert user_msg["role"] == "user"
    # Both A (real success) and B (synthesized interruption) must be present.
    ids = [b["toolResult"]["toolUseId"] for b in user_msg["content"]]
    assert set(ids) == {"tu-A", "tu-B"}
    # B is the synthesized one.
    tu_b = next(b for b in user_msg["content"] if b["toolResult"]["toolUseId"] == "tu-B")
    assert tu_b["toolResult"]["status"] == "error"
    assert tu_b["toolResult"]["content"] == [{"text": "Tool was interrupted."}]


def test_heal_internal_dangling_turn() -> None:
    """Dangling toolUse in the middle of history also gets healed."""
    from ai_functions.types.events import EventId
    from ai_functions.types.ids import MessageId

    # Cycle 1: assistant emits toolUse, never completes (dangling).
    # Cycle 2: user sends a fresh prompt, assistant replies with text.
    events = [
        MessageUserEvent(thread_id=ThreadId("t1"), text="call tool"),
        MessageAssistantCompleteEvent(
            thread_id=ThreadId("t1"),
            message_id=MessageId("m1"),
            id=EventId("e1"),
            content=[
                {"toolUse": {"toolUseId": "tu-1", "name": "add", "input": {"a": 1, "b": 1}}},
            ],
        ),
        # ← cancel happens here; no TOOL_RESULT emitted
        MessageUserEvent(thread_id=ThreadId("t1"), text="new prompt"),
        MessageAssistantCompleteEvent(
            thread_id=ThreadId("t1"),
            message_id=MessageId("m2"),
            id=EventId("e2"),
            content=[{"text": "ok here is my answer"}],
        ),
    ]
    messages = reconstruct_messages(events)
    # Expected: user / assistant(toolUse) / user(synthesized) / user(new prompt) / assistant
    # The healer is inserted between the dangling assistant and the following user.
    assert len(messages) == 5
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert "toolUse" in messages[1]["content"][0]
    # Healer message.
    assert messages[2]["role"] == "user"
    tr_block = messages[2]["content"][0]
    assert "toolResult" in tr_block
    assert tr_block["toolResult"]["toolUseId"] == "tu-1"
    assert tr_block["toolResult"]["content"] == [{"text": "Tool was interrupted."}]
    # Subsequent cycle intact.
    assert messages[3]["role"] == "user"
    assert messages[3]["content"][0].get("text") == "new prompt"
    assert messages[4]["role"] == "assistant"


def test_heal_is_idempotent_when_all_tools_matched() -> None:
    """No synthesis when every toolUse has a real matching toolResult."""
    from ai_functions.types.events import EventId
    from ai_functions.types.ids import MessageId

    events = [
        MessageUserEvent(thread_id=ThreadId("t1"), text="hi"),
        MessageAssistantCompleteEvent(
            thread_id=ThreadId("t1"),
            message_id=MessageId("m1"),
            id=EventId("e1"),
            content=[{"toolUse": {"toolUseId": "tu-1", "name": "add", "input": {}}}],
        ),
        ToolResultEvent(
            thread_id=ThreadId("t1"),
            message_id=MessageId("m1"),
            id=EventId("e2"),
            tool_use_id="tu-1",
            status="success",
            content=[{"text": "3"}],
        ),
    ]
    messages = reconstruct_messages(events)
    assert len(messages) == 3
    # Only the real result; no "Tool was interrupted." block.
    user_msg = messages[2]
    texts = [c.get("text") for b in user_msg["content"] for c in b["toolResult"]["content"]]
    assert "Tool was interrupted." not in texts


def test_heal_preserves_order_of_tooluse_ids() -> None:
    """Synthesized blocks appear in the order of the toolUse blocks they heal."""
    from ai_functions.types.events import EventId
    from ai_functions.types.ids import MessageId

    events = [
        MessageAssistantCompleteEvent(
            thread_id=ThreadId("t1"),
            message_id=MessageId("m1"),
            id=EventId("e1"),
            content=[
                {"toolUse": {"toolUseId": "tu-A", "name": "a", "input": {}}},
                {"toolUse": {"toolUseId": "tu-B", "name": "b", "input": {}}},
                {"toolUse": {"toolUseId": "tu-C", "name": "c", "input": {}}},
            ],
        ),
    ]
    messages = reconstruct_messages(events)
    assert len(messages) == 2
    ids_in_order = [b["toolResult"]["toolUseId"] for b in messages[1]["content"]]
    assert ids_in_order == ["tu-A", "tu-B", "tu-C"]


# ── End-to-end: cancel mid-tool call produces valid reconstructed history ─


@ai_function[str](structured_output=False)
def _simple(prompt: str) -> str:
    return prompt


async def test_cancel_mid_tool_call_leaves_healed_history() -> None:
    """Cancelling between toolUse emission and tool execution heals on read.

    The assistant emits a toolUse block that never gets a matching result
    in the event log (the cycle was cancelled before tools ran). The
    reconstructed history must have a synthesized toolResult so a follow-up
    model call sees a valid conversation.
    """
    import asyncio

    from strands.tools import tool

    @tool
    def slow_add(a: int, b: int) -> int:
        """Never completes — barrier blocks it indefinitely."""
        return a + b  # unreachable in this test, but needed for signature

    @ai_function[str](structured_output=False, tools=[slow_add])
    def _with_slow(prompt: str) -> str:
        return prompt

    async with RuntimeHarness() as h:
        # The model emits a tool call. A barrier inside text_chunks lets us
        # cancel after toolUse has been emitted but before the tool actually
        # runs — approximating the "cancel mid tool call" scenario.
        model = ScriptedModel(
            [
                Turn(
                    text_chunks=("thinking...", AwaitBarrier("mid")),
                    tool_calls=(("slow_add", {"a": 1, "b": 2}),),
                ),
                Turn(text="unused"),
            ]
        )
        handle = await h.spawn(_with_slow.replace(model=model))
        fut = handle.run("compute 1+2")
        from ai_functions.types import EventKind

        await h.wait_for(handle.id, EventKind.MESSAGE_ASSISTANT_TOKEN)
        await handle.cancel()
        h.release("mid")
        try:
            await fut
        except asyncio.CancelledError:
            pass

        events = await h.events(handle.id)
        messages = reconstruct_messages(events)

        # Every toolUse in the reconstructed history has a matching toolResult.
        for idx, msg in enumerate(messages):
            tu_ids = _tooluse_ids(msg)  # type: ignore[arg-type]
            if not tu_ids:
                continue
            tr_ids = _toolresult_ids_after(messages, idx)  # type: ignore[arg-type]
            assert set(tu_ids) <= set(tr_ids), f"dangling toolUse at message {idx}: tu_ids={tu_ids}, tr_ids={tr_ids}"


async def test_fork_during_tool_call_produces_healed_child_history() -> None:
    """A child forked from a thread with dangling toolUse sees a healed history."""
    import asyncio

    from strands.tools import tool

    @tool
    def slow_add(a: int, b: int) -> int:
        """Barrier-blocked tool."""
        return a + b

    @ai_function[str](structured_output=False, tools=[slow_add])
    def _with_slow(prompt: str) -> str:
        return prompt

    async with RuntimeHarness() as h:
        model = ScriptedModel(
            [
                Turn(
                    text_chunks=("thinking...", AwaitBarrier("mid")),
                    tool_calls=(("slow_add", {"a": 5, "b": 7}),),
                ),
                Turn(text="never reached"),
            ]
        )
        parent = await h.spawn(_with_slow.replace(model=model))
        fut = parent.run("compute 5+7")
        from ai_functions.types import EventKind

        await h.wait_for(parent.id, EventKind.MESSAGE_ASSISTANT_TOKEN)
        await parent.cancel()
        h.release("mid")
        try:
            await fut
        except asyncio.CancelledError:
            pass

        # Fork while the parent history is still dangling.
        child = await parent.fork()
        child_events = await h.events(child.id)
        child_messages = reconstruct_messages(child_events)

        # Child's reconstruction must also be self-consistent.
        for idx, msg in enumerate(child_messages):
            tu_ids = _tooluse_ids(msg)  # type: ignore[arg-type]
            if not tu_ids:
                continue
            tr_ids = _toolresult_ids_after(child_messages, idx)  # type: ignore[arg-type]
            assert set(tu_ids) <= set(tr_ids), (
                f"fork child has dangling toolUse at {idx}: tu_ids={tu_ids}, tr_ids={tr_ids}"
            )
