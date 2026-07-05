"""Tests for ``ai_functions.testing`` — ScriptedModel and RuntimeHarness.

The testing harness is infrastructure every downstream test depends on, so
these tests verify the behaviors that are easy to get subtly wrong:
script playback, barrier-based concurrency control, ``wait_for`` both
paths, teardown, and contextvar isolation between harnesses.
"""

from __future__ import annotations

import asyncio

import pytest

from ai_functions import ai_function
from ai_functions.testing import (
    AwaitBarrier,
    RuntimeHarness,
    ScriptedModel,
    Turn,
)
from ai_functions.testing._barriers import await_barrier
from ai_functions.types import EventKind
from ai_functions.types.events import (
    MessageAssistantCompleteEvent,
    MessageAssistantTokenEvent,
)


@ai_function[str](structured_output=False)
def _echo(prompt: str) -> str:
    """Minimal AI function used as a vehicle for driving ScriptedModel."""
    return prompt


# ── Turn validation ──────────────────────────────────────────────────────


def test_turn_rejects_both_text_and_chunks() -> None:
    with pytest.raises(ValueError, match="text or text_chunks"):
        Turn(text="hi", text_chunks=("hi",))


def test_turn_rejects_empty() -> None:
    with pytest.raises(ValueError, match="must have"):
        Turn()


# ── Basic playback ───────────────────────────────────────────────────────


async def test_scripted_model_plays_back_text() -> None:
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="hello world")])
        handle = await h.spawn(_echo.replace(model=model))
        result = await handle.run("prompt")
        assert result.strip() == "hello world"
        assert model.remaining_turns == 0


async def test_text_is_auto_chunked_by_words() -> None:
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="one two three")])
        handle = await h.spawn(_echo.replace(model=model))
        await handle.run("x")
        tokens = await h.events(handle.id, kinds=[EventKind.MESSAGE_ASSISTANT_TOKEN])
        # "one " / "two " / "three"
        assert len(tokens) == 3
        texts = [t.text for t in tokens if isinstance(t, MessageAssistantTokenEvent)]
        assert "".join(texts) == "one two three"


async def test_explicit_chunks_stream_verbatim() -> None:
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text_chunks=("a", "b", "c"))])
        handle = await h.spawn(_echo.replace(model=model))
        await handle.run("x")
        tokens = await h.events(handle.id, kinds=[EventKind.MESSAGE_ASSISTANT_TOKEN])
        texts = [t.text for t in tokens if isinstance(t, MessageAssistantTokenEvent)]
        assert texts == ["a", "b", "c"]


async def test_agent_messages_captures_turns() -> None:
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="reply")])
        handle = await h.spawn(_echo.replace(model=model))
        await handle.run("prompt")
        messages = h.agent_messages(handle.id)
        assert [m.get("role") for m in messages] == ["user", "assistant"]


async def test_agent_messages_content_matches_assistant_complete_event() -> None:
    """The captured assistant turn should carry the same content as the
    MESSAGE_ASSISTANT_COMPLETE event — the harness's view and the event log
    must agree on what the model produced.
    """
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="hello")])
        handle = await h.spawn(_echo.replace(model=model))
        await handle.run("x")
        complete_events = [
            e
            for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_ASSISTANT_COMPLETE])
            if isinstance(e, MessageAssistantCompleteEvent)
        ]
        assert len(complete_events) == 1
        messages = h.agent_messages(handle.id)
        assistant_turns = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_turns) == 1
        # Both should reference the same content payload.
        assert assistant_turns[0].get("content") == complete_events[0].content


# ── Barriers ─────────────────────────────────────────────────────────────


async def test_await_barrier_mid_stream_blocks_until_released() -> None:
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text_chunks=("part1", AwaitBarrier("mid"), "part2"))])
        handle = await h.spawn(_echo.replace(model=model))
        fut = handle.run("x")
        # The first chunk streams before the barrier.
        await h.wait_for(handle.id, EventKind.MESSAGE_ASSISTANT_TOKEN)
        # The cycle is suspended on "mid" — future cannot complete.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(fut), timeout=0.05)
        tokens_mid = await h.events(handle.id, kinds=[EventKind.MESSAGE_ASSISTANT_TOKEN])
        assert len(tokens_mid) == 1
        h.release("mid")
        await fut
        tokens_after = await h.events(handle.id, kinds=[EventKind.MESSAGE_ASSISTANT_TOKEN])
        assert len(tokens_after) == 2


async def test_turn_await_before_blocks_start_of_turn() -> None:
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="hi", await_before="gate")])
        handle = await h.spawn(_echo.replace(model=model))
        fut = handle.run("x")
        # The cycle reaches the model call and suspends on "gate".
        await h.wait_for(handle.id, EventKind.MESSAGE_ASSISTANT_START)
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(fut), timeout=0.05)
        assert await h.events(handle.id, kinds=[EventKind.MESSAGE_ASSISTANT_TOKEN]) == []
        h.release("gate")
        await fut


async def test_release_is_idempotent() -> None:
    async with RuntimeHarness() as h:
        h.release("nobody_waiting")
        h.release("nobody_waiting")
        model = ScriptedModel([Turn(text="ok", await_before="already_released")])
        h.release("already_released")  # release before the stream starts
        handle = await h.spawn(_echo.replace(model=model))
        result = await handle.run("x")
        assert result.strip() == "ok"


# ── wait_for ─────────────────────────────────────────────────────────────


async def test_wait_for_already_happened_resolves_immediately() -> None:
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="done")])
        handle = await h.spawn(_echo.replace(model=model))
        await handle.run("x")
        # COMPLETED is already in the log — wait_for should resolve without
        # hitting the subscription path. A very tight timeout proves it.
        event = await h.wait_for(handle.id, EventKind.COMPLETED, timeout=0.01)
        assert event.kind == EventKind.COMPLETED


async def test_wait_for_times_out_when_event_never_arrives() -> None:
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="done")])
        handle = await h.spawn(_echo.replace(model=model))
        await handle.run("x")
        # APPROVAL_REQUEST never fires in this script.
        with pytest.raises(asyncio.TimeoutError):
            await h.wait_for(handle.id, EventKind.APPROVAL_REQUEST, timeout=0.05)


# ── Script exhaustion ────────────────────────────────────────────────────


async def test_script_exhaustion_surfaces_as_failed_cycle() -> None:
    async with RuntimeHarness() as h:
        model = ScriptedModel([])  # no turns scripted
        handle = await h.spawn(_echo.replace(model=model))
        fut = handle.run("x")
        with pytest.raises(Exception):  # noqa: B017, PT011 -- Strands may wrap the error
            await fut
        # And the runtime should have recorded a FAILED event for the cycle.
        failed = await h.events(handle.id, kinds=[EventKind.FAILED])
        assert len(failed) == 1


# ── Context isolation and teardown ───────────────────────────────────────


async def test_two_nested_harnesses_have_independent_barrier_registries() -> None:
    async with RuntimeHarness() as h1, RuntimeHarness() as h2:
        model = ScriptedModel([Turn(text="x", await_before="gate")])
        handle = await h2.spawn(_echo.replace(model=model))
        fut = handle.run("x")
        await h2.wait_for(handle.id, EventKind.MESSAGE_ASSISTANT_START)
        # Release on the *outer* harness must not resolve the inner barrier.
        h1.release("gate")
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(fut), timeout=0.05)
        h2.release("gate")
        await fut


async def test_barrier_used_outside_any_harness_raises() -> None:
    with pytest.raises(RuntimeError, match="no RuntimeHarness is active"):
        await await_barrier("x")


async def test_teardown_unwinds_pending_barriers() -> None:
    """Exiting the harness while a model is suspended on a barrier must not
    hang teardown — ``__aexit__`` releases everything and terminates threads.
    """
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="x", await_before="forever")])
        handle = await h.spawn(_echo.replace(model=model))
        fut = handle.run("x")
        await h.wait_for(handle.id, EventKind.MESSAGE_ASSISTANT_START)
        # Deliberately do not release "forever".
    # After __aexit__, the future must be resolved (terminate_now cancels it).
    assert fut.done()


async def test_worker_property_raises_before_enter() -> None:
    h = RuntimeHarness()
    with pytest.raises(RuntimeError, match="before __aenter__"):
        _ = h.worker
