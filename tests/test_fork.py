"""Tests for ``ThreadHandle.fork`` / ``LocalWorker.fork``.

Fork is the runtime's mechanism for branching a conversation: the child gets a
copy of the parent's event history, runs its own cycles, and does not
affect the parent. The tests below exercise the behaviours that are
easy to get subtly wrong:

- The child log starts as a byte-equal copy of the parent's (modulo
  ``thread_id`` and event ``id``).
- The child is independent: running the child does not emit events on
  the parent's log, and vice versa.
- ``reconstruct_messages`` on both sides matches ``agent.messages`` of
  each (a round-trip invariant regardless of fork).
- Forking at N=0 (no prior events) works.
- Forking after multiple cycles preserves the full conversation prefix.
"""

from __future__ import annotations

from typing import Any

import pytest

from ai_functions import ai_function
from ai_functions.ai_thread.reconstruction import reconstruct_messages
from ai_functions.protocols import Spawnable
from ai_functions.testing import RuntimeHarness, ScriptedModel, Turn, assert_messages_equivalent
from ai_functions.types import EventKind, ThreadContext


@ai_function(str, structured_output=False)
def _fn(prompt: str) -> str:
    return prompt


# ── Happy path ───────────────────────────────────────────────────────────


async def test_fork_copies_history_into_new_thread() -> None:
    """Child thread's event log starts with the parent's events (new ids, new thread_id)."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="first reply")])
        parent = await h.spawn(_fn.replace(model=model))
        await parent.run("first prompt")

        parent_events = await h.events(parent.id)
        child = await parent.fork()

        child_events = await h.events(child.id)
        # Same kinds, in order.
        assert [e.kind for e in child_events] == [e.kind for e in parent_events]
        # thread_id is rewritten; event ids are freshly minted.
        for parent_e, child_e in zip(parent_events, child_events, strict=True):
            assert child_e.thread_id == child.id
            assert child_e.id != parent_e.id


async def test_fork_child_reconstructs_same_message_history() -> None:
    """``reconstruct_messages`` on the child log produces the parent's conversation shape."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="hello there")])
        parent = await h.spawn(_fn.replace(model=model))
        await parent.run("prompt")
        child = await parent.fork()

        parent_msgs = reconstruct_messages(await h.events(parent.id))
        child_msgs = reconstruct_messages(await h.events(child.id))
        assert_messages_equivalent(parent_msgs, child_msgs)


async def test_fork_before_any_cycle_copies_empty_log() -> None:
    """A thread with no events can still be forked; the child log is empty."""
    async with RuntimeHarness() as h:
        parent = await h.spawn(_fn.replace(model=ScriptedModel([Turn(text="unused")])))
        child = await parent.fork()
        assert await h.events(child.id) == []


async def test_fork_preserves_event_order() -> None:
    """Multi-cycle parent history copies into the child in the same order."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="reply one"), Turn(text="reply two"), Turn(text="reply three")])
        parent = await h.spawn(_fn.replace(model=model))
        await parent.run("one")
        await parent.run("two")
        await parent.run("three")

        child = await parent.fork()
        child_kinds = [e.kind for e in await h.events(child.id)]
        # Three complete cycles = three STARTED/COMPLETED pairs.
        assert child_kinds.count(EventKind.STARTED) == 3
        assert child_kinds.count(EventKind.COMPLETED) == 3
        # RESULT always precedes its COMPLETED.
        for idx, kind in enumerate(child_kinds):
            if kind == EventKind.COMPLETED:
                prev = child_kinds[:idx]
                # The most recent RESULT should appear before this COMPLETED.
                assert EventKind.RESULT in prev


# ── Independence ─────────────────────────────────────────────────────────


async def test_fork_child_and_parent_have_independent_futures() -> None:
    """Running the child adds events only to the child's log."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="parent reply"), Turn(text="child reply")])
        parent = await h.spawn(_fn.replace(model=model))
        await parent.run("parent prompt")
        parent_before_fork = len(await h.events(parent.id))

        child = await parent.fork()
        await child.run("child prompt")

        parent_after_child_ran = await h.events(parent.id)
        assert len(parent_after_child_ran) == parent_before_fork, (
            "child cycles must not emit events onto the parent's log"
        )
        # Child's own log has parent's prefix plus its new cycle.
        child_kinds = [e.kind for e in await h.events(child.id)]
        assert child_kinds.count(EventKind.COMPLETED) == 2


async def test_fork_child_does_not_see_parent_events_after_fork() -> None:
    """Events emitted on the parent *after* the fork do not appear on the child."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="first"), Turn(text="after fork"), Turn(text="child")])
        parent = await h.spawn(_fn.replace(model=model))
        await parent.run("p1")

        child = await parent.fork()
        child_events_at_fork = await h.events(child.id)

        # Run another cycle on the parent.
        await parent.run("p2")

        # Child log is unchanged by the parent's subsequent work.
        child_events_later = await h.events(child.id)
        assert [e.id for e in child_events_later] == [e.id for e in child_events_at_fork]


async def test_fork_child_event_log_reflects_inherited_prefix() -> None:
    """After the child runs its own cycle, its event log = parent's prefix + new cycle."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="parent reply"), Turn(text="child-only reply")])
        parent = await h.spawn(_fn.replace(model=model))
        await parent.run("shared prompt")

        child = await parent.fork()
        await child.run("diverged prompt")

        # Reconstructed history shows 2 user turns + 2 assistant turns.
        child_recon = reconstruct_messages(await h.events(child.id))
        users = [m for m in child_recon if m.get("role") == "user"]
        assistants = [m for m in child_recon if m.get("role") == "assistant"]
        assert len(users) == 2
        assert len(assistants) == 2
        # First user text is the parent's prompt; second is the child's divergence.
        first_user_text = users[0]["content"][0].get("text")
        second_user_text = users[1]["content"][0].get("text")
        assert first_user_text == "shared prompt"
        assert second_user_text == "diverged prompt"


# ── Thread.fork — unsupported thread propagates NotImplementedError ─────────


class _UnforkableThread:
    """Minimal ``Thread`` that refuses to fork — exercises propagation."""

    name: str = "unforkable"

    @property
    def input_shape(self) -> Any:
        from ai_functions.types import InputShape

        return InputShape.NO_ARGS

    def to_thread(self) -> _UnforkableThread:
        return self

    async def execute(self, ctx: ThreadContext, *args: Any, **kwargs: Any) -> str:
        del ctx, args, kwargs
        return "ok"

    async def notify(self, text: str) -> None:
        del text

    def serialize_result(self, result: str) -> str:
        return result

    def deserialize_result(self, payload: str) -> str:
        return payload

    async def fork(self) -> Spawnable[Any, Any]:
        raise NotImplementedError("this thread does not support forking")

    async def teardown(self) -> None:
        return None


async def test_thread_fork_not_implemented_propagates() -> None:
    """A thread whose ``fork`` raises ``NotImplementedError`` surfaces cleanly."""
    async with RuntimeHarness() as h:
        handle = await h.spawn(_UnforkableThread())
        with pytest.raises(NotImplementedError):
            await handle.fork()
        # The source thread remains fully usable after a failed fork.
        assert await handle.run() == "ok"


# ── Coordinator.copy_events — standalone ───────────────────────────────────


async def _make_coord_with_threads(*tids: str) -> Any:
    """Build a coordinator pre-populated with ``tids`` for copy_events tests."""
    from ai_functions.runtime import InMemoryCoordinator
    from ai_functions.types import InputShape, ThreadId, ThreadInfo, ThreadStatus, WorkerId

    coord = InMemoryCoordinator()
    wid = WorkerId("worker-test")
    # Bypass worker adapter registration by stuffing a sentinel; we only
    # exercise the event log here.
    coord._workers[wid] = object()  # type: ignore[attr-defined]
    for tid_str in tids:
        tid = ThreadId(tid_str)
        await coord.register_thread(
            ThreadInfo(
                thread_id=tid,
                worker_id=wid,
                thread_name=None,
                input_shape=InputShape.NO_ARGS,
                status=ThreadStatus.NOT_STARTED,
                parent_id=None,
            ),
        )
    return coord


async def test_coordinator_copy_events_rewrites_thread_id_and_event_id() -> None:
    """``copy_events`` preserves kind and order, rewrites ``thread_id`` and ``id``."""
    from ai_functions.types import ThreadId
    from ai_functions.types.events import MessageUserEvent

    coord = await _make_coord_with_threads("src", "dst")
    src = ThreadId("src")
    dst = ThreadId("dst")

    coord.append_event(MessageUserEvent(thread_id=src, text="one"))
    coord.append_event(MessageUserEvent(thread_id=src, text="two"))

    await coord.copy_events(source_id=src, target_id=dst)

    target_events = await coord.get_events(dst)
    assert [e.kind for e in target_events] == [EventKind.MESSAGE_USER, EventKind.MESSAGE_USER]
    texts = [e.text for e in target_events if isinstance(e, MessageUserEvent)]
    assert texts == ["one", "two"]
    source_events = await coord.get_events(src)
    for s, t in zip(source_events, target_events, strict=True):
        assert t.thread_id == dst
        assert t.id != s.id


async def test_coordinator_copy_events_does_not_notify_subscribers() -> None:
    """``copy_events`` is a bulk seed — live subscribers see no callbacks fire."""
    from ai_functions.types import Event, ThreadId
    from ai_functions.types.events import MessageUserEvent

    coord = await _make_coord_with_threads("src", "dst")
    src = ThreadId("src")
    dst = ThreadId("dst")
    coord.append_event(MessageUserEvent(thread_id=src, text="seed"))

    received: list[Event] = []

    def _cb(event: Event) -> None:
        received.append(event)

    with coord.on(_cb, thread_id=dst):
        await coord.copy_events(source_id=src, target_id=dst)

    assert received == []
    # Events did land on the target despite no broadcast.
    assert len(await coord.get_events(dst)) == 1


async def test_coordinator_copy_events_unregistered_raises() -> None:
    """``copy_events`` raises ``ThreadNotFoundError`` if either thread is not registered."""
    from ai_functions.runtime import InMemoryCoordinator
    from ai_functions.types import ThreadId

    coord = InMemoryCoordinator()
    src = ThreadId("src")
    dst = ThreadId("dst")

    with pytest.raises(KeyError):
        await coord.copy_events(source_id=src, target_id=dst)

    coord2 = await _make_coord_with_threads("src")
    with pytest.raises(KeyError):
        await coord2.copy_events(source_id=ThreadId("src"), target_id=dst)


async def test_coordinator_copy_events_until_event_id_out_of_range_raises() -> None:
    """A non-existent ``until_event_id`` raises ``ValueError``."""
    from ai_functions.types import EventId, ThreadId
    from ai_functions.types.events import MessageUserEvent

    coord = await _make_coord_with_threads("src", "dst")
    src = ThreadId("src")
    dst = ThreadId("dst")
    coord.append_event(MessageUserEvent(thread_id=src, text="one"))

    with pytest.raises(ValueError):
        await coord.copy_events(
            source_id=src,
            target_id=dst,
            until_event_id=EventId("evt-does-not-exist"),
        )
