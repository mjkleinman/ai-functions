"""Cross-thread messaging via the default runtime tools.

The default ``config_hook`` installed on every ``AIFunction`` injects two
runtime-facing tools: ``list_threads`` and ``send_message``. These tests
exercise peer discovery and the three ``send_message`` modes (``wait``,
``fire_and_forget``, ``continue_then_receive``).
"""

from __future__ import annotations

import asyncio
import json

from ai_functions import ai_function
from ai_functions.testing import RuntimeHarness, ScriptedModel, Turn
from ai_functions.types import EventKind
from ai_functions.types.events import MessageUserEvent


@ai_function(str, structured_output=False)
def _chat(message: str) -> str:
    """Simple chat responder: {message}"""
    return message


def _tool_result_text(event) -> str:  # type: ignore[no-untyped-def]
    """Pull the text payload out of a TOOL_RESULT event."""
    block = event.content[0]
    text = block.get("text") if isinstance(block, dict) else None
    assert isinstance(text, str), f"TOOL_RESULT content missing text: {event.content!r}"
    return text


async def test_list_threads_shows_peers_and_marks_self() -> None:
    """``list_threads`` returns every registered thread with a self flag."""
    async with RuntimeHarness() as h:
        alice_model = ScriptedModel(
            [
                Turn(tool_calls=(("list_threads", {}),)),
                Turn(text="done"),
            ],
        )
        bob_model = ScriptedModel([Turn(text="never runs")])

        alice = await h.spawn(_chat.replace(model=alice_model), thread_name="alice")
        _bob = await h.spawn(_chat.replace(model=bob_model), thread_name="bob")

        await alice.run("what threads exist?")

        tool_results = [e for e in await h.events(alice.id) if e.kind == EventKind.TOOL_RESULT]
        assert len(tool_results) == 1
        payload = json.loads(_tool_result_text(tool_results[0]))
        by_name = {t["thread_name"]: t for t in payload["threads"]}

        assert set(by_name) == {"alice", "bob"}
        assert by_name["alice"]["is_self"] is True
        assert by_name["bob"]["is_self"] is False


async def test_send_message_wait_returns_peer_reply() -> None:
    """``mode='wait'`` awaits the peer's cycle and returns its reply string."""
    async with RuntimeHarness() as h:
        bob = await h.spawn(
            _chat.replace(model=ScriptedModel([Turn(text="pong from bob")])),
            thread_name="bob",
        )
        alice_model = ScriptedModel(
            [
                Turn(
                    tool_calls=(
                        (
                            "send_message",
                            {"thread_id": str(bob.id), "message": "ping", "mode": "wait"},
                        ),
                    ),
                ),
                Turn(text="alice is done"),
            ],
        )
        alice = await h.spawn(_chat.replace(model=alice_model), thread_name="alice")

        result = await alice.run("ask bob")
        assert result.strip() == "alice is done"

        tool_results = [e for e in await h.events(alice.id) if e.kind == EventKind.TOOL_RESULT]
        assert len(tool_results) == 1
        ack = _tool_result_text(tool_results[0])
        # Bob's reply is returned directly as the tool result.
        assert "pong from bob" in ack

        # Bob actually ran a cycle.
        bob_completes = [e for e in await h.events(bob.id) if e.kind == EventKind.COMPLETED]
        assert len(bob_completes) == 1


async def test_send_message_fire_and_forget_returns_immediately() -> None:
    """``mode='fire_and_forget'`` schedules peer.run(...) and returns an ack."""
    async with RuntimeHarness() as h:
        bob = await h.spawn(
            _chat.replace(model=ScriptedModel([Turn(text="bob's reply nobody reads")])),
            thread_name="bob",
        )
        alice_model = ScriptedModel(
            [
                Turn(
                    tool_calls=(
                        (
                            "send_message",
                            {
                                "thread_id": str(bob.id),
                                "message": "go do something",
                                "mode": "fire_and_forget",
                            },
                        ),
                    ),
                ),
                Turn(text="alice carries on"),
            ],
        )
        alice = await h.spawn(_chat.replace(model=alice_model), thread_name="alice")

        result = await alice.run("dispatch to bob")
        assert result.strip() == "alice carries on"

        tool_results = [e for e in await h.events(alice.id) if e.kind == EventKind.TOOL_RESULT]
        ack = _tool_result_text(tool_results[0])
        assert ack.startswith("ok: dispatched"), f"unexpected ack: {ack!r}"

        # Bob's cycle runs independently; wait for it to complete.
        for _ in range(100):
            if any(e.kind == EventKind.COMPLETED for e in await h.events(bob.id)):
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("bob's fire_and_forget cycle never completed")


async def test_send_message_continue_then_receive_schedules_followup() -> None:
    """``continue_then_receive`` kicks a follow-up cycle on the sender with the peer reply."""
    async with RuntimeHarness() as h:
        bob = await h.spawn(
            _chat.replace(model=ScriptedModel([Turn(text="bob says hello back")])),
            thread_name="bob",
        )
        # Alice: first run dispatches via continue_then_receive and ends.
        # The runtime tool schedules a follow-up alice.run(notification)
        # automatically; that second cycle is driven by Alice's second
        # scripted turn.
        alice_model = ScriptedModel(
            [
                # Cycle 1 — driven by our own alice.run("greet bob"):
                Turn(
                    tool_calls=(
                        (
                            "send_message",
                            {
                                "thread_id": str(bob.id),
                                "message": "hi bob",
                                "mode": "continue_then_receive",
                            },
                        ),
                    ),
                ),
                Turn(text="alice cycle 1 done"),
                # Cycle 2 — triggered by the follow-up alice.run(notification):
                Turn(text="alice saw bob's reply"),
            ],
        )
        alice = await h.spawn(_chat.replace(model=alice_model), thread_name="alice")

        cycle1 = await alice.run("greet bob")
        assert cycle1.strip() == "alice cycle 1 done"

        # Wait for the follow-up cycle to complete.
        for _ in range(200):
            completeds = [e for e in await h.events(alice.id) if e.kind == EventKind.COMPLETED]
            if len(completeds) >= 2:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("alice's continue_then_receive follow-up never completed")

        # The follow-up cycle's user turn carries Bob's reply.
        alice_user_events = [
            e for e in await h.events(alice.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        # Turn 1's user prompt + Turn 2's user prompt (the auto-notification).
        assert len(alice_user_events) == 2
        notification = alice_user_events[1].text
        assert "bob says hello back" in notification
        assert str(bob.id) in notification


async def test_send_message_continue_then_receive_rejects_non_str_sender() -> None:
    """If the sender doesn't accept a single-str prompt, the tool returns an error."""
    async with RuntimeHarness() as h:
        bob = await h.spawn(
            _chat.replace(model=ScriptedModel([Turn(text="never reached")])),
            thread_name="bob",
        )

        @ai_function(str, structured_output=False)
        def _two_arg(a: str, b: str) -> str:
            """Structured: {a} / {b}"""
            return f"{a}-{b}"

        alice_model = ScriptedModel(
            [
                Turn(
                    tool_calls=(
                        (
                            "send_message",
                            {
                                "thread_id": str(bob.id),
                                "message": "hi bob",
                                "mode": "continue_then_receive",
                            },
                        ),
                    ),
                ),
                Turn(text="alice is done"),
            ],
        )
        alice = await h.spawn(
            _two_arg.replace(model=alice_model),
            thread_name="alice_multi",
        )

        result = await alice.run("foo", "bar")
        assert result.strip() == "alice is done"

        tool_results = [e for e in await h.events(alice.id) if e.kind == EventKind.TOOL_RESULT]
        ack = _tool_result_text(tool_results[0])
        assert ack.startswith("error:")
        assert "continue_then_receive" in ack
        assert "wait" in ack


async def test_send_message_to_self_returns_error() -> None:
    """The tool refuses self-sends."""
    async with RuntimeHarness() as h:
        alice_model = ScriptedModel(
            [
                Turn(
                    tool_calls=(
                        (
                            "send_message",
                            {"thread_id": "_placeholder_", "message": "hi me", "mode": "wait"},
                        ),
                    ),
                ),
                Turn(text="done"),
            ],
        )
        alice = await h.spawn(_chat.replace(model=alice_model), thread_name="alice")

        # Patch the scripted model's placeholder with alice's real id.
        alice_model._turns[0].tool_calls[0][1]["thread_id"] = str(alice.id)  # type: ignore[attr-defined]  # noqa: SLF001

        result = await alice.run("try to self-send")
        assert result.strip() == "done"

        tool_results = [e for e in await h.events(alice.id) if e.kind == EventKind.TOOL_RESULT]
        ack = _tool_result_text(tool_results[0])
        assert ack.startswith("error:"), f"expected error, got: {ack!r}"


async def test_send_message_unknown_mode_returns_error() -> None:
    """An unrecognised mode returns an error ack without invoking the peer."""
    async with RuntimeHarness() as h:
        bob = await h.spawn(
            _chat.replace(model=ScriptedModel([Turn(text="never runs")])),
            thread_name="bob",
        )
        alice_model = ScriptedModel(
            [
                Turn(
                    tool_calls=(
                        (
                            "send_message",
                            {"thread_id": str(bob.id), "message": "?", "mode": "async"},
                        ),
                    ),
                ),
                Turn(text="done"),
            ],
        )
        alice = await h.spawn(_chat.replace(model=alice_model), thread_name="alice")

        await alice.run("bad mode")
        tool_results = [e for e in await h.events(alice.id) if e.kind == EventKind.TOOL_RESULT]
        ack = _tool_result_text(tool_results[0])
        assert ack.startswith("error:")
        assert "unknown mode" in ack
        # Bob's cycle must not have been triggered.
        bob_completes = [e for e in await h.events(bob.id) if e.kind == EventKind.COMPLETED]
        assert len(bob_completes) == 0
