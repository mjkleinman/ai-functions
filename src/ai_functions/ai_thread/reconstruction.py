"""Rebuild Strands-format message history from a stored event sequence."""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never, cast

from strands.tools._tool_helpers import generate_missing_tool_result_content
from strands.types.content import ContentBlock, Message
from strands.types.tools import ToolResult

if TYPE_CHECKING:
    from strands.types.content import Messages

from ..types import (
    ContextSummarizedEvent,
    Event,
    MessageAssistantCompleteEvent,
    MessageUserEvent,
    RenderableEvent,
    ToolCallEvent,
    ToolResultEvent,
    is_renderable_event,
)


def render_renderable_events(events: list[RenderableEvent]) -> Messages:
    """Render a sequence of :data:`RenderableEvent` s into Strands ``Messages``.

    The pure-rendering core. Assumes the caller has already handled any
    ``ContextSummarizedEvent`` boundaries and filtered out
    observability-only events that carry no conversational content; this
    function walks what remains and emits the matching message list. It
    does not run the I10 healing pass — that is
    :func:`reconstruct_messages`'s responsibility, so the healer runs
    exactly once on the fully-assembled message list.

    Args:
        events: Renderable events in chronological order.

    Returns:
        The Strands ``Messages`` list these events produce.

    Ensures:
        - Each :class:`MessageUserEvent` yields one user ``Message`` with
          a single text block (see I7).
        - Each :class:`MessageAssistantCompleteEvent` yields one assistant
          ``Message`` whose content is the stored list of content blocks.
        - A contiguous run of :class:`ToolResultEvent` events collapses
          into one user ``Message`` carrying one ``toolResult`` block per
          event; a subsequent ``MessageUserEvent``,
          ``MessageAssistantCompleteEvent``, or end-of-list flushes the
          group.
        - :class:`ToolCallEvent` is observability-only and produces no
          message.
    """
    messages: list[Message] = []
    pending_tool_results: list[ContentBlock] = []

    def flush_tool_results() -> None:
        if not pending_tool_results:
            return
        messages.append(Message(role="user", content=list(pending_tool_results)))
        pending_tool_results.clear()

    for event in events:
        match event:
            case MessageUserEvent():
                flush_tool_results()
                messages.append(Message(role="user", content=[ContentBlock(text=event.text)]))
            case MessageAssistantCompleteEvent():
                flush_tool_results()
                messages.append(Message(role="assistant", content=list(event.content)))
            case ToolResultEvent():
                # Repack into a Strands ``ToolResult`` dict and splat as a
                # ``{"toolResult": ...}`` ``ContentBlock``. This matches
                # ``strands.event_loop.event_loop.py``'s ``tool_result_message``
                # builder byte-for-byte, so the reconstructed block is
                # identical to what Strands appends to ``agent.messages``
                # (preserves the cache prefix — I9).
                pending_tool_results.append(
                    ContentBlock(
                        toolResult=ToolResult(
                            toolUseId=event.tool_use_id,
                            status=event.status,
                            content=event.content,
                        )
                    )
                )
            case ToolCallEvent():
                # Observability-only: the tool call is already carried by the
                # preceding assistant turn's ``toolUse`` block.
                continue
            case _:
                assert_never(event)

    flush_tool_results()
    return messages


def reconstruct_messages(events: list[Event]) -> Messages:
    """Convert stored events into a Strands-format ``Messages`` list.

    Args:
        events: Events for one thread in chronological order.

    Returns:
        A message list ready for a Strands agent.

    Requires:
        ``events`` is sorted by append order (oldest first).

    Ensures:
        - If the input contains one or more
          :class:`ContextSummarizedEvent` events, the one with the
          greatest index (the last one in append order) defines a
          boundary: every event at that index or earlier is dropped from
          rendering and ``event.new_history`` is rendered in its place
          via :func:`render_renderable_events`, then events strictly
          after the boundary render normally and append to the result.
        - Events outside the renderable subset — lifecycle, token-usage,
          session, approval, streaming fragments — are filtered out
          before rendering.
        - The output obeys every ``Ensures`` of
          :func:`render_renderable_events`.
        - Every ``toolUse`` block in the reconstructed history is
          followed by a matching ``toolResult`` block (I10). Where the
          event log contains no matching ``TOOL_RESULT``, the missing
          blocks are synthesized by Strands'
          ``generate_missing_tool_result_content`` helper — the same text
          and shape Strands' own session manager produces when it detects
          orphaned ``toolUse``. The I10 healing pass runs exactly once,
          on the fully-assembled message list, after any summary splice.
    """
    # Find the last ``ContextSummarizedEvent`` — it defines the boundary.
    # Earlier summary events are superseded and contribute nothing.
    summary_idx: int | None = None
    for i in range(len(events) - 1, -1, -1):
        if isinstance(events[i], ContextSummarizedEvent):
            summary_idx = i
            break

    messages: list[Message] = []
    if summary_idx is not None:
        summary_event = events[summary_idx]
        assert isinstance(summary_event, ContextSummarizedEvent)
        messages.extend(render_renderable_events(list(summary_event.new_history)))
        tail = events[summary_idx + 1 :]
    else:
        tail = events

    # Filter the tail down to renderable events. The TypeGuard is
    # self-maintaining: widening ``RenderableEvent`` widens the filter
    # automatically (via ``typing.get_args``). The inner function's
    # exhaustive match is the real type-level safety net.
    renderable_tail: list[RenderableEvent] = [e for e in tail if is_renderable_event(e)]
    messages.extend(render_renderable_events(renderable_tail))
    _heal_dangling_tool_calls(messages)
    return messages


def _heal_dangling_tool_calls(messages: list[Message]) -> None:
    """Fill in missing ``toolResult`` blocks for orphaned ``toolUse`` blocks.

    Mirrors the combined behaviour of Strands'
    ``RepositorySessionManager._fix_dangling_tool_uses`` (internal
    dangling turns) and ``Agent._convert_prompt_to_messages`` (dangling
    last turn). Both call ``generate_missing_tool_result_content`` to
    produce the healing blocks; we do the same so the shape is
    byte-identical. See I10.

    Args:
        messages: Reconstructed messages to repair in place.

    Ensures:
        For every ``toolUse`` block in ``messages``, a ``toolResult``
        block with the matching ``toolUseId`` appears somewhere in the
        following message.
    """
    idx = 0
    # We may ``insert`` into ``messages`` as we walk it, so iterate by
    # index rather than enumerating the initial sequence.
    while idx < len(messages):
        message = messages[idx]
        content = message.get("content", [])
        tool_use_ids = [
            cast("dict[str, str]", block["toolUse"])["toolUseId"] for block in content if "toolUse" in block
        ]
        if not tool_use_ids:
            idx += 1
            continue

        # Collect the toolResult ids already present in the following message.
        next_idx = idx + 1
        existing_result_ids: list[str] = []
        if next_idx < len(messages):
            next_content = messages[next_idx].get("content", [])
            existing_result_ids = [
                cast("dict[str, str]", block["toolResult"])["toolUseId"]
                for block in next_content
                if "toolResult" in block
            ]

        missing_ids = [uid for uid in tool_use_ids if uid not in existing_result_ids]
        if not missing_ids:
            idx += 1
            continue

        missing_blocks = generate_missing_tool_result_content(missing_ids)
        if existing_result_ids:
            # Partial tool completion: extend the existing toolResult user message.
            cast("list[ContentBlock]", messages[next_idx]["content"]).extend(missing_blocks)
        else:
            # No matching toolResult message at all — insert a fresh one.
            messages.insert(next_idx, Message(role="user", content=list(missing_blocks)))
        idx += 1
