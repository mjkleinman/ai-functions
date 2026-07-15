"""Token-usage aggregation over the event log.

Usage accounting is derived, not stored: the coordinator's event log is the
single source of truth (I2), and this module folds ``TOKEN_USAGE`` events into
:class:`~ai_functions.types.TokenUsage` totals on demand. Working over the
:class:`~ai_functions.protocols.Coordinator` protocol (``get_events`` only)
keeps it usable against any implementation, including a remote
``CoordinatorClient``.

Warm-seeded threads (``spawn(seed_from=...)``) start with a *copy* of the
source thread's log, including its ``TOKEN_USAGE`` and ``THREAD_SPAWNED``
events. Counting those would attribute the source's spend to the new thread,
so callers measuring a seeded thread pass ``since_id`` — typically
:func:`last_event_id` captured right after spawn — to fold only events
appended after that point.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..types import EventKind, ThreadId, ThreadSpawnedEvent, TokenUsage, TokenUsageEvent

if TYPE_CHECKING:
    from ..protocols import Coordinator
    from ..types import EventId


async def last_event_id(coordinator: Coordinator, thread_id: ThreadId) -> EventId | None:
    """Return the id of the newest event in ``thread_id``'s log, or ``None`` if empty.

    Captured right after ``spawn(seed_from=...)`` it marks the end of the
    seed-copied prefix, making it the natural ``since_id`` baseline for
    :func:`subtree_token_usage`.

    Args:
        coordinator: Coordinator holding the event log.
        thread_id: Thread whose log to inspect.

    Returns:
        The last event's id, or ``None`` for an empty (or unknown) log.
    """
    events = await coordinator.get_events(thread_id)
    for event in reversed(events):
        # CustomEvent carries no id; only BaseEvent-backed events can anchor
        # a since_id cursor, so skip to the newest event that has one.
        event_id: EventId | None = getattr(event, "id", None)
        if event_id is not None:
            return event_id
    return None


async def subtree_token_usage(
    coordinator: Coordinator,
    thread_id: ThreadId,
    *,
    since_id: EventId | None = None,
) -> TokenUsage:
    """Fold token usage for ``thread_id`` and the sub-computations it spawned.

    Sums every ``TOKEN_USAGE`` event in the thread's log (strictly after
    ``since_id`` when given), then recurses into each child recorded by a
    ``THREAD_SPAWNED`` event in the same window. Fresh spawns are the only
    producers of ``THREAD_SPAWNED`` edges — forks (``seed_from``) record none
    — so each child's log is counted whole, exactly once.

    Because seeded spawns record no edge, a thread seeded from this one is
    *not* part of the subtree: its log opens with a copy of the source's
    usage events, so counting it would double-attribute the source's spend.
    Callers who orchestrate seeded threads (e.g. the cost-aware cascade) own
    that attribution themselves, measuring each seeded thread with a
    post-spawn ``since_id`` baseline.

    Args:
        coordinator: Coordinator holding the event logs.
        thread_id: Root of the subtree to aggregate.
        since_id: Count only events appended strictly after this id in the
            *root* log. Pass :func:`last_event_id` captured after a seeded
            spawn to exclude the seed-copied prefix.

    Returns:
        Componentwise total ``TokenUsage`` for the subtree.
    """
    usage, _ = await subtree_usage(coordinator, thread_id, since_id=since_id)
    return usage


async def subtree_usage(
    coordinator: Coordinator,
    thread_id: ThreadId,
    *,
    since_id: EventId | None = None,
) -> tuple[TokenUsage, int]:
    """Like :func:`subtree_token_usage`, additionally counting model turns.

    Turns are ``MESSAGE_ASSISTANT_COMPLETE`` events — one per assistant turn
    of the agentic loop (``TOKEN_USAGE`` is emitted once per *invocation*
    with accumulated usage, so it cannot resolve turns). Backends that skip
    the assistant-turn span (emitting only usage) fall back to one turn per
    usage event, so a turn count of at least 1 is reported whenever tokens
    were spent.

    Args:
        coordinator: Coordinator holding the event logs.
        thread_id: Root of the subtree to aggregate.
        since_id: Count only events appended strictly after this id in the
            *root* log (see :func:`subtree_token_usage`).

    Returns:
        ``(total_usage, turns)`` for the subtree.
    """
    total = TokenUsage()
    turns = 0
    usage_events = 0
    visited: set[ThreadId] = set()

    async def _fold(tid: ThreadId, cursor: EventId | None) -> None:
        nonlocal total, turns, usage_events
        if tid in visited:
            return
        visited.add(tid)
        events = await coordinator.get_events(
            tid,
            since_id=cursor,
            kinds=[EventKind.TOKEN_USAGE, EventKind.THREAD_SPAWNED, EventKind.MESSAGE_ASSISTANT_COMPLETE],
        )
        for event in events:
            if isinstance(event, TokenUsageEvent):
                total = total + event.token_usage
                usage_events += 1
            elif isinstance(event, ThreadSpawnedEvent):
                await _fold(event.child_thread_id, None)
            elif getattr(event, "kind", None) == EventKind.MESSAGE_ASSISTANT_COMPLETE:
                turns += 1

    await _fold(thread_id, since_id)
    return total, turns if turns > 0 else usage_events
