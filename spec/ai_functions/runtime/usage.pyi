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

from ..protocols import Coordinator
from ..types import EventId, ThreadId, TokenUsage

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
    ...

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

    Args:
        coordinator: Coordinator holding the event logs.
        thread_id: Root of the subtree to aggregate.
        since_id: Count only events appended strictly after this id in the
            *root* log. Pass :func:`last_event_id` captured after a seeded
            spawn to exclude the seed-copied prefix.

    Returns:
        Componentwise total ``TokenUsage`` for the subtree.
    """
    ...

async def subtree_usage(
    coordinator: Coordinator,
    thread_id: ThreadId,
    *,
    since_id: EventId | None = None,
) -> tuple[TokenUsage, int]:
    """Like :func:`subtree_token_usage`, additionally counting model turns.

    Turns are ``MESSAGE_ASSISTANT_COMPLETE`` events (one per assistant turn),
    with a fallback of one turn per usage event for backends that emit only
    usage, so at least 1 turn is reported whenever tokens were spent.

    Args:
        coordinator: Coordinator holding the event logs.
        thread_id: Root of the subtree to aggregate.
        since_id: Count only events appended strictly after this id in the
            *root* log (see :func:`subtree_token_usage`).

    Returns:
        ``(total_usage, turns)`` for the subtree.
    """
    ...
