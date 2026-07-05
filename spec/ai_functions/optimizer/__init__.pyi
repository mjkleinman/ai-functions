"""TextGrad-style optimization over reconstructed computation graphs."""

from __future__ import annotations

from typing import Any

from ..memory.base import MemoryBackend
from ..protocols import Coordinator
from ..types.graph import Result, ThreadNode
from ..types.ids import ThreadId
from .textgrad import TextGradOptimizer

__all__ = [
    "TextGradOptimizer",
    "build_graph",
    "build_graph_from_result",
]


async def build_graph(
    coordinator: Coordinator,
    thread_id: ThreadId,
    backends: list[MemoryBackend],
) -> ThreadNode:
    """Reconstruct a thread's computation graph, recursing into spawned children.

    Reads ``thread_id``'s event log from ``coordinator`` and builds its node,
    then follows each ``ThreadSpawnedEvent`` to reconstruct the child's subtree
    and wire the ``child_threads`` / ``parent`` edges. Cross-thread edges that
    live only in Python dataflow (one thread's result passed into another) are
    recorded in no event log; :func:`build_graph_from_result` wires them from a
    traced ``Result``, or the caller wires them by hand::

        email = await build_graph(coord, "email-1", [memory])
        joke1 = await build_graph(coord, "joke-1", [memory])
        joke2 = await build_graph(coord, "joke-2", [memory])
        email.child_threads = [joke1, joke2]  # sibling dataflow, wired by hand

    Args:
        coordinator: Coordinator holding the event logs.
        thread_id: Root thread to reconstruct.
        backends: Live memory backends whose ``backend_id`` is matched against
            each ``ParameterRecalledEvent`` so the resulting ``ParameterNode``
            can reference the owning backend for consolidation.

    Returns:
        The root :class:`ThreadNode` with its spawned-child subtree attached.

    Ensures:
        - One :class:`ParameterNode` per distinct ``(backend_id, name)`` per
          node; last-write-wins on repeated recalls, value kept in recalled type.
        - A ``ParameterRecalledEvent`` whose ``backend_id`` matches no entry in
          ``backends`` is skipped (no node, logged).
        - Each spawned child (one per ``ThreadSpawnedEvent`` in the thread's log)
          is reconstructed recursively and attached via ``child_threads`` with
          ``parent`` set; a child id already visited on this walk is skipped.
        - ``node.value`` is the text of the last ``MessageAssistantCompleteEvent``,
          or ``None`` if the thread produced no assistant turn.
    """
    ...


async def build_graph_from_result(
    result: Result[Any],
    backends: list[MemoryBackend],
) -> ThreadNode:
    """Build the full ``ThreadNode`` graph from a traced :class:`Result`.

    Combines the two edge sources: spawned children come from each thread's
    ``ThreadSpawnedEvent`` s (via :func:`build_graph`), sibling dataflow edges
    come from ``Result.inputs`` (discovered by argument scanning at trace
    time). ``ParameterView`` inputs are not grafted — their recall events were
    emitted by ``AIFunction.trace``, so reconstruction already materializes
    them as ``ParameterNode`` s.

    Args:
        result: The root ``Result`` returned by ``AIFunction.trace``.
        backends: Live memory backends, matched by ``backend_id``.

    Returns:
        The root ``ThreadNode`` with spawned and sibling subtrees attached.

    Ensures:
        - The graph is a DAG built once: a ``Result`` consumed by several
          traces (a diamond) resolves to a single shared node object reachable
          from every consumer, so ``backward`` accumulates feedback from all
          of them and ``consolidate`` reads those same nodes.
        - A thread reachable both as a spawned child and as a sibling
          ``Result`` resolves to one node object.
        - ``parent`` is set by the first consumer that grafts a node
          (informational; traversal follows ``child_threads``).
    """
    ...
