"""Reconstruct a single ``ThreadNode`` from one thread's event log.

The optimization graph is a pure function of the event log + the live memory
backends, built after a run. ``build_graph`` rebuilds one thread's node;
cross-thread structure (one thread's output feeding another) is wired by the
caller, since that Python-level dataflow is recorded in no event log.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from ..ai_thread.reconstruction import reconstruct_messages
from ..types.events import (
    MessageAssistantCompleteEvent,
    ParameterRecalledEvent,
    ResultEvent,
    ThreadSpawnedEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from ..types.graph import ParameterNode, Result, ThreadNode, ToolCallNode
from ._formatting import unique_name

if TYPE_CHECKING:
    from ..protocols import Coordinator
    from ..types.events import Event
    from ..types.graph import ParameterHost
    from ..types.ids import ThreadId

logger = logging.getLogger(__name__)


def leads_to_grad_parameter(node: ThreadNode, _cache: dict[int, bool] | None = None) -> bool:
    """Return whether ``node`` is (or reaches through children) a grad-enabled parameter.

    A node is a live optimization target when it either owns a
    ``requires_grad`` parameter or has a child thread that does. This is the
    grad-subtree pruning predicate shared by :func:`topological_sort` (which
    prunes grad-free subtrees from the walk) and the optimizer's backward pass
    (which offers only grad-reaching children as routable feedback targets).

    Args:
        node: The thread node to test.
        _cache: Optional memo of ``id(node) -> bool`` reused across a single
            traversal to keep the check linear on shared/diamond graphs.
    """
    cache = _cache if _cache is not None else {}
    nid = id(node)
    if nid in cache:
        return cache[nid]
    # Seed the cache before recursing so a cycle back to ``node`` terminates.
    cache[nid] = False
    # An economic-function run is kept because it owns a grad-enabled decision
    # parameter (its beliefs host), so no learnable-node special case is needed.
    result = any(p.requires_grad for p in node.parameters) or any(
        leads_to_grad_parameter(c, cache) for c in node.child_threads
    )
    cache[nid] = result
    return result


def topological_sort(node: ThreadNode) -> list[ThreadNode]:
    """Return ThreadNodes in reverse topological order, pruning grad-free subtrees."""
    visited: set[int] = set()
    order: list[ThreadNode] = []
    _has_grad_cache: dict[int, bool] = {}

    def _dfs(n: ThreadNode) -> None:
        nid = id(n)
        if nid in visited:
            return
        visited.add(nid)
        for child in n.child_threads:
            if leads_to_grad_parameter(child, _has_grad_cache):
                _dfs(child)
        order.append(n)

    _dfs(node)
    return list(reversed(order))


def _assistant_text(content: list[Any]) -> str:  # pyright: ignore[reportExplicitAny]
    """Concatenate the text blocks of an assistant turn's content."""
    return "".join(block["text"] for block in content if isinstance(block, dict) and "text" in block)


def _decode_result_payload(payload: str) -> Any:  # pyright: ignore[reportExplicitAny]
    """Decode a ``ResultEvent`` payload into a value for the node.

    ``Thread.serialize_result`` JSON-encodes the result (``dump_json``), so a
    string output type arrives as a quoted JSON string; decoding unquotes it and
    rehydrates dicts/lists. A non-JSON-serializable result is recorded as a
    best-effort ``str`` that will not parse — fall back to the raw payload.
    """
    try:
        return json.loads(payload)
    except (json.JSONDecodeError, TypeError):
        return payload


def _reconstruct_node(events: list[Event], backends: list[ParameterHost]) -> ThreadNode:
    """Reconstruct one thread's computation node from its event log.

    Builds exactly one ``ThreadNode`` from a single thread's pre-fetched events.
    Cross-thread edges (``child_threads``) are wired by the caller (or by
    :func:`build_graph`, which recurses spawned children).

    Args:
        events: One thread's events in append order (oldest first).
        backends: Live parameter hosts (memory backends, beliefs adapters), matched by ``backend_id``.

    Returns:
        A single childless ``ThreadNode``.
    """
    backend_map = {b.backend_id: b for b in backends}

    thread_id = ""
    func_name: str | None = None
    for evt in events:
        tid = getattr(evt, "thread_id", None)
        if tid:
            thread_id = str(tid)
        name = getattr(evt, "thread_name", None)
        if name and func_name is None:
            func_name = str(name)

    # Parameters: one ParameterNode per (backend_id, name); last recall wins.
    param_nodes: dict[tuple[str, str], ParameterNode] = {}
    tool_calls: list[ToolCallNode] = []
    tc_map: dict[str, ToolCallNode] = {}
    assistant_text: str = ""
    result_payload: Any = None  # pyright: ignore[reportExplicitAny]  # ResultEvent.payload (str) or None

    for evt in events:
        if isinstance(evt, ParameterRecalledEvent):
            backend = backend_map.get(evt.backend_id)
            if backend is None:
                logger.warning("build_graph: no backend for id '%s' (param '%s')", evt.backend_id, evt.name)
                continue
            value = backend.deserialize_value(evt.name, evt.value)
            key = (evt.backend_id, evt.name)
            param_nodes[key] = ParameterNode(
                node_id=unique_name(evt.name),
                value=value,
                requires_grad=evt.requires_grad,
                name=evt.name,
                derivation=evt.derivation,
                backend=backend,
                description=evt.description,
                procedural=backend._is_procedural(evt.name),  # noqa: SLF001
                meta=dict(evt.meta),
            )
        elif isinstance(evt, ToolCallEvent):
            tc = ToolCallNode(
                tool_use_id=evt.tool_use_id,
                tool_name=evt.tool_name,
                arguments=dict(evt.arguments),
            )
            tc_map[evt.tool_use_id] = tc
            tool_calls.append(tc)
        elif isinstance(evt, ToolResultEvent):
            tc = tc_map.get(evt.tool_use_id)
            if tc is not None:
                tc.result = "".join(
                    block["text"] for block in evt.content if isinstance(block, dict) and "text" in block
                )
                tc.status = "error" if evt.status == "error" else "success"
        elif isinstance(evt, MessageAssistantCompleteEvent):
            assistant_text = _assistant_text(list(evt.content)) or assistant_text
        elif isinstance(evt, ResultEvent):
            result_payload = evt.payload

    # Prefer the thread's serialized result over the assistant's turn text: for
    # a structured-output cycle the assistant text is only a preamble while the
    # real output lives in the ResultEvent, and the backward pass needs the true
    # output to tell one child result from another. Falls back to None.
    result_value: Any = None  # pyright: ignore[reportExplicitAny]
    if result_payload is not None:
        result_value = _decode_result_payload(result_payload)
    elif assistant_text:
        result_value = assistant_text

    messages = reconstruct_messages(events)

    # node_id must be injective on thread_id: render_inputs / the backward pass
    # key routable targets by node_id, so any collision silently drops a target
    # (e.g. two sibling joke Results would merge into one). thread_ids are
    # ``thread-<hex>``, so a fixed-length *prefix* is constant across threads —
    # use the unique trailing segment instead.
    suffix = thread_id.rsplit("-", 1)[-1] or thread_id
    nid = f"{func_name or 'thread'}-{suffix}"
    return ThreadNode(
        node_id=nid,
        thread_id=thread_id,
        func_name=func_name,
        messages=messages,
        value=result_value,
        parameters=list(param_nodes.values()),
        tool_calls=tool_calls,
        child_threads=[],
        events=list(events),
    )


def _splice_delegated_traces(node: ThreadNode) -> None:
    """Splice each delegated child's messages onto ``node`` at its marker.

    A supervisor thread (an economic run, a retry wrapper) emits a
    :class:`~ai_functions.types.events.TraceDelegationEvent` naming the child
    whose conversation the backward pass should read instead of the
    supervisor's telemetry. This appends that child's reconstructed messages to
    the node's own — any narration the supervisor logged as ordinary messages
    stays in order, and the delegated conversation follows. A marker whose child
    is not among ``child_threads`` (never spawned, or pruned) contributes
    nothing. Called after ``child_threads`` are wired; a no-op with no markers.
    """
    from ..types.events import TraceDelegationEvent

    by_id = {c.thread_id: c for c in node.child_threads}
    for evt in node.events:
        if not isinstance(evt, TraceDelegationEvent):
            continue
        child = by_id.get(str(evt.child_thread_id))
        if child is not None:
            node.messages = [*node.messages, *child.messages]


async def build_graph(
    coordinator: Coordinator,
    thread_id: ThreadId,
    backends: list[ParameterHost],
) -> ThreadNode:
    """Reconstruct a thread's computation graph, recursing into spawned children.

    Reads ``thread_id``'s event log from ``coordinator`` and builds its node,
    then follows each ``ThreadSpawnedEvent`` to reconstruct the child's subtree
    and wire the ``child_threads`` / ``parent`` edges. Cross-thread edges that
    live only in Python dataflow (one thread's result passed into another) are
    recorded in no event log and remain the caller's to wire.

    Args:
        coordinator: Coordinator holding the event logs.
        thread_id: Root thread to reconstruct.
        backends: Live parameter hosts (memory backends, beliefs adapters), matched by ``backend_id``.

    Returns:
        The root ``ThreadNode`` with its spawned-child subtree attached.
    """
    return await _build_subtree(coordinator, thread_id, backends, set())


async def _build_subtree(
    coordinator: Coordinator,
    thread_id: ThreadId,
    backends: list[ParameterHost],
    seen: set[str],
) -> ThreadNode:
    """Build ``thread_id``'s node and recurse its spawned children.

    ``seen`` guards against a thread id appearing twice in the spawn events
    (self-spawn or a cycle), so the walk stays finite.
    """
    seen.add(str(thread_id))
    events = await coordinator.get_events(thread_id)
    node = _reconstruct_node(events, backends)

    children: list[ThreadNode] = []
    for evt in events:
        if isinstance(evt, ThreadSpawnedEvent) and str(evt.child_thread_id) not in seen:
            child = await _build_subtree(coordinator, evt.child_thread_id, backends, seen)
            child.parent = node
            children.append(child)
    node.child_threads = children
    _splice_delegated_traces(node)
    return node


async def build_graph_from_result(
    result: Result[Any],  # pyright: ignore[reportExplicitAny]
    backends: list[ParameterHost],
) -> ThreadNode:
    """Build the full ``ThreadNode`` graph from a traced :class:`Result`.

    Combines the two edge sources: spawned children come from each thread's
    ``ThreadSpawnedEvent`` s (via :func:`build_graph`), sibling dataflow edges
    come from ``Result.inputs`` (discovered by argument scanning at trace
    time). ``ParameterView`` inputs are not grafted here — their recall events
    were emitted by ``AIFunction.trace``, so ``_reconstruct_node`` already
    materializes them as ``ParameterNode`` s.

    The returned graph is a DAG, built once: a ``Result`` consumed by several
    traces (a diamond) resolves to a **single shared node object** reachable
    from every consumer, so ``backward`` accumulates feedback from all of them
    and ``consolidate`` reads those same nodes. ``parent`` is set by the first
    consumer that grafts a node (first-consumer-wins; it is informational —
    traversal follows ``child_threads``).

    Args:
        result: The root ``Result`` returned by ``AIFunction.trace``.
        backends: Live parameter hosts (memory backends, beliefs adapters), matched by ``backend_id``.

    Returns:
        The root ``ThreadNode`` with spawned and sibling subtrees attached.
    """
    built: dict[str, ThreadNode] = {}
    return await _assemble(result, backends, built)


def _register_subtree(node: ThreadNode, built: dict[str, ThreadNode]) -> None:
    """Index ``node`` and its spawned descendants by thread id.

    Keeps the first node seen for an id, so a thread reachable both as a
    spawned child and as a sibling ``Result`` resolves to one object.
    """
    built.setdefault(node.thread_id, node)
    for child in node.child_threads:
        _register_subtree(child, built)


async def _assemble(
    result: Result[Any],  # pyright: ignore[reportExplicitAny]
    backends: list[ParameterHost],
    built: dict[str, ThreadNode],
) -> ThreadNode:
    """Build (or reuse) ``result``'s node and graft its sibling ``Result`` inputs."""
    tid = str(result.thread_id)
    node = built.get(tid)
    if node is None:
        node = await build_graph(result.coordinator, result.thread_id, backends)
        _register_subtree(node, built)
        node = built[tid]

    for inp in result.inputs:
        if not isinstance(inp, Result):
            continue
        child = await _assemble(inp, backends, built)
        if child is node or any(c.thread_id == child.thread_id for c in node.child_threads):
            continue
        if child.parent is None:
            child.parent = node
        node.child_threads.append(child)

    return node
