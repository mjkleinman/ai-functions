"""Graph types for post-hoc computation-graph reconstruction.

These types represent a graph reconstructed from the coordinator's event log
*after* execution completes, not built during it. The same reconstruction
works whether threads run standalone or on separate workers.

Three node types:
- ThreadNode: an AIThread / AIFunction execution (messages, tool calls, params).
- ParameterNode: a memory parameter that was recalled / queried / searched.
- ToolCallNode: a tool invocation within a thread.

All grad-bearing nodes share a common base carrying ``node_id``, ``value``,
``requires_grad``, and ``gradients``.

Two further types are *handles*, not graph nodes: :class:`ParameterView`
(returned by ``memory.recall/query/search``) and :class:`Result` (returned by
``AIFunction.trace``). They exist so Python dataflow between threads — one
function's output passed as another's input — can be discovered by scanning
arguments at trace time. They carry no gradients; the graph nodes above are
still reconstructed exclusively from the event log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from strands.types.content import Message

if TYPE_CHECKING:
    from ..memory.base import MemoryBackend
    from ..protocols import Coordinator
    from .ids import ThreadId


@dataclass
class Node:
    """Base class for all grad-bearing graph nodes."""

    node_id: str
    value: Any = None  # pyright: ignore[reportExplicitAny]
    requires_grad: bool = True
    gradients: list[str] = field(default_factory=list)


@dataclass
class ToolCallNode:
    """A tool invocation extracted from the event log."""

    tool_use_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportExplicitAny]
    result: str | None = None
    status: Literal["success", "error"] = "success"


@dataclass
class ParameterNode(Node):
    """A memory parameter that was recalled during thread execution.

    Reconstructed from ``ParameterRecalledEvent`` events. Holds a **direct
    reference** to the ``MemoryBackend`` that owns this parameter, so the
    optimizer can call ``backend.consolidate(name, feedbacks)`` with no lookup
    table.
    """

    name: str = ""
    derivation: Literal["full", "query", "search"] = "full"
    backend: MemoryBackend | None = field(default=None, repr=False)
    description: str = ""
    procedural: bool = False
    meta: dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportExplicitAny]


@dataclass
class ThreadNode(Node):
    """A thread execution reconstructed from the event log.

    The primary graph node. The optimizer walks these in reverse topological
    order, distributing feedback to parameters and child threads.
    """

    thread_id: str = ""
    func_name: str | None = None
    messages: list[Message] = field(default_factory=list)

    parameters: list[ParameterNode] = field(default_factory=list)
    tool_calls: list[ToolCallNode] = field(default_factory=list)
    child_threads: list[ThreadNode] = field(default_factory=list)
    parent: ThreadNode | None = field(default=None, repr=False)

    events: list[Any] = field(default_factory=list, repr=False)  # pyright: ignore[reportExplicitAny]


# ── Dataflow handles (not graph nodes) ────────────────────────────────────────


@dataclass(kw_only=True, eq=False)
class ParameterView[T]:
    """A recalled parameter value plus the metadata needed to link it into a graph.

    Returned by ``MemoryBackend.recall`` / ``query`` / ``search``. An opaque
    wrapper — **not** a ``str`` or ``list`` subclass and not a graph node.
    ``__str__`` returns ``str(value)`` so a view interpolates into f-strings
    and prompt templates unchanged; passing the view itself (not an f-string
    of it) to ``AIFunction.trace`` or ``__call__`` keeps its identity so the
    dataflow edge is preserved. The runtime unwraps views to ``value`` at the
    ``ThreadHandle.run`` boundary before any prompt is built or serialized.

    Attributes:
        value: The recalled value.
        name: Parameter name on the backend (slash-separated for nesting).
        backend: The live backend the value came from.
        derivation: How the value was derived (``full`` recall, ``query``,
            or ``search``).
        requires_grad: Whether the parameter participates in optimization.
        description: Schema description of the parameter.
        meta: Derivation metadata (e.g. the query string).
        emitted: Whether a ``ParameterRecalledEvent`` has already been
            appended for this view (at recall time, under an ambient or
            explicit thread scope). ``AIFunction.trace`` emits for views that
            were *not* emitted at recall time and skips the rest, so one
            logical recall never lands in two logs.
    """

    value: T
    name: str
    backend: MemoryBackend
    derivation: Literal["full", "query", "search"] = "full"
    requires_grad: bool = True
    description: str = ""
    meta: dict[str, Any] = field(default_factory=dict)  # pyright: ignore[reportExplicitAny]
    emitted: bool = False

    def __str__(self) -> str:
        """Render the wrapped value for casual printing (drops the dataflow edge)."""
        return str(self.value)


@dataclass(kw_only=True, eq=False)
class Result[T]:
    """The output of one ``AIFunction.trace`` call plus its provenance.

    Minimal by design: no agent reference, no messages — the thread's events
    are the source of truth, read back from ``coordinator`` by
    ``build_graph_from_result``. ``inputs`` carries the sibling dataflow edges
    discovered by scanning the trace call's arguments. ``__str__`` returns
    ``str(value)`` for casual printing; like :class:`ParameterView`, a
    ``Result`` interpolated into an f-string loses its identity (the
    computation still works, the optimization edge is dropped).

    Attributes:
        value: The typed cycle result.
        coordinator: Coordinator holding the traced thread's event log
            (kept alive by this reference).
        thread_id: Id of the traced thread.
        inputs: ``ParameterView`` / ``Result`` handles found in the call's
            arguments, in discovery order, deduplicated by identity.
    """

    value: T
    coordinator: Coordinator
    thread_id: ThreadId
    inputs: list[ParameterView[Any] | Result[Any]] = field(default_factory=list)  # pyright: ignore[reportExplicitAny]

    def __str__(self) -> str:
        """Render the wrapped value for casual printing (drops the dataflow edge)."""
        return str(self.value)


type Traceable[T] = T | ParameterView[T] | Result[T]
"""A value of type ``T``, or a dataflow handle wrapping one.

Names the union a dataflow edge can take — a plain ``T``, a recalled
``ParameterView[T]``, or a traced ``Result[T]``. It is a documentation and
introspection aid, **not** something to annotate prompt-function parameters
with: a prompt function should declare its parameters as the plain type it
actually receives (``def email_writer(jokes: str, ...)``), because the runtime
unwraps every handle to its ``.value`` at the ``ThreadHandle.run`` boundary
before ``prompt_fn`` runs. Passing handles is type-checked by ``trace``'s own
``*args: Any`` signature, not by widening the body's parameters.
"""


def collect_nodes(value: Any) -> list[ParameterView[Any] | Result[Any]]:  # pyright: ignore[reportExplicitAny]
    """Recursively find the dataflow handles in ``value``.

    Scans dicts (values), lists, tuples, and sets. Returns handles in
    discovery order, deduplicated by identity — the same view passed twice is
    one edge, not two.
    """
    out: list[ParameterView[Any] | Result[Any]] = []  # pyright: ignore[reportExplicitAny]
    seen: set[int] = set()

    def _walk(v: Any) -> None:  # pyright: ignore[reportExplicitAny]
        if isinstance(v, (ParameterView, Result)):
            if id(v) not in seen:
                seen.add(id(v))
                out.append(v)
        elif isinstance(v, dict):
            for item in v.values():  # pyright: ignore[reportUnknownVariableType]
                _walk(item)
        elif isinstance(v, (list, tuple, set, frozenset)):
            for item in v:  # pyright: ignore[reportUnknownVariableType]
                _walk(item)

    _walk(value)
    return out


def unwrap_nodes(value: Any) -> Any:  # pyright: ignore[reportExplicitAny]
    """Recursively replace ``ParameterView`` / ``Result`` handles with their ``.value``.

    Rebuilds dicts, lists, tuples (including ``NamedTuple``), and sets;
    returns every other value unchanged. Called at the ``ThreadHandle.run``
    boundary so handles never reach prompt construction or cross-process
    serialization.
    """
    if isinstance(value, (ParameterView, Result)):
        return value.value  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if isinstance(value, dict):
        return {k: unwrap_nodes(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, tuple):
        items = [unwrap_nodes(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
        # NamedTuple subclasses take positional fields, not an iterable.
        if hasattr(value, "_fields"):
            return type(value)(*items)
        return tuple(items) if type(value) is tuple else type(value)(items)
    if isinstance(value, list):
        items = [unwrap_nodes(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
        return items if type(value) is list else type(value)(items)
    if isinstance(value, (set, frozenset)):
        return type(value)(unwrap_nodes(item) for item in value)  # pyright: ignore[reportUnknownVariableType]
    return value
