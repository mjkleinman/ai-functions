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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TYPE_CHECKING

from strands.types.content import Message

if TYPE_CHECKING:
    from ..memory.base import MemoryBackend
    from ..protocols import Coordinator
    from .ids import ThreadId


@dataclass
class Node:
    """Base class for all grad-bearing graph nodes."""

    node_id: str
    value: Any = None
    requires_grad: bool = True
    gradients: list[str] = field(default_factory=list)


@dataclass
class ToolCallNode:
    """A tool invocation extracted from the event log."""

    tool_use_id: str
    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    status: Literal["success", "error"] = "success"


@dataclass
class ParameterNode(Node):
    """A memory parameter that was recalled during thread execution.

    Reconstructed from parameter-recall events. Holds a **direct reference**
    to the ``MemoryBackend`` that owns this parameter, so the optimizer can
    call ``backend.consolidate(name, feedbacks)`` with no lookup table.

    ``description`` is present on every parameter; ``meta`` carries
    backend-specific data (e.g. query, top_k, scores). ``value`` holds the
    deserialized parameter value, whose type depends on the backend schema.
    """

    name: str = ""
    derivation: Literal["full", "query", "search"] = "full"
    backend: MemoryBackend | None = field(default=None, repr=False)
    description: str = ""
    procedural: bool = False
    meta: dict[str, Any] = field(default_factory=dict)


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

    events: list[Any] = field(default_factory=list, repr=False)

@dataclass(kw_only=True, eq=False)
class ParameterView[T]:
    """A recalled parameter value plus the metadata needed to link it into a graph.

    Returned by ``MemoryBackend.recall`` / ``query`` / ``search``. An opaque
    wrapper — not a ``str``/``list`` subclass and not a graph node. ``__str__``
    returns ``str(value)`` so a view interpolates into f-strings and prompt
    templates; passing the view itself to ``AIFunction.trace`` / ``__call__``
    preserves the dataflow edge. ``emitted`` records whether the recall event
    already landed in some thread's log (``trace`` emits for un-emitted views
    and skips the rest).
    """

    value: T
    name: str
    backend: MemoryBackend
    derivation: Literal["full", "query", "search"] = "full"
    requires_grad: bool = True
    description: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    emitted: bool = False

    def __str__(self) -> str: ...

@dataclass(kw_only=True, eq=False)
class Result[T]:
    """The output of one ``AIFunction.trace`` call plus its provenance.

    ``inputs`` carries the sibling dataflow edges discovered by scanning the
    trace call's arguments; the thread's events (read from ``coordinator``)
    remain the source of truth for everything else.
    """

    value: T
    coordinator: Coordinator
    thread_id: ThreadId
    inputs: list[ParameterView[Any] | Result[Any]] = field(default_factory=list)

    def __str__(self) -> str: ...

type Traceable[T] = T | ParameterView[T] | Result[T]

def collect_nodes(value: Any) -> list[ParameterView[Any] | Result[Any]]:
    """Recursively find the dataflow handles in ``value``.

    Scans dicts, lists, tuples, and sets; returns handles in discovery order,
    deduplicated by identity.
    """

def unwrap_nodes(value: Any) -> Any:
    """Recursively replace ``ParameterView`` / ``Result`` handles with their ``.value``.

    Rebuilds dicts, lists, tuples (including ``NamedTuple``), and sets;
    returns every other value unchanged.
    """
