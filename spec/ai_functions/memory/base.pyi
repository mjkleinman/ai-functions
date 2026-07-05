"""Abstract memory backend with parameter-recall tracking.

A ``MemoryBackend`` exposes named, typed parameters over a Pydantic schema and
is, at its core, plain storage. ``recall`` / ``query`` / ``search`` return a
:class:`~ai_functions.types.graph.ParameterView` wrapping the value. When a
thread is identifiable — explicit ``coordinator`` / ``thread_id`` arguments or
the ambient :func:`~ai_functions.types.context.thread_scope` — the call
immediately emits a ``ParameterRecalledEvent`` into that thread's log so the
computation graph can be reconstructed post-hoc for optimization; otherwise it
is a pure fetch whose event is emitted later by ``AIFunction.trace`` when the
view is consumed (tracked by ``ParameterView.emitted``).

Emission works from anywhere — including outside an ``@ai_function`` body —
because ``Coordinator.append_event`` only requires the event's ``thread_id`` to
be set and creates the thread's log on demand. ``ParameterView.__str__``
returns ``str(value)``, so a view interpolates into prompts and f-strings
normally; the runtime also unwraps views to their ``.value`` at the
``ThreadHandle.run`` boundary.

Backend fetches (``_recall`` / ``_query`` / ``_search``) may block — a query is
a full model call, a network backend does real I/O — so the public methods run
them in a worker thread with the ambient thread scope cleared, keeping the
event loop responsive and library-internal model calls out of the caller's
event log.

The backend holds no reference to a coordinator or thread between calls. It is
identified by a stable ``backend_id`` that :func:`build_graph` uses to match
recall events back to the live backend at consolidation time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Self

from pydantic import BaseModel
from strands.tools import ToolProvider
from strands.types.tools import AgentTool

from ..protocols import Coordinator
from ..types.graph import ParameterView
from ..types.ids import ThreadId

# Value shape stored by the simple built-in backends (JSON, AgentCore).
ValueType = str | list[str]

# Backend-specific metadata attached to a fetch. Returned by the storage
# methods (``_recall`` / ``_query`` / ``_search``) alongside the value, merged
# into the recall event's ``meta``, and carried by the reconstructed
# ``ParameterNode`` back to ``consolidate``. The JSON backend's search puts
# ``{"results": {entry_id: value}}`` here so consolidation can target exactly
# the entries the forward pass retrieved.
ParameterMeta = dict[str, Any]


class MemoryBackend(ABC):
    """Abstract memory backend over a Pydantic schema.

    Subclasses implement storage, retrieval, and consolidation through the
    abstract ``_*`` methods. The base class owns the public API, schema
    introspection (frozen / procedural markers, descriptions), value
    serialization, and recall-event emission.
    """

    actor_id: str
    schema: type[BaseModel]

    def __init__(self, schema: type[BaseModel], actor_id: str) -> None:
        """Bind a schema and actor namespace to this backend.

        Args:
            schema: Pydantic model describing the memory parameters.
            actor_id: Namespaces this actor's values within the backend.
        """
        ...

    @property
    def backend_id(self) -> str:
        """Stable identifier used to match recall events back to this backend.

        Format ``"ClassName:actor_id"``. ``build_graph`` keys on this to
        reconnect a reconstructed ``ParameterNode`` to the live backend so
        ``consolidate`` can route feedback. Subclasses may override.
        """
        ...

    # ── Abstract storage contract (subclass implements) ──

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this backend (e.g. flush to disk)."""
        ...

    @abstractmethod
    def _save(self, name: str, value: Any) -> None:
        """Replace the stored value of parameter ``name``."""
        ...

    @abstractmethod
    def _recall(self, name: str) -> tuple[Any, ParameterMeta]:
        """Return ``(value, meta)`` for parameter ``name`` from storage."""
        ...

    @abstractmethod
    def _query(self, name: str, query: str) -> tuple[str, ParameterMeta]:
        """Answer ``query`` given the content of parameter ``name``, as ``(answer, meta)``."""
        ...

    @abstractmethod
    def _search(self, name: str, query: str, k: int = 5, **kwargs: Any) -> tuple[Any, ParameterMeta]:
        """Return ``(top_k_matches, meta)`` from parameter ``name``'s value."""
        ...

    @abstractmethod
    def _consolidate(
        self,
        name: str,
        feedback: list[str],
        retrieved: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Incorporate ``feedback`` into the stored value of parameter ``name``.

        Args:
            name: Parameter name.
            feedback: Feedback strings to merge into the stored value.
            retrieved: For list parameters, the ``{entry_id: value}`` mapping of
                the entries the forward pass actually retrieved (from the search
                derivation meta); ``None`` means no retrieval context.
            kwargs: Backend-specific options.
        """
        ...

    @abstractmethod
    def _delete(self, name: str) -> None:
        """Reset parameter ``name`` to its schema default."""
        ...

    # ── Value (de)serialization ──

    def deserialize_value(self, name: str, raw: Any) -> Any:
        """Reconstruct a parameter value from its event representation.

        The inverse of value serialization: validates ``raw`` against the
        field's full type annotation via ``TypeAdapter``, so arbitrarily nested
        shapes (``list[Model]``, ``list[list[Model]]``, ``dict[str, Model]``, a
        bare ``BaseModel``) rehydrate to models rather than raw dicts; plain
        types pass through. Falls back to the raw value if validation fails.

        Args:
            name: Parameter name (supports nested ``a/b/c`` paths).
            raw: The JSON-shaped value taken from a recall event.

        Returns:
            The deserialized value.
        """
        ...

    # ── Public API (async; each read emits a ParameterRecalledEvent) ──
    #
    # These are async so emission is durable before the call returns, even on
    # a network-backed coordinator whose ``append_event`` defers the write to
    # the event loop. After appending, the method awaits ``get_events`` until
    # the event is visible (read-after-write coherence), bounded so a stalled
    # connection degrades to best-effort rather than hanging. On an in-process
    # coordinator the append is already durable, so the confirmation returns
    # on its first read.

    async def recall(
        self,
        name: str,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
    ) -> ParameterView[Any]:
        """Recall a parameter's full value as a :class:`ParameterView`.

        Args:
            name: Parameter name (supports nested ``a/b/c`` paths).
            coordinator: Coordinator to append the recall event to. Defaults
                to the ambient thread scope's coordinator when one is active.
            thread_id: Thread the recalled value is being fed into; stamped
                onto the event. Defaults to the ambient scope's thread.
            requires_grad: Override gradient tracking for this read; defaults
                to ``False`` for ``Frozen`` fields and ``True`` otherwise.

        Returns:
            A ``ParameterView`` wrapping the stored value. Use ``.value`` (or
            ``str()``) for the raw value; pass the view itself to ``trace`` /
            ``__call__`` to preserve the graph edge.

        Ensures:
            - When a thread is identifiable (explicit args or ambient scope),
              one ``ParameterRecalledEvent`` with ``derivation="full"`` is
              appended to that thread's log and confirmed durable (visible via
              ``get_events``) before this call returns, subject to a bounded
              number of confirmation reads; the view's ``emitted`` is ``True``.
            - Otherwise no event is emitted (``emitted=False``) and
              ``AIFunction.trace`` emits it when the view is consumed.
            - ``view.value`` is identical to a direct storage read.
        """
        ...

    async def query(
        self,
        name: str,
        query: str,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
    ) -> ParameterView[str]:
        """Answer a question over a parameter's value, as a :class:`ParameterView`.

        Args:
            name: Parameter name.
            query: Natural-language question answered over the value.
            coordinator: Coordinator to append the recall event to; defaults
                to the ambient scope's coordinator.
            thread_id: Thread the answer is fed into; defaults to the ambient
                scope's thread.
            requires_grad: Override gradient tracking; default as in
                :meth:`recall`.

        Returns:
            A ``ParameterView[str]`` wrapping the model's answer.

        Ensures:
            - When a thread is identifiable, one ``ParameterRecalledEvent``
              with ``derivation="query"`` and the query in its ``meta`` is
              appended and confirmed durable before returning.
        """
        ...

    async def search(
        self,
        name: str,
        query: str,
        k: int = 5,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
        **kwargs: Any,
    ) -> ParameterView[Any]:
        """Return top-``k`` matches from a collection parameter, as a :class:`ParameterView`.

        Args:
            name: Parameter name (typically a collection-valued field).
            query: Retrieval query.
            k: Maximum number of matches.
            coordinator: Coordinator to append the recall event to; defaults
                to the ambient scope's coordinator.
            thread_id: Thread the matches are fed into; defaults to the
                ambient scope's thread.
            requires_grad: Override gradient tracking; default as in
                :meth:`recall`.
            kwargs: Backend-specific search options, forwarded to ``_search``
                and recorded in the event ``meta``.

        Returns:
            A ``ParameterView`` wrapping the top-``k`` results.

        Ensures:
            - When a thread is identifiable, one ``ParameterRecalledEvent``
              with ``derivation="search"`` carrying ``query``, ``top_k``, any
              extra options, and the backend's fetch meta (e.g. the JSON
              backend's ``{"results": {entry_id: value}}``) in its ``meta`` is
              appended and confirmed durable before returning.
        """
        ...

    async def emit_recall(
        self,
        view: ParameterView[Any],
        coordinator: Coordinator,
        thread_id: ThreadId,
    ) -> None:
        """Emit the recall event for a view consumed by a traced thread.

        Called by ``AIFunction.trace`` for views whose recall happened outside
        any thread scope (the thread did not exist yet). No-op when the view
        was already emitted at recall time, so one logical recall never lands
        in two logs.

        Args:
            view: The consumed ``ParameterView`` (produced by this backend).
            coordinator: Coordinator holding the traced thread's log.
            thread_id: The traced thread the view was fed into.
        """
        ...

    def consolidate(self, name: str, feedback: list[str], retrieved: dict[str, str] | None = None) -> None:
        """Incorporate ``feedback`` into parameter ``name``.

        Runs with the ambient thread scope cleared, so consolidation's
        internal model calls never attribute to a running user thread.

        Args:
            name: Parameter name.
            feedback: Feedback strings to merge into the stored value.
            retrieved: For list parameters, the ``{entry_id: value}`` mapping
                the forward pass retrieved (merged from the search derivation
                meta by the optimizer), so consolidation targets those entries;
                ``None`` consolidates against the full value.
        """
        ...

    def save(self, name: str, value: Any) -> None:
        """Store a parameter's value directly, without consolidation.

        Args:
            name: Parameter name.
            value: New value to store.
        """
        ...

    def delete(self, name: str) -> None:
        """Reset a parameter to its schema default.

        Args:
            name: Parameter name.
        """
        ...

    # ── Scoped tool provider ──

    def tool_provider(self, *names: str, operations: set[str] | None = None) -> DynamicToolProvider:
        """Return a ``ToolProvider`` with tools scoped to the given parameter names.

        For each parameter, generates uniquely-named tools (``recall_<name>``,
        ``query_<name>``, and — by field type — ``search_<name>`` for list
        parameters or ``save_<name>`` / ``delete_<name>`` for scalar ones) whose
        descriptions come from the schema, so an agent can read and write memory
        during a cycle.

        The generated tools take no coordinator/thread arguments, but recall
        operations pick up the ambient thread scope the runtime opens for each
        cycle — so a tool call inside a running thread still emits a
        ``ParameterRecalledEvent`` and feeds the optimization graph. Tools
        return the raw value (not a ``ParameterView``).

        Args:
            names: One or more parameter names (slash-separated for nested fields).
            operations: Restrict to this subset of ``{"recall", "query",
                "search", "save", "delete"}``; all applicable tools if ``None``.

        Returns:
            A ``DynamicToolProvider`` holding the generated tools.
        """
        ...

    # ── Context manager ──

    def __enter__(self) -> Self: ...
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...


class DynamicToolProvider(ToolProvider):
    """A ``ToolProvider`` holding a pre-built list of memory tools.

    Returned by :meth:`MemoryBackend.tool_provider`. The tools are already
    materialized, so consumer tracking is bookkeeping only.
    """

    def __init__(self, tools: list[AgentTool]) -> None:
        """Hold the pre-built ``tools``."""
        ...

    @property
    def tools(self) -> list[AgentTool]:
        """The pre-built tools (subclasses extend providers by combining these)."""
        ...

    async def load_tools(self, **kwargs: object) -> Sequence[AgentTool]:
        """Return the pre-built tools."""
        ...

    def add_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Register a consumer (bookkeeping only)."""
        ...

    def remove_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Deregister a consumer (bookkeeping only)."""
        ...
