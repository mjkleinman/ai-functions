"""Abstract memory backend with parameter-recall tracking.

A ``MemoryBackend`` exposes named, typed parameters over a Pydantic schema and
is, at its core, plain storage. ``recall`` / ``query`` / ``search`` return a
:class:`~ai_functions.types.graph.ParameterView` wrapping the value. When a
``coordinator`` and ``thread_id`` are available (explicitly or via the ambient
:func:`thread_scope`), the call immediately emits a ``ParameterRecalledEvent``
into that thread's log so the computation graph can be reconstructed post-hoc
for optimization; without them it is a pure fetch whose event is emitted later
by ``AIFunction.trace`` when the view is consumed (see ``ParameterView.emitted``).

Emission works from anywhere — including outside an ``@ai_function`` body —
because ``Coordinator.append_event`` only requires the event's ``thread_id`` to
be set and creates the thread's log on demand. ``ParameterView.__str__``
returns ``str(value)``, so a view interpolates into prompts and f-strings
normally; the runtime also unwraps views to their ``.value`` at the
``ThreadHandle.run`` boundary.

Backend fetches (``_recall`` / ``_query`` / ``_search``) may block — a query is
a full model call, a network backend does real I/O — so the public methods run
them in a worker thread (``asyncio.to_thread``) with the ambient thread scope
cleared, keeping the event loop responsive and library-internal model calls
out of the caller's event log.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, cast, get_args, get_origin

from pydantic import BaseModel, TypeAdapter
from pydantic.fields import FieldInfo
from strands.tools import ToolProvider
from strands.tools.decorator import tool as _strands_tool  # pyright: ignore[reportUnknownVariableType]

from ..types.context import current_thread_scope, no_thread_scope
from ..types.events import EventKind, ParameterRecalledEvent
from ..types.graph import GradFeedback, ParameterView
from .frozen import FrozenMarker
from .procedural import ProceduralMarker

if TYPE_CHECKING:
    from strands.types.tools import AgentTool

    from ..protocols import Coordinator
    from ..types.ids import ThreadId

logger = logging.getLogger(__name__)

# Value shape stored by the simple built-in backends (JSON, AgentCore).
ValueType = str | list[str]

# Backend-specific metadata attached to a fetch. Returned by the storage
# methods (``_recall`` / ``_query`` / ``_search``) alongside the value, merged
# into the recall event's ``meta``, and carried by the reconstructed
# ``ParameterNode`` back to ``consolidate``. The JSON backend's search puts
# ``{"results": {entry_id: value}}`` here so consolidation can target exactly
# the entries the forward pass retrieved.
ParameterMeta = dict[str, Any]


def _model_from_annotation(annotation: Any) -> type[BaseModel] | None:  # pyright: ignore[reportExplicitAny]
    """Extract the ``BaseModel`` subclass from a (possibly wrapped) annotation.

    Handles a bare model, ``Annotated[Model, ...]``, and ``Optional[Model]`` /
    ``Model | None`` (and other unions) by returning the single ``BaseModel``
    member. Returns ``None`` if no model is present (e.g. a plain ``str`` or a
    union of non-models), so callers can raise a clear error.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    # Unwrap Annotated[...] / Optional[...] / Union[...] — find a model member.
    for arg in get_args(annotation):
        model = _model_from_annotation(arg)
        if model is not None:
            return model
    return None


# Bounded read-after-write confirmation: how many times ``get_events`` is
# polled for an appended recall event before degrading to best-effort. An
# in-process coordinator confirms on the first read; a network-backed one may
# need the event loop to run its deferred append task (and a round-trip).
_CONFIRM_MAX_READS = 50

# Backoff between confirmation reads. Starts at 0 (one loop tick, enough for an
# in-process deferred task) and grows so a genuine network round-trip has time
# to land within the bounded number of reads rather than spinning instantly.
_CONFIRM_BACKOFF_STEP = 0.01
_CONFIRM_BACKOFF_MAX = 0.2


class MemoryBackend(ABC):
    """Abstract memory backend over a Pydantic schema.

    Subclasses implement storage, retrieval, and consolidation through the
    abstract ``_*`` methods. The base class owns the public API, schema
    introspection, value serialization, and recall-event emission.
    """

    def __init__(self, schema: type[BaseModel], actor_id: str) -> None:
        self.actor_id = actor_id
        self.schema = schema

    # -- Identification --------------------------------------------------------

    @property
    def backend_id(self) -> str:
        """Stable identifier used to match recall events back to this backend.

        Format ``"ClassName:actor_id"``. Subclasses may override.
        """
        return f"{type(self).__name__}:{self.actor_id}"

    # -- Schema introspection --------------------------------------------------

    def _resolve_field(self, name: str) -> FieldInfo:
        parts = name.split("/")
        current_model = self.schema
        for part in parts[:-1]:
            annotation = current_model.model_fields[part].annotation
            current_model = _model_from_annotation(annotation)
            if current_model is None:
                raise TypeError(
                    f"Cannot resolve nested parameter {name!r}: intermediate field {part!r} "
                    f"is not a Pydantic model (annotation: {annotation!r})."
                )
        return current_model.model_fields[parts[-1]]

    def _get_description(self, name: str) -> str:
        return self._resolve_field(name).description or ""

    def _is_procedural(self, name: str) -> bool:
        return any(isinstance(m, ProceduralMarker) for m in self._resolve_field(name).metadata)

    def _is_frozen(self, name: str) -> bool:
        return any(isinstance(m, FrozenMarker) for m in self._resolve_field(name).metadata)

    def _is_list_field(self, name: str) -> bool:
        """Return whether parameter ``name`` is a list-valued field."""
        return get_origin(self._resolve_field(name).annotation) is list

    def _leaf_parameter_names(self) -> list[str]:
        """Return every leaf parameter path in the schema, slash-separated.

        Nested Pydantic models are recursed into (``profile/tone``); non-model
        fields are leaves. This is the set of names a backend stores under.
        """

        def _walk(model: type[BaseModel], prefix: str) -> list[str]:
            names: list[str] = []
            for field_name, field_info in model.model_fields.items():
                path = f"{prefix}{field_name}"
                nested = _model_from_annotation(field_info.annotation)
                if nested is not None:
                    names.extend(_walk(nested, f"{path}/"))
                else:
                    names.append(path)
            return names

        return _walk(self.schema, "")

    # -- Abstract storage contract (subclass implements) -----------------------

    @abstractmethod
    def close(self) -> None:
        """Release resources held by this backend."""

    @abstractmethod
    def _save(self, name: str, value: Any) -> None:  # pyright: ignore[reportExplicitAny]
        """Save the value of a parameter, replacing whatever is stored."""

    @abstractmethod
    def _recall(self, name: str) -> tuple[Any, ParameterMeta]:  # pyright: ignore[reportExplicitAny]
        """Return ``(value, meta)`` for a parameter from storage."""

    @abstractmethod
    def _query(self, name: str, query: str) -> tuple[str, ParameterMeta]:
        """Answer a question given the content of parameter name, as ``(answer, meta)``."""

    @abstractmethod
    def _search(self, name: str, query: str, k: int = 5, **kwargs: Any) -> tuple[Any, ParameterMeta]:  # pyright: ignore[reportExplicitAny]
        """Return ``(top_k_matches, meta)`` from the parameter's stored value."""

    @abstractmethod
    def _consolidate(
        self,
        name: str,
        feedback: list[GradFeedback],
        retrieved: dict[str, str] | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> None:
        """Incorporate feedback into the stored value of parameter name.

        Args:
            name: Parameter name.
            feedback: Gradients to merge into the stored value. A text-rewriting
                backend reads each entry's ``text`` and ignores its ``score``.
            retrieved: For list parameters, the ``{entry_id: value}`` mapping of
                the entries the forward pass actually retrieved (from the search
                derivation meta), so consolidation can target them; ``None``
                means no retrieval context — consolidate against the full value.
            kwargs: Backend-specific options.
        """

    @abstractmethod
    def _delete(self, name: str) -> None:
        """Reset a parameter to its schema default."""

    # -- Value (de)serialization -----------------------------------------------

    def _serialize_value(self, name: str, value: Any) -> Any:  # pyright: ignore[reportExplicitAny]
        """Convert a parameter value to a JSON-serializable form."""
        annotation = self._resolve_field(name).annotation
        if annotation is None:
            return value
        try:
            return TypeAdapter(annotation).dump_python(value, mode="json")
        except Exception:  # noqa: BLE001 — fall back to the raw value
            return value

    def deserialize_value(self, name: str, raw: Any) -> Any:  # pyright: ignore[reportExplicitAny]
        """Reconstruct a parameter value from its event representation.

        The inverse of :meth:`_serialize_value`: validates ``raw`` against the
        field's full type annotation via ``TypeAdapter`` (so arbitrarily nested
        shapes — ``list[Model]``, ``list[list[Model]]``, ``dict[str, Model]``,
        a bare ``BaseModel`` — rehydrate to models, not raw dicts). Falls back
        to the raw value if validation is impossible.
        """
        annotation = self._resolve_field(name).annotation
        if annotation is None:
            return raw
        try:
            return TypeAdapter(annotation).validate_python(raw)
        except Exception:  # noqa: BLE001 — fall back to the raw value (mirrors _serialize_value)
            return raw

    # -- Event emission --------------------------------------------------------

    async def _emit_parameter_event(
        self,
        coordinator: Coordinator | None,
        thread_id: ThreadId | None,
        name: str,
        value: Any,  # pyright: ignore[reportExplicitAny]
        derivation: str,
        requires_grad: bool,
        description: str = "",
        meta: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    ) -> bool:
        """Append a ``ParameterRecalledEvent`` and confirm it is durable.

        Falls back to the ambient :func:`thread_scope` for whichever of
        ``coordinator`` / ``thread_id`` is ``None``; if both are still ``None``
        the call is a pure fetch and this is a no-op returning ``False``.
        Returns ``True`` when an event was appended (confirmed durable or
        degraded to best-effort with a warning), so callers can record whether
        the recall is already represented in some thread's log.

        ``Coordinator.append_event`` is synchronous, but a network-backed
        coordinator defers the actual write to the event loop and returns
        before it lands. Per I8 (read-after-write coherence), this method then
        awaits ``get_events`` until the appended event is visible, so a
        subsequent ``build_graph`` read cannot overtake the write. An
        in-process coordinator confirms on the first read. The poll uses a
        growing backoff and is bounded by ``_CONFIRM_MAX_READS``; if the event
        is still not visible, it logs a warning (the I8 guarantee is unmet for
        that event) rather than hanging or failing silently.
        """
        if coordinator is None or thread_id is None:
            scope = current_thread_scope()
            if scope is not None:
                coordinator = coordinator or scope.coordinator
                thread_id = thread_id or scope.thread_id
        if coordinator is None or thread_id is None:
            return False
        event = ParameterRecalledEvent(
            thread_id=thread_id,
            name=name,
            value=self._serialize_value(name, value),
            derivation=derivation,  # pyright: ignore[reportArgumentType]
            requires_grad=requires_grad,
            backend_id=self.backend_id,
            description=description,
            meta=meta or {},
        )
        coordinator.append_event(event)

        delay = 0.0
        for _ in range(_CONFIRM_MAX_READS):
            stored = await coordinator.get_events(thread_id, kinds=[EventKind.PARAMETER_RECALLED])
            if any(e.id == event.id for e in stored):
                return True
            await asyncio.sleep(delay)
            delay = min(delay + _CONFIRM_BACKOFF_STEP, _CONFIRM_BACKOFF_MAX)

        # Never confirmed within the bound. The I8 read-after-write guarantee is
        # not met for this event; warn rather than fail silently so a dropped
        # parameter recall (and the resulting under-optimization) is diagnosable.
        logger.warning(
            "Parameter recall event for %r (thread %s) was not confirmed durable after %d reads; "
            "the optimization graph may miss this parameter.",
            name,
            thread_id,
            _CONFIRM_MAX_READS,
        )
        return True

    # -- Public API ------------------------------------------------------------

    async def _build_view(
        self,
        name: str,
        value: Any,  # pyright: ignore[reportExplicitAny]
        derivation: str,
        coordinator: Coordinator | None,
        thread_id: ThreadId | None,
        requires_grad: bool,
        meta: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    ) -> ParameterView[Any]:  # pyright: ignore[reportExplicitAny]
        """Wrap a fetched value in a ``ParameterView`` and emit its recall event."""
        description = self._get_description(name)
        emitted = await self._emit_parameter_event(
            coordinator,
            thread_id,
            name,
            value,
            derivation,
            requires_grad,
            description=description,
            meta=meta,
        )
        return ParameterView(
            value=value,
            name=name,
            backend=self,
            derivation=derivation,  # pyright: ignore[reportArgumentType]
            requires_grad=requires_grad,
            description=description,
            meta=dict(meta) if meta else {},
            emitted=emitted,
        )

    async def recall(
        self,
        name: str,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
    ) -> ParameterView[Any]:  # pyright: ignore[reportExplicitAny]
        """Recall a parameter's full value as a :class:`ParameterView`.

        Emits a recall event immediately when a thread is identifiable
        (explicit args or ambient scope); otherwise the view is emitted later
        by ``AIFunction.trace`` when consumed. Use ``.value`` (or ``str()``)
        for the raw value.
        """
        if requires_grad is None:
            requires_grad = not self._is_frozen(name)
        with no_thread_scope():
            value, fetch_meta = await asyncio.to_thread(self._recall, name)  # pyright: ignore[reportAny]
        return await self._build_view(
            name, value, "full", coordinator, thread_id, requires_grad, meta=fetch_meta or None
        )

    async def query(
        self,
        name: str,
        query: str,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
    ) -> ParameterView[str]:
        """Answer a question over a parameter's value, as a :class:`ParameterView`."""
        if requires_grad is None:
            requires_grad = not self._is_frozen(name)
        with no_thread_scope():
            value, fetch_meta = await asyncio.to_thread(self._query, name, query)
        view = await self._build_view(
            name, value, "query", coordinator, thread_id, requires_grad, meta={"query": query, **fetch_meta}
        )
        return cast("ParameterView[str]", view)

    async def search(
        self,
        name: str,
        query: str,
        k: int = 5,
        coordinator: Coordinator | None = None,
        thread_id: ThreadId | None = None,
        requires_grad: bool | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> ParameterView[Any]:  # pyright: ignore[reportExplicitAny]
        """Return top-k matches from a collection parameter, as a :class:`ParameterView`."""
        if requires_grad is None:
            requires_grad = not self._is_frozen(name)
        with no_thread_scope():
            value, fetch_meta = await asyncio.to_thread(lambda: self._search(name, query, k, **kwargs))  # pyright: ignore[reportAny]
        meta: dict[str, Any] = {"query": query, "top_k": k}  # pyright: ignore[reportExplicitAny]
        if kwargs:
            meta.update(kwargs)
        meta.update(fetch_meta)
        return await self._build_view(name, value, "search", coordinator, thread_id, requires_grad, meta=meta)

    async def emit_recall(
        self,
        view: ParameterView[Any],  # pyright: ignore[reportExplicitAny]
        coordinator: Coordinator,
        thread_id: ThreadId,
    ) -> None:
        """Emit the recall event for a view consumed by a traced thread.

        Called by ``AIFunction.trace`` for views whose recall happened outside
        any thread scope (the thread did not exist yet). No-op when the view
        was already emitted at recall time, so one logical recall never lands
        in two logs. Serialization and the I8 durability confirmation are the
        same as an at-recall emission.
        """
        if view.emitted:
            return
        view.emitted = await self._emit_parameter_event(
            coordinator,
            thread_id,
            view.name,
            view.value,
            view.derivation,
            view.requires_grad,
            description=view.description,
            meta=view.meta or None,
        )

    def consolidate(
        self, name: str, feedback: list[GradFeedback], retrieved: dict[str, str] | None = None
    ) -> None:
        """Incorporate feedback into a parameter.

        Runs with the ambient thread scope cleared: consolidation's internal
        model calls must not attribute to whatever thread happens to be
        running (see :func:`~ai_functions.types.context.no_thread_scope`).

        Args:
            name: Parameter name.
            feedback: Gradients to merge into the stored value. A text-rewriting
                backend reads each entry's ``text``; the numeric ``score`` is
                for hosts (e.g. the economics beliefs adapter) that learn from it.
            retrieved: For list parameters, the ``{entry_id: value}`` mapping
                the forward pass retrieved (merged from the search derivation
                meta by the optimizer), so consolidation targets those entries;
                ``None`` consolidates against the full value.
        """
        with no_thread_scope():
            self._consolidate(name, feedback, retrieved=retrieved)

    def save(self, name: str, value: Any) -> None:  # pyright: ignore[reportExplicitAny]
        """Store a parameter's value directly, without consolidation."""
        self._save(name, value)

    def fetch(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        """Return a parameter's current value without emitting a recall event.

        The synchronous, event-free counterpart of :meth:`recall`, for
        machinery reading its own bookkeeping state (e.g. a belief provider
        loading persisted statistics): a plain read that must not appear as a
        dataflow edge in any thread's optimization graph.
        """
        value, _ = self._recall(name)  # pyright: ignore[reportAny]
        return value

    def delete(self, name: str) -> None:
        """Reset a parameter to its schema default."""
        self._delete(name)

    # -- Scoped tool provider --------------------------------------------------

    def tool_provider(self, *names: str, operations: set[str] | None = None) -> DynamicToolProvider:
        """Return a ``ToolProvider`` with tools scoped to the given parameter names.

        For each parameter, generates uniquely-named tools (e.g. ``recall_facts``,
        ``search_visited``) whose descriptions are derived from the schema so an
        agent can read and write memory directly during a cycle. ``search_<name>``
        is generated only for list parameters; ``save_<name>`` / ``delete_<name>``
        only for scalar parameters.

        The generated tools take no coordinator/thread arguments, but recall
        operations pick up the ambient :func:`thread_scope` the runtime opens
        for each cycle — so a tool call inside a running thread still emits a
        ``ParameterRecalledEvent`` and feeds the optimization graph.

        Args:
            names: One or more parameter names (slash-separated for nested fields).
            operations: Restrict to this subset of ``{"recall", "query",
                "search", "save", "delete"}``; all applicable tools if ``None``.

        Returns:
            A ``DynamicToolProvider`` holding the generated tools.
        """
        ops = operations or {"recall", "query", "search", "save", "delete"}
        tools: list[AgentTool] = []
        for name in names:
            desc = self._get_description(name) or name
            safe = name.replace("/", "_")
            is_list = self._is_list_field(name)
            if "recall" in ops:
                tools.append(
                    _strands_tool(name=f"recall_{safe}", description=f"Retrieve the full content of: {desc}")(
                        self._make_recall_tool(name)
                    )
                )
            if "query" in ops:
                tools.append(
                    _strands_tool(name=f"query_{safe}", description=f"Ask a natural-language question about: {desc}")(
                        self._make_query_tool(name)
                    )
                )
            if "search" in ops and is_list:
                tools.append(
                    _strands_tool(name=f"search_{safe}", description=f"Search for relevant entries in: {desc}")(
                        self._make_search_tool(name)
                    )
                )
            if "save" in ops and not is_list:
                tools.append(
                    _strands_tool(name=f"save_{safe}", description=f"Overwrite the content of: {desc}")(
                        self._make_save_tool(name)
                    )
                )
            if "delete" in ops and not is_list:
                tools.append(
                    _strands_tool(name=f"delete_{safe}", description=f"Reset to default: {desc}")(
                        self._make_delete_tool(name)
                    )
                )
        return DynamicToolProvider(tools)

    # -- Tool factories (capture the parameter name via closure) ---------------

    def _make_recall_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        async def _recall() -> Any:  # pyright: ignore[reportExplicitAny]
            """Retrieve the full content of this memory parameter."""
            # Tools hand the raw value to the agent; the recall event was
            # already emitted (via the ambient scope) by the public method.
            return (await self.recall(name)).value  # pyright: ignore[reportAny]

        return _recall

    def _make_query_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        async def _query(query: str) -> str:
            """Ask a question about this memory parameter.

            Args:
                query: The natural-language question to answer.
            """
            return (await self.query(name, query)).value

        return _query

    def _make_search_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        async def _search(query: str, k: int = 5) -> list[str]:
            """Search for relevant entries in this memory parameter.

            Args:
                query: Keywords or phrase to match against entries.
                k: Maximum number of results to return.
            """
            return cast("list[str]", (await self.search(name, query, k)).value)

        return _search

    def _make_save_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        def _save(value: str) -> str:
            """Overwrite this memory parameter with a new value.

            Args:
                value: The new value to store.
            """
            self.save(name, value)
            return "Saved"

        return _save

    def _make_delete_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        def _delete() -> str:
            """Reset this memory parameter to its default value."""
            self.delete(name)
            return "Deleted"

        return _delete

    # -- Context manager -------------------------------------------------------

    def __enter__(self) -> Self:
        """Enter the context, returning this backend."""
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit the context, closing the backend."""
        self.close()


class DynamicToolProvider(ToolProvider):
    """A ``ToolProvider`` holding a pre-built list of memory tools.

    Returned by :meth:`MemoryBackend.tool_provider`. Consumer tracking is a
    no-op set kept only to satisfy the Strands ``ToolProvider`` contract; the
    tools are already materialized so nothing is loaded lazily per consumer.
    """

    def __init__(self, tools: list[AgentTool]) -> None:
        self._tools = tools
        self._consumers: set[object] = set()

    @property
    def tools(self) -> list[AgentTool]:
        """The pre-built tools (subclasses extend providers by combining these)."""
        return list(self._tools)

    async def load_tools(self, **kwargs: object) -> Sequence[AgentTool]:
        """Return the pre-built tools."""
        return self._tools

    def add_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Register a consumer (no-op beyond bookkeeping)."""
        self._consumers.add(consumer_id)

    def remove_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Deregister a consumer (no-op beyond bookkeeping)."""
        self._consumers.discard(consumer_id)
