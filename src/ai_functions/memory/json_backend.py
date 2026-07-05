"""JSON-file-backed memory backend with stable per-entry ids for list parameters.

Stores a memory schema as a Pydantic model serialized to a JSON file,
namespaced per ``actor_id``. Scalar and procedural consolidation (merging
feedback into values) use internal AI functions that rewrite the value; list
consolidation is *agentic*: an internal AI function is handed CRUD tools
scoped to the parameter (:class:`MemoryToolProvider`) and edits the store
entry by entry, so untouched entries are never paraphrased or dropped.

Every list entry has a stable string id, allocated from a persisted
per-parameter monotonic counter and **never reused** — an id recorded in an
event log during the forward pass (``search`` puts ``{"results": {entry_id:
value}}`` in its derivation meta) still resolves to the same logical entry at
consolidation time, across saves, deletes, other consolidations, and reopens.

File format (version 2)::

    {actor_id: {"_format": 2,
                "data": <schema dump>,
                "lists": {param: {"next_id": int, "ids": [str, ...]}}}}

Legacy files (a bare schema dump per actor) are read transparently; their list
entries get fresh ids, and ``close()`` writes the new format.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from strands.models import Model
from strands.tools import ToolProvider
from strands.tools.decorator import tool as _strands_tool  # pyright: ignore[reportUnknownVariableType]

from ..ai_thread import ai_function
from ..ai_thread.postcondition import PostConditionResult
from .base import DynamicToolProvider, MemoryBackend, ParameterMeta, ValueType
from .procedural import validate_procedural

if TYPE_CHECKING:
    from collections.abc import Sequence

    from strands.types.tools import AgentTool

_FORMAT_KEY = "_format"
_FORMAT_VERSION = 2


def _bullet_points(values: list[str]) -> str:
    return "\n".join(f"- {v}" for v in values)


def _check_valid_python(response: str) -> PostConditionResult:
    """Post-condition: consolidated procedural code must parse as Python."""
    try:
        validate_procedural(response)
    except SyntaxError as exc:
        return PostConditionResult(passed=False, message=f"Code is not valid Python: {exc}")
    return PostConditionResult(passed=True)


@ai_function[str](structured_output=False)
def _consolidate_list(memories: str, feedback: str) -> str:
    """Drive the list-consolidation agent: edit entries with the memory tools."""
    return (
        "You are a memory manager. The memory entries listed below were retrieved from a "
        "memory store because they are relevant to the feedback. Use the provided tools to "
        "search, add, update, or delete entries as needed to incorporate the feedback.\n\n"
        f"<retrieved_memories>\n{memories}\n</retrieved_memories>\n\n"
        f"<feedback>\n{feedback}\n</feedback>\n\n"
        "Rules:\n"
        "- Update or delete entries by the entry_id shown above; use search_memories to find others.\n"
        "- Prefer updating an existing entry over adding a near-duplicate.\n"
        "- Keep each entry concise and self-contained.\n"
        '- After you have applied all necessary changes using the tools, answer exactly "done".'
    )


@ai_function[str]
def _consolidate_value(value: str, feedback: list[str]) -> str:
    """Build the prompt to merge feedback into a scalar-valued parameter."""
    return (
        "Update the following value with the feedback provided.\n"
        "Return the updated value.\n\n"
        f"<value>\n{value}\n</value>\n\n"
        f"<feedback>\n{_bullet_points(feedback)}\n</feedback>"
    )


@ai_function[str](post_conditions=[_check_valid_python])
def _consolidate_procedural(value: str, feedback: list[str]) -> str:
    """Build the prompt to merge feedback into a code-valued parameter."""
    return (
        "Update the following code with the feedback provided.\n"
        "Return the complete updated code as a string.\n\n"
        f"<code>\n{value}\n</code>\n\n"
        f"<feedback>\n{_bullet_points(feedback)}\n</feedback>"
    )


@ai_function[str]
def _query_value(value: str, query: str) -> str:
    """Build the prompt to answer a question over a parameter's value."""
    return (
        "Based on the content below, answer the following question:\n"
        f"<question>{query}</question>\n"
        f"<content>{value}</content>"
    )


def _get_nested_attr(obj: Any, path: str) -> Any:  # pyright: ignore[reportExplicitAny]
    for part in path.split("/"):
        obj = getattr(obj, part)  # pyright: ignore[reportAny]
    return obj  # pyright: ignore[reportAny]


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:  # pyright: ignore[reportExplicitAny]
    parts = path.split("/")
    for part in parts[:-1]:
        obj = getattr(obj, part)  # pyright: ignore[reportAny]
    setattr(obj, parts[-1], value)  # pyright: ignore[reportAny]


def _is_str_list_field(annotation: Any) -> bool:  # pyright: ignore[reportExplicitAny]
    """Return whether ``annotation`` is ``list[str]`` (possibly Optional-wrapped)."""
    import typing

    origin = typing.get_origin(annotation)
    if origin is list:
        args = typing.get_args(annotation)
        return bool(args) and args[0] is str
    # Unwrap Optional[list[str]] / Annotated[...] and re-check the members.
    for arg in typing.get_args(annotation):
        if _is_str_list_field(arg):
            return True
    return False


class JSONMemoryBackend(MemoryBackend):
    """File-backed memory using JSON serialization with stable list-entry ids."""

    def __init__(
        self,
        schema: type[BaseModel],
        actor_id: str,
        path: Path | str,
        model: Model | str | None = None,
    ) -> None:
        super().__init__(schema, actor_id)
        self.path = Path(path)

        self._consolidate_value_fn = _consolidate_value.replace(model=model)
        self._consolidate_procedural_fn = _consolidate_procedural.replace(model=model)
        self._consolidate_list_fn = _consolidate_list.replace(model=model)
        self._query_value_fn = _query_value.replace(model=model)

        self._all_actors: dict[str, dict[str, Any]] = {}  # pyright: ignore[reportExplicitAny]
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open() as f:
                self._all_actors = json.load(f)

        # Entry ledgers for list parameters: ids aligned with list order, plus
        # a persisted monotonic counter per parameter (ids are never reused).
        self._ids: dict[str, list[str]] = {}
        self._next_id: dict[str, int] = {}

        raw = self._all_actors.get(actor_id)
        if raw is not None:
            data, lists = self._split_record(raw)
            self._model = schema.model_validate(data)
            for name, ledger in lists.items():
                self._ids[name] = [str(i) for i in ledger.get("ids", [])]  # pyright: ignore[reportAny]
                self._next_id[name] = int(ledger.get("next_id", 1))  # pyright: ignore[reportAny]
        else:
            self._model = schema()
        self._ensure_ledgers()

    # -- Ledger bookkeeping ------------------------------------------------------

    @staticmethod
    def _split_record(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:  # pyright: ignore[reportExplicitAny]
        """Split a per-actor record into ``(schema data, list ledgers)``.

        Recognizes the versioned format; a legacy record (bare schema dump)
        yields empty ledgers, so its list entries get fresh ids.
        """
        if raw.get(_FORMAT_KEY) == _FORMAT_VERSION:
            return raw.get("data", {}), raw.get("lists", {})
        return raw, {}

    def _record(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Serialize this actor's state in the versioned file format."""
        lists = {
            name: {"next_id": self._next_id[name], "ids": list(self._ids[name])}
            for name in self._list_parameter_names()
        }
        return {_FORMAT_KEY: _FORMAT_VERSION, "data": self._model.model_dump(), "lists": lists}

    def _list_parameter_names(self) -> list[str]:
        """Leaf parameter paths whose fields are list-valued."""
        return [name for name in self._leaf_parameter_names() if self._is_list_field(name)]

    def _alloc_id(self, name: str) -> str:
        """Allocate the next entry id for ``name`` (monotonic, never reused)."""
        n = self._next_id.get(name, 1)
        self._next_id[name] = n + 1
        return str(n)

    def _ensure_ledgers(self) -> None:
        """Make every list parameter's ledger exist and align with its values.

        Every list parameter ends up with both an id list and a counter, even
        when the list is empty (an empty list is a valid, aligned ledger — not
        a missing one). A ledger that is absent or misaligned (legacy file,
        external edit) is rebuilt with fresh ids; an aligned one is left
        untouched so its ids stay stable.
        """
        for name in self._list_parameter_names():
            self._next_id.setdefault(name, 1)
            values = _get_nested_attr(self._model, name)  # pyright: ignore[reportAny]
            if not isinstance(values, list):
                self._ids.setdefault(name, [])
                continue
            if name not in self._ids or len(self._ids[name]) != len(values):  # pyright: ignore[reportUnknownArgumentType]
                self._ids[name] = [self._alloc_id(name) for _ in values]  # pyright: ignore[reportUnknownVariableType]

    def list_entries(self, name: str) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Return ``{entry_id: value}`` for a list parameter, in list order.

        Args:
            name: Parameter name (slash-separated for nested fields).

        Raises:
            TypeError: ``name`` is not a list parameter (search and entry CRUD
                are only supported for list parameters).
        """
        if not self._is_list_field(name):
            raise TypeError(f"Entry operations are only supported for list parameters, but '{name}' is not one.")
        values = _get_nested_attr(self._model, name)  # pyright: ignore[reportAny]
        return dict(zip(self._ids[name], values, strict=True))  # pyright: ignore[reportUnknownArgumentType]

    # -- Entry CRUD (used by MemoryToolProvider and the consolidation agent) -----

    def _list_add(self, name: str, value: str) -> str:
        """Append a new entry and return its stable entry id."""
        entries = _get_nested_attr(self._model, name)  # pyright: ignore[reportAny]
        entry_id = self._alloc_id(name)
        entries.append(value)  # pyright: ignore[reportUnknownMemberType]
        self._ids[name].append(entry_id)
        return entry_id

    def _list_update(self, name: str, entry_id: str, value: str) -> bool:
        """Replace an entry's value by id, keeping the id. True on success."""
        try:
            index = self._ids[name].index(entry_id)
        except (KeyError, ValueError):
            return False
        _get_nested_attr(self._model, name)[index] = value
        return True

    def _list_remove(self, name: str, entry_id: str) -> bool:
        """Remove an entry by id (the id is retired, never reused). True on success."""
        try:
            index = self._ids[name].index(entry_id)
        except (KeyError, ValueError):
            return False
        del _get_nested_attr(self._model, name)[index]
        del self._ids[name][index]
        return True

    # -- Abstract storage contract ------------------------------------------------

    def _save(self, name: str, value: ValueType) -> None:
        _set_nested_attr(self._model, name, value)
        if self._is_list_field(name) and isinstance(value, list):
            # A wholesale replace retires the old entries; new ones get fresh
            # ids (the counter is monotonic, so retired ids are never reused).
            self._ids[name] = [self._alloc_id(name) for _ in value]

    def _consolidate(
        self,
        name: str,
        feedback: list[str],
        retrieved: dict[str, str] | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny]
    ) -> None:
        if self._is_procedural(name):
            value = _get_nested_attr(self._model, name)  # pyright: ignore[reportAny]
            value = self._consolidate_procedural_fn.run_sync(value=value, feedback=feedback)
            _set_nested_attr(self._model, name, value)
        elif self._is_list_field(name):
            # The agentic consolidator's CRUD tools are typed for str entries;
            # routing a list[BaseModel] through them would corrupt the schema.
            if not _is_str_list_field(self._resolve_field(name).annotation):
                raise NotImplementedError(
                    f"JSONMemoryBackend cannot consolidate non-string list parameter '{name}'. "
                    f"Only list[str] (and str / Procedural) fields are supported."
                )
            self._consolidate_entries(name, feedback, retrieved)
        else:
            value = _get_nested_attr(self._model, name)  # pyright: ignore[reportAny]
            value = self._consolidate_value_fn.run_sync(value=value, feedback=feedback)
            _set_nested_attr(self._model, name, value)

    def _consolidate_entries(self, name: str, feedback: list[str], retrieved: dict[str, str] | None) -> None:
        """Agentic list consolidation: an AI function edits entries via CRUD tools.

        The prompt shows the *current* values of the retrieved entries (ids
        from the search derivation meta, values re-read from the store — an
        entry may have been updated since the search). Stale ids (entries
        deleted since the search) are dropped; with no retrieval context, or
        none of it still valid, the full entry set is shown.
        """
        from ..optimizer._formatting import to_yaml

        entries = self.list_entries(name)
        snapshot = {i: entries[i] for i in (retrieved or {}) if i in entries} or entries
        fn = self._consolidate_list_fn.replace(tools=[MemoryToolProvider(self, name)])
        fn.run_sync(memories=to_yaml(snapshot), feedback=_bullet_points(feedback))

    def _delete(self, name: str) -> None:
        """Reset a parameter to its schema default."""
        field_info = self._resolve_field(name)
        # A required field has no default to reset to; surface it clearly.
        if field_info.is_required():
            raise ValueError(f"Cannot delete required parameter '{name}': it has no schema default.")
        default = field_info.get_default(call_default_factory=True)  # pyright: ignore[reportAny]
        _set_nested_attr(self._model, name, default)
        if self._is_list_field(name) and isinstance(default, list):
            self._ids[name] = [self._alloc_id(name) for _ in default]  # pyright: ignore[reportUnknownVariableType]

    def _recall(self, name: str) -> tuple[ValueType, ParameterMeta]:
        return _get_nested_attr(self._model, name), {}

    def _query(self, name: str, query: str) -> tuple[str, ParameterMeta]:
        value = _get_nested_attr(self._model, name)  # pyright: ignore[reportAny]
        return self._query_value_fn.run_sync(value=value, query=query), {}

    def _search(self, name: str, query: str, k: int = 5, **kwargs: Any) -> tuple[list[str], ParameterMeta]:  # pyright: ignore[reportExplicitAny]
        """Return the top-k entries of a list parameter, ranked by BM25 against query.

        Only supported for ``list[str]`` parameters. The meta carries
        ``{"results": {entry_id: value}}`` for the returned entries, in rank
        order — recorded into the recall event so consolidation can target
        exactly the entries the forward pass retrieved.
        """
        ranked = self._search_entries(name, query, k)
        return [value for _, value in ranked], {"results": dict(ranked)}

    def _search_entries(self, name: str, query: str, k: int = 5) -> list[tuple[str, str]]:
        """Return the top-k ``(entry_id, value)`` pairs ranked by BM25."""
        entries = self.list_entries(name)
        if not entries:
            return []
        corpus = [(entry_id, str(value)) for entry_id, value in entries.items()]  # pyright: ignore[reportAny]

        from rank_bm25 import BM25Okapi

        bm25 = BM25Okapi([value.lower().split() for _, value in corpus])
        scores = bm25.get_scores(query.lower().split())  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        ranked = sorted(zip(corpus, scores, strict=True), key=lambda pair: pair[1], reverse=True)  # pyright: ignore[reportUnknownArgumentType, reportUnknownVariableType, reportUnknownLambdaType]
        return [pair for pair, _ in ranked[:k]]  # pyright: ignore[reportUnknownVariableType]

    # -- Tool provider (adds entry CRUD tools for list parameters) ---------------

    def tool_provider(self, *names: str, operations: set[str] | None = None) -> DynamicToolProvider:
        """Extend the base tools with entry-id-based CRUD tools for list parameters.

        In addition to the base ``recall_<name>`` / ``query_<name>`` /
        ``search_<name>`` (and scalar ``save_<name>`` / ``delete_<name>``),
        list parameters get ``add_to_<name>``, ``update_<name>``, and
        ``delete_from_<name>`` operating on stable entry ids.

        Args:
            names: One or more parameter names (slash-separated for nested fields).
            operations: Restrict to this subset of ``{"recall", "query",
                "search", "save", "delete", "add", "update"}``; all applicable
                tools if ``None``.

        Returns:
            A ``DynamicToolProvider`` holding the generated tools.
        """
        ops = operations or {"recall", "query", "search", "save", "delete", "add", "update"}
        provider = super().tool_provider(*names, operations=ops)
        extra: list[AgentTool] = []
        for name in names:
            if not self._is_list_field(name):
                continue
            desc = self._get_description(name) or name
            safe = name.replace("/", "_")
            if "add" in ops:
                extra.append(
                    _strands_tool(name=f"add_to_{safe}", description=f"Add a new entry to: {desc}")(
                        self._make_entry_add_tool(name)
                    )
                )
            if "update" in ops:
                extra.append(
                    _strands_tool(name=f"update_{safe}", description=f"Update an entry by entry_id in: {desc}")(
                        self._make_entry_update_tool(name)
                    )
                )
            if "delete" in ops:
                extra.append(
                    _strands_tool(name=f"delete_from_{safe}", description=f"Delete an entry by entry_id from: {desc}")(
                        self._make_entry_delete_tool(name)
                    )
                )
        return DynamicToolProvider(provider.tools + extra)

    def _make_entry_add_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        def _add(value: str) -> str:
            """Add a new entry to this list.

            Args:
                value: The text content of the new entry.
            """
            return f"Added with entry_id={self._list_add(name, value)}"

        return _add

    def _make_entry_update_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        def _update(entry_id: str, value: str) -> str:
            """Update an existing entry by its stable entry_id.

            Args:
                entry_id: The stable identifier of the entry to update.
                value: The new text content.
            """
            if not self._list_update(name, entry_id, value):
                raise ValueError(f"entry_id={entry_id} not found")
            return f"Updated entry_id={entry_id}"

        return _update

    def _make_entry_delete_tool(self, name: str) -> Any:  # pyright: ignore[reportExplicitAny]
        def _delete(entry_id: str) -> str:
            """Delete an entry by its stable entry_id.

            Args:
                entry_id: The stable identifier of the entry to delete.
            """
            if not self._list_remove(name, entry_id):
                raise ValueError(f"entry_id={entry_id} not found")
            return f"Deleted entry_id={entry_id}"

        return _delete

    # -- Persistence ---------------------------------------------------------------

    def close(self) -> None:
        """Persist this actor's current values back to the JSON file.

        Re-reads the file and merges in only this actor's key, so concurrent
        backends sharing one file do not clobber each other (the in-memory
        snapshot taken at construction may be stale). Writes atomically
        (temp-file + rename) so a crash mid-write cannot corrupt the file for
        other actors. Other actors' records are preserved byte-for-byte,
        whatever format they are in.
        """
        on_disk: dict[str, dict[str, Any]] = {}  # pyright: ignore[reportExplicitAny]
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open() as f:
                on_disk = json.load(f)

        on_disk[self.actor_id] = self._record()
        self._all_actors = on_disk

        # Atomic write: temp file in the same dir, then os.replace (matches the
        # pattern in discovery.py / session.py). A crash mid-write leaves the
        # prior file intact for all actors.
        fd, tmp_path = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(on_disk, fh)
            os.replace(tmp_path, self.path)
        except Exception:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def dump(self) -> BaseModel:
        """Return the underlying Pydantic model.

        Read-only by contract: mutating a list field on the returned model
        directly bypasses the entry-id ledger. Go through ``save`` or the
        entry CRUD tools instead.
        """
        return self._model

    def __str__(self) -> str:
        """Return a human-readable YAML dump of the current values.

        Uses ``to_yaml`` so multi-line values (e.g. ``Procedural`` code) render
        as unquoted literal blocks rather than escaped one-line scalars.
        """
        from ..optimizer._formatting import to_yaml

        return to_yaml(self._model.model_dump())


class MemoryToolProvider(ToolProvider):
    """CRUD tools scoped to one ``list[str]`` parameter on a :class:`JSONMemoryBackend`.

    Handed to the list-consolidation agent (and usable directly on any agent)
    so it can search, add, update, and delete entries by their stable
    ``entry_id`` instead of rewriting the whole list.
    """

    def __init__(self, backend: JSONMemoryBackend, name: str) -> None:
        """Scope the tools to ``name`` on ``backend``."""
        self._backend = backend
        self._name = name
        self._consumers: set[object] = set()
        self._tools: list[AgentTool] = self._build_tools()

    def _build_tools(self) -> list[AgentTool]:
        backend, name = self._backend, self._name

        def search_memories(query: str, k: int = 5) -> list[dict[str, str]]:
            """Search entries by keyword relevance.

            Args:
                query: Keywords to match against entry texts.
                k: Maximum number of results.

            Returns:
                A list of ``{"entry_id": ..., "value": ...}`` dicts, most
                relevant first.
            """
            return [{"entry_id": i, "value": v} for i, v in backend._search_entries(name, query, k)]  # noqa: SLF001

        def add_memory(value: str) -> str:
            """Add a new memory entry to the list.

            Args:
                value: The text content of the new entry.
            """
            return f"Added with entry_id={backend._list_add(name, value)}"  # noqa: SLF001

        def update_memory(entry_id: str, value: str) -> str:
            """Update an existing memory entry by its stable entry_id.

            Args:
                entry_id: The stable identifier of the entry to update.
                value: The new text content.
            """
            if not backend._list_update(name, entry_id, value):  # noqa: SLF001
                raise ValueError(f"entry_id={entry_id} not found")
            return f"Updated entry_id={entry_id}"

        def delete_memory(entry_id: str) -> str:
            """Delete a memory entry by its stable entry_id.

            Args:
                entry_id: The stable identifier of the entry to delete.
            """
            if not backend._list_remove(name, entry_id):  # noqa: SLF001
                raise ValueError(f"entry_id={entry_id} not found")
            return f"Deleted entry_id={entry_id}"

        return [
            _strands_tool(name="search_memories", description="Search entries by keyword relevance.")(search_memories),
            _strands_tool(name="add_memory", description="Add a new entry to the list.")(add_memory),
            _strands_tool(name="update_memory", description="Update an entry by its entry_id.")(update_memory),
            _strands_tool(name="delete_memory", description="Delete an entry by its entry_id.")(delete_memory),
        ]

    async def load_tools(self, **kwargs: object) -> Sequence[AgentTool]:
        """Return the CRUD tools."""
        return self._tools

    def add_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Register a consumer (bookkeeping only)."""
        self._consumers.add(consumer_id)

    def remove_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Deregister a consumer (bookkeeping only)."""
        self._consumers.discard(consumer_id)
