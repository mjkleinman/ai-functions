"""JSON-file-backed memory backend with stable per-entry ids for list parameters.

Stores a memory schema as a Pydantic model serialized to a JSON file,
namespaced per ``actor_id``. Scalar and procedural consolidation use internal
AI functions that rewrite the value; list consolidation is *agentic*: an
internal AI function edits the store entry by entry through CRUD tools
(:class:`MemoryToolProvider`), so untouched entries are never paraphrased or
dropped.

Every list entry has a stable string id, allocated from a persisted
per-parameter monotonic counter and never reused — an id recorded in an event
log during the forward pass (``search`` puts ``{"results": {entry_id: value}}``
in its derivation meta) still resolves to the same logical entry at
consolidation time. Legacy files (a bare schema dump per actor) are read
transparently; ``close()`` writes the versioned format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from collections.abc import Sequence

from pydantic import BaseModel
from strands.models import Model
from strands.tools import ToolProvider
from strands.types.tools import AgentTool

from .base import DynamicToolProvider, MemoryBackend


class JSONMemoryBackend(MemoryBackend):
    """File-backed memory using JSON serialization with stable list-entry ids.

    The file holds a mapping ``{actor_id: record}``; multiple actors may share
    one file. Construction loads this actor's slice (or an empty schema
    instance); :meth:`close` writes it back atomically, merging by actor.
    """

    path: Path

    def __init__(
        self,
        schema: type[BaseModel],
        actor_id: str,
        path: Path | str,
        model: Model | str | None = None,
    ) -> None:
        """Open a JSON-backed memory store for one actor.

        Args:
            schema: Pydantic model describing the memory parameters.
            actor_id: Namespaces this actor's values within ``path``.
            path: JSON file backing the store; created on :meth:`close` if
                absent.
            model: Model (or model id) the consolidation / query AI functions
                run on. ``None`` uses the library default provider.

        Ensures:
            - If ``path`` exists and contains ``actor_id``, the stored values
              are loaded and validated against ``schema``; a versioned record
              restores the entry-id ledgers, a legacy record gets fresh ids.
            - Otherwise the store starts from a default ``schema()`` instance.
        """
        ...

    def list_entries(self, name: str) -> dict[str, Any]:
        """Return ``{entry_id: value}`` for a list parameter, in list order.

        Args:
            name: Parameter name (slash-separated for nested fields).

        Raises:
            TypeError: ``name`` is not a list parameter.
        """
        ...

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
        ...

    def close(self) -> None:
        """Persist this actor's current values back to ``path``.

        Ensures:
            - ``path`` contains this actor's serialized schema and entry-id
              ledgers under ``actor_id``, preserving any other actors already
              in the file (whatever format their records are in), written
              atomically.
        """
        ...

    def dump(self) -> BaseModel:
        """Return the underlying Pydantic model holding current values.

        Read-only by contract: mutating a list field on the returned model
        directly bypasses the entry-id ledger. Go through ``save`` or the
        entry CRUD tools instead.
        """
        ...

    def __str__(self) -> str:
        """Return a human-readable YAML dump of the current values."""
        ...


class MemoryToolProvider(ToolProvider):
    """CRUD tools scoped to one ``list[str]`` parameter on a :class:`JSONMemoryBackend`.

    Handed to the list-consolidation agent (and usable directly on any agent)
    so it can search, add, update, and delete entries by their stable
    ``entry_id`` instead of rewriting the whole list. Provides
    ``search_memories``, ``add_memory``, ``update_memory``, ``delete_memory``.
    """

    def __init__(self, backend: JSONMemoryBackend, name: str) -> None:
        """Scope the tools to ``name`` on ``backend``."""
        ...

    async def load_tools(self, **kwargs: object) -> Sequence[AgentTool]:
        """Return the CRUD tools."""
        ...

    def add_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Register a consumer (bookkeeping only)."""
        ...

    def remove_consumer(self, consumer_id: object, **kwargs: object) -> None:
        """Deregister a consumer (bookkeeping only)."""
        ...
