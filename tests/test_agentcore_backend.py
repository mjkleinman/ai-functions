"""Offline tests for the AgentCore memory backend.

The real backend talks to AWS Bedrock AgentCore; these tests substitute a
minimal in-memory fake for ``bedrock_agentcore.memory`` (installed into
``sys.modules`` per test), so they cover the backend's own logic — error
semantics, list round-trips, legacy-format compatibility, nested schemas,
strict save-deletes — without credentials or the optional dependency.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any

import pytest
from pydantic import BaseModel, Field

from ai_functions import AgentCoreMemoryBackend, Procedural

# ── Fake bedrock_agentcore.memory ────────────────────────────────────────────


class _FakeClientError(Exception):
    """Mimics botocore ClientError: carries response["Error"]["Code"]."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


@dataclass
class _FakeMessage:
    text: str
    role: str


class _FakeConstants:
    ConversationalMessage = _FakeMessage
    MessageRole = types.SimpleNamespace(USER="USER", ASSISTANT="ASSISTANT")


@dataclass
class _ActorStore:
    """Per-actor STM events + LTM records."""

    events: list[dict[str, Any]] = field(default_factory=list)
    ltm_records: list[dict[str, Any]] = field(default_factory=list)
    next_id: int = 0


@dataclass
class _FakeState:
    """Shared fake-service state, pokeable by tests."""

    actors: dict[str, _ActorStore] = field(default_factory=dict)
    list_events_error: Exception | None = None
    delete_event_error: Exception | None = None

    def store(self, actor_id: str) -> _ActorStore:
        return self.actors.setdefault(actor_id, _ActorStore())


class _FakeSession:
    def __init__(self, actor_id: str, state: _FakeState) -> None:
        self.actor_id = actor_id
        self.state = state

    def add_turns(self, messages: list[_FakeMessage], metadata: object, event_timestamp: object) -> None:
        store = self.state.store(self.actor_id)
        store.events.append(
            {
                "eventId": f"evt-{store.next_id}",
                "payload": [{"conversational": {"content": {"text": m.text}}} for m in messages],
            }
        )
        store.next_id += 1

    def list_events(self, include_payload: bool, max_results: int) -> list[dict[str, Any]]:
        if self.state.list_events_error is not None:
            raise self.state.list_events_error
        return list(self.state.store(self.actor_id).events[:max_results])

    def search_long_term_memories(self, query: str, namespace_prefix: str, top_k: int) -> list[dict[str, Any]]:
        return list(self.state.store(self.actor_id).ltm_records[:top_k])


class _FakeManager:
    def __init__(self, memory_id: str, region_name: str) -> None:
        self.memory_id = memory_id
        self.state = _FakeState()

    def create_memory_session(self, actor_id: str, session_id: str) -> _FakeSession:
        return _FakeSession(actor_id, self.state)

    def list_long_term_memory_records(self, namespace_prefix: str, max_results: int) -> list[dict[str, Any]]:
        actor = namespace_prefix.strip("/")
        return list(self.state.store(actor).ltm_records[:max_results])

    def delete_event(self, actor_id: str, session_id: str, event_id: str) -> None:
        if self.state.delete_event_error is not None:
            raise self.state.delete_event_error
        store = self.state.store(actor_id)
        store.events = [e for e in store.events if e["eventId"] != event_id]

    def delete_all_long_term_memories_in_namespace(self, namespace: str) -> None:
        self.state.store(namespace.strip("/")).ltm_records = []


@pytest.fixture
def fake_acm(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``bedrock_agentcore.memory`` and return it."""
    memory_mod = types.ModuleType("bedrock_agentcore.memory")
    memory_mod.MemorySessionManager = _FakeManager  # type: ignore[attr-defined]
    memory_mod.constants = _FakeConstants  # type: ignore[attr-defined]
    pkg = types.ModuleType("bedrock_agentcore")
    pkg.memory = memory_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bedrock_agentcore", pkg)
    monkeypatch.setitem(sys.modules, "bedrock_agentcore.memory", memory_mod)
    return memory_mod


# ── Schemas ──────────────────────────────────────────────────────────────────


class _Profile(BaseModel):
    tone: str = Field(default="neutral", description="Writing tone")


class _Memory(BaseModel):
    guidelines: str = Field("No guidelines yet.", description="Guidelines")
    tags: list[str] = Field(default_factory=list, description="Tags")
    profile: _Profile = Field(default_factory=lambda: _Profile(), description="Nested profile")


class _RequiredMemory(BaseModel):
    mandatory: str = Field(description="Required, no default")


class _NestedProcedural(BaseModel):
    class Inner(BaseModel):
        code: Procedural = Field(description="Nested procedural code")

    inner: Inner = Field(default_factory=lambda: _NestedProcedural.Inner(code=""), description="Inner model")


def _backend(fake_acm: types.ModuleType, schema: type[BaseModel] = _Memory) -> AgentCoreMemoryBackend:
    return AgentCoreMemoryBackend(schema, actor_id="a1", memory_id="mem-1")


def _state(backend: AgentCoreMemoryBackend) -> _FakeState:
    return backend.manager.state  # type: ignore[no-any-return]


# ── Round trips ──────────────────────────────────────────────────────────────


def test_scalar_save_recall_roundtrip(fake_acm: types.ModuleType) -> None:
    mem = _backend(fake_acm)
    mem.save("guidelines", "Always about cats.")
    assert mem._recall("guidelines")[0] == "Always about cats."  # noqa: SLF001


def test_list_roundtrip_preserves_items_with_newlines(fake_acm: types.ModuleType) -> None:
    """Items are stored one per message, so single newlines inside items survive."""
    mem = _backend(fake_acm)
    items = ["first tip\nwith a second line", "second tip", "third: a, b, c"]
    mem.save("tags", items)
    assert mem._recall("tags")[0] == items  # noqa: SLF001


def test_list_item_blank_lines_collapse_but_items_stay_whole(fake_acm: types.ModuleType) -> None:
    """A blank line inside an item is collapsed (it is the legacy separator), not split."""
    mem = _backend(fake_acm)
    mem.save("tags", ["para one\n\npara two", "other"])
    assert mem._recall("tags")[0] == ["para one\npara two", "other"]  # noqa: SLF001


def test_recall_splits_legacy_format_messages(fake_acm: types.ModuleType) -> None:
    """A legacy message holds all items joined by blank lines; recall splits it."""
    mem = _backend(fake_acm)
    actor = mem._parameter_actor("tags")  # noqa: SLF001
    _state(mem).store(actor).events.append(
        {"eventId": "legacy-evt", "payload": [{"conversational": {"content": {"text": "alpha\n\nbeta\n\ngamma"}}}]}
    )
    assert mem._recall("tags")[0] == ["alpha", "beta", "gamma"]  # noqa: SLF001


def test_save_replaces_previous_value(fake_acm: types.ModuleType) -> None:
    mem = _backend(fake_acm)
    mem.save("guidelines", "old value")
    mem.save("guidelines", "new value")
    assert mem._recall("guidelines")[0] == "new value"  # noqa: SLF001


# ── Defaults, nested paths, required fields ──────────────────────────────────


def test_recall_empty_returns_schema_default(fake_acm: types.ModuleType) -> None:
    mem = _backend(fake_acm)
    assert mem._recall("guidelines")[0] == "No guidelines yet."  # noqa: SLF001
    assert mem._recall("tags")[0] == []  # noqa: SLF001


def test_nested_path_roundtrip_and_default(fake_acm: types.ModuleType) -> None:
    """Slash paths resolve through nested models for defaults and round trips."""
    mem = _backend(fake_acm)
    assert mem._recall("profile/tone")[0] == "neutral"  # noqa: SLF001
    mem.save("profile/tone", "formal")
    assert mem._recall("profile/tone")[0] == "formal"  # noqa: SLF001


def test_required_field_schema_constructs_and_recalls(fake_acm: types.ModuleType) -> None:
    """Schemas with required fields work; an empty required field raises clearly."""
    mem = _backend(fake_acm, schema=_RequiredMemory)
    with pytest.raises(ValueError, match="no schema default"):
        mem._recall("mandatory")  # noqa: SLF001
    mem.save("mandatory", "now set")
    assert mem._recall("mandatory")[0] == "now set"  # noqa: SLF001
    assert "now set" in str(mem)


def test_str_renders_without_instantiating_schema(fake_acm: types.ModuleType) -> None:
    mem = _backend(fake_acm, schema=_RequiredMemory)
    assert "unset" in str(mem)  # required + empty renders a placeholder, no crash


def test_nested_procedural_field_rejected(fake_acm: types.ModuleType) -> None:
    """Procedural detection recurses into nested models."""
    with pytest.raises(ValueError, match="Procedural"):
        _backend(fake_acm, schema=_NestedProcedural)


# ── Error semantics ──────────────────────────────────────────────────────────


def test_not_found_reads_as_empty(fake_acm: types.ModuleType) -> None:
    """ResourceNotFound means 'no records yet' — recall returns the default."""
    mem = _backend(fake_acm)
    _state(mem).list_events_error = _FakeClientError("ResourceNotFoundException")
    assert mem._recall("guidelines")[0] == "No guidelines yet."  # noqa: SLF001


def test_service_error_propagates_instead_of_reading_empty(fake_acm: types.ModuleType) -> None:
    """An auth error must raise — not silently read as the schema default."""
    mem = _backend(fake_acm)
    mem.save("guidelines", "real stored value")
    _state(mem).list_events_error = _FakeClientError("AccessDeniedException")
    with pytest.raises(_FakeClientError, match="AccessDeniedException"):
        mem._recall("guidelines")  # noqa: SLF001


def test_save_raises_when_stale_delete_fails(fake_acm: types.ModuleType) -> None:
    """Replacing a value must not silently leave the old records behind."""
    mem = _backend(fake_acm)
    mem.save("guidelines", "old value")
    _state(mem).delete_event_error = _FakeClientError("ThrottlingException")
    with pytest.raises(_FakeClientError, match="ThrottlingException"):
        mem.save("guidelines", "new value")


def test_delete_all_is_best_effort(fake_acm: types.ModuleType) -> None:
    """Cleanup deletion logs failures instead of raising."""
    mem = _backend(fake_acm)
    mem.save("guidelines", "value")
    _state(mem).delete_event_error = _FakeClientError("ThrottlingException")
    mem.delete_all(wait=False)  # must not raise


# ── Search / counts ──────────────────────────────────────────────────────────


def test_search_ranks_stm_by_query(fake_acm: types.ModuleType) -> None:
    """STM texts are BM25-ranked against the query, not returned chronologically."""
    mem = _backend(fake_acm)
    mem.save(
        "tags",
        [
            "let meat rest before slicing",
            "add a pinch of sugar to tomato sauces",
            "toast spices in a dry pan",
        ],
    )
    top, meta = mem._search("tags", "tomato sauce sugar", k=2)  # noqa: SLF001
    assert top[0] == "add a pinch of sugar to tomato sauces"
    assert len(top) <= 2
    assert meta["stm_count"] >= 1  # fetch meta reports record provenance


def test_record_counts_covers_nested_leaves(fake_acm: types.ModuleType) -> None:
    mem = _backend(fake_acm)
    mem.save("guidelines", "v")
    mem.save("profile/tone", "formal")
    total_stm, total_ltm = mem.record_counts()
    assert total_stm == 2
    assert total_ltm == 0
