"""Pure-function correctness: serialize/deserialize, CustomEvent, I2."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, TypeAdapter

from ai_functions import ai_function
from ai_functions.testing import RuntimeHarness, ScriptedModel, Turn
from ai_functions.types import Event, ThreadId
from ai_functions.types.events import CustomEvent, MessageUserEvent

# ── serialize_result / deserialize_result round-trip ──────────────────────


@pytest.mark.parametrize(
    "output_type, value",
    [
        (str, "hello"),
        (int, 42),
        (float, 3.14),
        (list[int], [1, 2, 3]),
        (dict[str, int], {"a": 1, "b": 2}),
    ],
)
async def test_serialize_deserialize_roundtrip_primitives(output_type: type, value: Any) -> None:  # type: ignore[type-arg]
    """Primitives, lists, and dicts round-trip through the serializer."""

    @ai_function[output_type](structured_output=True)
    def _fn(x: str) -> str:
        return x

    async with RuntimeHarness() as h:
        # The harness's AIThread initialization needs a live worker; the
        # thread itself exposes serialize/deserialize on the instance.
        handle = await h.spawn(_fn)
        thread = h.worker._threads[handle.id]  # noqa: SLF001 -- accessing for a pure test
        payload = thread.serialize_result(value)
        recovered = thread.deserialize_result(payload)
        assert recovered == value


class _PydModel(BaseModel):
    name: str
    count: int


async def test_serialize_deserialize_roundtrip_pydantic() -> None:
    """Pydantic models round-trip preserving field values and type."""

    @ai_function[_PydModel](structured_output=True)
    def _fn(x: str) -> str:
        return x

    async with RuntimeHarness() as h:
        handle = await h.spawn(_fn)
        thread = h.worker._threads[handle.id]  # noqa: SLF001
        value = _PydModel(name="foo", count=7)
        payload = thread.serialize_result(value)
        recovered = thread.deserialize_result(payload)
        assert isinstance(recovered, _PydModel)
        assert recovered == value


async def test_deserialize_malformed_payload_raises() -> None:
    """A malformed payload raises ``AIFunctionError``."""
    from ai_functions.ai_thread import AIFunctionError

    @ai_function[int](structured_output=True)
    def _fn(x: str) -> str:
        return x

    async with RuntimeHarness() as h:
        handle = await h.spawn(_fn)
        thread = h.worker._threads[handle.id]  # noqa: SLF001
        with pytest.raises(AIFunctionError):
            thread.deserialize_result("{not json")


# ── CustomEvent discriminated-union fallback ──────────────────────────────


def test_unknown_kind_parses_as_custom_event() -> None:
    """An unknown ``kind`` routes the discriminated union to ``CustomEvent``."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    parsed = ta.validate_python({"kind": "my_custom_kind"})
    assert isinstance(parsed, CustomEvent)
    assert parsed.kind == "my_custom_kind"


def test_custom_event_round_trips() -> None:
    """``CustomEvent`` serializes and re-parses to an equal instance."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    original = CustomEvent(kind="my_kind", payload={"a": "hello", "b": 1})
    dumped = ta.dump_python(original)
    reparsed = ta.validate_python(dumped)
    assert isinstance(reparsed, CustomEvent)
    assert reparsed.kind == original.kind
    assert reparsed.payload == original.payload


def test_known_kind_does_not_hit_custom_event() -> None:
    """A known SystemEvent ``kind`` parses to the matching concrete class."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    parsed = ta.validate_python({"kind": "message_user", "thread_id": "thr-xyz", "text": "hi"})
    assert isinstance(parsed, MessageUserEvent)
    assert parsed.text == "hi"


def test_custom_event_collects_extra_fields_into_payload() -> None:
    """Extra top-level fields are folded into ``payload`` by the before-validator."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    parsed = ta.validate_python({"kind": "my_kind", "a": "hello", "b": 1})
    assert isinstance(parsed, CustomEvent)
    assert parsed.payload == {"a": "hello", "b": 1}


def test_custom_event_serializes_flat() -> None:
    """``model_dump`` emits ``{"kind": ..., **payload}`` — no ``"payload"`` key."""
    dumped = CustomEvent(kind="my_kind", payload={"a": "hello", "b": 1}).model_dump()
    assert dumped == {"kind": "my_kind", "a": "hello", "b": 1}
    assert "payload" not in dumped


def test_custom_event_flat_wire_format_round_trips() -> None:
    """Flat wire dict → validator → serializer reproduces the same flat dict."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    flat = {"kind": "my_kind", "a": "hello", "b": 1}
    parsed = ta.validate_python(flat)
    dumped = ta.dump_python(parsed)
    assert dumped == flat


def test_custom_event_explicit_payload_still_collects_extras() -> None:
    """Input may mix an explicit ``payload`` with extras; extras win on key clash."""
    ta: TypeAdapter[Event] = TypeAdapter(Event)
    parsed = ta.validate_python(
        {"kind": "my_kind", "payload": {"a": "from_payload", "c": 3}, "a": "from_extra", "b": 2}
    )
    assert isinstance(parsed, CustomEvent)
    # ``a`` is specified both ways — the extra takes precedence; ``c`` only
    # exists in the explicit payload; ``b`` only as an extra.
    assert parsed.payload == {"a": "from_extra", "c": 3, "b": 2}


# ── I2: event immutability ────────────────────────────────────────────────


async def test_events_are_frozen() -> None:
    """Assigning to a field on an emitted event raises (pydantic frozen)."""
    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="ok")])

        @ai_function[str](structured_output=False)
        def _fn(x: str) -> str:
            return x

        handle = await h.spawn(_fn.replace(model=model))
        await handle.run("x")
        events = await h.events(handle.id)
        assert events  # sanity
        sample = events[0]
        with pytest.raises(Exception):  # noqa: B017, PT011 -- pydantic raises ValidationError
            sample.thread_id = ThreadId("thr-injected")  # type: ignore[misc]
