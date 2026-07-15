"""Offline tests for the JSON backend's list-entry ledger and agentic consolidation.

Covers: stable entry ids (persisted monotonic counter, never reused), the
versioned file format and legacy migration, entry CRUD, search meta
(``{"results": {entry_id: value}}``), the agentic list consolidator driven by
a ScriptedModel, retrieval-targeted consolidation snapshots, and the optimizer
plumbing that merges retrieval meta into ``backend.consolidate``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel, Field

from ai_functions import JSONMemoryBackend, TextGradOptimizer, ai_function, build_graph_from_result
from ai_functions.types.graph import GradFeedback, ParameterNode, ThreadNode


class CookingMemory(BaseModel):
    tips: list[str] = Field(
        default=["salt the pasta water", "rest the meat", "toast the spices"],
        description="Cooking tips.",
    )
    style: str = Field(default="plain", description="Writing style.")


def _backend(tmp_path: Path, actor: str = "chef") -> JSONMemoryBackend:
    return JSONMemoryBackend(CookingMemory, actor_id=actor, path=tmp_path / "mem.json")


class _CaptureConsolidateFn:
    """Stands in for a consolidation AI function; records its inputs.

    ``returns`` is what ``run_sync`` yields — ``"done"`` for the agentic list
    consolidator (whose return value is discarded; the tools do the work), or
    the rewritten value for the scalar consolidator (whose return value is
    stored).
    """

    def __init__(self, returns: str = "done") -> None:
        self.kwargs: dict[str, Any] | None = None
        self.tools: list[object] | None = None
        self._returns = returns

    def replace(self, **kwargs: Any) -> _CaptureConsolidateFn:
        self.tools = kwargs.get("tools")
        return self

    def run_sync(self, **kwargs: Any) -> str:
        self.kwargs = kwargs
        return self._returns


# ── Entry ledger ─────────────────────────────────────────────────────────────


def test_defaults_get_sequential_ids(tmp_path: Path) -> None:
    mem = _backend(tmp_path)
    assert mem.list_entries("tips") == {
        "1": "salt the pasta water",
        "2": "rest the meat",
        "3": "toast the spices",
    }


def test_entries_of_scalar_param_rejected(tmp_path: Path) -> None:
    mem = _backend(tmp_path)
    with pytest.raises(TypeError, match="list parameters"):
        mem.list_entries("style")


def test_crud_roundtrip(tmp_path: Path) -> None:
    mem = _backend(tmp_path)
    new_id = mem._list_add("tips", "deglaze the pan")  # noqa: SLF001
    assert new_id == "4"
    assert mem._list_update("tips", "2", "rest the meat before slicing")  # noqa: SLF001
    assert mem._list_remove("tips", "1")  # noqa: SLF001
    assert mem.list_entries("tips") == {
        "2": "rest the meat before slicing",
        "3": "toast the spices",
        "4": "deglaze the pan",
    }
    assert not mem._list_update("tips", "99", "x")  # noqa: SLF001
    assert not mem._list_remove("tips", "1")  # noqa: SLF001 -- already gone


def test_ids_never_reused_after_delete(tmp_path: Path) -> None:
    """Deleting the newest entry must not free its id for reuse."""
    mem = _backend(tmp_path)
    assert mem._list_remove("tips", "3")  # noqa: SLF001 -- delete the max id
    assert mem._list_add("tips", "fresh herbs at the end") == "4"  # noqa: SLF001


def test_save_retires_old_ids(tmp_path: Path) -> None:
    """A wholesale save replaces the entries; new ones get fresh ids."""
    mem = _backend(tmp_path)
    mem.save("tips", ["only tip"])
    assert mem.list_entries("tips") == {"4": "only tip"}


def test_delete_reseeds_defaults_with_fresh_ids(tmp_path: Path) -> None:
    mem = _backend(tmp_path)
    mem.delete("tips")
    assert list(mem.list_entries("tips").keys()) == ["4", "5", "6"]
    assert list(mem.list_entries("tips").values()) == list(CookingMemory().tips)


# ── Persistence and migration ────────────────────────────────────────────────


def test_ids_and_counter_persist_across_reopen(tmp_path: Path) -> None:
    mem = _backend(tmp_path)
    mem._list_remove("tips", "3")  # noqa: SLF001
    mem._list_add("tips", "pat proteins dry")  # noqa: SLF001 -- takes id 4
    mem.close()

    reopened = _backend(tmp_path)
    assert reopened.list_entries("tips") == {
        "1": "salt the pasta water",
        "2": "rest the meat",
        "4": "pat proteins dry",
    }
    # The counter persisted too: the next id continues, never reusing "3".
    assert reopened._list_add("tips", "new") == "5"  # noqa: SLF001


def test_legacy_file_migrates_with_fresh_ids(tmp_path: Path) -> None:
    """A v2-legacy record (bare schema dump) loads; entries get fresh ids."""
    path = tmp_path / "mem.json"
    legacy = {"chef": {"tips": ["a", "b"], "style": "casual"}}
    path.write_text(json.dumps(legacy))

    mem = JSONMemoryBackend(CookingMemory, actor_id="chef", path=path)
    assert mem.list_entries("tips") == {"1": "a", "2": "b"}
    assert (mem._recall("style"))[0] == "casual"  # noqa: SLF001
    mem.close()

    on_disk = json.loads(path.read_text())
    assert on_disk["chef"]["_format"] == 2
    assert on_disk["chef"]["lists"]["tips"] == {"next_id": 3, "ids": ["1", "2"]}


def test_empty_default_list_gets_a_ledger(tmp_path: Path) -> None:
    """A list parameter that defaults empty still gets a ledger + counter.

    An empty list is a valid, aligned ledger — not a missing one. ``_ids`` /
    ``_next_id`` must be seeded for it, or the first access — ``close()``
    writing the record, ``list_entries``, or ``search`` — would raise
    ``KeyError``.
    """

    class EmptyDefaults(BaseModel):
        notes: list[str] = Field(default_factory=list, description="Notes.")

    path = tmp_path / "mem.json"
    mem = JSONMemoryBackend(EmptyDefaults, actor_id="a", path=path)
    assert mem.list_entries("notes") == {}
    # Ids allocate from 1 and the ledger round-trips through close/reopen.
    assert mem._list_add("notes", "first") == "1"  # noqa: SLF001
    mem.close()

    reopened = JSONMemoryBackend(EmptyDefaults, actor_id="a", path=path)
    assert reopened.list_entries("notes") == {"1": "first"}
    assert reopened._list_add("notes", "second") == "2"  # noqa: SLF001


def test_close_preserves_other_actor_records_verbatim(tmp_path: Path) -> None:
    """Merging by actor must not rewrite records it does not own."""
    path = tmp_path / "mem.json"
    legacy_other = {"tips": ["other actor tip"], "style": "loud"}
    path.write_text(json.dumps({"other": legacy_other}))

    mem = JSONMemoryBackend(CookingMemory, actor_id="chef", path=path)
    mem.close()

    on_disk = json.loads(path.read_text())
    assert on_disk["other"] == legacy_other  # untouched, still legacy format
    assert on_disk["chef"]["_format"] == 2


# ── Search meta ──────────────────────────────────────────────────────────────


def test_search_meta_carries_ranked_entry_ids(tmp_path: Path) -> None:
    mem = _backend(tmp_path)
    values, meta = mem._search("tips", "meat resting", k=2)  # noqa: SLF001
    assert values[0] == "rest the meat"
    assert list(meta["results"].items())[0] == ("2", "rest the meat")
    assert len(meta["results"]) == 2


def test_search_empty_list_has_empty_results(tmp_path: Path) -> None:
    mem = _backend(tmp_path)
    mem.save("tips", [])
    values, meta = mem._search("tips", "anything")  # noqa: SLF001
    assert values == []
    assert meta == {"results": {}}


async def test_public_search_merges_fetch_meta_into_view(tmp_path: Path) -> None:
    """The view (and hence the recall event) carries query, top_k, and results."""
    mem = _backend(tmp_path)
    view = await mem.search("tips", "meat resting", k=2)
    assert view.meta["query"] == "meat resting"
    assert view.meta["top_k"] == 2
    assert "2" in view.meta["results"]


# ── Agentic consolidation ────────────────────────────────────────────────────


def test_agentic_consolidation_edits_entries_by_id(tmp_path: Path) -> None:
    """The consolidation agent updates/deletes/adds entries; the rest are untouched."""
    from ai_functions.testing import ScriptedModel, Turn

    mem = _backend(tmp_path)
    model = ScriptedModel(
        [
            Turn(
                tool_calls=(("update_memory", {"entry_id": "1", "value": "salt the water; save a cup for the sauce"}),)
            ),
            Turn(tool_calls=(("delete_memory", {"entry_id": "3"}),)),
            Turn(tool_calls=(("add_memory", {"value": "use San Marzano tomatoes"}),)),
            Turn(text="done"),
        ]
    )
    mem._consolidate_list_fn = mem._consolidate_list_fn.replace(model=model)  # noqa: SLF001

    mem.consolidate("tips", [GradFeedback(text="fix the salt tip; drop the spice tip; add a tomato tip")])

    assert mem.list_entries("tips") == {
        "1": "salt the water; save a cup for the sauce",  # updated in place, id kept
        "2": "rest the meat",  # untouched — never paraphrased
        "4": "use San Marzano tomatoes",  # added with a fresh id
    }


def test_consolidation_snapshot_targets_retrieved_entries(tmp_path: Path) -> None:
    """With retrieval context, the agent is shown only the retrieved entries,
    re-read from the store (an entry may have changed since the search);
    stale ids are dropped."""
    mem = _backend(tmp_path)
    capture = _CaptureConsolidateFn()
    mem._consolidate_list_fn = cast("Any", capture)  # noqa: SLF001

    mem.consolidate(
        "tips", [GradFeedback(text="fb")], retrieved={"2": "stale text from search time", "99": "deleted since"}
    )

    assert capture.kwargs is not None
    memories = capture.kwargs["memories"]
    assert "rest the meat" in memories  # current value, not the stale search-time text
    assert "stale text" not in memories
    assert "salt the pasta water" not in memories  # non-retrieved entry not shown
    assert "99" not in memories  # stale id dropped
    assert capture.tools is not None  # CRUD tools were attached


def test_consolidation_without_context_shows_all_entries(tmp_path: Path) -> None:
    mem = _backend(tmp_path)
    capture = _CaptureConsolidateFn()
    mem._consolidate_list_fn = cast("Any", capture)  # noqa: SLF001

    mem.consolidate("tips", [GradFeedback(text="fb")])

    assert capture.kwargs is not None
    for value in CookingMemory().tips:
        assert value in capture.kwargs["memories"]


def test_scalar_consolidation_routes_through_rewrite(tmp_path: Path) -> None:
    """A scalar parameter goes through the value-rewrite consolidator (not the
    agentic list path), and the rewritten value is stored verbatim.

    The real ``_consolidate_value_fn`` is a *structured* ``@ai_function[str]``,
    so its result comes back clean (no trailing newline from the plain-str
    ``str(AgentResult)`` path). We stand in a stub that returns a clean string
    to mirror that — driving it with a ScriptedModel would force
    ``structured_output=False`` and exercise a path production never uses.
    """
    mem = _backend(tmp_path)
    capture = _CaptureConsolidateFn(returns="a warmer style")
    mem._consolidate_value_fn = cast("Any", capture)  # noqa: SLF001

    mem.consolidate("style", [GradFeedback(text="be warmer")])

    assert capture.kwargs == {"value": "plain", "feedback": ["be warmer"], "description": "Writing style."}
    assert mem._recall("style")[0] == "a warmer style"  # noqa: SLF001 -- stored verbatim, no trailing newline


# ── Optimizer plumbing ───────────────────────────────────────────────────────


def test_optimizer_consolidate_merges_retrieved_across_nodes(tmp_path: Path) -> None:
    """meta["results"] from every grouped node reaches backend.consolidate."""
    mem = _backend(tmp_path)
    capture = _CaptureConsolidateFn()
    mem._consolidate_list_fn = cast("Any", capture)  # noqa: SLF001

    def _param(results: dict[str, str]) -> ParameterNode:
        return ParameterNode(
            node_id="tips", name="tips", backend=mem, gradients=[GradFeedback(text="fb")], meta={"results": results}
        )

    child = ThreadNode(node_id="c", thread_id="c", parameters=[_param({"2": "rest the meat"})])
    root = ThreadNode(node_id="r", thread_id="r", parameters=[_param({"3": "toast the spices"})])
    root.child_threads = [child]
    child.parent = root

    TextGradOptimizer().consolidate(root)

    assert capture.kwargs is not None
    memories = capture.kwargs["memories"]
    assert "rest the meat" in memories and "toast the spices" in memories
    assert "salt the pasta water" not in memories  # never retrieved, not shown


async def test_meta_flows_search_to_graph_node(tmp_path: Path) -> None:
    """End-to-end: search meta lands on the reconstructed ParameterNode."""
    from ai_functions import Traceable
    from ai_functions.testing import ScriptedModel, Turn

    mem = _backend(tmp_path)

    @ai_function[str](structured_output=False)
    def _writer(tips: Traceable[list[str]]):
        """Write a recipe using {tips}."""

    view = await mem.search("tips", "meat resting", k=1)
    result = await _writer.replace(model=ScriptedModel([Turn(text="a recipe")])).trace(tips=view)
    graph = await build_graph_from_result(result, [mem])

    assert len(graph.parameters) == 1
    node_meta = graph.parameters[0].meta
    assert node_meta["results"] == {"2": "rest the meat"}
    assert node_meta["query"] == "meat resting"
