"""Offline tests for the trace dataflow API.

Covers: ``ParameterView`` / ``Result`` handles, ``collect_nodes`` /
``unwrap_nodes``, ``AIFunction.trace`` (emission of un-emitted views,
skip-if-emitted, dedup), unwrapping at the ``ThreadHandle.run`` boundary,
``build_graph_from_result`` (sibling edges, diamond sharing), gradient
accumulation through a diamond, and ``TextGradOptimizer.step`` wiring.

None of these require a live model: cycles run on ``ScriptedModel`` and the
optimizer's backward function is stubbed where needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple, cast

import yaml
from pydantic import BaseModel, Field

from ai_functions import (
    JSONMemoryBackend,
    Procedural,
    TextGradOptimizer,
    Traceable,
    ai_function,
    build_graph_from_result,
)
from ai_functions.optimizer.textgrad import Feedback, Feedbacks
from ai_functions.runtime import InMemoryCoordinator
from ai_functions.types import (
    EventKind,
    ParameterRecalledEvent,
    ThreadId,
    current_thread_scope,
    no_thread_scope,
    thread_scope,
)
from ai_functions.types.graph import (
    ParameterNode,
    ParameterView,
    Result,
    ThreadNode,
    collect_nodes,
    unwrap_nodes,
)

# ── Schema / helpers ─────────────────────────────────────────────────────────


class WritingMemory(BaseModel):
    joke_guidelines: str = Field("No specific guidelines yet.", description="Guidelines to write a good joke")
    formatting_guidelines: str = Field("No specific guidelines yet.", description="Email layout guidelines.")
    tags: list[str] = Field(default_factory=list, description="Free-form tags.")


def _backend(tmp_path: Path) -> JSONMemoryBackend:
    return JSONMemoryBackend(WritingMemory, actor_id="w1", path=tmp_path / "mem.json")


def _scripted(text: str = "done") -> Any:
    from ai_functions.testing import ScriptedModel, Turn

    return ScriptedModel([Turn(text=text)])


async def _recalled_events(result: Result[Any]) -> list[ParameterRecalledEvent]:
    events = await result.coordinator.get_events(result.thread_id, kinds=[EventKind.PARAMETER_RECALLED])
    return [e for e in events if isinstance(e, ParameterRecalledEvent)]


# ── collect_nodes / unwrap_nodes ─────────────────────────────────────────────


def _view(mem: JSONMemoryBackend, name: str = "joke_guidelines", value: str = "v") -> ParameterView[str]:
    return ParameterView(value=value, name=name, backend=mem)


def test_collect_nodes_recurses_containers_and_dedupes(tmp_path: Path) -> None:
    """Handles are found in nested containers; the same handle counts once."""
    mem = _backend(tmp_path)
    v = _view(mem)
    r = Result(value="x", coordinator=InMemoryCoordinator(), thread_id=ThreadId("t"))

    found = collect_nodes(({"a": v, "b": [r, (v,)]}, [v]))

    assert found == [v, r]  # discovery order, deduplicated by identity


def test_collect_nodes_ignores_plain_values() -> None:
    """Plain values, including strings, produce no handles."""
    assert collect_nodes(("text", 3, {"k": [1, 2]})) == []


class _Pair(NamedTuple):
    left: Any
    right: Any


def test_unwrap_nodes_rebuilds_containers(tmp_path: Path) -> None:
    """Views/Results are replaced by .value in dicts, lists, tuples, NamedTuples."""
    mem = _backend(tmp_path)
    v = _view(mem, value="guidelines")
    r = Result(value=42, coordinator=InMemoryCoordinator(), thread_id=ThreadId("t"))

    out = unwrap_nodes({"a": v, "b": [r, "keep"], "c": _Pair(left=v, right=(r,))})

    assert out == {"a": "guidelines", "b": [42, "keep"], "c": _Pair(left="guidelines", right=(42,))}
    assert isinstance(out["c"], _Pair)


def test_procedural_marker_detected_through_traceable() -> None:
    """A ``Traceable[Procedural]`` param is still recognized as procedural code.

    Prompt functions should annotate parameters with the plain type (``Procedural``),
    but ``Traceable[Procedural]`` remains a valid annotation: the marker then lives in
    the ``Annotated`` metadata of a union member, so detection must recurse into the
    union rather than only inspecting the top-level hint.
    """

    @ai_function[str](code_execution_mode="local")
    def _task(helpers: Traceable[Procedural]):
        """Run a task using {helpers}."""

    @ai_function[str](code_execution_mode="local")
    def _plain(note: Traceable[str]):
        """Run a task using {note}."""

    from ai_functions.ai_thread.code_execution import detect_procedural_params

    assert detect_procedural_params(_task.prompt_fn) == {"helpers"}
    assert detect_procedural_params(_plain.prompt_fn) == set()


def test_parameter_view_str_and_identity(tmp_path: Path) -> None:
    """str(view) is str(value); equality is identity (usable in id-keyed sets)."""
    mem = _backend(tmp_path)
    a, b = _view(mem, value="same"), _view(mem, value="same")
    assert str(a) == "same"
    assert a != b  # identity semantics, not value equality
    assert len({a, b}) == 2


# ── no_thread_scope / fetch isolation ────────────────────────────────────────


def test_no_thread_scope_clears_and_restores() -> None:
    """no_thread_scope hides the ambient scope and restores it on exit."""
    coord = InMemoryCoordinator()
    with thread_scope(coord, ThreadId("outer")):
        with no_thread_scope():
            assert current_thread_scope() is None
        assert current_thread_scope() is not None


class _ScopeProbeBackend(JSONMemoryBackend):
    """Records the ambient scope visible to the storage fetch."""

    seen_scope: object = "unset"

    def _recall(self, name: str) -> Any:
        self.seen_scope = current_thread_scope()
        return super()._recall(name)


async def test_fetch_runs_scope_free_but_still_emits(tmp_path: Path) -> None:
    """The storage fetch must not see the ambient scope (its internal model
    calls would pollute the caller's log), while emission still uses it."""
    mem = _ScopeProbeBackend(WritingMemory, actor_id="w1", path=tmp_path / "mem.json")
    coord = InMemoryCoordinator()
    tid = ThreadId("scoped-1")

    with thread_scope(coord, tid):
        view = await mem.recall("joke_guidelines")

    assert mem.seen_scope is None  # fetch saw no scope
    assert view.emitted is True  # emission used the scope
    events = await coord.get_events(tid, kinds=[EventKind.PARAMETER_RECALLED])
    assert len(events) == 1


# ── Memory tools return raw values ───────────────────────────────────────────


async def test_recall_tool_returns_raw_value(tmp_path: Path) -> None:
    """The generated recall tool hands the agent the value, not a ParameterView."""
    mem = _backend(tmp_path)
    tool_fn = mem._make_recall_tool("joke_guidelines")  # noqa: SLF001
    out = await tool_fn()
    assert out == "No specific guidelines yet."
    assert not isinstance(out, ParameterView)


async def test_search_tool_returns_raw_list(tmp_path: Path) -> None:
    """The generated search tool returns a plain list of entries."""
    mem = _backend(tmp_path)
    mem.save("tags", ["alpha beta", "gamma delta"])
    tool_fn = mem._make_search_tool("tags")  # noqa: SLF001
    out = await tool_fn("alpha", k=1)
    assert out == ["alpha beta"]
    assert not isinstance(out, ParameterView)


# ── emit_recall ──────────────────────────────────────────────────────────────


async def test_emit_recall_noop_when_already_emitted(tmp_path: Path) -> None:
    """emit_recall skips views already represented in some log."""
    mem = _backend(tmp_path)
    coord = InMemoryCoordinator()
    view = await mem.recall("joke_guidelines", coordinator=coord, thread_id=ThreadId("first"))
    assert view.emitted is True

    await mem.emit_recall(view, coord, ThreadId("second"))

    assert ThreadId("second") not in coord._events  # noqa: SLF001 -- no second event


async def test_emit_recall_emits_for_pure_fetch_view(tmp_path: Path) -> None:
    """emit_recall materializes the event for a view fetched outside any scope."""
    mem = _backend(tmp_path)
    coord = InMemoryCoordinator()
    view = await mem.recall("joke_guidelines")
    assert view.emitted is False

    await mem.emit_recall(view, coord, ThreadId("traced"))

    assert view.emitted is True
    events = await coord.get_events(ThreadId("traced"), kinds=[EventKind.PARAMETER_RECALLED])
    recalls = [e for e in events if isinstance(e, ParameterRecalledEvent)]
    assert len(recalls) == 1
    assert recalls[0].name == "joke_guidelines"


# ── AIFunction.trace ─────────────────────────────────────────────────────────


async def test_trace_returns_result_with_provenance() -> None:
    """trace runs the cycle and wraps value + coordinator + thread id + inputs."""

    @ai_function[str](structured_output=False)
    def _writer(topic: str):
        """Write about {topic}."""

    result = await _writer.replace(model=_scripted("a joke")).trace(topic="cats")

    assert isinstance(result, Result)
    # The plain-str path returns str(AgentResult), which ends each text block
    # with a newline; rstrip to compare the content.
    assert result.value.rstrip() == "a joke"
    assert str(result).rstrip() == "a joke"
    assert result.inputs == []
    # The event log survives teardown for build_graph_from_result.
    events = await result.coordinator.get_events(result.thread_id)
    assert events


async def test_trace_emits_unemitted_views_on_traced_thread(tmp_path: Path) -> None:
    """A pure-fetch recall passed to trace lands in the traced thread's log."""
    mem = _backend(tmp_path)

    @ai_function[str](structured_output=False)
    def _writer(topic: str, joke_guidelines: str):
        """Write about {topic} using {joke_guidelines}."""

    view = await mem.recall("joke_guidelines")
    assert view.emitted is False

    result = await _writer.replace(model=_scripted()).trace(topic="cats", joke_guidelines=view)

    assert view.emitted is True
    assert result.inputs == [view]
    recalls = await _recalled_events(result)
    assert len(recalls) == 1
    assert recalls[0].name == "joke_guidelines"
    assert recalls[0].thread_id == result.thread_id


async def test_trace_does_not_reemit_views_emitted_elsewhere(tmp_path: Path) -> None:
    """One logical recall, one event: an in-scope recall is not re-emitted by trace."""
    mem = _backend(tmp_path)
    coord = InMemoryCoordinator()
    other_tid = ThreadId("other-thread")

    @ai_function[str](structured_output=False)
    def _writer(joke_guidelines: str):
        """Use {joke_guidelines}."""

    with thread_scope(coord, other_tid):
        view = await mem.recall("joke_guidelines")
    assert view.emitted is True

    result = await _writer.replace(model=_scripted()).trace(joke_guidelines=view)

    assert result.inputs == [view]
    assert await _recalled_events(result) == []  # traced thread's log stays clean
    original = await coord.get_events(other_tid, kinds=[EventKind.PARAMETER_RECALLED])
    assert len(original) == 1  # the one event stays where it was emitted


async def test_trace_dedupes_duplicate_views(tmp_path: Path) -> None:
    """The same view passed twice is one input and one emitted event."""
    mem = _backend(tmp_path)

    @ai_function[str](structured_output=False)
    def _writer(a: str, b: str):
        """Combine {a} and {b}."""

    view = await mem.recall("joke_guidelines")
    result = await _writer.replace(model=_scripted()).trace(a=view, b=view)

    assert result.inputs == [view]
    assert len(await _recalled_events(result)) == 1


async def test_prompt_fn_receives_unwrapped_values(tmp_path: Path) -> None:
    """Handles are unwrapped at the run boundary; prompt_fn sees plain values."""
    mem = _backend(tmp_path)
    seen: dict[str, object] = {}

    @ai_function[str](structured_output=False)
    def _writer(joke_guidelines: str) -> str:
        seen["arg"] = joke_guidelines
        return f"Guidelines: {joke_guidelines}"

    view = await mem.recall("joke_guidelines")
    await _writer.replace(model=_scripted()).trace(joke_guidelines=view)

    assert seen["arg"] == "No specific guidelines yet."
    assert not isinstance(seen["arg"], ParameterView)


async def test_call_also_unwraps_views(tmp_path: Path) -> None:
    """__call__ goes through the same run boundary, so views unwrap there too."""
    mem = _backend(tmp_path)
    seen: dict[str, object] = {}

    @ai_function[str](structured_output=False)
    def _writer(joke_guidelines: str) -> str:
        seen["arg"] = joke_guidelines
        return f"Guidelines: {joke_guidelines}"

    view = await mem.recall("joke_guidelines")
    # Passing a handle to __call__ is off-type by design (only trace() widens to
    # Any), but the runtime still unwraps it at the run boundary — the point here.
    await _writer.replace(model=_scripted())(joke_guidelines=view)  # pyright: ignore[reportArgumentType]

    assert seen["arg"] == "No specific guidelines yet."


# ── build_graph_from_result ──────────────────────────────────────────────────


async def _traced_joke(mem: JSONMemoryBackend, topic: str) -> Result[str]:
    @ai_function[str](structured_output=False)
    def _joke_writer(topic: str, joke_guidelines: str):
        """Write a joke about {topic} using {joke_guidelines}."""

    return await _joke_writer.replace(model=_scripted(f"a {topic} joke")).trace(
        topic=topic,
        joke_guidelines=await mem.recall("joke_guidelines"),
    )


async def test_build_graph_from_result_wires_sibling_edges(tmp_path: Path) -> None:
    """Results passed as arguments become child_threads; params come from events."""
    mem = _backend(tmp_path)

    cat = await _traced_joke(mem, "cats")
    prog = await _traced_joke(mem, "programmers")

    @ai_function[str](structured_output=False)
    def _email_writer(joke_1: str, joke_2: str, formatting_guidelines: str):
        """Email with {joke_1} and {joke_2}, formatted per {formatting_guidelines}."""

    email = await _email_writer.replace(model=_scripted("an email")).trace(
        joke_1=cat,
        joke_2=prog,
        formatting_guidelines=await mem.recall("formatting_guidelines"),
    )

    graph = await build_graph_from_result(email, [mem])

    assert {p.name for p in graph.parameters} == {"formatting_guidelines"}
    assert len(graph.child_threads) == 2
    assert {c.thread_id for c in graph.child_threads} == {str(cat.thread_id), str(prog.thread_id)}
    for child in graph.child_threads:
        assert child.parent is graph
        assert {p.name for p in child.parameters} == {"joke_guidelines"}


async def test_build_graph_from_result_diamond_shares_one_node(tmp_path: Path) -> None:
    """A Result consumed by two traces resolves to one shared node object."""
    mem = _backend(tmp_path)

    shared = await _traced_joke(mem, "cats")

    @ai_function[str](structured_output=False)
    def _consumer(joke: str):
        """Use {joke}."""

    left = await _consumer.replace(model=_scripted("left")).trace(joke=shared)
    right = await _consumer.replace(model=_scripted("right")).trace(joke=shared)

    @ai_function[str](structured_output=False)
    def _root(a: str, b: str):
        """Combine {a} and {b}."""

    root_result = await _root.replace(model=_scripted("combined")).trace(a=left, b=right)

    graph = await build_graph_from_result(root_result, [mem])

    assert len(graph.child_threads) == 2
    left_node, right_node = graph.child_threads
    assert len(left_node.child_threads) == 1
    assert len(right_node.child_threads) == 1
    # The diamond fix: both consumers hold the *same* node object, so gradients
    # from both accumulate on it and consolidate reads them once.
    assert left_node.child_threads[0] is right_node.child_threads[0]


# ── backward through a diamond (pure graph logic, stubbed model) ─────────────


class _EchoBackwardFn:
    """Stands in for the backward AI function: routes each incoming feedback item
    verbatim to every listed input id, and records how many times it ran.

    ``replace`` returns ``self`` so the optimizer's per-node
    ``.replace(post_conditions=[...])`` is a no-op offline."""

    def __init__(self) -> None:
        self.calls = 0

    def replace(self, **kwargs: object) -> _EchoBackwardFn:
        del kwargs
        return self

    def run_sync(self, *, inputs: str, feedback: list[str], **kwargs: object) -> Feedbacks:
        del kwargs
        self.calls += 1
        target_ids = list(yaml.safe_load(inputs).keys())
        return Feedbacks(
            feedbacks=[Feedback(node_id=tid, feedback=f, score=0.5) for tid in target_ids for f in feedback]
        )


def _node(node_id: str, *, params: bool = False) -> ThreadNode:
    parameters = [ParameterNode(node_id=f"{node_id}-p", name=f"{node_id}-p")] if params else []
    return ThreadNode(node_id=node_id, thread_id=node_id, parameters=parameters)


def test_backward_diamond_accumulates_from_both_parents() -> None:
    """A shared child is visited once, yet accumulates feedback from both parents.

    With refined routing every grad-reaching node distributes through the
    backward fn (root → b1/b2, b1/b2 → shared, shared → its param), so the echo
    stub runs once per such node. ``shared`` still appears once in the
    topological walk, so it collects exactly one contribution per parent."""
    shared = _node("shared", params=True)
    b1, b2 = _node("b1"), _node("b2")
    root = _node("root")
    root.child_threads = [b1, b2]
    b1.child_threads = [shared]
    b2.child_threads = [shared]
    b1.parent = b2.parent = root
    shared.parent = b1  # first-consumer-wins; informational

    opt = TextGradOptimizer()
    stub = _EchoBackwardFn()
    opt._backward_fn = cast("Any", stub)  # noqa: SLF001 -- offline stand-in for the model call

    opt.backward(root, "fix it")

    assert [g.text for g in shared.gradients] == ["fix it", "fix it"]  # one contribution per parent
    # root, b1, b2, shared each distribute exactly once (shared visited once).
    assert stub.calls == 4
    assert [g.text for g in shared.parameters[0].gradients] == ["fix it", "fix it"]  # refined onto the leaf param
    assert opt.last_dropped_feedback == []


def test_backward_records_dropped_feedback() -> None:
    """Feedback for an unknown target id is dropped and surfaced in ``last_dropped_feedback``."""
    node = _node("root", params=True)

    class _MismatchedFn:
        def replace(self, **kwargs: object) -> _MismatchedFn:
            del kwargs
            return self

        def run_sync(self, **kwargs: object) -> Feedbacks:
            del kwargs
            return Feedbacks(feedbacks=[Feedback(node_id="no-such-param", feedback="x", score=0.5)])

    opt = TextGradOptimizer()
    opt._backward_fn = cast("Any", _MismatchedFn())  # noqa: SLF001

    opt.backward(node, "fix it")

    assert opt.last_dropped_feedback == ["no-such-param"]
    assert node.parameters[0].gradients == []


# ── TextGradOptimizer.step ───────────────────────────────────────────────────


async def test_step_builds_once_and_returns_same_graph(tmp_path: Path) -> None:
    """step feeds one graph object through backward and consolidate, and returns it."""
    mem = _backend(tmp_path)
    result = await _traced_joke(mem, "cats")

    opt = TextGradOptimizer()
    seen: dict[str, ThreadNode] = {}
    opt.backward = lambda graph, fb: seen.setdefault("backward", graph)  # type: ignore[method-assign]
    opt.consolidate = lambda graph: seen.setdefault("consolidate", graph)  # type: ignore[method-assign]

    graph = await opt.step(result, "feedback", backends=[mem])

    assert seen["backward"] is graph
    assert seen["consolidate"] is graph
    assert {p.name for p in graph.parameters} == {"joke_guidelines"}
