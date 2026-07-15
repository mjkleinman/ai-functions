"""The code-execution prompt preamble that advertises the sandbox namespace.

When ``code_execution_mode="local"``, the runtime tells the agent which modules
are importable, which recalled ``Procedural`` helpers are already defined —
**by signature and docstring**, so the agent knows *when* to call each — and
which other bound variables are in scope. Without this, a recalled helper is
defined in the sandbox but never surfaced to the model.
"""

from __future__ import annotations

import importlib.util

import pytest
from pydantic import BaseModel, Field

from ai_functions import ai_function
from ai_functions.ai_thread.code_execution import CodeExecutionPlan, detect_procedural_params
from ai_functions.memory import Procedural
from ai_functions.testing import RuntimeHarness, ScriptedModel, Turn
from ai_functions.tools.local_python_executor import procedural_signatures
from ai_functions.types import EventKind, MessageUserEvent

_HAS_SMOLAGENTS = importlib.util.find_spec("smolagents") is not None


# ── procedural_signatures helper ──────────────────────────────────────────────


def test_procedural_signatures_captures_signature_docstring_and_return() -> None:
    """Each block is the def line (with return annotation) plus the docstring."""
    code = 'def greet(name: str) -> str:\n    """Say hello to someone by name."""\n    return f\'hi {name}\'\n'
    (block,) = procedural_signatures(code)
    assert block == 'def greet(name: str) -> str:\n    """Say hello to someone by name."""'


def test_procedural_signatures_handles_async_and_missing_docstring() -> None:
    code = "async def fetch(url, *, timeout=5):\n    return url\n"
    (block,) = procedural_signatures(code)
    # No docstring → a `...` body stands in; async prefix preserved.
    assert block == "async def fetch(url, *, timeout=5):\n    ..."


def test_procedural_signatures_skips_underscored_and_nested_defs() -> None:
    """Only top-level, non-underscored defs are advertised (they are the callable
    names in the namespace); classes and private helpers are omitted."""
    code = (
        "def public(x):\n"
        '    """Public helper."""\n'
        "    def _inner():\n"
        "        return 1\n"
        "    return _inner()\n\n"
        "def _private(y):\n"
        "    return y\n\n"
        "class Helper:\n"
        "    pass\n"
    )
    blocks = procedural_signatures(code)
    assert len(blocks) == 1
    assert blocks[0].startswith("def public(x):")
    assert "_private" not in " ".join(blocks)
    assert "_inner" not in " ".join(blocks)
    assert "class Helper" not in " ".join(blocks)


def test_procedural_signatures_returns_empty_on_syntax_error() -> None:
    assert procedural_signatures("def broken(") == []


# ── CodeExecutionPlan.preamble ────────────────────────────────────────────────


def _build_plan(fn, bound_args: dict[str, object]):
    """Build a CodeExecutionPlan for direct preamble testing."""
    proc_names = detect_procedural_params(fn.prompt_fn)
    output_model = fn.to_thread()._output_spec.structured_output_model  # noqa: SLF001
    return CodeExecutionPlan.build(fn.config, output_model, proc_names, bound_args, fn.name)


def test_preamble_empty_when_code_execution_disabled() -> None:
    @ai_function[str]
    def f(helpers: Procedural):
        """{helpers}"""

    plan = CodeExecutionPlan.build(f.config, None, set(), {"helpers": "def h():\n    return 1\n"}, f.name)
    assert plan.preamble() == ""


def test_preamble_lists_helper_signature_with_docstring() -> None:
    @ai_function[str](code_execution_mode="local", code_executor_additional_imports=["numpy.*"])
    def f(helpers: Procedural, topic: str):
        """Work on {topic}."""

    plan = _build_plan(
        f,
        {
            "helpers": (
                "def secret_greeting(name):\n"
                '    """Return the secret greeting for a person."""\n'
                "    return f'Zphqr, {name}!'\n"
            ),
            "topic": "cats",
            "_hidden": "x",
        },
    )
    preamble = plan.preamble()

    # Helper advertised by signature AND docstring — but not its body.
    assert "def secret_greeting(name):" in preamble
    assert "Return the secret greeting for a person." in preamble
    assert "Zphqr" not in preamble
    # Importable modules listed (both the extra and a built-in one).
    assert "numpy.*" in preamble
    assert "math" in preamble
    # Regular arg listed; private (_-prefixed) arg skipped.
    assert "topic" in preamble
    assert "_hidden" not in preamble


# ── Integration: the preamble (with docstring) reaches the emitted user turn ──


@pytest.mark.skipif(not _HAS_SMOLAGENTS, reason="LOCAL execution requires smolagents")
async def test_preamble_with_docstring_appears_in_message_user_event() -> None:
    """In LOCAL mode the preamble — including the helper docstring — is folded
    into the prompt's single MESSAGE_USER turn."""

    class Answer(BaseModel):
        answer: str = Field(description="the answer")

    @ai_function[Answer](code_execution_mode="local")
    def run_task(helpers: Procedural):
        """Use the helper to answer."""

    async with RuntimeHarness() as h:
        # The scripted model calls the executor with code that returns via
        # final_answer, so the cycle completes cleanly.
        model = ScriptedModel(
            [Turn(tool_calls=(("python_executor", {"code": "final_answer(answer=greet('x'))"}),))],
        )
        handle = await h.spawn(run_task.replace(model=model), thread_name="run_task")
        await handle.run(
            helpers=('def greet(name):\n    """Greet a person warmly by name."""\n    return f\'hello {name}\'\n')
        )

        user_events = [
            e for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        # A single prompt turn carries the task, the environment preamble, the
        # helper signature, AND its docstring.
        assert len(user_events) == 1
        text = user_events[0].text
        assert "Use the helper to answer." in text
        assert "python execution environment" in text
        assert "def greet(name):" in text
        assert "Greet a person warmly by name." in text
        # Prompt layout: environment block BEFORE the task prompt, and the
        # final-result instruction (listing both output channels) after it.
        assert text.index("<environment>") < text.index("Use the helper to answer.")
        assert "IMPORTANT: To provide your final result" in text
        assert "use the Answer tool" in text
        assert "final_answer" in text
        assert text.index("IMPORTANT:") > text.index("Use the helper to answer.")


# ── prompt layout without code execution ──────────────────────────────────────


async def test_prompt_without_code_execution_has_result_instruction_only() -> None:
    """With code execution off, the turn is prompt + structured-output
    instruction: no environment block, no final_answer mention."""

    class Answer(BaseModel):
        answer: str = Field(description="the answer")

    @ai_function[Answer]
    def ask(topic: str):
        """Say something about {topic}."""

    async with RuntimeHarness() as h:
        model = ScriptedModel(
            [Turn(tool_calls=(("Answer", {"answer": "ok"}),))],
        )
        handle = await h.spawn(ask.replace(model=model), thread_name="ask")
        await handle.run(topic="cats")

        user_events = [
            e for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        assert len(user_events) == 1
        text = user_events[0].text
        assert text.startswith("Say something about cats.")
        assert "<environment>" not in text
        assert "IMPORTANT: To provide your final result, use the Answer tool." in text
        assert "python_executor" not in text


async def test_plain_str_prompt_has_no_result_instruction() -> None:
    """Plain-str output (structured_output=False) has no output channel to
    advertise — the turn is exactly the rendered prompt."""

    @ai_function[str](structured_output=False, coordinator_tools_enabled=False)
    def ask(topic: str):
        """Say something about {topic}."""

    async with RuntimeHarness() as h:
        model = ScriptedModel([Turn(text="something about cats")])
        handle = await h.spawn(ask.replace(model=model), thread_name="ask")
        await handle.run(topic="cats")

        user_events = [
            e for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        assert len(user_events) == 1
        assert user_events[0].text == "Say something about cats."


# ── Procedural params require code execution ──────────────────────────────────


def test_procedural_param_requires_local_mode() -> None:
    """A Procedural parameter with code execution DISABLED is rejected at
    thread construction — the code would silently interpolate as inert text."""
    from ai_functions.ai_thread.errors import AIFunctionError

    @ai_function[str]  # code_execution_mode defaults to DISABLED
    def f(helpers: Procedural):
        """{helpers}"""

    with pytest.raises(AIFunctionError, match=r"Procedural parameter\(s\) \[helpers\]"):
        f.to_thread()


# ── no-result retries with guidance under code execution ──────────────────────


@pytest.mark.skipif(not _HAS_SMOLAGENTS, reason="LOCAL execution requires smolagents")
async def test_no_result_retries_with_guidance() -> None:
    """A cycle that ends without final_answer retries with guidance.

    Only reachable with an executor-only output channel (a non-JSON-serializable
    return type), where final_answer is the sole channel: turn 1 ends without
    calling it, turn 2 answers.
    """

    class Thing:
        """Non-JSON-serializable marker result."""

    thing = Thing()

    @ai_function[Thing](code_execution_mode="local")
    def run_task(payload: Thing):
        """Return the payload object."""

    model = ScriptedModel(
        [
            Turn(text="Let me think about this instead of answering."),
            Turn(tool_calls=(("python_executor", {"code": "final_answer(answer=payload)"}),)),
        ],
    )

    async with RuntimeHarness() as h:
        handle = await h.spawn(run_task.replace(model=model), thread_name="run_task")
        result = await handle.run(payload=thing)
        # The bound arg round-tripped through the sandbox into final_answer.
        assert result is thing

        user_events = [
            e for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        # Prompt turn + one no-result guidance turn.
        assert len(user_events) == 2
        guidance = user_events[1].text
        assert "No result was produced" in guidance
        assert "IMPORTANT: To provide your final result" in guidance
        assert "final_answer" in guidance
        # Executor-only output: the structured-output tool must NOT be offered.
        assert "use the FinalAnswer tool" not in guidance


# ── Integration: ai_function tools callable inside the sandbox ────────────────


@pytest.mark.skipif(not _HAS_SMOLAGENTS, reason="LOCAL execution requires smolagents")
async def test_ai_function_tool_callable_in_sandbox() -> None:
    """An ``AIFunction`` in ``tools`` is callable from executed code.

    The parent result must round-trip the child's answer, and the preamble must
    advertise the callable.
    """

    class Answer(BaseModel):
        answer: str = Field(description="the answer")

    # str output is wrapped in a FinalAnswer model; the scripted structured
    # answer targets that wrapper tool.
    child_model = ScriptedModel(
        [Turn(tool_calls=(("FinalAnswer", {"answer": "CHILD-SAYS-HI"}),))],
    )

    @ai_function[str](coordinator_tools_enabled=False)
    def shout(text: str):
        """Shout {text}."""

    child = shout.replace(model=child_model)

    parent_model = ScriptedModel(
        [Turn(tool_calls=(("python_executor", {"code": "r = shout('hi')\nfinal_answer(answer=r)"}),))],
    )

    @ai_function[Answer](code_execution_mode="local", tools=[child])
    def run_task(topic: str):
        """Work on {topic}."""

    async with RuntimeHarness() as h:
        handle = await h.spawn(run_task.replace(model=parent_model), thread_name="run_task")
        result = await handle.run(topic="x")

        # The child's structured answer round-tripped through the sandbox
        # callable into the parent's final_answer.
        assert result.answer == "CHILD-SAYS-HI"

        # The preamble advertises the sandbox callable by name.
        user_events = [
            e for e in await h.events(handle.id, kinds=[EventKind.MESSAGE_USER]) if isinstance(e, MessageUserEvent)
        ]
        assert "The following tools are also callable as functions in the python environment: `shout`" in (
            user_events[0].text
        )
