"""Per-cycle code-execution plan.

Owns the code-execution logic for a cycle: mode validation, procedural-parameter
detection, sandbox preamble rendering, fresh-executor-per-attempt creation, and
result extraction (``final_answer`` precedence). A ``CodeExecutionPlan`` is built
once per cycle from the resolved cycle config and bound arguments; a
``DisabledPlan`` null-object ensures call sites remain unconditional.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import typing
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from ..types import unwrap_nodes
from .config import CodeExecutionMode, ThreadConfig
from .errors import AIFunctionError

if TYPE_CHECKING:
    from collections.abc import Callable

    from strands.agent.agent_result import AgentResult

    from .ai_function import AIFunction


def _sandbox_tool_callables(
    cycle_config: ThreadConfig,
) -> dict[str, Callable[..., object]]:
    """Wrap ``AIFunction`` tools as blocking callables for the sandbox namespace.

    Must be called from inside the running cycle (an active event loop): each
    wrapper captures that loop and, when invoked from the sandbox, schedules
    ``fn(*args, **kwargs)`` on it via ``asyncio.run_coroutine_threadsafe`` and
    blocks until the child cycle completes.

    Sandbox code runs on a worker thread (the sync ``python_executor`` tool is
    dispatched via ``asyncio.to_thread``) while the cycle's loop awaits the tool
    future, so blocking the worker cannot deadlock the loop; this relies on sync
    tools running off the event loop.

    Returns:
        Empty dict when the loop is not running (sync test construction) —
        sandbox tool calls are only reachable through a live cycle anyway.
    """
    from .ai_function import AIFunction

    ai_fns = [t for t in cycle_config.tools if isinstance(t, AIFunction)]
    if not ai_fns:
        return {}
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return {}

    def _make_wrapper(fn: AIFunction[..., object]) -> Callable[..., object]:
        def wrapper(*args: object, **kwargs: object) -> object:
            future = asyncio.run_coroutine_threadsafe(fn(*args, **kwargs), loop)
            return future.result()

        wrapper.__name__ = fn.name
        wrapper.__doc__ = fn.config.description or fn.prompt_fn.__doc__
        return wrapper

    return {fn.name: _make_wrapper(fn) for fn in ai_fns}


class CodeExecutionPlan:
    """Immutable per-cycle plan describing how code execution participates.

    Constructed in :meth:`build`; call sites use the three methods without
    caring whether execution is enabled (``DisabledPlan`` returns inert
    defaults).
    """

    __slots__ = (
        "_cycle_config",
        "_output_model",
        "_procedural_names",
        "_bound_args",
        "_function_name",
        "_sandbox_tools",
    )

    def __init__(
        self,
        cycle_config: ThreadConfig,
        output_model: type[BaseModel],
        procedural_names: set[str],
        bound_args: dict[str, object],
        function_name: str,
    ) -> None:
        self._cycle_config = cycle_config
        self._output_model = output_model
        self._procedural_names = procedural_names
        self._bound_args = bound_args
        self._function_name = function_name
        # Captured once per plan (= per cycle): wrappers bind the cycle's
        # event loop, so they must be built inside the running cycle.
        self._sandbox_tools = _sandbox_tool_callables(cycle_config)

    # ── Public interface ─────────────────────────────────────────────────────

    def preamble(self) -> str:
        """Render the environment block advertising the sandbox namespace.

        Returns the ``<environment>...</environment>`` text the model sees. It
        lists importable modules, already-defined ``Procedural`` helper
        signatures (with docstrings), and other bound variables. Underscore-
        prefixed args are excluded.
        """
        from ..tools.local_python_executor import SAFE_BUILTINS, procedural_signatures

        def _truncate(text: str, limit: int = 200) -> str:
            return text if len(text) <= limit else text[:limit] + "..."

        modules = SAFE_BUILTINS + list(self._cycle_config.code_executor_additional_imports)
        parts = [
            "You have access to a python execution environment.",
            "Use it if needed, but prefer using tool calls directly if the task can be accomplished "
            "without writing code.",
            f"The following modules are available for import: {', '.join(modules)}.",
            "Modules not listed above are not available for security reasons. "
            "You cannot use the `os` module. You cannot use the `open(...)` builtin.",
        ]

        helper_blocks: list[str] = []
        variables: list[str] = []
        for name, raw in self._bound_args.items():
            if name.startswith("_"):
                continue
            value = unwrap_nodes(raw)
            if name in self._procedural_names and isinstance(value, str):
                helper_blocks.extend(procedural_signatures(value))
            else:
                variables.append(f" - {name}: {_truncate(repr(value))}")

        if helper_blocks:
            parts.append(
                "\nThe following functions have already been executed and are available "
                "in the python environment's namespace. You can call them directly:"
            )
            parts.extend(f"```python\n{block}\n```" for block in helper_blocks)
        if variables:
            parts.append(
                "\nThe following variables are already available in the python environment (DO NOT redefine them):"
            )
            parts.extend(variables)
        if self._sandbox_tools:
            parts.append(
                "\nThe following tools are also callable as functions in the python environment: "
                + ", ".join(f"`{n}`" for n in self._sandbox_tools)
            )

        return "<environment>\n" + "\n".join(parts) + "\n</environment>"

    def final_answer_channel(self) -> str:
        """Describe the executor's ``final_answer`` output channel for the prompt.

        Advertising ``final_answer`` in the prompt itself — not only in the tool
        description — helps the model call it before its first ``python_executor``
        invocation.
        """
        from ..tools.local_python_executor import generate_signature_from_model

        signature = generate_signature_from_model(self._output_model)
        return f"call {signature} from inside the python_executor tool"

    def fresh_tool(self) -> object:
        """Create a fresh ``python_executor`` tool for one attempt.

        Each attempt gets an independent sandbox: ``Procedural`` parameter code
        is re-defined from ``initial_code``, but no ad-hoc state from a prior
        attempt leaks in. ``AIFunction`` tools are injected as blocking
        callables (see ``_sandbox_tool_callables``) alongside the bound
        arguments.
        """
        from ..tools.local_python_executor import LocalPythonExecutorTool

        initial_code = [
            str(v) for k, v in self._bound_args.items() if k in self._procedural_names and isinstance(v, str)
        ]
        initial_state = {k: v for k, v in self._bound_args.items() if k not in self._procedural_names}
        # Sandbox tool wrappers are merged last so a name collision resolves
        # to the callable tool.
        initial_state.update(self._sandbox_tools)
        executor = LocalPythonExecutorTool(
            output_type=self._output_model,
            initial_state=initial_state,
            initial_code=initial_code,
            additional_authorized_imports=list(self._cycle_config.code_executor_additional_imports),
            executor_kwargs=dict(self._cycle_config.code_executor_kwargs),
        )
        return executor.python_executor

    def claim_result(self, response: AgentResult, state: dict[str, object]) -> BaseModel | None:
        """Return the executor's ``final_answer`` result, or ``None`` to defer.

        When code execution is enabled and the executor produced a result, it
        takes precedence over structured output — the agent explicitly committed
        to this answer via ``final_answer(...)``.
        """
        executor_result = cast("BaseModel | None", state.get("python_executor_result"))
        if executor_result is not None:
            return executor_result
        return None

    def config_with_tool(self, cycle_config: ThreadConfig) -> ThreadConfig:
        """Return ``cycle_config`` with a fresh executor tool appended."""
        return dataclasses.replace(
            cycle_config,
            tools=(*cycle_config.tools, self.fresh_tool()),
        )

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        cycle_config: ThreadConfig,
        output_model: type[BaseModel] | None,
        procedural_names: set[str],
        bound_args: dict[str, object],
        function_name: str,
    ) -> CodeExecutionPlan | DisabledPlan:
        """Build the appropriate plan for this cycle.

        Returns a ``DisabledPlan`` when ``code_execution_mode != LOCAL``, and a
        fully-initialized ``CodeExecutionPlan`` otherwise. Validates eagerly:
        raises ``AIFunctionError`` if LOCAL mode is requested but the output
        model is missing (plain-str return).
        """
        if cycle_config.code_execution_mode != CodeExecutionMode.LOCAL:
            return _DISABLED

        if output_model is None:
            raise AIFunctionError(
                "code_execution_mode=LOCAL is not supported for a plain str return type "
                "(the python_executor's final_answer needs a typed model).",
                function_name=function_name,
            )
        return cls(
            cycle_config=cycle_config,
            output_model=output_model,
            procedural_names=procedural_names,
            bound_args=bound_args,
            function_name=function_name,
        )


class DisabledPlan:
    """Null-object: code execution is off. All methods return inert defaults."""

    __slots__ = ()

    def preamble(self) -> str:
        """Return empty string — no preamble when code execution is disabled."""
        return ""

    def final_answer_channel(self) -> str:
        """Return empty string — no executor output channel when disabled."""
        return ""

    def fresh_tool(self) -> object:
        """Return None — no executor tool when code execution is disabled."""
        return None

    def claim_result(self, response: AgentResult, state: dict[str, object]) -> BaseModel | None:
        """Return None — disabled plan never claims a result."""
        return None

    def config_with_tool(self, cycle_config: ThreadConfig) -> ThreadConfig:
        """Return config unchanged — no tool to append."""
        return cycle_config


_DISABLED = DisabledPlan()


# ── Procedural-parameter detection ──────────────────────────────────────────


def detect_procedural_params(prompt_fn: Callable[..., Any]) -> set[str]:
    """Return the names of ``prompt_fn`` params annotated as ``Procedural``.

    Detected via ``ProceduralMarker`` in the parameter's ``Annotated``
    metadata — either directly (``x: Procedural``) or on a union member
    (``x: Traceable[Procedural]``, i.e. ``Procedural | ParameterView[...]
    | Result[...]``). These hold Python source that the python_executor
    should *define* (run at setup) rather than inject as a plain string
    variable.
    """
    from ..memory.procedural import ProceduralMarker

    def _is_procedural(hint: object) -> bool:
        metadata = getattr(hint, "__metadata__", ())
        if any(isinstance(m, ProceduralMarker) for m in metadata):
            return True
        return any(_is_procedural(arg) for arg in typing.get_args(hint))

    try:
        hints = typing.get_type_hints(prompt_fn, include_extras=True)
    except Exception:  # noqa: BLE001 — annotations may reference missing names
        return set()
    return {name for name, hint in hints.items() if _is_procedural(hint)}


def bind_call_args(
    prompt_fn: Callable[..., Any],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> dict[str, object]:
    """Bind call args to ``prompt_fn``'s signature, returning a name→value dict.

    Used to seed the optional ``python_executor`` namespace. If strict
    binding fails (e.g. an unexpected positional count), fall back to a
    best-effort mapping that still names positional args by parameter
    position — so positionally-passed inputs are not silently dropped.
    """
    try:
        sig = inspect.signature(prompt_fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except TypeError:
        try:
            sig = inspect.signature(prompt_fn)
            names = [
                p.name
                for p in sig.parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            result: dict[str, object] = dict(kwargs)
            for name, value in zip(names, args, strict=False):
                result.setdefault(name, value)
            return result
        except (TypeError, ValueError):
            return dict(kwargs)
