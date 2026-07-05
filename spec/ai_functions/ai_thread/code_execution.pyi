"""Per-cycle code-execution plan.

Owns the code-execution logic for a cycle: mode validation, procedural-parameter
detection, sandbox preamble rendering, fresh-executor-per-attempt creation, and
result extraction (``final_answer`` precedence). A ``CodeExecutionPlan`` is built
once per cycle from the resolved cycle config and bound arguments; a
``DisabledPlan`` null-object ensures call sites remain unconditional.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from strands.agent.agent_result import AgentResult

from .config import ThreadConfig


class CodeExecutionPlan:
    """Immutable per-cycle plan describing how code execution participates.

    Constructed in :meth:`build`; call sites use the three methods without
    caring whether execution is enabled (``DisabledPlan`` returns inert
    defaults).
    """

    def __init__(
        self,
        cycle_config: ThreadConfig,
        output_model: type[BaseModel],
        procedural_names: set[str],
        bound_args: dict[str, object],
        function_name: str,
    ) -> None: ...

    def preamble(self) -> str:
        """Render the environment block advertising the sandbox namespace.

        Returns the ``<environment>...</environment>`` text the model sees. It
        lists importable modules, already-defined ``Procedural`` helper
        signatures (with docstrings), and other bound variables. Underscore-
        prefixed args are excluded.
        """
        ...

    def final_answer_channel(self) -> str:
        """Describe the executor's ``final_answer`` output channel for the prompt.

        Advertising ``final_answer`` in the prompt itself — not only in the tool
        description — helps the model call it before its first ``python_executor``
        invocation.
        """
        ...

    def fresh_tool(self) -> object:
        """Create a fresh ``python_executor`` tool for one attempt.

        Each attempt gets an independent sandbox: ``Procedural`` parameter code
        is re-defined from ``initial_code``, but no ad-hoc state from a prior
        attempt leaks in. ``AIFunction`` tools are injected as blocking
        callables alongside the bound arguments.
        """
        ...

    def claim_result(self, response: AgentResult, state: dict[str, object]) -> BaseModel | None:
        """Return the executor's ``final_answer`` result, or ``None`` to defer.

        When code execution is enabled and the executor produced a result, it
        takes precedence over structured output — the agent explicitly committed
        to this answer via ``final_answer(...)``.
        """
        ...

    def config_with_tool(self, cycle_config: ThreadConfig) -> ThreadConfig:
        """Return ``cycle_config`` with a fresh executor tool appended."""
        ...

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
        ...


class DisabledPlan:
    """Null-object: code execution is off. All methods return inert defaults."""

    def preamble(self) -> str:
        """Return empty string — no preamble when code execution is disabled."""
        ...

    def final_answer_channel(self) -> str:
        """Return empty string — no executor output channel when disabled."""
        ...

    def fresh_tool(self) -> object:
        """Return None — no executor tool when code execution is disabled."""
        ...

    def claim_result(self, response: AgentResult, state: dict[str, object]) -> BaseModel | None:
        """Return None — disabled plan never claims a result."""
        ...

    def config_with_tool(self, cycle_config: ThreadConfig) -> ThreadConfig:
        """Return config unchanged — no tool to append."""
        ...


def detect_procedural_params(prompt_fn: Callable[..., Any]) -> set[str]:
    """Return the names of ``prompt_fn`` params annotated as ``Procedural``.

    Detected via ``ProceduralMarker`` in the parameter's ``Annotated``
    metadata — either directly (``x: Procedural``) or on a union member
    (``x: Traceable[Procedural]``, i.e. ``Procedural | ParameterView[...]
    | Result[...]``). These hold Python source that the python_executor
    should *define* (run at setup) rather than inject as a plain string
    variable.
    """
    ...


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
    ...
