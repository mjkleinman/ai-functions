"""Immutable ``@ai_function`` decorator and ``AIFunction`` template."""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Unpack, final, overload, override  # pyright: ignore[reportAny]

from strands.tools import ToolProvider
from strands.tools.decorator import tool as _strands_tool  # pyright: ignore[reportUnknownVariableType]
from strands.types.tools import AgentTool

from ..handle import ThreadHandle
from ..protocols import Spawnable
from ..types import InputShape
from ..utils import run_blocking
from .ai_thread import AIThread
from .config import (
    AgentKwargs,
    ThreadConfig,
    ThreadMergedKwargs,
    split_config_and_agent_kwargs,
)


def _merge_config(
    base: ThreadConfig,
    **kwargs: Unpack[ThreadMergedKwargs],
) -> ThreadConfig:
    """Return ``base`` with ``kwargs`` merged into it.

    ``ThreadKwargs`` fields replace the base field directly; unknown keys are
    treated as ``strands.Agent`` kwargs and merged into ``base.agent_kwargs``.
    """
    thread_kwargs, agent_kwargs = split_config_and_agent_kwargs(**kwargs)
    combined: dict[str, Any] = {**base.agent_kwargs, **agent_kwargs}  # pyright: ignore[reportExplicitAny]
    merged_agent = AgentKwargs(**combined)  # type: ignore[typeddict-item]  # dynamic dict merge
    return dataclasses.replace(
        base,
        **dict(thread_kwargs),  # type: ignore[arg-type]  # TypedDict -> dataclass fields
        agent_kwargs=merged_agent,
    )


def _infer_input_shape(prompt_fn: Callable[..., Any]) -> InputShape:  # pyright: ignore[reportExplicitAny]
    """Derive an :class:`InputShape` from ``prompt_fn``'s signature.

    Rules:

    - Zero positional parameters → ``NO_ARGS``.
    - Exactly one positional parameter resolved to ``str`` → ``STR_PROMPT``.
    - Anything else → ``STRUCTURED``.

    Uses :func:`typing.get_type_hints` to resolve string-form annotations
    introduced by ``from __future__ import annotations``, ``TypeAlias``
    indirection, etc.
    """
    import typing

    try:
        sig = inspect.signature(prompt_fn)
    except (TypeError, ValueError):
        return InputShape.STRUCTURED
    positional = [
        p
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not positional:
        return InputShape.NO_ARGS
    if len(positional) == 1:
        try:
            hints = typing.get_type_hints(prompt_fn)
        except Exception:  # noqa: BLE001 — annotations may reference missing names
            return InputShape.STRUCTURED
        if hints.get(positional[0].name) is str:
            return InputShape.STR_PROMPT
    return InputShape.STRUCTURED


@final
class AIFunction[**P, T](ToolProvider, Spawnable[P, T]):
    """Immutable AI function template; factory for ``AIThread`` instances.

    Args:
        prompt_fn: User function that builds the prompt; ``None`` means
            use the function's docstring.
        output_type: The declared output type for this function.
        config: Default per-thread configuration.

    Implements:
        Spawnable, strands.tools.ToolProvider.
    """

    __slots__ = (
        "_prompt_fn",
        "_output_type",
        "_config",
        "_executor",
        "_name",
        "_doc",
        "_input_shape",
    )

    def __init__(
        self,
        prompt_fn: Callable[P, str | None],
        output_type: type[T],
        config: ThreadConfig,
    ) -> None:
        self._prompt_fn: Callable[P, str | None] = prompt_fn
        self._output_type: type[T] = output_type
        self._config: ThreadConfig = config
        self._executor: object = None
        self._name: str = getattr(prompt_fn, "__name__", "ai_function")
        self._doc: str | None = inspect.getdoc(prompt_fn)
        self._input_shape: InputShape = _infer_input_shape(prompt_fn)

    @property
    def name(self) -> str:
        """Name of the wrapped function."""
        return self._name

    @property
    def config(self) -> ThreadConfig:
        """The default ``ThreadConfig`` attached to this template."""
        return self._config

    @property
    def output_type(self) -> type[T]:
        """The declared output type for this function."""
        return self._output_type

    @property
    def prompt_fn(self) -> Callable[P, str | None]:
        """The user-provided prompt builder."""
        return self._prompt_fn

    @property
    def input_shape(self) -> InputShape:
        """Coarse classification of this template's ``prompt_fn`` signature."""
        return self._input_shape

    # ── Spawnable ──

    def to_thread(
        self,
        **config_overrides: Unpack[ThreadMergedKwargs],
    ) -> AIThread[P, T]:
        """Produce a fresh ``AIThread`` bound to this template.

        Args:
            config_overrides: Kwargs matching ``ThreadConfig`` fields update
                the config directly; others merge into ``agent_kwargs``.

        Returns:
            A new ``AIThread`` with its own buffer and resolved config.

        Ensures:
            - The returned thread is unbound.
            - Successive calls return independent instances with no
              shared state.
        """
        merged = _merge_config(self._config, **config_overrides)
        return AIThread(self, merged)

    # ── One-shot execution (sugar) ──

    async def spawn(
        self,
        **config_overrides: Unpack[ThreadMergedKwargs],
    ) -> ThreadHandle[P, T]:
        """Spawn this template on a private in-process worker and return its handle.

        Convenience for scripts and examples that want a multi-turn
        session without standing up a coordinator and worker pair
        explicitly. The returned handle is backed by a coordinator owned
        by a fresh :class:`LocalWorker`; both stay alive for as long as
        the handle is reachable.

        Args:
            config_overrides: Kwargs matching ``ThreadConfig`` fields update
                the config directly; others merge into ``agent_kwargs``.

        Returns:
            A ``ThreadHandle`` in ``NOT_STARTED`` state bound to a
            private worker.

        Ensures:
            - Each call returns a handle on an independent worker with
              no shared state.
            - The caller owns the handle's lifecycle; teardown requires
              ``await handle.terminate()`` or
              ``await handle.terminate_now()``.

        Strategy:
            1. Merge ``config_overrides`` into this template
               (``self.replace``) if any.
            2. Construct a fresh :class:`InMemoryCoordinator` and
               :class:`LocalWorker`.
            3. ``await worker.spawn_locally(target)`` and return the handle.
        """
        from .. import InMemoryCoordinator
        from ..runtime.worker import LocalWorker

        target = self.replace(**config_overrides) if config_overrides else self
        coord = InMemoryCoordinator()
        worker = LocalWorker(coord)
        return await worker.spawn_locally(target)

    async def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Run one throwaway cycle standalone and return the result.

        Args:
            args: Positional arguments forwarded to ``prompt_fn``.
            kwargs: Keyword arguments forwarded to ``prompt_fn``.

        Returns:
            The typed cycle result.

        Strategy:
            1. ``handle = self.spawn()``.
            2. ``await handle.run(*args, **kwargs)``.
            3. ``await handle.terminate_now()``.
        """
        handle = await self.spawn()
        try:
            return await handle.run(*args, **kwargs)
        finally:
            await handle.terminate_now()

    def run_sync(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Blocking wrapper around ``__call__``.

        Args:
            args: Positional arguments forwarded to ``prompt_fn``.
            kwargs: Keyword arguments forwarded to ``prompt_fn``.

        Returns:
            The typed cycle result.

        Concurrency:
            Blocks the calling thread until the cycle completes.
        """
        return run_blocking(lambda: self(*args, **kwargs))

    # ── Template variants ──

    def replace(self, **kwargs: Unpack[ThreadMergedKwargs]) -> AIFunction[P, T]:
        """Return a new template with ``kwargs`` merged into its config.

        Args:
            kwargs: Kwargs matching ``ThreadConfig`` fields update the
                config directly; others merge into ``agent_kwargs``.

        Returns:
            A new ``AIFunction`` that differs from ``self`` only in the
            merged fields.
        """
        merged = _merge_config(self._config, **kwargs)
        return AIFunction(self._prompt_fn, self._output_type, merged)

    # ── ToolProvider ──

    @override
    async def load_tools(self, **kwargs: object) -> Sequence[AgentTool]:
        """Expose this template as a Strands tool for other agents.

        Args:
            kwargs: Ignored; present for protocol compatibility.

        Returns:
            One or more ``AgentTool`` instances whose invocation calls
            ``__call__``.
        """
        del kwargs
        template = self
        tool_name = self._config.name or self._name
        tool_description = self._config.description or self._doc or self._name

        async def _invoke(**call_kwargs: object) -> T:
            return await template(**call_kwargs)  # type: ignore[arg-type]  # pyright: ignore[reportCallIssue]

        _invoke.__name__ = tool_name
        _invoke.__doc__ = tool_description
        _invoke.__signature__ = inspect.signature(self._prompt_fn)  # type: ignore[attr-defined]
        annotations = dict(getattr(self._prompt_fn, "__annotations__", {}))
        annotations["return"] = self._output_type
        _invoke.__annotations__ = annotations

        agent_tool = _strands_tool(
            name=tool_name,
            description=tool_description,
        )(_invoke)
        return [agent_tool]

    @override
    def add_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Register a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent consuming this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        del consumer_id, kwargs
        return None

    @override
    def remove_consumer(self, consumer_id: Hashable, **kwargs: object) -> None:
        """Deregister a tool-provider consumer.

        Args:
            consumer_id: Identifier of the agent releasing this tool.
            kwargs: Ignored; present for protocol compatibility.
        """
        del consumer_id, kwargs
        return None


# ── Decorator ──


@overload
def ai_function[**P, T](
    output: type[T],
) -> Callable[[Callable[P, str | None]], AIFunction[P, T]]: ...


@overload
def ai_function[**P, T](
    output: type[T],
    *,
    config: ThreadConfig | None = None,
    **kwargs: Unpack[ThreadMergedKwargs],
) -> Callable[[Callable[P, str | None]], AIFunction[P, T]]: ...


def ai_function[**P, T](  # type: ignore[misc]  # overload implementation not visible to checker
    output: type[T],
    *,
    config: ThreadConfig | None = None,
    **kwargs: Unpack[ThreadMergedKwargs],
) -> Callable[[Callable[P, str | None]], AIFunction[P, T]]:
    """Wrap a prompt function as an ``AIFunction`` with the given output type.

    Args:
        output: The declared output type for the wrapped function.
        config: Optional base ``ThreadConfig``; a fresh one is used if
            ``None``.
        kwargs: Overrides merged into ``config``; ``ThreadKwargs`` keys
            update config fields directly, others merge into
            ``agent_kwargs``.

    Returns:
        A decorator that turns the prompt function into an ``AIFunction``.
    """
    base = config if config is not None else ThreadConfig()
    merged = _merge_config(base, **kwargs)

    def _decorator(prompt_fn: Callable[P, str | None]) -> AIFunction[P, T]:
        return AIFunction(prompt_fn, output, merged)

    return _decorator
