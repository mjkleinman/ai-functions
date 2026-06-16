"""AIThread — live, per-spawn instance backing an ``AIFunction``."""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast, final

import tstr
from pydantic import BaseModel, Field, TypeAdapter, create_model
from strands import Agent
from strands.agent.agent_result import AgentResult
from strands.agent.conversation_manager import NullConversationManager
from strands.hooks import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
    HookProvider,
    HookRegistry,
)
from strands.types.content import Messages
from strands.types.exceptions import (
    ContextWindowOverflowException,
    MaxTokensReachedException,
)

from ..protocols import Thread
from ..types import (
    ContextSummarizedEvent,
    MessageAssistantCompleteEvent,
    MessageAssistantStartEvent,
    MessageAssistantThinkingEvent,
    MessageAssistantTokenEvent,
    MessageId,
    MessageUserEvent,
    ThreadContext,
    TokenUsage,
    TokenUsageEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from .config import ThreadConfig, ThreadKwargs
from .errors import AIFunctionError
from .postcondition import PostCondition
from .reconstruction import reconstruct_messages
from .summarization import (
    DefaultSummarizationStrategy,
    SummarizationFailedError,
    SummarizationStrategy,
)

if TYPE_CHECKING:
    from .ai_function import AIFunction


# Hard cap on reactive summarization attempts per cycle. Bounds the worst
# case when every strategy configuration still overflows.
_MAX_SUMMARIZATION_ATTEMPTS: int = 3


@dataclass(frozen=True)
class OutputSpec[T]:
    """Describes the AI output type and how to present it to the model."""

    output_type: type[T]
    """The user-specified output type (e.g. ``str``, ``float``, a pydantic model)."""

    is_pydantic: bool
    """True iff ``output_type`` is a ``pydantic.BaseModel`` subclass."""

    is_structured: bool
    """True iff the output can be expressed as a JSON schema for the model."""

    structured_output_model: type[BaseModel] | None
    """Pydantic model passed to Strands as the structured output model."""

    is_wrapped: bool
    """True iff ``structured_output_model`` wraps ``output_type`` in a ``FinalAnswer``."""

    @classmethod
    def from_type(cls, output_type: type[T], is_structured: bool = True) -> OutputSpec[T]:
        """Derive an ``OutputSpec`` from a type and a structured-output flag.

        Args:
            output_type: The user-declared output type.
            is_structured: Whether structured output should be used if
                possible.

        Returns:
            An ``OutputSpec`` describing how to request and parse
            ``output_type``.

        Ensures:
            - ``result.is_pydantic`` is true iff ``output_type`` is a
              ``BaseModel`` subclass.
            - ``result.structured_output_model is not None`` iff
              ``result.is_structured``.
            - ``result.is_wrapped`` is true iff
              ``structured_output_model`` wraps ``output_type``.
        """
        if not is_structured:
            assert output_type is str, "is_structured=True can only be used with str return type"
            return cls(
                output_type=output_type,
                is_pydantic=False,
                is_structured=False,
                structured_output_model=None,
                is_wrapped=False,
            )

        if isinstance(output_type, type) and issubclass(output_type, BaseModel):  # pyright: ignore[reportUnnecessaryIsInstance]
            return cls(
                output_type=output_type,
                is_pydantic=True,
                is_structured=True,
                structured_output_model=output_type,
                is_wrapped=False,
            )

        # Non-pydantic, non-str type: wrap in a FinalAnswer model
        wrapped = create_model("FinalAnswer", answer=(output_type, Field(...)))
        return cls(
            output_type=output_type,
            is_pydantic=False,
            is_structured=True,
            structured_output_model=wrapped,
            is_wrapped=True,
        )


def _new_message_id() -> MessageId:
    return MessageId(f"msg-{uuid.uuid4().hex}")


class _EventBridgeHook(HookProvider):
    """Bridge Strands model/tool hooks and streaming callbacks to AI Functions events.

    Responsibilities:
      1. Work-boundary management (before every model call): drain the
         owning thread's inject buffer into ``MESSAGE_USER`` events,
         check ``cancel_signal``, await ``pause_signal``.
      2. Assistant-turn span emission: ``MESSAGE_ASSISTANT_START`` before a
         model call, ``MESSAGE_ASSISTANT_COMPLETE`` after it — both carry a
         fresh ``message_id`` that ties in any tool events emitted during
         that model turn.
      3. Tool-call telemetry: ``TOOL_CALL`` before each tool invocation,
         ``TOOL_RESULT`` / ``TOOL_ERROR`` after.
      4. Streaming deltas: the ``callback_handler`` passed to the ``Agent``
         fans ``data`` chunks into ``MESSAGE_ASSISTANT_TOKEN`` and
         ``reasoningText`` chunks into ``MESSAGE_ASSISTANT_THINKING``.
    """

    def __init__(self, ctx: ThreadContext, inject_buffer: list[str]) -> None:
        self._ctx = ctx
        self._inject_buffer = inject_buffer
        self._current_message_id: MessageId | None = None

    def register_hooks(self, registry: HookRegistry, **kwargs: object) -> None:
        """Register every Strands hook this bridge cares about."""
        registry.add_callback(BeforeModelCallEvent, self._on_before_model_call)
        registry.add_callback(AfterModelCallEvent, self._on_after_model_call)
        registry.add_callback(BeforeToolCallEvent, self._on_before_tool_call)
        registry.add_callback(AfterToolCallEvent, self._on_after_tool_call)

    # ── Streaming callback ──

    def stream_callback(self, **kwargs: object) -> None:
        """Fan one Strands streaming chunk into assistant token/thinking events.

        Passed to ``Agent(callback_handler=self.stream_callback)``. Strands
        invokes this once per streaming chunk with a kwargs payload whose keys
        vary by provider. We care about ``data`` (text delta) and
        ``reasoningText`` (extended-thinking delta); everything else is ignored.
        """
        ctx = self._ctx
        message_id = self._current_message_id
        if message_id is None:
            return
        complete = bool(kwargs.get("complete", False))
        data = kwargs.get("data")
        if isinstance(data, str) and data:
            ctx.on_event(
                MessageAssistantTokenEvent(
                    message_id=message_id,
                    text=data,
                    complete=complete,
                )
            )
        reasoning = kwargs.get("reasoningText")
        if isinstance(reasoning, str) and reasoning:
            ctx.on_event(
                MessageAssistantThinkingEvent(
                    message_id=message_id,
                    text=reasoning,
                    complete=complete,
                )
            )

    # ── Model hooks ──

    async def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        ctx = self._ctx
        # Drain the owning thread's inject buffer. Each drained entry is
        # simultaneously (1) emitted as a MESSAGE_USER event for the durable
        # log and (2) appended to the live agent's message list so the next
        # model iteration observes it. The two actions must be paired: any
        # divergence between the event log order and agent-observation order
        # would invalidate the agent's cached prefix on the next rehydration
        # (see I7).
        #
        # We append one Message per drained text. Strands does NOT coalesce
        # consecutive same-role messages (see ``_normalize_messages`` in
        # ``strands.event_loop.streaming`` — it only touches assistant-side
        # blank-text handling). ``reconstruct_messages`` must therefore also
        # emit one user Message per ``MESSAGE_USER`` event to preserve I7's
        # cache-prefix stability property.
        from strands.types.content import ContentBlock, Message

        while self._inject_buffer:
            text = self._inject_buffer.pop(0)
            ctx.on_event(MessageUserEvent(text=text))
            event.agent.messages.append(Message(role="user", content=[ContentBlock(text=text)]))
        # Cooperative cancel check
        if ctx.cancel_signal.is_set():
            raise asyncio.CancelledError
        # Await rate-limit / manual pause
        await ctx.coordinator.wait_until_unpaused(ctx.thread_id)
        # Open a fresh assistant-turn span
        self._current_message_id = _new_message_id()
        ctx.on_event(MessageAssistantStartEvent(message_id=self._current_message_id))

    async def _on_after_model_call(self, event: AfterModelCallEvent) -> None:
        ctx = self._ctx
        message_id = self._current_message_id
        if message_id is None:
            return
        stop_response = event.stop_response
        content: list[object] = []
        if stop_response is not None:
            # stop_response.message is a Strands Message (TypedDict).
            message = cast("dict[str, object]", stop_response.message)
            raw_content = message.get("content", [])
            if isinstance(raw_content, list):
                content = list(cast("list[object]", raw_content))
        ctx.on_event(
            MessageAssistantCompleteEvent(
                message_id=message_id,
                content=cast("list[Any]", content),  # pyright: ignore[reportExplicitAny]
            )
        )
        self._current_message_id = None

    # ── Tool hooks ──

    async def _on_before_tool_call(self, event: BeforeToolCallEvent) -> None:
        ctx = self._ctx
        tool_use = cast("dict[str, object]", event.tool_use)
        tool_use_id = str(tool_use.get("toolUseId", ""))
        tool_name = str(tool_use.get("name", ""))
        arguments_raw = tool_use.get("input", {})
        arguments: dict[str, object] = (
            dict(cast("dict[str, object]", arguments_raw))
            if isinstance(arguments_raw, dict)
            else {"input": arguments_raw}
        )
        ctx.on_event(
            ToolCallEvent(
                message_id=self._current_message_id,
                tool_use_id=tool_use_id,
                tool_name=tool_name,
                arguments=arguments,
            )
        )

    async def _on_after_tool_call(self, event: AfterToolCallEvent) -> None:
        ctx = self._ctx
        # ``event.result`` is always a Strands ``ToolResult`` dict
        # (``{"toolUseId", "status", "content"}``), even on exception — the
        # tool executor wraps exceptions into
        # ``{"status": "error", "content": [{"text": "Error: ..."}]}`` before
        # the hook fires. We split the dict across top-level event fields
        # (typed attribute access for consumers) and repack it at
        # reconstruction time so the on-wire ``toolResult`` block remains
        # byte-identical to what Strands appends to ``agent.messages``.
        tool_result = cast("dict[str, object]", event.result)
        ctx.on_event(
            ToolResultEvent(
                message_id=self._current_message_id,
                tool_use_id=str(tool_result.get("toolUseId", "")),
                status=cast("Any", tool_result.get("status", "success")),  # pyright: ignore[reportExplicitAny]
                content=cast("Any", tool_result.get("content", [])),  # pyright: ignore[reportExplicitAny]
            )
        )


@final
class AIThread[**P, T](Thread):  # type: ignore[type-arg]
    """Live stateful thread instance bound to an ``AIFunction`` template.

    Args:
        template: The ``AIFunction`` whose ``prompt_fn`` and
            ``output_type`` this thread invokes.
        config: The resolved ``ThreadConfig`` (template config merged
            with overrides).

    Raises:
        AIFunctionError: The resolved config sets
            ``agent_kwargs["conversation_manager"]`` to a non-``None``
            value (the runtime owns conversation history; a user-supplied
            manager would desynchronize ``agent.messages`` from the
            event log and break I7/I9).

    Implements:
        Thread.
    """

    __slots__ = (
        "_template",
        "_config",
        "_output_spec",
        "_summarization_strategy",
        "_inject_buffer",
    )

    def __init__(
        self,
        template: AIFunction[P, T],
        config: ThreadConfig,
    ) -> None:
        # Reject user-supplied conversation_manager (I7/I9): the runtime
        # installs NullConversationManager on every Agent it builds. Any
        # non-None value is a misconfiguration.
        user_manager = config.agent_kwargs.get("conversation_manager")
        if user_manager is not None:
            raise AIFunctionError(
                "agent_kwargs['conversation_manager'] must be unset: the "
                "runtime manages conversation history through the event log "
                "and installs NullConversationManager on every Agent. A "
                "user-supplied conversation manager would mutate "
                "agent.messages behind the runtime's back and break I7/I9.",
                function_name=getattr(template, "name", ""),
            )

        self._template = template
        self._config = config
        self._output_spec = OutputSpec[T].from_type(
            output_type=template.output_type,
            is_structured=config.structured_output,
        )

        self._summarization_strategy: SummarizationStrategy = (
            config.summarization_strategy
            if config.summarization_strategy is not None
            else DefaultSummarizationStrategy()
        )

        # Thread-owned buffer of pending user turns. execute() appends its
        # generated prompt here; notify() appends arbitrary text
        # (e.g. watcher nudges). The event-bridge hook drains the buffer
        # at the next BeforeModelCallEvent and emits one MESSAGE_USER per
        # entry — atomic pairing preserves I7.
        self._inject_buffer: list[str] = []

    # ── Thread ──

    @property
    def name(self) -> str:
        """Thread name, taken from the owning ``AIFunction``."""
        return self._template.name

    async def notify(self, text: str) -> None:
        """Buffer ``text`` for observation at the next model-call boundary.

        Args:
            text: Message body delivered by the runtime or an external sender.

        Ensures:
            - ``text`` is appended to the thread-local inject buffer.
            - The next :meth:`execute` cycle observes ``text`` on its first
              model-call boundary via the event-bridge hook.
        """
        self._inject_buffer.append(text)

    async def execute(self, ctx: ThreadContext, *args: P.args, **kwargs: P.kwargs) -> T:
        """Render a prompt and drive the executor for one cycle.

        Builds the prompt via ``self._generate_prompt`` and appends it to
        the thread's inject buffer (after any messages already pending from
        :meth:`notify`); the event-bridge hook atomically emits one
        ``MESSAGE_USER`` event per buffer entry and injects the matching
        user turn into the live Strands agent at the first
        ``BeforeModelCallEvent`` boundary (I7).

        Args:
            ctx: Freshly built per-cycle context.
            args: Positional arguments forwarded to ``template.prompt_fn``.
            kwargs: Keyword arguments forwarded to ``template.prompt_fn``.

        Returns:
            The typed cycle result.

        Requires:
            ``ctx`` is a fresh context built by the runtime for this cycle.

        Emits:
            MESSAGE_USER — one per drained buffer entry, via the
            event-bridge hook.

        Strategy:
            1. Call ``self._generate_prompt`` to produce the prompt string.
            2. Append the prompt to ``self._inject_buffer`` so the
               event-bridge hook emits it (and injects it into the live
               agent) atomically at the first ``BeforeModelCallEvent``,
               after any pending inject messages.
            3. Call ``self._run_cycle(ctx)`` and return its result.
        """
        prompt = self._generate_prompt(*args, **kwargs)
        self._inject_buffer.append(prompt)
        return await self._run_cycle(ctx)

    # ── Internal pipeline steps ──

    async def _run_cycle(self, ctx: ThreadContext) -> T:
        """Run the shared agent execution loop for one cycle.

        ``ResultEvent`` is emitted by the runtime dispatcher around the
        cycle, not by the thread itself (see ``ResultEvent`` invariant).

        Args:
            ctx: Freshly built per-cycle context.

        Returns:
            The typed result produced by the agent.

        Strategy:
            1. Call ``self._config.config_hook(ctx)`` if set and merge
               the returned ``ThreadKwargs`` into a cycle-local config
               (all fields replaced, ``config_hook`` key itself
               ignored).
            2. Call ``reconstruct_messages(await
               ctx.coordinator.get_events(ctx.thread_id))`` to obtain
               the up-to-date message history.
            3. Call ``self._build_agent(messages, cycle_config, ...)``
               to create a Strands ``Agent``.
            4. ``await agent.invoke_async(messages=messages)``.
            5. On interrupts, ``await ctx.on_interrupt(batch)`` and
               resume with decisions.
            6. Call ``self._extract_result`` to extract the typed
               result.
            7. Emit ``TOKEN_USAGE`` via ``ctx.on_event``.
            8. Run post-conditions via ``self._validate_result``; on
               failure, emit the errors as a ``MESSAGE_USER`` turn and
               retry from step 2 up to ``cycle_config.max_attempts``
               times.
            9. Return the typed and validated result.
        """
        # Build cycle-local config by applying config_hook if set
        cycle_config = self._config
        if self._config.config_hook is not None:
            patch: ThreadKwargs = self._config.config_hook(ctx)
            patch_dict = dict(patch)
            patch_dict.pop("config_hook", None)
            cycle_config = dataclasses.replace(self._config, **patch_dict)  # type: ignore[arg-type]

        # Inject the default runtime-facing tools (list_threads, send_message).
        # Bound to this cycle's ctx so the LLM can discover peer threads and
        # nudge them via notify. Appended after user tools so a
        # user-supplied tool with the same name wins on resolution.
        from .tools import coordinator_tools

        cycle_config = dataclasses.replace(
            cycle_config,
            tools=(*cycle_config.tools, *coordinator_tools(ctx)),
        )

        post_conditions = cycle_config.post_conditions
        function_name = self._template.name
        # Build bound_args for post-condition matching (kwargs only; no positional at this stage)
        bound_args: dict[str, object] = {}

        for attempt in range(cycle_config.max_attempts):
            response = await self._invoke_with_summarization(ctx, cycle_config)

            # Handle interrupts if the agent surface them
            while response.interrupts:
                decisions = await ctx.on_interrupt(list(response.interrupts))
                response = await self._invoke_with_summarization(ctx, cycle_config, resume_messages=decisions)

            state: dict[str, object] = response.state or {}
            result = self._extract_result(response, state)

            # Emit token usage
            usage = response.metrics.accumulated_usage
            ctx.on_event(
                TokenUsageEvent(
                    token_usage=TokenUsage(
                        input_tokens=usage.get("inputTokens", 0),
                        output_tokens=usage.get("outputTokens", 0),
                        cache_read_tokens=usage.get("cacheReadInputTokens", 0),
                        cache_write_tokens=usage.get("cacheWriteInputTokens", 0),
                    ),
                )
            )

            if not post_conditions:
                return result

            errors = await self._validate_result(result, bound_args, post_conditions, function_name)
            if not errors:
                return result

            # Enqueue the validation errors as a user turn so the next cycle's
            # hook emits the MESSAGE_USER event. Sole-emitter discipline (I7)
            # keeps the event log and agent-observation order aligned even
            # across retries.
            failures = "\n".join(f"- {e}" for e in errors)
            error_text = (
                f"[{function_name}] Post-condition failures"
                f" (attempt {attempt + 1}/{cycle_config.max_attempts}):\n{failures}"
            )
            self._inject_buffer.append(error_text)

        # Exhausted all attempts — raise with the last set of errors
        raise AIFunctionError(
            f"Post-conditions not satisfied after {cycle_config.max_attempts} attempt(s)",
            function_name=function_name,
        )

    async def _invoke_with_summarization(
        self,
        ctx: ThreadContext,
        cycle_config: ThreadConfig,
        resume_messages: object = None,
    ) -> AgentResult:
        """Build a Strands Agent from the current event log and invoke it.

        Wraps ``agent.invoke_async`` with a bounded reactive summarization
        loop: on ``ContextWindowOverflowException`` we run the summarization
        pipeline, emit a ``ContextSummarizedEvent`` on this thread, then
        re-build a fresh agent (rehydrating from the updated event log) and
        retry. ``MaxTokensReachedException`` is a hard failure —
        summarization does not help if the model cannot produce its current
        output within the max-tokens budget.

        Args:
            ctx: Per-cycle runtime context.
            cycle_config: The already-resolved cycle-local config.
            resume_messages: If not ``None``, passed as ``messages=`` to
                ``invoke_async`` (interrupt-resume path).

        Returns:
            The completed ``AgentResult``.

        Raises:
            SummarizationFailedError: Every summarization attempt in
                this cycle failed to bring the history under the
                context window.
            MaxTokensReachedException: Propagated unchanged;
                unrecoverable.
        """
        summarizations_so_far = 0
        while True:
            events = await ctx.coordinator.get_events(ctx.thread_id)
            messages: Messages = reconstruct_messages(events)
            bridge = _EventBridgeHook(ctx=ctx, inject_buffer=self._inject_buffer)
            agent = self._build_agent(messages, cycle_config, [bridge], bridge.stream_callback)

            try:
                if resume_messages is not None:
                    return await agent.invoke_async(messages=resume_messages)  # type: ignore[arg-type]
                return await agent.invoke_async()
            except ContextWindowOverflowException as exc:
                if summarizations_so_far >= _MAX_SUMMARIZATION_ATTEMPTS:
                    raise SummarizationFailedError(
                        function_name=self._template.name,
                        reason=(
                            f"Context window still overflowed after "
                            f"{_MAX_SUMMARIZATION_ATTEMPTS} summarization attempts: {exc}"
                        ),
                    ) from exc
                summarizations_so_far += 1
                # Re-fetch events: the failed model call ran the event-bridge
                # hook's pre-call phase, which drained the message queue and
                # emitted MESSAGE_USER events. Those are the events we want
                # to summarize — the pre-drain snapshot would miss them.
                fresh_events = await ctx.coordinator.get_events(ctx.thread_id)
                try:
                    new_history = await self._summarization_strategy.summarize(fresh_events, ctx, cycle_config)
                except SummarizationFailedError:
                    raise
                except Exception as strategy_exc:
                    raise SummarizationFailedError(
                        function_name=self._template.name,
                        reason=f"summarization strategy raised: {strategy_exc!r}",
                    ) from strategy_exc
                ctx.on_event(ContextSummarizedEvent(new_history=new_history))
                # The next iteration re-reads events (now with the new
                # ContextSummarizedEvent) and rebuilds messages — this is
                # the cache-invalidation point for I9.
                continue
            except MaxTokensReachedException:
                # Hard failure: the model exhausted its output budget
                # mid-cycle, meaning its own reply could not fit. Propagate
                # unchanged; summarization would not help.
                raise

    def _build_agent(
        self,
        messages: Messages,
        cycle_config: ThreadConfig,
        extra_hooks: list[HookProvider] | None,
        callback_handler: Callable[..., None] | None = None,
    ) -> Agent:
        """Assemble a configured Strands ``Agent`` for one cycle.

        Args:
            messages: Current conversation history passed to the agent.
            cycle_config: The merged per-cycle config (base config
                patched by ``config_hook``).
            extra_hooks: Extra Strands hooks wired into the agent.
            callback_handler: Strands ``callback_handler`` for per-chunk
                streaming callbacks; passed through unchanged when not
                ``None``.

        Returns:
            A freshly constructed Strands ``Agent``.

        Strategy:
            1. If ``cycle_config.structured_output`` is ``False``,
               check that ``output_type`` is ``str`` or raise an
               exception.
            2. If ``cycle_config.structured_output`` is ``True`` and
               ``output_type`` is not a pydantic model, create a
               pydantic wrapper: ``class FinalAnswer: answer:
               output_type``.
            3. Build a Strands agent with:

               - The extra hooks wired in (these include the
                 event-bridge hook that drains ``message_queue``,
                 checks ``cancel_signal``, awaits ``pause_signal``,
                 and emits assistant/tool telemetry events).
               - If ``structured_output`` is ``True``, set
                 ``structured_output_model`` to ``output_type`` or the
                 pydantic wrapper.
               - Pass any field from ``cycle_config`` that applies to
                 ``strands.Agent.__init__``.
        """
        spec = self._output_spec
        if not spec.is_structured and spec.output_type is not str:
            raise AIFunctionError(
                f"structured_output=False is only supported when output_type is str, got {spec.output_type!r}",
                function_name=self._template.name,
            )

        tools = list(cycle_config.tools)

        hooks: list[HookProvider] = list(extra_hooks) if extra_hooks else []
        if "hooks" in cycle_config.agent_kwargs and cycle_config.agent_kwargs["hooks"]:
            hooks.extend(cycle_config.agent_kwargs["hooks"])

        # Drop ``hooks`` and ``conversation_manager`` from the forwarded kwargs:
        # we merged hooks above, and we install ``NullConversationManager``
        # unconditionally (I7/I9) — any user-supplied manager was already
        # rejected in ``__init__``.
        agent_kwargs = {
            k: v for k, v in cycle_config.agent_kwargs.items() if k not in ("hooks", "conversation_manager")
        }

        if callback_handler is not None:
            return Agent(
                model=cycle_config.model,
                messages=list(messages),
                system_prompt=cycle_config.system_prompt,
                tools=tools or None,
                structured_output_model=spec.structured_output_model,
                hooks=hooks,
                callback_handler=callback_handler,
                conversation_manager=NullConversationManager(),
                **agent_kwargs,  # pyright: ignore[reportArgumentType]
            )
        return Agent(
            model=cycle_config.model,
            messages=list(messages),
            system_prompt=cycle_config.system_prompt,
            tools=tools or None,
            structured_output_model=spec.structured_output_model,
            hooks=hooks,
            conversation_manager=NullConversationManager(),
            **agent_kwargs,  # pyright: ignore[reportArgumentType]
        )

    def serialize_result(self, result: T) -> str:
        """Encode ``result`` as a string for storage in a ``ResultEvent``.

        Args:
            result: The typed cycle result to serialize.

        Returns:
            A string representation of ``result`` that
            ``deserialize_result`` can round-trip.

        Ensures:
            ``deserialize_result(serialize_result(result))`` returns a
            value equal to ``result``.

        Concurrency:
            Must be synchronous and side-effect-free.
        """
        adapter: TypeAdapter[T] = TypeAdapter(self._output_spec.output_type)
        return adapter.dump_json(result).decode("utf-8")

    def deserialize_result(self, payload: str) -> T:
        """Recover a result from the string stored in a ``ResultEvent``.

        Args:
            payload: Value previously produced by ``serialize_result``.

        Returns:
            The deserialized result of type ``T``.

        Raises:
            AIFunctionError: The payload is malformed or cannot be
                decoded as ``T``.
        """
        try:
            adapter: TypeAdapter[T] = TypeAdapter(self._output_spec.output_type)
            return adapter.validate_json(payload)
        except Exception as exc:
            raise AIFunctionError(
                f"Failed to deserialize result payload: {exc}",
                function_name=self._template.name,
            ) from exc

    async def fork(self) -> AIFunction[P, T]:
        """Return the owning template as a resumption ``Spawnable``.

        ``AIThread`` carries no per-instance state beyond the event log
        (which the runtime seeds separately via
        :meth:`Coordinator.copy_events`), so the original template is
        itself a valid resumption spawnable.

        Note:
            The fork intentionally does NOT inherit the current thread's
            inject buffer. Pending ``notify`` entries that have
            not yet been observed by a cycle are dropped from the forked
            thread's perspective — the fork starts with an empty buffer.

        Returns:
            ``self.template`` — the ``AIFunction`` this thread was
            spawned from.
        """
        return self._template

    async def teardown(self) -> None:
        """Release per-instance state on termination.

        Drops the inject buffer. ``AIThread`` owns no other external
        resources.
        """
        self._inject_buffer.clear()
        return None

    def _generate_prompt(self, *args: P.args, **kwargs: P.kwargs) -> str:
        """Render the prompt string from template arguments.

        Args:
            args: Positional arguments forwarded from ``execute``.
            kwargs: Keyword arguments forwarded from ``execute``.

        Returns:
            The rendered prompt string.

        Strategy:
            1. Call ``self._template.prompt_fn(*args, **kwargs)``.
            2. If ``prompt_fn`` returns ``None``, interpret
               ``prompt_fn.__doc__`` as a ``tstr`` template and
               interpolate it using the function arguments and their
               enclosing globals as context.
            3. If there is no docstring either, raise
               ``AIFunctionError``.
        """
        result = self._template.prompt_fn(*args, **kwargs)
        if result is not None:
            return result

        doc = self._template.prompt_fn.__doc__
        if not doc:
            raise AIFunctionError(
                "prompt_fn returned None and has no docstring to use as a template",
                function_name=self._template.name,
            )

        # Build context from bound arguments
        sig = inspect.signature(self._template.prompt_fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        context: dict[str, object] = dict(bound.arguments)

        if hasattr(self._template.prompt_fn, "__globals__"):
            fn_globals = self._template.prompt_fn.__globals__
        else:
            fn_globals = dict[str, Any]()  # pyright: ignore[reportExplicitAny]
        template = tstr.generate_template(doc, context, globals=fn_globals, use_eval=False)  # pyright: ignore[reportUnknownMemberType]
        return tstr.render(template)

    def _extract_result(self, response: AgentResult, state: dict[str, object]) -> T:
        """Extract the typed result from a Strands ``AgentResult``.

        Args:
            response: The Strands agent result.
            state: Per-cycle execution state.

        Returns:
            The typed result as declared by ``output_type``.
        """
        spec = self._output_spec

        if not spec.is_structured:
            return cast(T, str(response))
        # output is structured
        assert response.structured_output is not None, "Agent did not return a structured output"
        if spec.is_wrapped:
            return response.structured_output.answer  # pyright: ignore
        return cast(T, response.structured_output)

    async def _validate_result(
        self,
        result: T,
        bound_args: dict[str, object],
        post_conditions: tuple[PostCondition, ...],
        function_name: str,
    ) -> list[str]:
        """Evaluate every post-condition against ``result`` in parallel.

        Returns the list of all error messages from failed conditions.
        If a condition raises an exception, it is considered failed and
        the exception text is used as error message.

        Args:
            result: The candidate typed result.
            bound_args: Bound arguments for condition callables that
                accept them.
            post_conditions: Ordered tuple of validators from
                ``ThreadConfig``.
            function_name: Name of the owning function (for error
                attribution).

        Returns:
            A list of error messages from failed post conditions.
        """

        async def _run_one(cond: PostCondition) -> str | None:
            # Pass only the keyword args whose names match the condition's signature
            sig = inspect.signature(cond)
            extra = {k: v for k, v in bound_args.items() if k in sig.parameters}
            try:
                cond_result = cond(result, **extra)
                if asyncio.iscoroutine(cond_result):
                    cond_result = await cond_result
            except Exception as exc:
                return str(exc)
            if cond_result is None or cond_result.passed:
                return None
            return cond_result.message

        outcomes = await asyncio.gather(*(_run_one(c) for c in post_conditions))
        return [msg for msg in outcomes if msg is not None]

    # ── Introspection ──

    @property
    def template(self) -> AIFunction[P, T]:
        """The template this thread was created from."""
        return self._template

    @property
    def config(self) -> ThreadConfig:
        """Resolved config.

        ``template.config`` merged with ``to_thread`` overrides.
        """
        return self._config
