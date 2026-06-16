"""Scripted ``strands.Model`` implementation for deterministic agent tests.

``ScriptedModel`` plays back a fixed sequence of model turns. Each ``Turn`` describes
one model call: streamed text chunks followed by optional tool calls. Tests drive
agent behaviour by writing a small script; streaming is real (chunks are yielded
between ``await`` points) so concurrency tests can observe partial state.

Barriers (``await_before``/``await_after`` on a ``Turn``, and ``AwaitBarrier``
sentinels inside ``text_chunks``) suspend the stream until ``RuntimeHarness.release``
is called. The connection between model and harness is carried by a ``ContextVar``
set in ``RuntimeHarness.__aenter__`` — tests don't attach models explicitly.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncGenerator, AsyncIterable
from dataclasses import dataclass
from typing import Any, cast, final

from pydantic import BaseModel
from strands.models.model import Model
from strands.types.content import Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolChoice, ToolSpec

from ._barriers import await_barrier


@final
@dataclass(frozen=True)
class AwaitBarrier:
    """Sentinel placed inside ``Turn.text_chunks`` to suspend streaming mid-turn.

    When the scripted model reaches this sentinel it yields every chunk emitted so
    far, then awaits the harness barrier named ``name``. The next chunk after the
    sentinel is only streamed once ``RuntimeHarness.release(name)`` is called.
    """

    name: str


@final
@dataclass(frozen=True)
class Turn:
    """One scripted model call.

    A ``Turn`` describes what the model will emit on one invocation of
    ``Model.stream``: zero-or-more text chunks, then zero-or-more tool calls. Stop
    reason is ``"tool_use"`` if ``tool_calls`` is non-empty, ``"end_turn"`` otherwise.

    Exactly one of ``text`` or ``text_chunks`` may be provided (or neither,
    if the turn is a pure tool call). When ``text`` is given, ``ScriptedModel``
    splits it into word-sized chunks automatically; when ``text_chunks`` is given,
    each entry becomes one streamed chunk verbatim. A mid-stream ``AwaitBarrier``
    must appear inside ``text_chunks``.
    """

    text: str | None = None
    text_chunks: tuple[str | AwaitBarrier, ...] | None = None
    tool_calls: tuple[tuple[str, dict[str, object]], ...] = ()
    await_before: str | None = None
    await_after: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0

    def __post_init__(self) -> None:
        """Reject ambiguous or empty configurations.

        Raises:
            ValueError: Both ``text`` and ``text_chunks`` are set, or the
                turn has no text and no tool calls.
        """
        if self.text is not None and self.text_chunks is not None:
            raise ValueError("Turn: specify text or text_chunks, not both")
        if self.text is None and self.text_chunks is None and not self.tool_calls:
            raise ValueError("Turn: must have text, text_chunks, or tool_calls")

    def _resolved_chunks(self) -> tuple[str | AwaitBarrier, ...]:
        """Materialize the chunk sequence that ``stream`` will yield."""
        if self.text_chunks is not None:
            return self.text_chunks
        if self.text is not None:
            return _split_words(self.text)
        return ()


def _split_words(text: str) -> tuple[str, ...]:
    """Split ``text`` into whitespace-delimited word chunks, preserving spaces.

    Each returned chunk ends with a trailing space except the last, so
    concatenation reconstructs ``text`` exactly.
    """
    if not text:
        return ()
    words = text.split(" ")
    chunks: list[str] = []
    for i, w in enumerate(words):
        if i < len(words) - 1:
            chunks.append(w + " ")
        elif w:
            chunks.append(w)
    return tuple(chunks)


class ScriptExhausted(RuntimeError):
    """Raised when the agent requests more model calls than the script provides."""


@final
class ScriptedModel(Model):
    """Deterministic ``strands.Model`` that plays back a ``list[Turn]``.

    The script is fixed at construction; per-call state (cursor) is mutable.

    Args:
        turns: One ``Turn`` per expected model call, in order.

    Ensures:
        ``self.remaining_turns == len(turns)``.
    """

    def __init__(self, turns: list[Turn]) -> None:
        super().__init__()
        self._turns: list[Turn] = list(turns)
        self._cursor: int = 0

    @property
    def remaining_turns(self) -> int:
        """Turns not yet consumed by a ``stream`` call."""
        return len(self._turns) - self._cursor

    def update_config(self, **model_config: Any) -> None:  # pyright: ignore[reportExplicitAny, reportAny]
        """No-op for compatibility with ``strands.Model``.

        Args:
            model_config: Ignored; ``ScriptedModel`` has no mutable config.
        """
        del model_config

    def get_config(self) -> dict[str, object]:
        """Return a snapshot of the model's state.

        Returns:
            A dict with ``{"remaining_turns": int}``.
        """
        return {"remaining_turns": self.remaining_turns}

    def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        invocation_state: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
        **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    ) -> AsyncIterable[StreamEvent]:
        """Yield the Bedrock-shaped stream for the next scripted turn.

        Args:
            messages: Conversation history; not consulted by ``ScriptedModel``.
            tool_specs: Available tool specs; ignored (scripted turns name their target
                tool directly).
            system_prompt: Ignored.
            tool_choice: Ignored.
            system_prompt_content: Ignored.
            invocation_state: Ignored.
            kwargs: Ignored.

        Returns:
            An async iterable that yields one ``messageStart``, the turn's text and
            tool-use content blocks (with barriers honored), then ``messageStop`` and
            ``metadata``.

        Raises:
            ScriptExhausted: The script has no more turns.
        """
        del messages, tool_specs, system_prompt, tool_choice, system_prompt_content, invocation_state, kwargs
        if self._cursor >= len(self._turns):
            raise ScriptExhausted(
                f"ScriptedModel: agent requested turn {self._cursor + 1} but script has only {len(self._turns)} turns"
            )
        turn = self._turns[self._cursor]
        self._cursor += 1
        return _stream_turn(turn)

    def structured_output(
        self,
        output_model: type[BaseModel],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,  # pyright: ignore[reportExplicitAny, reportAny]
    ) -> AsyncGenerator[dict[str, BaseModel | Any]]:  # pyright: ignore[reportExplicitAny]
        """Unsupported path — ``ScriptedModel`` drives agents via ``stream`` only.

        Args:
            output_model: Ignored.
            prompt: Ignored.
            system_prompt: Ignored.
            kwargs: Ignored.

        Raises:
            NotImplementedError: Always; tests should rely on tool-call turns targeting
                the structured-output wrapper instead.
        """
        del output_model, prompt, system_prompt, kwargs
        raise NotImplementedError("ScriptedModel.structured_output is not supported; use a tool_calls turn")


# ── Stream emission ──


async def _stream_turn(turn: Turn) -> AsyncIterable[StreamEvent]:
    """Emit the full Bedrock-shaped stream for one ``Turn``, honoring barriers."""
    if turn.await_before is not None:
        await await_barrier(turn.await_before)

    yield cast(StreamEvent, {"messageStart": {"role": "assistant"}})

    chunks = turn._resolved_chunks()  # pyright: ignore[reportPrivateUsage]
    if chunks:
        yield cast(StreamEvent, {"contentBlockStart": {"start": {}}})
        for chunk in chunks:
            if isinstance(chunk, AwaitBarrier):
                await await_barrier(chunk.name)
            else:
                yield cast(StreamEvent, {"contentBlockDelta": {"delta": {"text": chunk}}})
        yield cast(StreamEvent, {"contentBlockStop": {}})

    for tool_name, tool_input in turn.tool_calls:
        tool_use_id = f"scripted-{uuid.uuid4().hex[:12]}"
        yield cast(
            StreamEvent,
            {"contentBlockStart": {"start": {"toolUse": {"toolUseId": tool_use_id, "name": tool_name}}}},
        )
        yield cast(
            StreamEvent,
            {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_input)}}}},
        )
        yield cast(StreamEvent, {"contentBlockStop": {}})

    stop_reason = "tool_use" if turn.tool_calls else "end_turn"
    yield cast(StreamEvent, {"messageStop": {"stopReason": stop_reason}})
    yield cast(
        StreamEvent,
        {
            "metadata": {
                "usage": {
                    "inputTokens": turn.input_tokens,
                    "outputTokens": turn.output_tokens,
                    "totalTokens": turn.input_tokens + turn.output_tokens,
                },
                "metrics": {"latencyMs": 0},
            }
        },
    )

    if turn.await_after is not None:
        await await_barrier(turn.await_after)
