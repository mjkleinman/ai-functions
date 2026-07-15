"""Opt-in console dump of the exact per-model-call request.

Set ``AI_FUNCTIONS_SHOW_PROMPTS=1`` to print, before every model call, the
request the model is about to receive: the system prompt and the full message
list (including turns injected by the runtime — prompt, notify entries,
validation feedback — and by Strands itself, e.g. tool results and the
structured-output forcing turn). Because it prints on every call, retries,
summarization splices, and mid-cycle injections are all visible.

Output is plain text (no rich markup or wrapping) so it is byte-exact and
copy-pasteable. Tool specs are not printed — only the conversational payload.

The first model call of each agent build prints the whole request; subsequent
calls print only the messages appended since the previous printed call (the
full history is unchanged prefix — reprinting it would drown the delta).
"""

from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from strands import Agent

_ENV_FLAG = "AI_FUNCTIONS_SHOW_PROMPTS"

_RULE = "═" * 72


def prompts_enabled() -> bool:
    """True when ``AI_FUNCTIONS_SHOW_PROMPTS`` opts into prompt dumps."""
    return os.getenv(_ENV_FLAG, "").strip().lower() in ("1", "true", "enabled")


def _render_block(block: dict[str, object]) -> str:
    """Render one content block verbatim-ish: text as-is, tool traffic compact."""
    if "text" in block:
        return cast("str", block["text"])
    if "toolUse" in block:
        tool_use = cast("dict[str, object]", block["toolUse"])
        args = json.dumps(tool_use.get("input", {}), default=str)
        return f"[toolUse {tool_use.get('name')}#{tool_use.get('toolUseId')}] {args}"
    if "toolResult" in block:
        tool_result = cast("dict[str, object]", block["toolResult"])
        body = json.dumps(tool_result.get("content", []), default=str)
        return f"[toolResult #{tool_result.get('toolUseId')} {tool_result.get('status')}] {body}"
    if "reasoningContent" in block:
        return "[reasoningContent]"
    (key,) = list(block.keys())[:1] or ["<empty>"]
    return f"[{key}]"


def print_model_request(agent: Agent, *, printed_upto: int, call_index: int, thread_name: str) -> int:
    """Print the request ``agent`` is about to send; return the new printed-upto index.

    Args:
        agent: The live Strands agent, after all pre-call injections.
        printed_upto: Number of ``agent.messages`` entries already printed for
            this agent build (0 on the first call — prints system prompt and
            the full history).
        call_index: 1-based model-call counter within this agent build, for
            the header.
        thread_name: Label for the header.

    Returns:
        ``len(agent.messages)`` — pass back as ``printed_upto`` next call.
    """
    messages = agent.messages
    out = sys.stderr
    print(_RULE, file=out)
    print(f"MODEL REQUEST · {thread_name} · call #{call_index}", file=out)
    if printed_upto == 0:
        print("--- system ---", file=out)
        print(agent.system_prompt or "<none>", file=out)
    elif printed_upto < len(messages):
        print(f"(messages 1..{printed_upto} unchanged, showing new)", file=out)
    for message in messages[printed_upto:]:
        role = message.get("role", "?")
        print(f"--- {role} ---", file=out)
        for raw_block in message.get("content", []):
            print(_render_block(cast("dict[str, object]", raw_block)), file=out)
    print(_RULE, file=out)
    return len(messages)


def print_model_response(content: list[object], *, thread_name: str, call_index: int) -> None:
    """Print the assistant turn a model call just produced.

    The response-side counterpart of :func:`print_model_request`, called by the
    event bridge after each model call with the completed turn's content
    blocks. Tool calls render compactly (as in requests); the next request dump
    does not repeat this turn — it prints only the messages appended after it.

    Args:
        content: The assistant message's content blocks.
        thread_name: Label for the header.
        call_index: 1-based model-call counter within this agent build.
    """
    out = sys.stderr
    print(f"MODEL RESPONSE · {thread_name} · call #{call_index}", file=out)
    for raw_block in content:
        if isinstance(raw_block, dict):
            print(_render_block(cast("dict[str, object]", raw_block)), file=out)
    print(_RULE, file=out)
