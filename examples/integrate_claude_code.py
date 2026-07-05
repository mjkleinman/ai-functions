"""Claude Agent thread — run a Claude Agent session, visualize events, print the result.

``ClaudeAgent`` is a ``Spawnable`` that drives a ``claude`` subprocess via
``claude_agent_sdk``. The SDK owns the conversation transcript; ai_functions
observes the SDK's message stream and re-emits each element as a ai_functions
event for observability.

This example:

1. Spawns a ``ClaudeAgent`` thread on a ``LocalWorker``.
2. Subscribes to events on the coordinator and pretty-prints them as
   they arrive.
3. Sends a task as a prompt and awaits the result.
4. Tears the thread down so the ``claude`` subprocess exits cleanly.
"""

from __future__ import annotations

import asyncio

from _utils import console, display
from claude_agent_sdk import ClaudeAgentOptions
from strands.types.tools import ToolResultContent

from ai_functions.claude_code import ClaudeAgent
from ai_functions.runtime import InMemoryCoordinator, LocalWorker
from ai_functions.types import (
    CompletedEvent,
    CustomEvent,
    Event,
    FailedEvent,
    MessageAssistantCompleteEvent,
    MessageUserEvent,
    StartedEvent,
    TokenUsageEvent,
    ToolCallEvent,
    ToolResultEvent,
)


def log_event(event: Event) -> None:
    """Render one ai_functions event to the console as it streams in."""
    match event:
        case StartedEvent(thread_name=name):
            console.print(f"  ▶ {name or 'thread'} started", markup=False)
        case CompletedEvent(thread_name=name):
            console.print(f"  ✓ {name or 'thread'} completed", markup=False)
        case FailedEvent(thread_name=name, error=error):
            console.print(f"  ✗ {name or 'thread'} failed: {error}", markup=False)
        case MessageUserEvent(text=text):
            preview = text if len(text) <= 120 else text[:117] + "..."
            console.print(f"  ▷ user: {preview}", markup=False)
        case MessageAssistantCompleteEvent(content=content):
            texts: list[str] = []
            for block in content:
                text = block.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
            if texts:
                joined = "\n".join(texts)
                preview = joined if len(joined) <= 240 else joined[:237] + "..."
                console.print(f"  ◁ assistant: {preview}", markup=False)
            else:
                kinds = [key for block in content for key in block]
                summary = ", ".join(kinds) if kinds else "empty"
                console.print(f"  ◁ assistant: <{summary}>", markup=False)
        case ToolCallEvent(tool_name=tool_name, arguments=arguments):
            console.print(f"    ⚙ tool call: {tool_name}({_format_args(arguments)})", markup=False)
        case ToolResultEvent(status=status, content=content):
            preview = _format_tool_result(content) or "<empty>"
            error = "[error] " if status == "error" else ""
            console.print(f"    ⚙ tool result: {error}{preview}", markup=False)
        case TokenUsageEvent(token_usage=usage):
            total_input = usage.input_tokens + usage.cache_read_tokens + usage.cache_write_tokens
            total = total_input + usage.output_tokens
            console.print(
                f"  Σ tokens: in={usage.input_tokens} "
                f"cache_r={usage.cache_read_tokens} cache_w={usage.cache_write_tokens} "
                f"out={usage.output_tokens} (total={total})",
                markup=False,
            )
        case CustomEvent(kind=kind):
            console.print(f"  • custom event: {kind}", markup=False)
        case _:
            pass


def _format_args(arguments: dict[str, object]) -> str:
    """Render tool-call arguments compactly for log output."""
    parts: list[str] = []
    for key, value in arguments.items():
        rendered = repr(value)
        if len(rendered) > 60:
            rendered = rendered[:57] + "..."
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)


def _format_tool_result(content: list[ToolResultContent]) -> str:
    """Render a tool result's content as a compact one-line preview."""
    texts = [block["text"] for block in content if isinstance(block.get("text"), str)]
    if texts:
        joined = " ".join(" ".join(t.split()) for t in texts)
        return joined if len(joined) <= 120 else joined[:117] + "..."
    kinds = [key for block in content for key in block]
    return f"<{', '.join(kinds)}>" if kinds else ""


async def main() -> None:
    # Bypass the Claude Agent permission prompt so the example runs non-interactively.
    # Don't do this outside trusted environments.
    options = ClaudeAgentOptions(permission_mode="bypassPermissions")
    template = ClaudeAgent(options=options, name="claude_agent_example")

    coord = InMemoryCoordinator()
    coord.on(log_event)

    worker = await LocalWorker(coord).register()

    handle = await worker.spawn_locally(template, thread_name="claude_agent_example")
    try:
        result = await handle.run(
            "List the three files in the current directory whose names end in .md, "
            "and in one short sentence say what the project is about based on README.md.",
        )
        display("Result", str(result))
    finally:
        await handle.terminate_now()
        await worker.close()


if __name__ == "__main__":
    asyncio.run(main())
