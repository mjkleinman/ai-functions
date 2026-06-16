"""Multi-turn conversation — same thread, accumulated history.

Use handle.run() to drive each turn. The thread keeps its conversation
history across calls. Each run() is a complete cycle that returns a
typed result.
"""

import asyncio

from ai_functions import ai_function


@ai_function(str)
def assistant(message: str):
    """{message}"""


async def main():
    # spawn() creates a handle with its own session — no runtime needed
    handle = await assistant.spawn()

    r1 = await handle.run(message="What is the capital of France?")
    print(f"Turn 1: {r1}")

    # The agent sees the full conversation history from turn 1
    r2 = await handle.run(message="What about Germany?")
    print(f"Turn 2: {r2}")

    # Inject context without starting a cycle
    await handle.notify("The user prefers short answers.")

    # The agent sees the injected message on the next run
    r3 = await handle.run(message="And Spain?")
    print(f"Turn 3: {r3}")

    await handle.terminate_now()


if __name__ == "__main__":
    asyncio.run(main())
