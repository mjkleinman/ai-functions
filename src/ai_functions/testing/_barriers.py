"""Shared barrier registry used by ``ScriptedModel`` and ``RuntimeHarness``.

Private to ``ai_functions.testing``. The contextvar is set by
``RuntimeHarness.__aenter__`` so scripted models spawned inside the
``async with`` block resolve barrier names against the active harness.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar


class BarrierRegistry:
    """Named ``asyncio.Event`` store shared by a harness and its scripted models.

    Creates an empty registry.
    """

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}

    def get(self, name: str) -> asyncio.Event:
        """Return the (lazily created) event for ``name``.

        Args:
            name: Barrier name.

        Returns:
            The ``asyncio.Event`` for ``name``; created on first access.
        """
        ev = self._events.get(name)
        if ev is None:
            ev = asyncio.Event()
            self._events[name] = ev
        return ev

    def release(self, name: str) -> None:
        """Set the event for ``name``, unblocking all current and future awaits.

        Args:
            name: Barrier name.
        """
        self.get(name).set()

    def release_all(self) -> None:
        """Set every known event so pending awaiters unwind on harness teardown."""
        for ev in self._events.values():
            ev.set()


current_registry: ContextVar[BarrierRegistry | None] = ContextVar(
    "_ai_functions_scripted_barrier_registry", default=None
)


async def await_barrier(name: str) -> None:
    """Block until the named barrier is released by the active harness.

    Args:
        name: Barrier name.

    Raises:
        RuntimeError: No ``RuntimeHarness`` is active in this context.
    """
    registry = current_registry.get()
    if registry is None:
        raise RuntimeError(f"ScriptedModel: barrier {name!r} requested but no RuntimeHarness is active")
    await registry.get(name).wait()
