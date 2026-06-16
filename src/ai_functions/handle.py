"""Thread reference returned by ``coordinator.spawn`` / ``worker.spawn_locally``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast, final

from .types import ThreadId, ThreadStatus

if TYPE_CHECKING:
    from .protocols import Coordinator


@final
class ThreadHandle[**P, T]:
    """Thin ``(thread_id, coordinator)`` reference delegating every operation.

    Every method forwards to the coordinator, which routes to the
    hosting worker via its registered adapter. The handle itself holds
    no per-thread state; one handle type works across any coordinator
    implementation (in-memory, remote, test double).

    Args:
        thread_id: Runtime-assigned id of an already-registered thread.
        coordinator: Coordinator responsible for routing operations on
            this thread.

    Lifecycle:
        NOT_STARTED -> RUNNING -> {IDLE, PAUSED, CANCELLED} ->
        {TERMINATED, FAILED}.
    """

    __slots__ = ("_thread_id", "_coordinator")

    def __init__(self, thread_id: ThreadId, coordinator: Coordinator) -> None:
        self._thread_id: ThreadId = thread_id
        self._coordinator: Coordinator = coordinator

    @property
    def id(self) -> ThreadId:
        """Thread id this handle refers to."""
        return self._thread_id

    async def status(self) -> ThreadStatus:
        """Return the runtime-maintained status of the thread.

        Returns:
            The status the coordinator currently attributes to the thread.

        Raises:
            ThreadNotFoundError: The thread is no longer registered.

        Concurrency:
            Awaits the coordinator; remote coordinators round-trip to the server.
        """
        return await self._coordinator.get_thread_status(self._thread_id)

    async def is_done(self) -> bool:
        """Check whether the thread has reached a terminal status.

        Returns:
            ``True`` iff the thread's current status is terminal
            (``TERMINATED`` or ``FAILED``).
        """
        return (await self.status()).is_done

    # ── Execution ────────────────────────────────────────────────

    def run(self, *args: P.args, **kwargs: P.kwargs) -> asyncio.Future[T]:
        """Enqueue a ``PromptRequest`` on the thread's work queue.

        Args:
            args: Positional arguments forwarded to the thread's prompt function.
            kwargs: Keyword arguments forwarded to the thread's prompt function.

        Returns:
            A future resolved by the dispatcher once the cycle completes.

        Ensures:
            - One ``PromptRequest(args, kwargs, future)`` is appended to
              the thread's FIFO work queue.
            - Each call returns its own independent future.
            - The returned future resolves with the result on success.
            - The returned future rejects with the raised exception on error.
            - The returned future rejects with ``CancelledError`` on
              cooperative or hard cancel.

        Raises:
            ThreadNotFoundError: The thread is no longer registered.

        Concurrency:
            Synchronous enqueue via the coordinator; the cycle runs
            later on the hosting worker's dispatcher.
        """
        return cast("asyncio.Future[T]", self._coordinator.submit(self._thread_id, *args, **kwargs))

    # ── Messaging ────────────────────────────────────────────────

    async def notify(self, text: str) -> None:
        """Deliver a side-channel message to this thread.

        Best-effort: the thread decides whether and when to surface the
        message. No cycle is started by this call.

        Args:
            text: Message body to route to the thread.

        Raises:
            ThreadNotFoundError: The thread is no longer registered.
        """
        await self._coordinator.notify(self._thread_id, text)

    # ── Lifecycle ────────────────────────────────────────────────

    async def pause(self) -> None:
        """Pause the in-flight cycle at its next work boundary.

        Ensures:
            - The thread's pause signal is set.
            - Queued work is not consumed until ``resume`` is called.

        Raises:
            ThreadNotFoundError: The thread is no longer registered.

        Concurrency:
            Idempotent.
        """
        await self._coordinator.pause(self._thread_id)

    async def resume(self) -> None:
        """Clear the pause signal so the executor continues.

        Raises:
            ThreadNotFoundError: The thread is no longer registered.

        Concurrency:
            Idempotent.
        """
        await self._coordinator.resume(self._thread_id)

    async def cancel(self) -> None:
        """Cooperatively cancel the in-flight cycle.

        Ensures:
            - The thread's cancel signal is set.
            - The in-flight ``PromptRequest``'s future rejects with
              ``CancelledError``.
            - The thread remains registered and continues to accept new work.

        Raises:
            ThreadNotFoundError: The thread is no longer registered.

        Concurrency:
            No-op when no cycle is in flight.
        """
        await self._coordinator.cancel(self._thread_id)

    async def terminate(self) -> None:
        """Schedule graceful termination behind currently-queued work.

        Ensures:
            - A ``TerminateAfterIdle`` marker is appended to the work queue.
            - Items queued ahead of the marker run to completion.
            - Items queued after the marker are rejected with ``CancelledError``.
            - Status transitions to ``TERMINATED`` once the dispatcher reaches the marker.
            - Subsequent ``run`` calls on this handle raise ``ThreadNotFoundError``.
        """
        await self._coordinator.terminate(self._thread_id)

    async def terminate_now(self) -> None:
        """Tear the thread down immediately without draining the work queue.

        Ensures:
            - The in-flight task (if any) is cancelled.
            - The in-flight ``PromptRequest``'s future rejects with ``CancelledError``.
            - Every queued ``PromptRequest``'s future rejects with ``CancelledError``.
            - The thread is removed from the coordinator before this call returns.
        """
        await self._coordinator.terminate_now(self._thread_id)

    # ── Forking ──────────────────────────────────────────────────

    async def fork(self) -> ThreadHandle[P, T]:
        """Fork into a new thread seeded with a copy of this thread's history.

        Returns:
            A new handle in ``NOT_STARTED`` state referring to the forked thread.

        Raises:
            ThreadNotFoundError: The thread is no longer registered.
            NotImplementedError: This thread type does not support forking.
        """
        forked = await self._coordinator.fork(self._thread_id)
        return cast("ThreadHandle[P, T]", forked)
