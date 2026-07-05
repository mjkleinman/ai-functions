"""Per-method pydantic params / result models for the Coordinator/Worker RPC.

Every RPC method on the wire gets exactly one params model here. The
shape of each model matches the method signature on the corresponding
``Coordinator`` or ``WorkerAdapter`` method one-to-one. Results that
aren't already pydantic-serializable are wrapped in a result model
here too (typed ``Binary`` for cloudpickle blobs, etc.).

The dispatch layer in :mod:`ai_functions.network.channel` validates incoming
``CallFrame.params`` against the matching model before invoking the
handler, so handlers receive fully-typed values and can drop the
``params["..."]`` / ``cast(...)`` noise that the old ``dict[str, Any]``
dispatch required.

Method names are fixed strings under two namespaces:

- ``coordinator.<op>`` — calls the client makes on the endpoint.
- ``worker.<wid>.<op>`` — calls the endpoint makes on a hosted worker
  (``<wid>`` is the worker's id so the same channel can route to
  multiple workers hosted behind the same client).
"""

from __future__ import annotations

from pydantic import BaseModel

from ..types import Event, EventId, EventKind, ThreadId, ThreadInfo, WorkerId
from .wire import Binary

# ── Coordinator params ───────────────────────────────────────────────────────


class RegisterWorkerParams(BaseModel):
    worker_id: WorkerId


class DeregisterWorkerParams(BaseModel):
    worker_id: WorkerId


class RegisterThreadParams(BaseModel):
    info: ThreadInfo


class DeregisterThreadParams(BaseModel):
    thread_id: ThreadId


class ListThreadsParams(BaseModel):
    pass


class GetThreadInfoParams(BaseModel):
    thread_id: ThreadId


class GetThreadStatusParams(BaseModel):
    thread_id: ThreadId


class SpawnParams(BaseModel):
    target_pickle: Binary
    seed_from: ThreadId | None = None
    seed_events: list[Event] | None = None
    worker_id: WorkerId | None = None
    thread_id: ThreadId | None = None
    thread_name: str | None = None
    parent_id: ThreadId | None = None
    metadata: dict[str, object] | None = None


class SubmitParams(BaseModel):
    thread_id: ThreadId
    args_kwargs_pickle: Binary


class InjectMessageParams(BaseModel):
    thread_id: ThreadId
    text: str


class ThreadIdOnlyParams(BaseModel):
    """Shared by pause / resume / cancel / terminate / terminate_now / is_paused / etc."""

    thread_id: ThreadId


class ForkParams(BaseModel):
    """Params for ``coordinator.fork``; ``parent_id`` overrides the inherited default."""

    thread_id: ThreadId
    parent_id: ThreadId | None = None


class ResolveApprovalParams(BaseModel):
    thread_id: ThreadId
    approval_id: str
    decision: str


class GetEventsParams(BaseModel):
    thread_id: ThreadId
    since_id: EventId | None = None
    kinds: list[EventKind] | None = None
    limit: int | None = None


class AppendEventParams(BaseModel):
    event: Event


class NewMessageIdParams(BaseModel):
    pass


class CopyEventsParams(BaseModel):
    source_id: ThreadId
    target_id: ThreadId
    until_event_id: EventId | None = None


# ── Worker params ────────────────────────────────────────────────────────────


class WorkerSpawnParams(BaseModel):
    target_pickle: Binary
    thread_id: ThreadId
    thread_name: str | None = None
    parent_id: ThreadId | None = None
    metadata: dict[str, object] = {}


class WorkerSubmitParams(BaseModel):
    thread_id: ThreadId
    args_kwargs_pickle: Binary


class WorkerInjectMessageParams(BaseModel):
    thread_id: ThreadId
    text: str


class WorkerThreadIdParams(BaseModel):
    """Shared by cancel/pause/resume/terminate/terminate_now/get_fork_spawnable."""

    thread_id: ThreadId


class WorkerResolveApprovalParams(BaseModel):
    thread_id: ThreadId
    approval_id: str
    decision: str
