"""Legacy MCP sidecar lock for multi-step holographic writes."""

from __future__ import annotations

import contextlib
import os
import threading
from pathlib import Path
from typing import Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover - posix-only deployment target
    fcntl = None


_held_lock = threading.RLock()
_held_condition = threading.Condition(_held_lock)
_held_by_path = {}


def _reset_after_fork() -> None:
    """Discard ownership and file descriptors inherited from the parent."""
    global _held_lock, _held_condition, _held_by_path
    inherited = tuple(_held_by_path.values())
    _held_lock = threading.RLock()
    _held_condition = threading.Condition(_held_lock)
    _held_by_path = {}
    for state in inherited:
        try:
            state["fh"].close()
        except Exception:
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)


@contextlib.contextmanager
def cross_process_write_lock(db_path: str) -> Iterator[None]:
    """Serialize legacy MCP/Hermes multi-transaction writes on *db_path*.

    This sidecar (``.mcp-write.lock``) is taken by the v0 MCP provider and
    holographic helpers. The v1 daemon does not take it: process exclusion
    uses ``.enfold.lock`` for the daemon lifetime, and each mutation uses
    ``BEGIN IMMEDIATE``. Maintenance waits on both sidecars. Do not treat
    this file as proof that the v1 writer is idle.

    The lock is keyed by the canonical database path and is reentrant
    within the owning thread. Other threads wait for the owner to release it.
    """
    if fcntl is None:  # pragma: no cover
        yield
        return

    process_id = os.getpid()
    canonical = str(Path(db_path).expanduser().resolve())
    lock_path = f"{canonical}.mcp-write.lock"
    owner = threading.get_ident()

    with _held_condition:
        while True:
            state = _held_by_path.get(canonical)
            if state is None:
                nested = False
                fh = open(lock_path, "a+")
                _held_by_path[canonical] = {
                    "count": 1,
                    "fh": fh,
                    "owner": owner,
                }
                break
            if state["owner"] == owner:
                state["count"] += 1
                nested = True
                fh = state["fh"]
                break
            _held_condition.wait()

    if not nested:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        except Exception:
            fh.close()
            with _held_condition:
                _held_by_path.pop(canonical, None)
                _held_condition.notify_all()
            raise

    try:
        yield
    finally:
        if os.getpid() == process_id:
            release = False
            with _held_condition:
                state = _held_by_path[canonical]
                state["count"] -= 1
                if state["count"] == 0:
                    _held_by_path.pop(canonical)
                    release = True
            if release:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                finally:
                    fh.close()
                    with _held_condition:
                        _held_condition.notify_all()
