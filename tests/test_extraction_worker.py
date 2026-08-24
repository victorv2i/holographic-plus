from __future__ import annotations

import threading
import time

import pytest

from enfold.extraction_processor import ExtractionProcessResult
from enfold.extraction_worker import SupervisedExtractionWorker


class BlockingProcessor:
    def __init__(self):
        self.calls = 0
        self.entered = threading.Event()
        self.release = threading.Event()

    def process_one(self):
        self.calls += 1
        self.entered.set()
        self.release.wait(1.0)
        return ExtractionProcessResult("completed", self.calls, writes=1)


def test_shutdown_stops_new_claims_and_wait_is_bounded():
    processor = BlockingProcessor()
    worker = SupervisedExtractionWorker(
        processor, poll_seconds=0.01, drain_limit=8
    )
    worker.start()
    assert processor.entered.wait(0.5)

    with pytest.raises(RuntimeError, match="did not stop cleanly"):
        worker.stop(0.01)
    assert worker.health()["stopping"] is True

    processor.release.set()
    worker.stop(0.5)
    assert processor.calls == 1


class BrokenProcessor:
    def process_one(self):
        raise RuntimeError("secret raw exception details")


def test_health_redacts_unexpected_worker_errors():
    worker = SupervisedExtractionWorker(
        BrokenProcessor(), poll_seconds=0.01, drain_limit=1
    )
    worker.start()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if worker.health()["last_error"] is not None:
            break
        time.sleep(0.005)
    state = worker.health()
    worker.stop(0.5)
    assert state["last_error"] == "worker_failure"
    assert "secret" not in str(state)


def test_failed_prerequisite_pauses_before_claim_and_recovers():
    class IdleProcessor:
        def __init__(self):
            self.calls = 0

        def process_one(self):
            self.calls += 1
            return ExtractionProcessResult("idle", None)

    ready = threading.Event()
    processor = IdleProcessor()

    def verify():
        if not ready.is_set():
            raise RuntimeError("private attestor payload")

    worker = SupervisedExtractionWorker(
        processor,
        poll_seconds=0.01,
        drain_limit=1,
        prerequisite_check=verify,
        prerequisite_check_interval_seconds=0.05,
    )
    worker.start()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if worker.health()["last_error"] == "worker_prerequisite_failed":
            break
        time.sleep(0.005)

    state = worker.health()
    assert processor.calls == 0
    assert state["last_error"] == "worker_prerequisite_failed"
    assert "private attestor payload" not in str(state)

    ready.set()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and processor.calls == 0:
        time.sleep(0.005)
    worker.stop(0.5)

    assert processor.calls > 0
    assert worker.health()["last_error"] is None


def test_failed_attestation_recheck_pauses_until_a_later_verification():
    class IdleProcessor:
        def __init__(self):
            self.calls = 0

        def process_one(self):
            self.calls += 1
            return ExtractionProcessResult("idle", None)

    verified = threading.Event()
    verified.set()
    processor = IdleProcessor()

    def verify():
        if not verified.is_set():
            raise RuntimeError("private attestation detail")

    worker = SupervisedExtractionWorker(
        processor,
        poll_seconds=0.005,
        drain_limit=1,
        prerequisite_check=verify,
        prerequisite_check_interval_seconds=0.02,
    )
    worker.start()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and processor.calls == 0:
        time.sleep(0.005)
    assert processor.calls > 0

    verified.clear()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if worker.health()["last_error"] == "worker_prerequisite_failed":
            break
        time.sleep(0.005)
    paused_calls = processor.calls
    time.sleep(0.05)

    assert worker.health()["last_error"] == "worker_prerequisite_failed"
    assert processor.calls == paused_calls

    verified.set()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline and processor.calls == paused_calls:
        time.sleep(0.005)
    worker.stop(0.5)

    assert processor.calls > paused_calls
    assert worker.health()["last_error"] is None


def test_prerequisite_deadline_is_rechecked_between_jobs_in_drain_batch():
    class SlowProcessor:
        def __init__(self):
            self.calls = 0

        def process_one(self):
            self.calls += 1
            time.sleep(0.03)
            return ExtractionProcessResult("completed", self.calls, writes=1)

    checks = 0

    def verify():
        nonlocal checks
        checks += 1
        if checks > 1:
            raise RuntimeError("expired attestation")

    processor = SlowProcessor()
    worker = SupervisedExtractionWorker(
        processor,
        poll_seconds=0.2,
        drain_limit=8,
        prerequisite_check=verify,
        prerequisite_check_interval_seconds=0.01,
    )
    worker.start()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if worker.health()["last_error"] == "worker_prerequisite_failed":
            break
        time.sleep(0.005)
    worker.stop(0.5)

    assert checks == 2
    assert processor.calls == 1
    assert worker.health()["last_error"] == "worker_prerequisite_failed"
