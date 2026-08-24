from __future__ import annotations

import contextlib
import threading

import enfold.reflection as reflection


def _stop_background_worker(provider) -> None:
    if provider._queue_stop is not None:
        provider._queue_stop.set()
    if provider._queue_wake is not None:
        provider._queue_wake.set()
    if provider._queue_worker is not None:
        provider._queue_worker.join(timeout=1.0)


def test_reflection_insert_does_not_invert_writer_lock_order(
    make_provider, hp, monkeypatch
):
    provider = make_provider(
        extraction_provider="testprov",
        extraction_model="testmodel",
        reflection_enabled=True,
    )
    _stop_background_worker(provider)
    provider._store.add_fact("Alex Rivera moved to Springfield in March.")
    provider._store.add_fact("Alex Rivera started a new job at Skylark.")

    reflection_in_llm = threading.Event()
    writer_has_write_lock = threading.Event()
    reflection_waiting_for_write_lock = threading.Event()
    abort = threading.Event()
    write_lock = threading.Lock()

    @contextlib.contextmanager
    def coordinated_write_lock(db_path):
        is_writer = threading.current_thread().name == "store-writer"
        if not is_writer:
            reflection_waiting_for_write_lock.set()
        while not write_lock.acquire(timeout=0.01):
            if abort.is_set():
                raise RuntimeError("test aborted a lock-order deadlock")
        try:
            if is_writer:
                writer_has_write_lock.set()
                assert reflection_waiting_for_write_lock.wait(timeout=1.0)
            yield
        finally:
            write_lock.release()

    def reflect_on_cluster(facts, **kwargs):
        reflection_in_llm.set()
        assert writer_has_write_lock.wait(timeout=1.0)
        return {
            "insight": "Alex Rivera's move coincided with the Skylark role.",
            "source_fact_ids": [fact["fact_id"] for fact in facts],
        }

    monkeypatch.setattr(hp, "cross_process_write_lock", coordinated_write_lock)
    monkeypatch.setattr(reflection, "reflect_on_cluster", reflect_on_cluster)

    results = {}

    def run_reflection():
        results["reflection"] = provider.run_reflection(now=1_000_000.0)

    def write_fact():
        results["writer"] = provider._handle_fact_store({
            "action": "add",
            "content": "Jordan Blake keeps weekly planning notes.",
        })

    reflection_thread = threading.Thread(target=run_reflection, name="reflection")
    writer_thread = threading.Thread(target=write_fact, name="store-writer")
    reflection_thread.start()
    assert reflection_in_llm.wait(timeout=1.0)
    writer_thread.start()

    writer_thread.join(timeout=0.5)
    reflection_thread.join(timeout=0.5)
    completed_without_deadlock = (
        not writer_thread.is_alive() and not reflection_thread.is_alive()
    )
    abort.set()
    writer_thread.join(timeout=1.0)
    reflection_thread.join(timeout=1.0)

    assert completed_without_deadlock
    assert results["reflection"] == 1
    assert '"status": "added"' in results["writer"]
    row = provider._store._conn.execute(
        "SELECT fact_id FROM facts WHERE content = ?",
        ("Jordan Blake keeps weekly planning notes.",),
    ).fetchone()
    assert row is not None
