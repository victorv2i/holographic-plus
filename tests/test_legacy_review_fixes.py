import json
import os
import select
import sqlite3
import threading
import time
import types

import numpy as np
import pytest

from enfold import _is_semantic_duplicate
from enfold.extract_queue import ExtractQueue
from enfold.llm_extract import insert_facts
from enfold.schema import migrate
from enfold.write_lock import cross_process_write_lock


def _llm_response(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


def _stop_provider_worker(provider):
    provider._queue_stop.set()
    provider._queue_wake.set()
    provider._queue_worker.join(timeout=1)
    assert not provider._queue_worker.is_alive()


def test_semantic_dedup_keeps_content_value_changes_and_true_paraphrases():
    assert not _is_semantic_duplicate(
        "Avery prefers tea.", "Avery prefers coffee.", 1.0, 0.92
    )
    assert _is_semantic_duplicate(
        "The user prefers Postgres over MySQL for new projects.",
        "For new work the user always reaches for Postgres instead of MySQL.",
        1.0,
        0.92,
    )


def test_add_path_stores_changed_nonnumeric_preference(make_provider):
    old = "Avery prefers tea."
    new = "Avery prefers coffee."
    vector = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    provider = make_provider(dedup_cosine=0.92)
    old_id = provider._store.add_fact(old, category="user_pref")
    provider._fake_embedder.table[new] = vector
    provider._embed_store.upsert(
        old_id,
        vector,
        embedding_identity=provider._embedding_identity("document"),
    )

    result = json.loads(
        provider._handle_fact_store(
            {"action": "add", "content": new, "category": "user_pref"}
        )
    )

    assert result["status"] == "added"
    assert result["fact_id"] != old_id


def test_interactive_supersede_failure_is_returned_to_caller(
    make_provider, monkeypatch
):
    provider = make_provider()
    first = json.loads(
        provider._handle_fact_store(
            {
                "action": "add",
                "content": "The Skylark dashboard port is 3100.",
                "category": "project",
            }
        )
    )

    def fail_supersede(*_args):
        raise sqlite3.OperationalError("forced supersede failure")

    monkeypatch.setattr(provider, "_supersede_fact", fail_supersede)
    result = json.loads(
        provider._handle_fact_store(
            {
                "action": "add",
                "content": "The Skylark dashboard port is 3200.",
                "category": "project",
            }
        )
    )

    assert result["status"] == "supersede_failed"
    assert "forced supersede failure" in result["error"]
    assert result["fact_id"] != first["fact_id"]
    assert result["rolled_back"] is False
    assert result["compensated"] is True
    active = provider._store._conn.execute(
        "SELECT fact_id FROM facts WHERE invalid_at IS NULL ORDER BY fact_id"
    ).fetchall()
    assert [row["fact_id"] for row in active] == [first["fact_id"]]


def test_interactive_add_rolls_back_with_transaction_aware_storage(
    make_provider, monkeypatch
):
    provider = make_provider()
    old_id = provider._store.add_fact(
        "The Skylark dashboard port is 3100.", category="project"
    )
    conn = provider._store._conn

    def transaction_aware_add(content, category="general", tags=""):
        cur = conn.execute(
            "INSERT INTO facts (content, category, tags, trust_score) "
            "VALUES (?, ?, ?, ?)",
            (content, category, tags, provider._store.default_trust),
        )
        return int(cur.lastrowid)

    def fail_inside_transaction(old_fact_id, new_fact_id):
        conn.execute(
            "UPDATE facts SET invalid_at = CURRENT_TIMESTAMP, superseded_by = ? "
            "WHERE fact_id = ?",
            (new_fact_id, old_fact_id),
        )
        raise sqlite3.OperationalError("forced transactional failure")

    monkeypatch.setattr(provider._store, "add_fact", transaction_aware_add)
    monkeypatch.setattr(provider, "_supersede_fact", fail_inside_transaction)
    result = json.loads(
        provider._handle_fact_store(
            {
                "action": "add",
                "content": "The Skylark dashboard port is 3200.",
                "category": "project",
            }
        )
    )

    assert result["status"] == "supersede_failed"
    assert result["rolled_back"] is True
    rows = conn.execute(
        "SELECT fact_id, invalid_at, superseded_by FROM facts ORDER BY fact_id"
    ).fetchall()
    assert [row["fact_id"] for row in rows] == [old_id]
    assert rows[0]["invalid_at"] is None
    assert rows[0]["superseded_by"] is None


def test_schema_probe_encodes_sqlite_uri_metacharacters(make_provider, tmp_path):
    db_path = tmp_path / "facts?#%.db"
    conn = sqlite3.connect(str(db_path))
    try:
        migrate(conn)
    finally:
        conn.close()

    provider = make_provider(init=False, db_path=str(db_path))
    with pytest.raises(RuntimeError, match="schema-v0 writer"):
        provider.initialize("legacy-writer")
    assert provider._store is None


def test_write_lock_blocks_other_threads_but_remains_reentrant(tmp_path):
    db_path = str(tmp_path / "threaded.db")
    entered = threading.Event()
    release = threading.Event()
    second_entered = threading.Event()

    def owner():
        with cross_process_write_lock(db_path):
            with cross_process_write_lock(db_path):
                entered.set()
                release.wait(timeout=2)

    def contender():
        entered.wait(timeout=2)
        with cross_process_write_lock(db_path):
            second_entered.set()

    first = threading.Thread(target=owner)
    second = threading.Thread(target=contender)
    first.start()
    second.start()
    assert entered.wait(timeout=1)
    time.sleep(0.05)
    assert not second_entered.is_set()
    release.set()
    first.join(timeout=1)
    second.join(timeout=1)
    assert second_entered.is_set()


def test_write_lock_does_not_suppress_protected_exception(tmp_path):
    with pytest.raises(RuntimeError, match="protected failure"):
        with cross_process_write_lock(str(tmp_path / "exception.db")):
            raise RuntimeError("protected failure")


def test_write_lock_resets_inherited_ownership_after_fork(tmp_path):
    if not hasattr(os, "fork"):
        pytest.skip("requires os.fork")

    db_path = str(tmp_path / "forked.db")
    read_fd, write_fd = os.pipe()
    child_pid = None
    child_failed = False
    try:
        with cross_process_write_lock(db_path):
            child_pid = os.fork()
            if child_pid == 0:
                os.close(read_fd)
                try:
                    with cross_process_write_lock(db_path):
                        os.write(write_fd, b"acquired")
                except BaseException:
                    child_failed = True
            else:
                os.close(write_fd)
                assert select.select([read_fd], [], [], 0.2)[0] == []

        if child_pid == 0:
            os.close(write_fd)
            os._exit(int(child_failed))

        assert select.select([read_fd], [], [], 2.0)[0] == [read_fd]
        assert os.read(read_fd, 8) == b"acquired"
    finally:
        os.close(read_fd)
        if child_pid:
            _, status = os.waitpid(child_pid, 0)
            assert os.waitstatus_to_exitcode(status) == 0


def test_malformed_extraction_marks_queue_row_retryable(
    make_provider, aux_module
):
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel"
    )
    _stop_provider_worker(provider)
    aux_module.call_llm = lambda **_kwargs: _llm_response('{"content": "not a list"}')
    row_id = provider._extract_queue.enqueue(
        [{"role": "user", "content": "durable but malformed extraction"}]
    )
    stop = threading.Event()
    original_mark_failed = provider._extract_queue.mark_failed

    def mark_failed_once(*args, **kwargs):
        attempts = original_mark_failed(*args, **kwargs)
        stop.set()
        return attempts

    provider._extract_queue.mark_failed = mark_failed_once
    provider._drain_extract_queue(stop, provider._extract_queue)

    row = provider._store._conn.execute(
        "SELECT status, attempts FROM extract_queue WHERE id = ?", (row_id,)
    ).fetchone()
    assert tuple(row) == ("pending", 1)


def test_partial_insert_retry_replays_saved_proposals_without_model_recall(
    make_provider, aux_module
):
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel"
    )
    _stop_provider_worker(provider)
    calls = []
    facts = [
        {"content": "The first durable replay fact is retained.", "category": "general", "tags": "replay"},
        {"content": "The second durable replay fact is retained.", "category": "general", "tags": "replay"},
    ]

    def extract_once(**_kwargs):
        calls.append(1)
        return _llm_response(json.dumps(facts))

    aux_module.call_llm = extract_once
    original_add = provider._store.add_fact
    failed_once = False

    def flaky_add(content, **kwargs):
        nonlocal failed_once
        if content == facts[1]["content"] and not failed_once:
            failed_once = True
            raise sqlite3.OperationalError("forced partial insert")
        return original_add(content, **kwargs)

    provider._store.add_fact = flaky_add
    row_id = provider._extract_queue.enqueue(
        [{"role": "user", "content": "two durable replay facts"}]
    )
    stop = threading.Event()
    original_mark_failed = provider._extract_queue.mark_failed

    def mark_failed_once(*args, **kwargs):
        attempts = original_mark_failed(*args, **kwargs)
        stop.set()
        return attempts

    provider._extract_queue.mark_failed = mark_failed_once
    provider._drain_extract_queue(stop, provider._extract_queue)
    snapshot = provider._store._conn.execute(
        "SELECT proposal_json FROM extract_queue WHERE id = ?", (row_id,)
    ).fetchone()[0]
    assert json.loads(snapshot) == facts

    provider._store.add_fact = original_add
    provider._extract_queue.mark_failed = original_mark_failed
    provider._drain_extract_queue(threading.Event(), provider._extract_queue)

    assert calls == [1]
    assert provider._extract_queue.pending_count() == 0
    stored = {
        row["content"]
        for row in provider._store.list_facts(min_trust=0.0, limit=20)
    }
    assert {fact["content"] for fact in facts} <= stored


def test_legacy_extraction_drops_credential_shaped_proposals(make_provider):
    provider = make_provider()
    result = insert_facts(
        provider._store,
        [
            {
                "content": "The deployment api_key = abcdefghijklmnopqrstuv.",
                "category": "tool",
                "tags": "deploy",
            }
        ],
    )

    assert result.inserted == 0
    assert result.skipped == 1
    assert provider._store.list_facts(min_trust=0.0, limit=10) == []


def test_legacy_dead_letter_snapshot_omits_credential_proposals(tmp_path):
    conn = sqlite3.connect(tmp_path / "screened-snapshot.db")
    queue = ExtractQueue(conn)
    row_id = queue.enqueue("legacy credential transcript")
    claimed = queue.next_pending(max_attempts=1, lease_owner="worker")
    safe = {
        "content": "Avery prefers local tools.",
        "category": "user_pref",
        "tags": "local,tools",
    }
    secret = "api_key = abcdefghijklmnopqrstuv"

    assert queue.save_proposals(
        row_id,
        [
            safe,
            {"content": secret, "category": "tool", "tags": "deploy"},
        ],
        claimed["lease_owner"],
    )
    assert queue.mark_failed(
        row_id,
        "insert failed",
        max_attempts=1,
        lease_owner=claimed["lease_owner"],
    ) == 1

    proposal_json, status = conn.execute(
        "SELECT proposal_json, status FROM extract_queue WHERE id = ?", (row_id,)
    ).fetchone()
    assert json.loads(proposal_json) == [safe]
    assert secret not in proposal_json
    assert status == "dead"
    conn.close()


def test_extraction_value_update_is_skipped_and_existing_fact_stays_current(
    make_provider, caplog
):
    provider = make_provider()
    old_id = provider._store.add_fact(
        "The Skylark dashboard port is 3100.", category="project"
    )
    proposal = {
        "content": "The Skylark dashboard port is 3200.",
        "category": "project",
        "tags": "skylark,port",
    }
    calls = []

    def supersede_once(old_fact_id, new_fact_id):
        calls.append((old_fact_id, new_fact_id))
        if len(calls) == 1:
            return False
        return provider._supersede_fact(old_fact_id, new_fact_id)

    first = insert_facts(
        provider._store,
        [proposal],
        dedup_check=provider._find_near_duplicate,
        update_check=provider._find_update_target,
        supersede=supersede_once,
    )
    assert first.inserted == 0
    assert first.skipped == 1
    assert first.failed == 0
    assert calls == []
    assert "reported failure" not in caplog.text

    second = insert_facts(
        provider._store,
        [proposal],
        dedup_check=provider._find_near_duplicate,
        update_check=provider._find_update_target,
        supersede=supersede_once,
    )
    assert second.failed == 0
    assert second.skipped == 1
    assert calls == []
    old = provider._store._conn.execute(
        "SELECT superseded_by FROM facts WHERE fact_id = ?", (old_id,)
    ).fetchone()
    assert old["superseded_by"] is None


@pytest.mark.parametrize("method", ["mark_failed", "mark_quota_failed"])
def test_stale_failure_update_cannot_clobber_reclaimed_lease(
    tmp_path, method
):
    db_path = tmp_path / f"{method}.db"
    first_conn = sqlite3.connect(str(db_path), check_same_thread=False)
    queue = ExtractQueue(first_conn)
    row_id = queue.enqueue("leased payload")
    claimed = queue.next_pending(max_attempts=5, lease_owner="worker-a")
    second_conn = sqlite3.connect(str(db_path), check_same_thread=False)

    def reclaim_during_check(_created_at):
        second_conn.execute(
            "UPDATE extract_queue SET lease_owner = ?, attempts = ? WHERE id = ?",
            ("worker-b", 7, row_id),
        )
        second_conn.commit()
        return False

    queue._age_exceeded = reclaim_during_check
    try:
        if method == "mark_failed":
            assert queue.mark_failed(
                row_id, "stale", max_attempts=10, lease_owner=claimed["lease_owner"]
            ) == 0
        else:
            assert queue.mark_quota_failed(
                row_id, "429 quota", time.time() + 60, claimed["lease_owner"]
            ) is False
        row = second_conn.execute(
            "SELECT status, lease_owner, attempts, last_error FROM extract_queue WHERE id = ?",
            (row_id,),
        ).fetchone()
        assert tuple(row) == ("processing", "worker-b", 7, None)
    finally:
        second_conn.close()
        first_conn.close()


@pytest.mark.parametrize(
    "operation",
    [
        "mark_done",
        "save_proposals",
        "release_claim",
        "mark_failed",
        "mark_quota_failed",
    ],
)
def test_same_owner_stale_lease_incarnation_cannot_mutate_reclaimed_row(
    tmp_path, operation
):
    conn = sqlite3.connect(tmp_path / f"same-owner-{operation}.db")
    queue = ExtractQueue(conn)
    row_id = queue.enqueue("same owner ABA payload")
    stale = queue.next_pending(
        max_attempts=5, lease_owner="stable-worker", lease_seconds=0
    )
    current = queue.next_pending(
        max_attempts=5, lease_owner="stable-worker", lease_seconds=60
    )

    if operation == "mark_done":
        assert queue.mark_done(row_id, stale["lease_owner"]) is False
        assert queue.mark_done(row_id, current["lease_owner"]) is True
    elif operation == "save_proposals":
        proposals = [
            {
                "content": "Avery prefers local tools.",
                "category": "user_pref",
                "tags": "local,tools",
            }
        ]
        assert queue.save_proposals(
            row_id, proposals, stale["lease_owner"]
        ) is False
        assert queue.save_proposals(
            row_id, proposals, current["lease_owner"]
        ) is True
    else:
        if operation == "release_claim":
            assert queue.release_claim(row_id, stale["lease_owner"]) is False
        elif operation == "mark_failed":
            assert queue.mark_failed(
                row_id, "stale failure", 5, stale["lease_owner"]
            ) == 0
        else:
            assert queue.mark_quota_failed(
                row_id, "stale quota", time.time() + 60, stale["lease_owner"]
            ) is False

        row = conn.execute(
            "SELECT status, lease_owner, lease_until, attempts, last_error "
            "FROM extract_queue WHERE id = ?",
            (row_id,),
        ).fetchone()
        assert tuple(row) == (
            "processing",
            "stable-worker",
            current["lease_until"],
            0,
            None,
        )

        if operation == "release_claim":
            assert queue.release_claim(row_id, current["lease_owner"]) is True
        elif operation == "mark_failed":
            assert queue.mark_failed(
                row_id, "current failure", 5, current["lease_owner"]
            ) == 1
        else:
            assert queue.mark_quota_failed(
                row_id, "current quota", time.time() + 60, current["lease_owner"]
            ) is True
    conn.close()


def test_plain_owner_string_remains_unfenced_legacy_compatibility(tmp_path):
    conn = sqlite3.connect(tmp_path / "plain-owner-compatibility.db")
    queue = ExtractQueue(conn)
    row_id = queue.enqueue("legacy owner payload")
    queue.next_pending(max_attempts=5, lease_owner="legacy-worker")

    assert queue.mark_done(row_id, "legacy-worker") is True
    conn.close()
