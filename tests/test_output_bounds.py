from __future__ import annotations

import json
import sqlite3

from enfold.context import TOKEN_ESTIMATE_METHOD, estimate_tokens
from enfold.core_store import insert_fact
from enfold.policy import MemoryPolicy
from enfold.protocol import ClientContext, Request
from enfold.schema import migrate
from enfold.service import EnfoldService, OutputBounds, TRUNCATION_MARKER


class RecordingRetriever:
    metadata = {"retrieval_stack": "bounds-fixture"}

    def __init__(self, rows):
        self._rows = rows
        self.calls = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return [dict(row) for row in self._rows[:kwargs["limit"]]]


def _service(tmp_path, rows, bounds):
    conn = sqlite3.connect(tmp_path / "output-bounds.db")
    migrate(conn)
    retriever = RecordingRetriever(rows)
    service = EnfoldService(
        conn,
        MemoryPolicy({"bounds-client": ("private",)}),
        retriever_factory=lambda _conn, _scopes: retriever,
        output_bounds=bounds,
    )
    context = ClientContext(
        client_id="bounds-client",
        surface="client-a",
        agent_id="worker",
        session_id="bounds-session",
        access_scopes=("private",),
    )
    return conn, service, context, retriever


def _row(fact_id, content, *, trust=0.9):
    return {
        "fact_id": fact_id,
        "content": content,
        "category": "general",
        "tags": "bounds",
        "trust_score": trust,
        "created_at": "2026-07-12 12:00:00",
        "updated_at": "2026-07-12 12:00:00",
        "memory_kind": None,
        "scope": "private",
        "invalid_at": None,
        "superseded_by": None,
        "conflict_group": None,
        "correction_status": "human_confirmed",
        "score": 0.9 - fact_id / 100,
    }


def _request(request_id, method, **params):
    return Request(request_id, method, params)


def _write(service, context, key, content, **params):
    return service.handle(
        context,
        _request(
            f"write-{key}",
            "memory.write",
            idempotency_key=key,
            content=content,
            source_type="agent_report",
            **params,
        ),
    )


def test_search_defaults_filter_trust_but_explicit_zero_is_preserved(tmp_path):
    bounds = OutputBounds()
    conn, service, context, retriever = _service(
        tmp_path, [_row(1, "low trust", trust=0.1)], bounds
    )

    service.handle(context, _request("default", "memory.search", query="trust"))
    service.handle(
        context,
        _request("explicit", "memory.search", query="trust", min_trust=0),
    )
    service.handle(
        context,
        _request("context-default", "memory.context", query="trust", token_budget=64),
    )
    service.handle(
        context,
        _request("context", "memory.context", query="trust", token_budget=64, min_trust=0),
    )

    assert retriever.calls[0][1]["min_trust"] == bounds.default_min_trust
    assert retriever.calls[1][1]["min_trust"] == 0
    assert retriever.calls[2][1]["min_trust"] == bounds.default_min_trust
    assert retriever.calls[3][1]["min_trust"] == 0
    conn.close()


def test_search_caps_results_fact_content_and_total_serialized_chars(tmp_path):
    bounds = OutputBounds(
        search_max_results=2,
        max_fact_chars=48,
        search_max_total_chars=900,
    )
    rows = [_row(index, f"fact-{index} " + "x" * 400) for index in range(1, 5)]
    conn, service, context, retriever = _service(tmp_path, rows, bounds)

    result = service.handle(
        context,
        _request("bounded", "memory.search", query="fact", limit=200, min_trust=0),
    )

    assert retriever.calls[0][1]["limit"] == 3
    assert len(result["facts"]) <= 2
    assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) <= 900
    assert all(len(fact["content"]) <= 48 for fact in result["facts"])
    assert all(fact["content"].endswith(TRUNCATION_MARKER) for fact in result["facts"])
    assert all(fact["content_truncated"] is True for fact in result["facts"])
    assert result["output_truncated"] is True
    conn.close()


def test_context_uses_chars_per_four_estimate_and_caps_full_payload(tmp_path):
    bounds = OutputBounds(
        context_max_results=2,
        max_fact_chars=64,
        context_max_total_chars=1200,
    )
    rows = [_row(index, f"context-{index} " + "y" * 800) for index in range(1, 5)]
    conn, service, context, retriever = _service(tmp_path, rows, bounds)

    result = service.handle(
        context,
        _request(
            "bounded-context",
            "memory.context",
            query="context",
            token_budget=256,
            min_trust=0,
        ),
    )

    assert TOKEN_ESTIMATE_METHOD == "unicode_chars_divided_by_four"
    assert estimate_tokens("abcdefgh") == 2
    assert retriever.calls[0][1]["limit"] == 9
    assert len(result["facts"]) <= 2
    assert len(json.dumps(result, ensure_ascii=False, separators=(",", ":"))) <= 1200
    assert result["markdown"].endswith(TRUNCATION_MARKER + "\n")
    assert result["facts"][0]["content"].endswith(TRUNCATION_MARKER)
    assert result["facts"][0]["context_truncated"] is True
    assert result["output_truncated"] is True
    conn.close()


def test_evidence_caps_full_utf8_payload_and_marks_truncated_items(tmp_path):
    bounds = OutputBounds(
        max_fact_chars=96,
        evidence_max_total_bytes=1_800,
    )
    conn, service, context, _retriever = _service(tmp_path, [], bounds)
    content = "é" * 1_000
    first = _write(service, context, "evidence-1", content, source_uri="commit:a")
    _write(service, context, "evidence-2", content, source_uri="commit:b")
    _write(service, context, "evidence-3", content, source_uri="commit:c")

    result = service.handle(
        context,
        _request("bounded-evidence", "memory.evidence", fact_id=first["fact_id"]),
    )

    payload_bytes = len(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    assert payload_bytes <= bounds.evidence_max_total_bytes
    assert 0 < len(result["evidence"]) < 3
    assert result["fact"]["content"].endswith(TRUNCATION_MARKER)
    assert result["fact"]["content_truncated"] is True
    assert all(
        item["content"].endswith(TRUNCATION_MARKER) for item in result["evidence"]
    )
    assert all(item["content_truncated"] is True for item in result["evidence"])
    assert result["output_truncated"] is True
    conn.close()


def test_history_caps_full_utf8_payload_and_marks_truncated_items(tmp_path):
    bounds = OutputBounds(
        max_fact_chars=96,
        history_max_total_bytes=1_500,
    )
    conn, service, context, _retriever = _service(tmp_path, [], bounds)
    state = {"subject_key": "agent:worker", "predicate_key": "current_task"}
    for index in range(3):
        _write(
            service,
            context,
            f"history-{index}",
            f"task-{index} " + "界" * 1_000,
            source_authority=0.8,
            state={
                **state,
                "object_value": f"task-{index}",
                "valid_from": f"2026-07-{index + 1:02d}T12:00:00Z",
            },
        )

    result = service.handle(
        context,
        _request("bounded-history", "memory.history", **state, scope="private"),
    )

    payload_bytes = len(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    assert payload_bytes <= bounds.history_max_total_bytes
    assert 0 < len(result["facts"]) < 3
    assert all(fact["content"].endswith(TRUNCATION_MARKER) for fact in result["facts"])
    assert all(fact["content_truncated"] is True for fact in result["facts"])
    assert result["output_truncated"] is True
    conn.close()


def test_evidence_stops_materializing_rows_when_byte_budget_is_exhausted(
    tmp_path,
):
    bounds = OutputBounds(
        max_fact_chars=48,
        evidence_max_total_bytes=900,
    )
    conn, service, context, _retriever = _service(tmp_path, [], bounds)
    first = _write(service, context, "stream-evidence-0", "bounded evidence")
    for index in range(40):
        _write(
            service,
            context,
            f"stream-evidence-{index + 1}",
            "bounded evidence",
            source_uri=f"commit:{index}",
        )

    materialized = 0

    def counting_rows(cursor, row):
        nonlocal materialized
        materialized += 1
        return sqlite3.Row(cursor, row)

    conn.row_factory = counting_rows
    result = service.handle(
        context,
        _request("stream-evidence", "memory.evidence", fact_id=first["fact_id"]),
    )

    assert result["output_truncated"] is True
    assert materialized < 10
    conn.close()


def test_state_history_stops_materializing_rows_at_byte_budget(tmp_path):
    bounds = OutputBounds(
        max_fact_chars=48,
        history_max_total_bytes=700,
    )
    conn, service, context, _retriever = _service(tmp_path, [], bounds)
    state = {"subject_key": "agent:worker", "predicate_key": "streamed_task"}
    for index in range(30):
        _write(
            service,
            context,
            f"stream-history-{index}",
            f"task-{index} " + "x" * 200,
            source_authority=0.8,
            state={
                **state,
                "object_value": f"task-{index}",
                "valid_from": f"2026-06-{index + 1:02d}T12:00:00Z",
            },
        )

    materialized = 0

    def counting_rows(cursor, row):
        nonlocal materialized
        materialized += 1
        return sqlite3.Row(cursor, row)

    conn.row_factory = counting_rows
    result = service.handle(
        context,
        _request("stream-history", "memory.history", **state, scope="private"),
    )

    assert result["output_truncated"] is True
    assert materialized < 10
    conn.close()


def test_non_state_history_limits_the_supersession_chain_query(tmp_path):
    conn, service, context, _retriever = _service(tmp_path, [], OutputBounds())
    fact_ids = [
        insert_fact(conn, f"chain node {index}", scope="private")
        for index in range(30)
    ]
    for index, fact_id in enumerate(fact_ids):
        conn.execute(
            "UPDATE facts SET created_at = ?, updated_at = ? WHERE fact_id = ?",
            (f"2026-06-{index + 1:02d}T12:00:00Z",) * 2 + (fact_id,),
        )
    for current, following in zip(fact_ids, fact_ids[1:]):
        conn.execute(
            "UPDATE facts SET invalid_at = CURRENT_TIMESTAMP, superseded_by = ? "
            "WHERE fact_id = ?",
            (following, current),
        )
    conn.commit()
    statements = []
    conn.set_trace_callback(statements.append)

    result = service.handle(
        context,
        _request(
            "limited-chain", "memory.history", fact_id=fact_ids[0], limit=2
        ),
    )

    conn.set_trace_callback(None)
    assert [fact["fact_id"] for fact in result["facts"]] == fact_ids[:2]
    assert result["output_truncated"] is True
    chain_reads = [
        statement
        for statement in statements
        if "FROM facts WHERE fact_id =" in statement
        or "WITH RECURSIVE chain" in statement
    ]
    assert len(chain_reads) == 2
    assert chain_reads[-1].count("LIMIT 3") == 2
    conn.close()


def test_conflicts_cap_members_per_selected_conflict(tmp_path):
    bounds = OutputBounds()
    conn, service, context, _retriever = _service(tmp_path, [], bounds)
    conn.execute(
        "INSERT INTO fact_conflicts("
        "conflict_id, scope, subject_key, predicate_key, detected_at"
        ") VALUES ('wide-conflict', 'private', 'agent:worker', 'setting', "
        "'2026-07-20T00:00:00Z')"
    )
    fact_ids = [
        insert_fact(conn, f"conflict member {index}", scope="private")
        for index in range(8)
    ]
    conn.executemany(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) "
        "VALUES ('wide-conflict', ?)",
        ((fact_id,) for fact_id in fact_ids),
    )
    conn.commit()

    result = service.handle(
        context,
        _request("bounded-conflict-members", "memory.conflicts", limit=1),
    )

    assert result["conflicts"][0]["member_fact_ids"] == fact_ids[:1]
    assert [
        member["fact_id"] for member in result["conflicts"][0]["members"]
    ] == fact_ids[:1]
    assert result["conflicts"][0]["members_truncated"] is True
    assert result["output_truncated"] is True
    conn.close()


def test_conflicts_stop_adding_members_at_byte_budget(tmp_path):
    bounds = OutputBounds(
        max_fact_chars=256,
        conflicts_max_total_bytes=900,
    )
    conn, service, context, _retriever = _service(tmp_path, [], bounds)
    conn.execute(
        "INSERT INTO fact_conflicts("
        "conflict_id, scope, subject_key, predicate_key, detected_at"
        ") VALUES ('large-conflict', 'private', 'agent:worker', 'setting', "
        "'2026-07-20T00:00:00Z')"
    )
    fact_ids = [
        insert_fact(conn, f"member {index} " + "é" * 400, scope="private")
        for index in range(8)
    ]
    conn.executemany(
        "INSERT INTO fact_conflict_members(conflict_id, fact_id) "
        "VALUES ('large-conflict', ?)",
        ((fact_id,) for fact_id in fact_ids),
    )
    conn.commit()

    result = service.handle(
        context,
        _request("byte-bounded-conflict", "memory.conflicts", limit=8),
    )

    conflict = result["conflicts"][0]
    assert len(conflict["members"]) < len(fact_ids)
    assert conflict["member_fact_ids"] == [
        member["fact_id"] for member in conflict["members"]
    ]
    assert conflict["members_truncated"] is True
    assert result["output_truncated"] is True
    assert len(json.dumps(result, ensure_ascii=False).encode("utf-8")) <= 900
    conn.close()
