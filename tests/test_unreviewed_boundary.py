from __future__ import annotations

import sqlite3

from enfold.context import pack_context
from enfold.hybrid_retrieval import DeterministicFeatureHashEmbedder, HybridRetriever
from enfold.policy import MemoryPolicy
from enfold.protocol import Request
from enfold.provenance import ConnectionContext
from enfold.schema import migrate
from enfold.service import EnfoldService


def _setup(tmp_path):
    conn = sqlite3.connect(tmp_path / "unreviewed-boundary.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    context = ConnectionContext(
        client_id="hermes-install",
        surface="hermes",
        agent_id="avery",
        session_id="boundary-session",
        access_scopes=("private", "work"),
    )
    service = EnfoldService(conn, MemoryPolicy({"hermes-install": ("private", "work")}))
    return conn, context, service


def test_null_correction_status_agent_write_is_returned_by_default_search(tmp_path):
    """Agent writes carry NULL correction_status and must keep flowing."""

    conn, context, service = _setup(tmp_path)
    written = service.handle(
        context,
        Request(
            "write-null",
            "memory.write",
            {
                "idempotency_key": "null-status",
                "content": "Dana keeps the Cedar runbook in the private vault.",
                "source_type": "agent_report",
            },
        ),
    )
    status = conn.execute(
        "SELECT correction_status FROM facts WHERE fact_id = ?",
        (written["fact_id"],),
    ).fetchone()[0]
    assert status is None

    found = service.handle(
        context,
        Request(
            "search-null",
            "memory.search",
            {"query": "Cedar runbook", "min_trust": 0},
        ),
    )["facts"]

    assert any(fact["fact_id"] == written["fact_id"] for fact in found)
    conn.close()


def _write_unreviewed(service, context, *, key, content):
    return service.handle(
        context,
        Request(
            f"write-{key}",
            "memory.write",
            {
                "idempotency_key": key,
                "content": content,
                "source_type": "automatic_extraction",
                "correction_status": "unreviewed",
            },
        ),
    )


def test_unreviewed_fact_is_excluded_from_default_search(tmp_path):
    conn, context, service = _setup(tmp_path)
    written = _write_unreviewed(
        service,
        context,
        key="unreviewed-hidden",
        content="Avery is the chief executive of the Cedar registry.",
    )
    assert written["outcome"] == "inserted"
    status = conn.execute(
        "SELECT correction_status FROM facts WHERE fact_id = ?",
        (written["fact_id"],),
    ).fetchone()[0]
    assert status == "unreviewed"

    found = service.handle(
        context,
        Request(
            "search-default",
            "memory.search",
            {"query": "chief executive", "min_trust": 0},
        ),
    )["facts"]

    assert found == []
    conn.close()


def test_unreviewed_fact_is_returned_only_with_include_unreviewed(tmp_path):
    conn, context, service = _setup(tmp_path)
    written = _write_unreviewed(
        service,
        context,
        key="unreviewed-opt-in",
        content="Avery is the chief executive of the Cedar registry.",
    )

    found = service.handle(
        context,
        Request(
            "search-opt-in",
            "memory.search",
            {
                "query": "chief executive",
                "min_trust": 0,
                "include_unreviewed": True,
            },
        ),
    )["facts"]

    assert [fact["fact_id"] for fact in found] == [written["fact_id"]]
    conn.close()


def test_human_confirmed_restores_default_recall(tmp_path):
    conn, context, service = _setup(tmp_path)
    written = _write_unreviewed(
        service,
        context,
        key="unreviewed-promote",
        content="Avery is the chief executive of the Cedar registry.",
    )
    conn.execute(
        "UPDATE facts SET correction_status = 'human_confirmed' WHERE fact_id = ?",
        (written["fact_id"],),
    )
    conn.commit()

    found = service.handle(
        context,
        Request(
            "search-promoted",
            "memory.search",
            {"query": "chief executive", "min_trust": 0},
        ),
    )["facts"]

    assert any(fact["fact_id"] == written["fact_id"] for fact in found)
    conn.close()


def test_hybrid_retriever_excludes_unreviewed_on_both_sql_paths(tmp_path):
    conn, _context, _service = _setup(tmp_path)
    conn.execute(
        """
        INSERT INTO facts (
            content, category, tags, trust_score, scope, sensitivity,
            correction_status
        ) VALUES
            ('Avery prefers tea in the Cedar kitchen.', 'general', '', 0.8,
             'private', 'normal', NULL),
            ('Avery is the chief executive of Cedar.', 'general', '', 0.8,
             'private', 'normal', 'unreviewed')
        """
    )
    conn.commit()
    retriever = HybridRetriever(
        conn,
        DeterministicFeatureHashEmbedder(),
        allowed_scopes=("private",),
        min_score=0.0,
        vector_backend="brute",
    )

    default_ids = {
        row["fact_id"]
        for row in retriever.search("Avery Cedar", min_trust=0)
    }
    opted_in = {
        row["fact_id"]
        for row in retriever.search(
            "Avery Cedar", min_trust=0, include_unreviewed=True
        )
    }
    statuses = {
        fact_id: status
        for fact_id, status in conn.execute(
            "SELECT fact_id, correction_status FROM facts"
        )
    }
    null_id = next(fid for fid, status in statuses.items() if status is None)
    unreviewed_id = next(
        fid for fid, status in statuses.items() if status == "unreviewed"
    )

    assert null_id in default_ids
    assert unreviewed_id not in default_ids
    assert {null_id, unreviewed_id} <= opted_in
    conn.close()


def test_pack_context_redacts_explicit_unreviewed_status():
    fact = {
        "fact_id": 10,
        "content": "Avery is the chief executive of Cedar.",
        "score": 0.99,
        "memory_kind": None,
        "subject_key": None,
        "predicate_key": None,
        "scope": "private",
        "invalid_at": None,
        "superseded_by": None,
        "conflict_group": None,
        "correction_status": "unreviewed",
        "attribution": {"performed_by": "extractor", "agent_id": "extractor"},
    }

    pack = pack_context([fact], token_budget=256)

    assert "chief executive" not in pack.markdown
    assert pack.facts == (
        {
            "fact_id": 10,
            "prompt_eligible": False,
            "content_omitted": True,
            "exclusion_reason": "unreviewed_content",
        },
    )
