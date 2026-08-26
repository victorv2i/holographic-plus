from __future__ import annotations

import json
import sqlite3

import pytest

from enfold.schema import migrate
from memory_eval.personal_arena import run_personal_arena


def _database(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    migrate(conn)
    conn.executemany(
        """
        INSERT INTO facts(
            fact_id, content, category, tags, trust_score, scope, sensitivity,
            schema_version, superseded_by
        ) VALUES (?, ?, ?, '', 0.9, 'private', 'normal', 1, ?)
        """,
        [
            (1, "The Atlas launch date is 2026-07-18.", "project", None),
            (2, "The retired Atlas launch date was 2026-06-30.", "project", 1),
            (3, "Mina owns the Atlas release checklist.", "project", None),
        ],
    )
    conn.commit()
    return conn


def _write_cases(path):
    rows = [
        {
            "id": "atlas-date",
            "query": "When is Atlas scheduled to launch?",
            "expected_fact_ids": [1],
            "forbidden_content_regexes": ["retired Atlas"],
            "category": "project",
        },
        {
            "id": "unknown",
            "query": "What is the approved budget for Project Zaffre?",
            "category": "project",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_personal_arena_hash_run_is_not_a_retrieval_claim(tmp_path):
    source = tmp_path / "live.db"
    _database(source).close()
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(cases_path)

    run = run_personal_arena(cases_path, source, dimensions=64)

    assert run.metadata["embedder_mode"] == "hash"
    assert run.metadata["claimable_retrieval"] is False
    assert run.metadata["retrieval"]["embedder_production_ready"] is False


def test_personal_arena_production_mode_refuses_without_identities(tmp_path):
    source = tmp_path / "live.db"
    _database(source).close()
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(cases_path)

    with pytest.raises(RuntimeError, match="will not fall back"):
        run_personal_arena(cases_path, source, embedder_mode="production")


def test_personal_arena_production_mode_does_not_fall_back_to_hash(tmp_path):
    source = tmp_path / "live.db"
    _database(source).close()
    cases_path = tmp_path / "cases.jsonl"
    _write_cases(cases_path)
    config = {
        "provider": "ollama",
        "model": "embeddinggemma:latest",
        "dimensions": 768,
        "query_identity": "ollama:embeddinggemma:latest:query:none:v1",
        "document_identity": "ollama:embeddinggemma:latest:document:none:v1",
        "embedding_version": "v1",
    }

    with pytest.raises(RuntimeError, match="will not fall back"):
        run_personal_arena(
            cases_path,
            source,
            embedder_mode="production",
            embedder_config=config,
        )
