from __future__ import annotations

import json
import sqlite3

import pytest

from enfold.core_store import connect_database
from enfold.hybrid_retrieval import HybridRetriever
from enfold.schema import migrate

import memory_eval.baseline as baseline
from memory_eval.baseline import prepare_eval_db, production_like_config, resolve_cases


def test_production_like_config_points_to_supplied_db_and_disables_mutating_adds(tmp_path):
    db = tmp_path / "copy.db"
    cfg = production_like_config(db)

    assert cfg["db_path"] == str(db)
    assert cfg["retriever_mode"] == "stored"
    assert "embedding_weight" not in cfg
    assert "hrr_weight" not in cfg
    assert cfg["fts_weight"] + cfg["jaccard_weight"] + cfg["dense_weight"] == 1.0
    assert cfg["embed_on_add"] is False


def test_load_eval_retriever_builds_hybrid_retriever_not_enfold_provider(tmp_path):
    db = tmp_path / "eval.db"
    conn = connect_database(db)
    migrate(conn)
    conn.execute(
        "INSERT INTO facts(fact_id, content, category, tags, trust_score, scope, sensitivity, schema_version) "
        "VALUES (1, 'Avery prefers dark theme', 'user_pref', '', 0.9, 'private', 'normal', 1)"
    )
    conn.commit()
    conn.close()

    config = production_like_config(db)
    config["retriever_mode"] = "ci"
    config["allow_nonproduction"] = True
    provider = baseline.load_eval_retriever(db, config)
    try:
        assert isinstance(provider._retriever, HybridRetriever)
        assert type(provider).__name__ != "EnfoldProvider"
        rows = provider.search("What is known about prefers dark theme Avery?", limit=5, bump=False)
        assert isinstance(rows, list)
    finally:
        provider.shutdown()


def test_load_eval_retriever_stored_mode_refuses_without_production_identities(tmp_path):
    db = tmp_path / "eval.db"
    conn = connect_database(db)
    migrate(conn)
    conn.close()

    with pytest.raises(RuntimeError, match="will not fall back to EnfoldProvider"):
        baseline.load_eval_retriever(db, production_like_config(db))


def test_resolve_cases_loads_file_instead_of_generating(tmp_path):
    db = tmp_path / "unused.db"
    cases_file = tmp_path / "cases.json"
    cases_file.write_text('[{"id":"one","query":"q","gold_fact_id":1}]')

    cases = resolve_cases(db_path=db, cases_path=cases_file, sample=99, min_trust=0.0)

    assert len(cases) == 1
    assert cases[0].id == "one"


def test_prepare_eval_db_always_creates_a_backup_snapshot(tmp_path):
    src = tmp_path / "source.db"
    scratch = tmp_path / "scratch.db"
    conn = sqlite3.connect(src)
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO facts (content) VALUES ('source row')")
    conn.commit()
    conn.close()

    prepared = prepare_eval_db(src, scratch)

    assert prepared.path == scratch
    assert prepared.backup.source == src
    assert prepared.backup.destination == scratch
    assert prepared.backup.quick_check == "ok"

    with sqlite3.connect(src) as conn:
        conn.execute("INSERT INTO facts (content) VALUES ('after snapshot')")
        conn.commit()
    with sqlite3.connect(scratch) as conn:
        rows = conn.execute("SELECT content FROM facts ORDER BY fact_id").fetchall()
    assert rows == [("source row",)]


def test_run_baseline_uses_snapshot_path_for_provider_and_report(tmp_path, monkeypatch):
    src = tmp_path / "source.db"
    out = tmp_path / "report.json"
    scratch = tmp_path / "scratch.db"
    conn = sqlite3.connect(src)
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT, trust_score REAL)")
    conn.execute("INSERT INTO facts VALUES (1, 'source row', 'general', 0.9)")
    conn.commit()
    conn.close()

    class FakeProvider:
        def search(self, query, *, category=None, min_trust=0.3, limit=10, bump=False):
            assert bump is False
            return [{"fact_id": 1, "content": "source row"}]

        def shutdown(self):
            pass

    def fake_load_provider(repo_root, config, *, hermes_src, test_stubs):
        assert config["db_path"] == str(scratch)
        return FakeProvider()

    monkeypatch.setattr(baseline, "load_provider", fake_load_provider)

    result = baseline.run_baseline(
        db_path=src,
        out_path=out,
        scratch_db=scratch,
        sample=1,
        repo_root=tmp_path,
    )

    report = json.loads(out.read_text())
    assert result["metadata"]["db_path"] == str(scratch)
    assert report["metadata"]["snapshot"]["source"] == str(src)
    assert report["metadata"]["snapshot"]["destination"] == str(scratch)
    assert report["summary"]["recall@1"] == 1.0


def test_run_baseline_clears_pending_extract_queue_only_on_snapshot(tmp_path, monkeypatch):
    src = tmp_path / "source.db"
    out = tmp_path / "report.json"
    scratch = tmp_path / "scratch.db"
    conn = sqlite3.connect(src)
    conn.execute("CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT, trust_score REAL)")
    conn.execute("CREATE TABLE extract_queue (id INTEGER PRIMARY KEY, payload TEXT, status TEXT)")
    conn.execute("INSERT INTO facts VALUES (1, 'source row', 'general', 0.9)")
    conn.execute("INSERT INTO extract_queue VALUES (1, 'private transcript', 'pending')")
    conn.execute("INSERT INTO extract_queue VALUES (2, 'private retry transcript', 'retrying')")
    conn.commit()
    conn.close()

    class FakeProvider:
        def search(self, query, *, category=None, min_trust=0.3, limit=10, bump=False):
            return [{"fact_id": 1, "content": "source row"}]

        def shutdown(self):
            pass

    def fake_load_provider(repo_root, config, *, hermes_src, test_stubs):
        with sqlite3.connect(config["db_path"]) as conn:
            assert conn.execute("SELECT COUNT(*) FROM extract_queue").fetchone()[0] == 0
        return FakeProvider()

    monkeypatch.setattr(baseline, "load_provider", fake_load_provider)

    result = baseline.run_baseline(
        db_path=src,
        out_path=out,
        scratch_db=scratch,
        sample=1,
        repo_root=tmp_path,
    )

    with sqlite3.connect(src) as conn:
        assert conn.execute("SELECT COUNT(*) FROM extract_queue").fetchone()[0] == 2
    assert result["metadata"]["schema"]["extract_queue"]["pending"] == 2
    assert result["metadata"]["eval_safety"]["cleared_extract_queue_rows"] == 2
