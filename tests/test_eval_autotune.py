from __future__ import annotations

import json
from inspect import signature

from enfold.core_store import connect_database
from enfold.hybrid_retrieval import HybridRetriever, RankingConfig
from enfold.schema import migrate

from memory_eval.autotune import (
    TrialScore,
    _base_eval_config,
    _eligible_knobs,
    _is_better,
    _normalize_inactive_knobs,
    _parse_scalar,
    run_autotune,
)


def test_parse_scalar_matches_flat_yaml_fallback_needs():
    assert _parse_scalar("true") is True
    assert _parse_scalar("False") is False
    assert _parse_scalar("null") is None
    assert _parse_scalar("42") == 42
    assert _parse_scalar("0.45") == 0.45
    assert _parse_scalar('"embeddinggemma"') == "embeddinggemma"


def test_base_eval_config_uses_live_retrieval_values_but_forces_safe_eval_flags(tmp_path):
    db = tmp_path / "scratch.db"
    config = _base_eval_config(
        {
            "fts_weight": 0.5,
            "jaccard_weight": 0.2,
            "dense_weight": 0.3,
            "embed_on_add": True,
            "dedup_on_add": True,
            "reflection_enabled": True,
            "extract_drain_batch": 10,
        },
        db,
    )

    assert config["db_path"] == str(db)
    assert config["retriever_mode"] == "stored"
    assert config["fts_weight"] == 0.5
    assert "embedding_weight" not in config
    assert "hrr_weight" not in config
    assert config["embed_on_add"] is False
    assert config["dedup_on_add"] is False
    assert config["reflection_enabled"] is False
    assert config["extract_drain_batch"] == 0


def test_eligible_knobs_map_only_to_hybrid_retriever_fields():
    knobs = _eligible_knobs({"dense_embeddings": True}, {})
    names = {spec.key for spec in knobs}
    ranking_fields = set(RankingConfig.__dataclass_fields__)
    retriever_params = set(signature(HybridRetriever.__init__).parameters) - {"self"}
    allowed = ranking_fields | retriever_params

    assert "embedding_weight" not in names
    assert "hrr_weight" not in names
    assert "entity_boost_weight" not in names
    assert names <= allowed
    assert "fts_weight" in names
    assert "dense_weight" in names
    assert "recency_half_life_days" in names


def test_is_better_rejects_any_stale_leak_increase_over_baseline():
    baseline = TrialScore(0.6, 0, 0.0, 20.0)
    incumbent = TrialScore(0.6, 0, 0.0, 20.0)
    challenger = TrialScore(0.8, 1, 0.1, 10.0)

    accepted, decision = _is_better(challenger, incumbent, baseline)

    assert accepted is False
    assert "stale_leak@1 increased" in decision


def test_normalize_inactive_knobs_does_not_revive_legacy_provider_keys():
    baseline = {
        "fts_weight": 0.35,
        "jaccard_weight": 0.25,
        "dense_weight": 0.40,
    }
    config = {
        "fts_weight": 0.5,
        "jaccard_weight": 0.2,
        "dense_weight": 0.3,
    }

    normalized = _normalize_inactive_knobs(
        config,
        baseline,
        active_backend={"dense_embeddings": True},
    )

    assert "embedding_weight" not in normalized
    assert normalized["fts_weight"] == 0.5


def _seed_eval_db(path, rows):
    conn = connect_database(path)
    migrate(conn)
    for fact_id, content in rows:
        conn.execute(
            "INSERT INTO facts(fact_id, content, category, tags, trust_score, scope, sensitivity, schema_version) "
            "VALUES (?, ?, 'general', '', 0.9, 'private', 'normal', 1)",
            (fact_id, content),
        )
    conn.commit()
    conn.close()


def test_run_autotune_uses_hybrid_retriever_and_describes_actual_cases(tmp_path):
    db = tmp_path / "memory.db"
    facts = [
        (1, "Avery prefers dark theme in the editor"),
        (2, "The Atlas launch is scheduled for September"),
        (3, "Mina owns the release checklist"),
        (4, "The current editor font is Iosevka"),
    ]
    _seed_eval_db(db, facts)
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([
        {"id": "p1", "query": "What editor theme does Avery prefer?", "gold_fact_id": 1, "case_type": "paraphrase"},
        {"id": "p2", "query": "When is the Atlas launch scheduled?", "gold_fact_id": 2, "case_type": "paraphrase"},
        {"id": "p3", "query": "Who owns the release checklist?", "gold_fact_id": 3, "case_type": "paraphrase"},
        {"id": "p4", "query": "Which font is set in the editor?", "gold_fact_id": 4, "case_type": "paraphrase"},
    ]))
    report_dir = tmp_path / "report"

    result = run_autotune(
        db_path=db,
        report_dir=report_dir,
        max_experiments=2,
        max_minutes=1,
        cases_path=cases_path,
        sample=4,
        limit=5,
        min_trust=0.3,
        repo_root=tmp_path,
        hermes_src=None,
        test_stubs=False,
        seed=1701,
        retriever_mode="ci",
    )

    recommendation = (report_dir / "RECOMMENDATION.md").read_text()
    log_rows = [json.loads(line) for line in (report_dir / "experiments.jsonl").read_text().splitlines()]
    knob_names = {row["proposal"]["knob"] for row in log_rows if row["proposal"]["knob"]}

    assert result["trials"] >= 1
    assert result["holdout_case_count"] >= 1
    assert result["tune_case_count"] >= 1
    assert "ci-feature-hash" in result["active_backend"]["active"]
    assert "embedding_weight" not in knob_names
    assert "hrr_weight" not in knob_names
    assert "this run used exact-fact cases" not in recommendation
    assert "paraphrase" in recommendation.lower()


def test_run_autotune_refuses_improvement_on_self_scoring_cases(tmp_path):
    db = tmp_path / "memory.db"
    facts = [
        (1, "Avery prefers dark theme in the editor"),
        (2, "The Atlas launch is scheduled for September"),
        (3, "Mina owns the release checklist"),
        (4, "The current editor font is Iosevka"),
    ]
    _seed_eval_db(db, facts)
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([
        {"id": "e1", "query": content, "gold_fact_id": fact_id, "case_type": "exact_fact"}
        for fact_id, content in facts
    ]))
    report_dir = tmp_path / "report"

    result = run_autotune(
        db_path=db,
        report_dir=report_dir,
        max_experiments=3,
        max_minutes=1,
        cases_path=cases_path,
        sample=4,
        limit=5,
        min_trust=0.3,
        repo_root=tmp_path,
        hermes_src=None,
        test_stubs=False,
        seed=1701,
        retriever_mode="ci",
    )

    recommendation = (report_dir / "RECOMMENDATION.md").read_text()
    assert result["beat_baseline"] is False
    assert result["improvement_reported"] is False
    assert "refused" in recommendation.lower()
    assert "self-scoring" in recommendation.lower() or "self-referential" in recommendation.lower()
