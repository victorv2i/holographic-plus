from __future__ import annotations

from memory_eval.extraction_arena import (
    DEFAULT_CASES_PATH,
    DEFAULT_OUTPUTS_PATH,
    load_candidate_outputs,
    load_extraction_arena,
    score_extraction_arena,
)
from memory_eval.runner import EvalCase, EvalResult
from memory_eval.truth_scorecard import truth_scorecard


def _result(case: EvalCase, ranked: list[int], *, latency_ms: float = 4.0) -> EvalResult:
    gold_rank = ranked.index(case.gold_fact_id) + 1 if case.gold_fact_id in ranked else None
    stale = set(case.stale_fact_ids)
    return EvalResult(
        case=case,
        ranked_fact_ids=ranked,
        gold_rank=gold_rank,
        stale_leak_ranks=[idx + 1 for idx, fact_id in enumerate(ranked) if fact_id in stale],
        latency_ms=latency_ms,
        results=[{"fact_id": fact_id, "score": 0.9 - idx * 0.1} for idx, fact_id in enumerate(ranked)],
    )


def test_truth_scorecard_reports_required_metrics_and_blocks_missing_sources():
    results = [
        _result(
            EvalCase(
                id="stale",
                query="current runtime",
                gold_fact_id=1,
                case_type="stale_fact_exclusion",
                stale_fact_ids=[2],
            ),
            [2, 1],
        ),
        _result(
            EvalCase(
                id="conflict",
                query="coverage end",
                gold_fact_id=3,
                case_type="contradiction",
                expected_current_fact_ids=[3, 4],
            ),
            [3, 4],
        ),
        _result(
            EvalCase(id="abs", query="unknown", gold_fact_id=-1, case_type="abstention", should_abstain=True),
            [],
        ),
        _result(
            EvalCase(
                id="asof",
                query="current employer",
                gold_fact_id=5,
                case_type="current_state_update",
                stale_fact_ids=[6],
            ),
            [5],
        ),
    ]

    card = truth_scorecard(retrieval_results=results)

    assert card["kind"] == "truth_model"
    assert card["fabricated"] is False
    metrics = card["metrics"]
    # Exposed-case leak rate: stale-exclusion leaks, current-state update does not.
    assert metrics["stale_fact_leak_rate"]["value"] == 0.5
    assert metrics["stale_fact_leak_rate"]["cases"] == 2
    assert metrics["stale_fact_leak_rate"]["leaks"] == 1
    assert metrics["contradiction_detection_rate"]["value"] == 1.0
    assert metrics["abstention_correctness"]["value"] == 1.0
    assert metrics["temporal_asof_correctness"]["value"] == 1.0
    assert metrics["injection_resistance"]["status"] == "blocked"
    assert metrics["tokens_per_query"]["status"] == "blocked"
    assert "p50" in metrics["latency_ms"]


def test_truth_scorecard_scores_injection_from_extraction_arena():
    extraction = score_extraction_arena(
        load_extraction_arena(DEFAULT_CASES_PATH),
        load_candidate_outputs(DEFAULT_OUTPUTS_PATH),
    )

    card = truth_scorecard(extraction_score=extraction)

    assert card["metrics"]["injection_resistance"]["status"] == "ok"
    assert card["metrics"]["injection_resistance"]["value"] == 1.0
    assert card["metrics"]["stale_fact_leak_rate"]["status"] == "blocked"
