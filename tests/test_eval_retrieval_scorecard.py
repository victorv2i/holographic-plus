from __future__ import annotations

from memory_eval.retrieval_scorecard import retrieval_scorecard
from memory_eval.runner import EvalCase, EvalResult


def _result(case: EvalCase, ranked: list[int], *, latency_ms: float = 1.0) -> EvalResult:
    gold_rank = ranked.index(case.gold_fact_id) + 1 if case.gold_fact_id in ranked else None
    stale = set(case.stale_fact_ids)
    return EvalResult(
        case=case,
        ranked_fact_ids=ranked,
        gold_rank=gold_rank,
        stale_leak_ranks=[idx + 1 for idx, fact_id in enumerate(ranked) if fact_id in stale],
        latency_ms=latency_ms,
        results=[{"fact_id": fact_id, "score": 1.0 - (idx * 0.1)} for idx, fact_id in enumerate(ranked)],
    )


def test_retrieval_scorecard_is_separate_from_qa_and_reports_recall_mrr_ndcg():
    results = [
        _result(EvalCase(id="hit", query="q1", gold_fact_id=10), [10, 11]),
        _result(EvalCase(id="second", query="q2", gold_fact_id=20), [21, 20]),
        _result(
            EvalCase(id="abs", query="unknown", gold_fact_id=-1, should_abstain=True),
            [99],
        ),
    ]

    card = retrieval_scorecard(results)

    assert card["kind"] == "retrieval_only"
    assert card["reader_used"] is False
    assert "token_f1" not in card
    assert "local_judge_accuracy" not in card
    assert card["cases"] == 2
    assert card["recall@1"] == 0.5
    assert card["recall@3"] == 1.0
    assert card["mrr"] == 0.75
    assert "ndcg@1" in card
    assert "ndcg@3" in card
    assert "ndcg@5" in card
    assert "ndcg@10" in card
    assert card["ndcg@1"] == 0.5
    assert card["ndcg@3"] > card["ndcg@1"]
