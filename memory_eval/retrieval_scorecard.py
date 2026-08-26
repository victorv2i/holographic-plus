"""Retrieval-only scorecard, kept separate from reader QA accuracy.

HippoRAG splits ``retrieval_eval`` from ``qa_eval``. Enfold's product is the
retriever, so published retrieval numbers must not be mixed with judge F1.
"""

from __future__ import annotations

from typing import Any, Sequence

from .metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k
from .protocol import RETRIEVAL_K
from .runner import EvalResult


def retrieval_scorecard(
    results: Sequence[EvalResult],
    *,
    k_values: Sequence[int] = RETRIEVAL_K,
) -> dict[str, Any]:
    """Score ranked fact ids only. Abstention cases are excluded from ranking."""

    answerable = [result for result in results if not result.case.should_abstain]
    ranked = [result.ranked_fact_ids for result in answerable]
    gold = [result.case.gold_fact_id for result in answerable]
    card: dict[str, Any] = {
        "kind": "retrieval_only",
        "reader_used": False,
        "cases": len(answerable),
        "mrr": mean_reciprocal_rank(ranked, gold),
    }
    for k in k_values:
        card[f"recall@{k}"] = recall_at_k(ranked, gold, k)
        card[f"ndcg@{k}"] = ndcg_at_k(ranked, gold, k)
    return card
