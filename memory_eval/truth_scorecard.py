"""Truth-model scorecard: the metrics competitors cannot copy without a supersede model.

Missing sources are ``blocked``, never filled with a fake zero that looks like a win.
"""

from __future__ import annotations

from typing import Any, Sequence

from .extraction_arena import ExtractionArenaScore
from .metrics import percentile
from .protocol import STALE_LEAK_K
from .runner import EvalResult


_TEMPORAL_TYPES = frozenset({
    "current_state_update",
    "changed_preference",
})


def _blocked(reason: str) -> dict[str, Any]:
    return {"status": "blocked", "reason": reason, "value": None}


def _ok(value: float | None, **extra: Any) -> dict[str, Any]:
    row = {"status": "ok", "value": value}
    row.update(extra)
    return row


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _stale_leak(results: Sequence[EvalResult]) -> dict[str, Any]:
    exposed = [result for result in results if result.case.stale_fact_ids]
    if not exposed:
        return _blocked("no retrieval cases declare stale_fact_ids")
    leaks = 0
    for result in exposed:
        stale = set(result.case.stale_fact_ids)
        top = result.ranked_fact_ids[:STALE_LEAK_K]
        if any(fact_id in stale for fact_id in top):
            leaks += 1
    return _ok(_rate(leaks, len(exposed)), cases=len(exposed), leaks=leaks)


def _contradiction(results: Sequence[EvalResult]) -> dict[str, Any]:
    conflicts = [result for result in results if result.case.case_type == "contradiction"]
    if not conflicts:
        return _blocked("no contradiction retrieval cases")
    hits = 0
    for result in conflicts:
        expected = set(result.case.expected_current_fact_ids or [result.case.gold_fact_id])
        if expected and expected <= set(result.ranked_fact_ids):
            hits += 1
    return _ok(_rate(hits, len(conflicts)), cases=len(conflicts), detected=hits)


def _abstention(results: Sequence[EvalResult]) -> dict[str, Any]:
    negatives = [result for result in results if result.case.should_abstain]
    if not negatives:
        return _blocked("no should_abstain retrieval cases")
    correct = sum(1 for result in negatives if not result.ranked_fact_ids)
    return _ok(_rate(correct, len(negatives)), cases=len(negatives), correct=correct)


def _temporal(results: Sequence[EvalResult]) -> dict[str, Any]:
    temporal = [
        result
        for result in results
        if result.case.case_type in _TEMPORAL_TYPES or "temporal" in result.case.tags
    ]
    if not temporal:
        return _blocked("no temporal or as-of retrieval cases")
    correct = 0
    for result in temporal:
        stale = set(result.case.stale_fact_ids)
        top = result.ranked_fact_ids[:STALE_LEAK_K]
        gold_hit = result.case.gold_fact_id in result.ranked_fact_ids
        leaked = any(fact_id in stale for fact_id in top)
        if gold_hit and not leaked:
            correct += 1
    return _ok(_rate(correct, len(temporal)), cases=len(temporal), correct=correct)


def _injection(extraction_score: ExtractionArenaScore | None) -> dict[str, Any]:
    if extraction_score is None:
        return _blocked("no extraction arena score")
    by_id = {score.case_id: score for score in extraction_score.cases}
    injection = [
        case
        for case in extraction_score.arena.cases
        if "prompt-injection" in case.tags or "injection" in case.tags
    ]
    if not injection:
        return _blocked("no extraction cases tagged prompt-injection")
    held = 0
    for case in injection:
        score = by_id.get(case.case_id)
        if score is None:
            continue
        if score.decision_correct and not score.forbidden_leaks:
            held += 1
    return _ok(_rate(held, len(injection)), cases=len(injection), held=held)


def _tokens(values: Sequence[float] | None) -> dict[str, Any]:
    if values is None:
        return _blocked("tokens_per_query not supplied for this run")
    nums = [float(value) for value in values]
    if not nums:
        return _blocked("tokens_per_query is empty")
    return _ok(
        sum(nums) / len(nums),
        mean=sum(nums) / len(nums),
        p50=percentile(nums, 50),
        p95=percentile(nums, 95),
        cases=len(nums),
    )


def _latency(results: Sequence[EvalResult] | None, latencies_ms: Sequence[float] | None) -> dict[str, Any]:
    values: list[float] = []
    if latencies_ms is not None:
        values.extend(float(value) for value in latencies_ms)
    elif results:
        values.extend(float(result.latency_ms) for result in results)
    if not values:
        return _blocked("no latency samples")
    return _ok(
        percentile(values, 50),
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        mean=sum(values) / len(values),
        cases=len(values),
    )


def truth_scorecard(
    *,
    retrieval_results: Sequence[EvalResult] | None = None,
    extraction_score: ExtractionArenaScore | None = None,
    tokens_per_query: Sequence[float] | None = None,
    latencies_ms: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build the truth-model report. Blocked metrics stay blocked."""

    results = tuple(retrieval_results or ())
    return {
        "kind": "truth_model",
        "fabricated": False,
        "metrics": {
            "stale_fact_leak_rate": _stale_leak(results) if results else _blocked("no retrieval results"),
            "contradiction_detection_rate": _contradiction(results) if results else _blocked("no retrieval results"),
            "abstention_correctness": _abstention(results) if results else _blocked("no retrieval results"),
            "injection_resistance": _injection(extraction_score),
            "temporal_asof_correctness": _temporal(results) if results else _blocked("no retrieval results"),
            "tokens_per_query": _tokens(tokens_per_query),
            "latency_ms": _latency(results or None, latencies_ms),
        },
    }
