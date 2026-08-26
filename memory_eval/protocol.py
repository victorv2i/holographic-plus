"""Pinned constants for Enfold's published benchmark protocol.

Code and reports must conform to these values. The human-readable page lives
at ``docs/BENCHMARK_PROTOCOL.md``. Numbers that cannot be reproduced from a
command in this tree are not scores; they are blocked measurements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


BENCHMARK_PROTOCOL_PATH = Path(__file__).resolve().parents[1] / "docs" / "BENCHMARK_PROTOCOL.md"

RETRIEVAL_K: tuple[int, ...] = (1, 3, 5, 10)
TOKEN_BUDGET = 256
READER_MAX_FACTS = 10
STALE_LEAK_K = 3

# ACL/snap-research locomo10.json as counted in the 2026-08-24 scout.
# Hash is of the AgenticMemory-bundled copy of that file (2,805,274 bytes).
LOCOMO_DATASET = {
    "name": "LOCOMO",
    "file": "locomo10.json",
    "source_url": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    "sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
    "bytes": 2805274,
    "conversations": 10,
    "questions": 1986,
}

# Official LME-S cleaned release. Hash is recorded at acquisition time.
LONGMEMEVAL_S_DATASET = {
    "name": "LongMemEval-S",
    "file": "longmemeval_s_cleaned.json",
    "source": "HuggingFace xiaowu0162/longmemeval-cleaned",
    "sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    "questions": 500,
    "comparable_split": "S",
    "forbidden_as_comparable": ("oracle", "M"),
}

LOCOMO_CATEGORY_TABLE: dict[int, dict[str, Any]] = {
    1: {"id": 1, "name": "category_1", "count": 282, "vendor_aliases": ("multi_hop", "single_hop")},
    2: {"id": 2, "name": "temporal", "count": 321, "vendor_aliases": ("temporal",)},
    3: {"id": 3, "name": "category_3", "count": 96, "vendor_aliases": ("open_domain", "multi_hop")},
    4: {"id": 4, "name": "category_4", "count": 841, "vendor_aliases": ("single_hop", "open_domain")},
    5: {"id": 5, "name": "adversarial", "count": 446, "vendor_aliases": ("adversarial",)},
}

LONGMEMEVAL_QUESTION_TYPES: tuple[str, ...] = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "abstention",
)

PROTOCOL_RULES: tuple[str, ...] = (
    "never_drop_category_because_it_scores_badly",
    "never_report_oracle_as_real",
    "always_publish_same_reader_full_context_baseline",
    "hash_embedder_is_not_a_retrieval_claim",
    "never_compare_local_judge_to_vendor_gpt4o_j",
    "never_fabricate_a_score",
    "seed_extraction_arena_is_not_the_capture_ship_gate",
    "capture_ships_only_when_real_transcript_gate_is_green",
)

# Capture may ship only when the real-transcript gate is green. The
# bundled 7-case synthetic seed is smoke, not this gate. Corrupting
# errors (assistant/tool facts, speaker misattribution, silent typed
# demotion, wrong typed slot) must be zero. Misses are friction.
TRANSCRIPT_GATE = {
    "name": "enfold-transcript-gate",
    "fixture": "memory_eval/fixtures/transcript_gate_cases.jsonl",
    "production_failure_case_id": "prod-autoextract-junk-replay",
    "seed_is_not_the_gate": True,
    "metrics": (
        "speaker_attribution",
        "typed_slot_completeness",
        "incidental_durable_recall",
        "forbidden_assistant_tool",
    ),
    "thresholds": {
        "forbidden_assistant_tool_rate": 0.0,
        "speaker_misattribution_rate": 0.0,
        "typed_slot_precision_min": 1.0,
        "typed_slot_completeness_min": 0.90,
        "silent_demotion_rate": 0.0,
        "incidental_durable_recall_min": 0.70,
        "incidental_durable_precision_min": 0.90,
    },
}

DEFAULT_MODELS = {
    "embedder": "ollama:embeddinggemma:latest (production stored identity)",
    "extractor": "local instruct model pinned in the extraction adapter recipe",
    "reader": "local instruct model, temperature 0, identity published per run",
    "judge": "second local model or the reader at temperature 0; not GPT-4o-J",
}

RETRIEVAL_METRICS: tuple[str, ...] = (
    "recall@1",
    "recall@3",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@1",
    "ndcg@3",
    "ndcg@5",
    "ndcg@10",
)

QA_METRICS: tuple[str, ...] = (
    "token_f1",
    "local_judge_accuracy",
    "retrieval_recall@k_on_evidence_ids",
    "tokens_per_query",
    "search_ms",
    "end_to_end_ms",
)

TRUTH_METRICS: tuple[str, ...] = (
    "stale_fact_leak_rate",
    "contradiction_detection_rate",
    "abstention_correctness",
    "injection_resistance",
    "temporal_asof_correctness",
    "tokens_per_query",
    "latency_ms",
)


def load_benchmark_protocol() -> dict[str, Any]:
    """Return the machine-readable standing rules the harness must obey."""

    return {
        "protocol_path": str(BENCHMARK_PROTOCOL_PATH),
        "never_drop_locomo_category_5": True,
        "never_report_oracle_as_real": True,
        "require_same_reader_full_context_baseline": True,
        "hash_embedder_is_not_a_retrieval_claim": True,
        "retrieval_k": RETRIEVAL_K,
        "token_budget": TOKEN_BUDGET,
        "locomo": LOCOMO_DATASET,
        "longmemeval_s": LONGMEMEVAL_S_DATASET,
        "locomo_categories": LOCOMO_CATEGORY_TABLE,
        "longmemeval_question_types": LONGMEMEVAL_QUESTION_TYPES,
        "models": DEFAULT_MODELS,
        "retrieval_metrics": RETRIEVAL_METRICS,
        "qa_metrics": QA_METRICS,
        "truth_metrics": TRUTH_METRICS,
        "rules": PROTOCOL_RULES,
        "transcript_gate": TRANSCRIPT_GATE,
        "seed_extraction_arena_is_not_the_capture_ship_gate": True,
    }
