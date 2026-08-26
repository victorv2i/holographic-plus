from __future__ import annotations

from memory_eval.protocol import (
    BENCHMARK_PROTOCOL_PATH,
    LOCOMO_CATEGORY_TABLE,
    PROTOCOL_RULES,
    RETRIEVAL_K,
    TOKEN_BUDGET,
    TRANSCRIPT_GATE,
    load_benchmark_protocol,
)


def test_benchmark_protocol_document_exists_and_encodes_standing_rules():
    text = BENCHMARK_PROTOCOL_PATH.read_text(encoding="utf-8")

    assert BENCHMARK_PROTOCOL_PATH.name == "BENCHMARK_PROTOCOL.md"
    assert "never drop" in text.lower()
    assert "category 5" in text.lower() or "cat 5" in text.lower()
    assert "oracle" in text.lower()
    assert "full-context" in text.lower()
    assert "locomo" in text.lower()
    assert "longmemeval" in text.lower()
    assert "sha256" in text.lower()
    assert "ndcg" in text.lower()
    assert "recall@" in text.lower()
    assert "mrr" in text.lower()
    assert "real-transcript capture gate" in text.lower()
    assert "seed" in text.lower() and "not the ship gate" in text.lower()


def test_protocol_constants_match_the_standing_rules():
    proto = load_benchmark_protocol()

    assert proto["never_drop_locomo_category_5"] is True
    assert proto["never_report_oracle_as_real"] is True
    assert proto["require_same_reader_full_context_baseline"] is True
    assert proto["hash_embedder_is_not_a_retrieval_claim"] is True
    assert RETRIEVAL_K == (1, 3, 5, 10)
    assert TOKEN_BUDGET == 256
    assert LOCOMO_CATEGORY_TABLE[1]["id"] == 1
    assert LOCOMO_CATEGORY_TABLE[5]["name"] == "adversarial"
    assert "never_drop_category_because_it_scores_badly" in PROTOCOL_RULES
    assert proto["seed_extraction_arena_is_not_the_capture_ship_gate"] is True
    assert proto["transcript_gate"]["seed_is_not_the_gate"] is True
    assert proto["transcript_gate"]["thresholds"] == TRANSCRIPT_GATE["thresholds"]
    assert "capture_ships_only_when_real_transcript_gate_is_green" in PROTOCOL_RULES
