from __future__ import annotations

from collections.abc import Sequence

from enfold.core_store import connect_database
import pytest

from enfold.hybrid_retrieval import HybridRetriever, RankingConfig
from enfold.schema import migrate


class TableEmbedder:
    identity = "fusion-quality-fixture"
    production_ready = False

    def __init__(self, table: dict[str, Sequence[float]]):
        self._table = table

    def embed_query(self, text: str) -> Sequence[float]:
        return self._table[text]

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._table[text] for text in texts)


def _store(tmp_path):
    conn = connect_database(tmp_path / "fusion-quality.db")
    migrate(conn)
    return conn


def _fact(conn, fact_id: int, content: str, **fields):
    values = {
        "category": "general",
        "tags": "",
        "trust_score": 0.8,
        "created_at": "2026-07-12 12:00:00",
        "updated_at": "2026-07-12 12:00:00",
        "memory_kind": None,
        "scope": "private",
        "sensitivity": "normal",
        "schema_version": 1,
        **fields,
    }
    columns = ("fact_id", "content", *values)
    conn.execute(
        f"INSERT INTO facts({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        (fact_id, content, *values.values()),
    )


def test_rrf_prefers_dense_leader_over_fts_rank_one_with_weak_dense(tmp_path):
    conn = _store(tmp_path)
    query = "alpha beta ranking"
    decoy = "alpha beta ranking alpha beta ranking alpha beta ranking"
    gold = "alpha ranking paraphrase of the intended topic"
    table = {
        query: (1.0, 0.0),
        decoy: (0.15, 0.99),
        gold: (1.0, 0.0),
    }
    _fact(conn, 1, decoy)
    _fact(conn, 2, gold)
    for index in range(3, 8):
        filler = f"alpha beta ranking filler {index}"
        _fact(conn, index, filler)
        table[filler] = (0.1, 0.995)
    for index in range(8, 22):
        filler = f"unrelated pantry note {index}"
        _fact(conn, index, filler)
        table[filler] = (0.5, 0.87)
    conn.commit()
    ranking = RankingConfig(
        fts_query_coverage_weight=0.0,
        trust_weight=0.0,
        memory_kind_weight=0.0,
        recency_weight=0.0,
        review_weight=0.0,
        named_subject_weight=0.0,
        score_floor=0.0,
        ambiguity_margin=0.0,
    )

    rows = HybridRetriever(
        conn, TableEmbedder(table), ranking_config=ranking
    ).search(query, min_trust=0, limit=10)

    assert rows[0]["fact_id"] == 2
    assert rows[0]["dense_score"] > next(
        row["dense_score"] for row in rows if row["fact_id"] == 1
    )
    conn.close()


def test_weak_unique_dense_hit_does_not_clear_the_score_floor(tmp_path):
    conn = _store(tmp_path)
    content = "Avery's job status is on leave."
    query = "active"
    _fact(conn, 1, content, trust_score=0.5, memory_kind="state")
    conn.commit()
    embedder = TableEmbedder({query: (1.0, 0.0), content: (0.08, 0.997)})

    rows = HybridRetriever(conn, embedder).search(query, min_trust=0)

    assert rows == []
    conn.close()
