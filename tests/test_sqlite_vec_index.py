from __future__ import annotations

import logging

import numpy as np
import pytest

from enfold.core_store import connect_database
from enfold.context import pack_context
from enfold.embeddings import embedding_to_bytes
from enfold.embed_store import EmbedStore
from enfold.hybrid_retrieval import (
    HybridRetriever,
    RankingConfig,
    SQLiteVersionedEmbeddingBackend,
    VersionedStoredEmbeddingAdapter,
)
from enfold.schema import migrate
from enfold.sqlite_vec_index import (
    IDENTITY_KEY,
    SQLiteVecIndex,
    load_sqlite_vec,
    rebuild_sqlite_vec_index,
)


IDENTITY = "fake:model:document:none:v1"


class QueryEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, text):
        return self.vectors[text]


def _database(tmp_path):
    conn = connect_database(tmp_path / "vectors.db")
    migrate(conn)
    return conn


def _fact(conn, fact_id, content, vector, *, scope="private"):
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, trust_score, schema_version) "
        "VALUES (?, ?, ?, 0.8, 1)",
        (fact_id, content, scope),
    )
    array = np.asarray(vector, dtype=np.float32)
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (fact_id, embedding_to_bytes(array), len(array), IDENTITY),
    )


def _retriever(conn, query_embedder, vector_backend):
    backend = SQLiteVersionedEmbeddingBackend(
        conn,
        query_embedder,
        query_identity="fake:model:query:none:v1",
        document_identity=IDENTITY,
        embedding_version="v1",
        dimensions=3,
    )
    return HybridRetriever(
        conn,
        VersionedStoredEmbeddingAdapter(backend),
        vector_backend=vector_backend,
        min_score=0.0,
        ranking_config=RankingConfig(
            trust_weight=0.0,
            memory_kind_weight=0.0,
            recency_weight=0.0,
            ambiguity_margin=0.0,
        ),
    )


def test_sqlite_vec_dense_scores_match_brute_real_retrieval_path(tmp_path):
    conn = _database(tmp_path)
    _fact(conn, 1, "alpha one", (1.0, 0.0, 0.0))
    _fact(conn, 2, "beta two", (0.5, 0.5, 0.0))
    _fact(conn, 3, "gamma three", (-1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    query_embedder = QueryEmbedder({"unmatched": (1.0, 0.0, 0.0)})

    brute = _retriever(conn, query_embedder, "brute").search("unmatched")
    indexed = _retriever(conn, query_embedder, "sqlite-vec").search("unmatched")

    assert [row["fact_id"] for row in indexed] == [row["fact_id"] for row in brute]
    assert [row["dense_score"] for row in indexed] == pytest.approx(
        [row["dense_score"] for row in brute], abs=1e-6
    )
    assert (
        _retriever(conn, query_embedder, "auto").metadata["vector_backend"]
        == "sqlite-vec"
    )


def test_sqlite_vec_global_dense_candidates_include_old_semantic_only_memory(tmp_path):
    conn = _database(tmp_path)
    _fact(conn, 1, "old semantic-only memory", (1.0, 0.0, 0.0))
    for fact_id in range(2, 6):
        _fact(conn, fact_id, f"new unrelated memory {fact_id}", (0.0, 1.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    query_embedder = QueryEmbedder({"unmatched terms": (1.0, 0.0, 0.0)})

    retriever = _retriever(conn, query_embedder, "sqlite-vec")
    retriever._candidate_limit = 3
    rows = retriever.search("unmatched terms")

    assert rows[0]["fact_id"] == 1
    assert retriever.metadata["dense_candidate_coverage"] == "global"
    assert retriever.metadata["candidate_generation"] == "global-index-plus-lexical"


def test_sqlite_vec_global_dense_path_leaves_unembedded_records_lexical_only(tmp_path):
    conn = _database(tmp_path)
    _fact(conn, 1, "old stored semantic-only memory", (1.0, 0.0, 0.0))
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, trust_score, schema_version) "
        "VALUES (2, 'new queued record', 'private', 0.8, 1)"
    )
    conn.execute(
        "INSERT INTO embedding_jobs("
        "fact_id, document_identity, embedding_version, dimensions, "
        "content_sha256, status, attempts, available_at, created_at, updated_at"
        ") VALUES (2, ?, 'v1', 3, 'queued', 'pending', 0, 'now', 'now', 'now')",
        (IDENTITY,),
    )
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    query_embedder = QueryEmbedder(
        {
            "unmatched terms": (1.0, 0.0, 0.0),
            "queued": (1.0, 0.0, 0.0),
        }
    )

    retriever = _retriever(conn, query_embedder, "sqlite-vec")
    rows = retriever.search("unmatched terms")

    assert [row["fact_id"] for row in rows] == [1]
    assert retriever.metadata["vector_fallback_active"] is False
    lexical_rows = retriever.search("queued")
    assert lexical_rows[0]["fact_id"] == 2
    assert lexical_rows[0]["dense_score"] == 0.0


def test_sqlite_vec_global_path_does_not_apply_candidate_limit_before_hybrid_ranking(
    tmp_path,
):
    conn = _database(tmp_path)
    # Fact one ranks fourth by cosine, but its trusted state-kind prior makes
    # it the correct final hybrid result. A dense pre-ranking window of three
    # must not hide it.
    _fact(conn, 1, "old high-trust state", (0.9, 0.4358899, 0.0))
    for fact_id in range(2, 5):
        _fact(conn, fact_id, f"new low-trust memory {fact_id}", (1.0, 0.0, 0.0))
    conn.execute(
        "UPDATE facts SET trust_score = 1.0, memory_kind = 'state' WHERE fact_id = 1"
    )
    conn.execute("UPDATE facts SET trust_score = 0.3 WHERE fact_id IN (2, 3, 4)")
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)

    backend = SQLiteVersionedEmbeddingBackend(
        conn,
        QueryEmbedder({"unmatched terms": (1.0, 0.0, 0.0)}),
        query_identity="fake:model:query:none:v1",
        document_identity=IDENTITY,
        embedding_version="v1",
        dimensions=3,
    )
    retriever = HybridRetriever(
        conn,
        VersionedStoredEmbeddingAdapter(backend),
        vector_backend="sqlite-vec",
        ranking_config=RankingConfig(ambiguity_margin=0.0),
    )
    retriever._candidate_limit = 3
    rows = retriever.search("unmatched terms")

    assert rows[0]["fact_id"] == 1


def test_sqlite_vec_global_path_does_not_limit_lexical_candidates_before_ranking(
    tmp_path,
):
    conn = _database(tmp_path)
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, trust_score, schema_version) "
        "VALUES (1, 'needle', 'private', 0.8, 1)"
    )
    conn.execute(
        "INSERT INTO embedding_jobs("
        "fact_id, document_identity, embedding_version, dimensions, "
        "content_sha256, status, attempts, available_at, created_at, updated_at"
        ") VALUES (1, ?, 'v1', 3, 'queued', 'pending', 0, 'now', 'now', 'now')",
        (IDENTITY,),
    )
    _fact(conn, 2, "needle", (0.0, 1.0, 0.0))
    _fact(conn, 3, "stored unrelated", (0.0, 1.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)

    retriever = _retriever(
        conn, QueryEmbedder({"needle": (1.0, 0.0, 0.0)}), "sqlite-vec"
    )
    retriever._candidate_limit = 1
    rows = retriever.search("needle", limit=3)

    assert 1 in [row["fact_id"] for row in rows]


def test_sqlite_vec_global_dense_path_matches_brute_force_above_ten_thousand_mixed_scopes(
    tmp_path,
):
    conn = _database(tmp_path)
    private_vectors = {1: (1.0, 0.0, 0.0)}
    _fact(conn, 1, "old eligible semantic-only memory", private_vectors[1])
    for fact_id in range(2, 10_003):
        private_vectors[fact_id] = (0.0, 1.0, 0.0)
        _fact(conn, fact_id, f"new eligible memory {fact_id}", private_vectors[fact_id])
    # These closer-looking records must not displace the authorized global
    # candidate set. They model another scope sharing the same vec0 table.
    for fact_id in range(10_003, 10_504):
        _fact(
            conn, fact_id, f"team-only memory {fact_id}", (1.0, 0.0, 0.0), scope="team"
        )
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None

    query = (1.0, 0.0, 0.0)
    indexed_scores = index.scores(query, tuple(private_vectors))
    brute_scores = {
        fact_id: max(0.0, sum(left * right for left, right in zip(query, vector)))
        for fact_id, vector in private_vectors.items()
    }
    assert indexed_scores.keys() == brute_scores.keys()
    assert all(
        indexed_scores[fact_id] == pytest.approx(score, abs=1e-6)
        for fact_id, score in brute_scores.items()
    )

    retriever = _retriever(
        conn, QueryEmbedder({"unmatched terms": query}), "sqlite-vec"
    )
    retriever._candidate_limit = 3
    rows = retriever.search("unmatched terms")

    assert rows[0]["fact_id"] == 1


def test_sqlite_vec_scores_exact_single_authorized_candidate(tmp_path):
    conn = _database(tmp_path)
    _fact(conn, 1, "authorized but orthogonal", (0.0, 1.0, 0.0))
    _fact(conn, 2, "unauthorized exact match", (1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None

    scores = index.scores((1.0, 0.0, 0.0), (1,))

    assert scores == {1: pytest.approx(0.0, abs=1e-6)}


def test_sqlite_vec_scores_every_authorized_candidate_when_global_neighbors_are_denied(
    tmp_path,
):
    """Filtering must happen before exact scoring, not after vec0 global KNN."""

    conn = _database(tmp_path)
    # More than one score batch of authorized vectors are orthogonal to the
    # query. More global (but denied) exact neighbors used to consume vec0's
    # per-batch KNN window, making the first batch look incomplete.
    for fact_id in range(1, 502):
        _fact(conn, fact_id, f"authorized {fact_id}", (0.0, 1.0, 0.0))
    for fact_id in range(502, 1_103):
        _fact(conn, fact_id, f"denied {fact_id}", (1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None
    statements = []
    conn.set_trace_callback(statements.append)

    scores = index.scores((1.0, 0.0, 0.0), tuple(range(1, 502)))

    assert len(scores) == 501
    assert set(scores) == set(range(1, 502))
    assert all(score == pytest.approx(0.0, abs=1e-6) for score in scores.values())
    assert not any("embedding MATCH" in statement for statement in statements)
    assert any("vec_distance_cosine" in statement for statement in statements)


def test_sqlite_vec_fresh_open_uses_generation_ledger_without_global_membership_diff(
    tmp_path,
):
    conn = _database(tmp_path)
    for fact_id in range(1, 8):
        _fact(conn, fact_id, f"memory {fact_id}", (1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    statements = []
    conn.set_trace_callback(statements.append)

    assert SQLiteVecIndex.open(conn, IDENTITY, 3) is not None

    assert not any("EXCEPT SELECT" in statement for statement in statements)


def test_sqlite_vec_generation_detects_a_canonical_change_outside_index_write_through(
    tmp_path,
):
    conn = _database(tmp_path)
    for fact_id in range(1, 8):
        _fact(conn, fact_id, f"memory {fact_id}", (1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    replacement = embedding_to_bytes(np.asarray((0.0, 1.0, 0.0), dtype=np.float32))
    # Fact seven is outside the historical five-row payload sample. A new
    # index handle must still reject the stale derived index cheaply.
    conn.execute(
        "UPDATE fact_embeddings SET embedding = ? WHERE fact_id = 7 "
        "AND embedding_identity = ?",
        (replacement, IDENTITY),
    )
    conn.commit()

    assert SQLiteVecIndex.open(conn, IDENTITY, 3) is None


def test_sqlite_vec_mmr_matches_brute_when_token_and_embedding_diversity_disagree(
    tmp_path,
):
    conn = _database(tmp_path)
    _fact(conn, 1, "shared alpha", (1.0, 0.0, 0.0))
    _fact(conn, 2, "unique beta", (1.0, 0.0, 0.0))
    _fact(conn, 3, "shared alpha gamma", (0.0, 1.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    query_embedder = QueryEmbedder({"what is relevant": (1.0, 0.0, 0.0)})

    def selected(vector_backend):
        rows = _retriever(conn, query_embedder, vector_backend).search(
            "what is relevant", limit=3
        )
        return [
            fact["fact_id"]
            for fact in pack_context(
                rows, token_budget=512, max_facts=2, mmr_lambda=0.2
            ).facts
        ]

    assert selected("brute") == [1, 3]
    assert selected("sqlite-vec") == selected("brute")


def test_auto_falls_back_honestly_when_extension_is_missing(
    tmp_path, monkeypatch, caplog
):
    conn = _database(tmp_path)
    _fact(conn, 1, "only memory", (1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    query_embedder = QueryEmbedder({"unmatched": (1.0, 0.0, 0.0)})
    expected = _retriever(conn, query_embedder, "brute").search("unmatched")

    def unavailable(_conn):
        raise RuntimeError("extension missing")

    monkeypatch.setattr("enfold.sqlite_vec_index.load_sqlite_vec", unavailable)
    with caplog.at_level(logging.WARNING, logger="enfold.sqlite_vec_index"):
        actual = _retriever(conn, query_embedder, "auto").search("unmatched")

    assert actual == expected
    assert "falling back to brute" in caplog.text


@pytest.mark.parametrize("corruption", ["identity", "population"])
def test_auto_falls_back_on_invalid_index_with_identical_results(
    tmp_path, caplog, corruption
):
    conn = _database(tmp_path)
    _fact(conn, 1, "only memory", (1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None
    if corruption == "identity":
        conn.execute(
            "UPDATE enfold_meta SET value='wrong' WHERE key=?", (IDENTITY_KEY,)
        )
    else:
        conn.execute(f'DELETE FROM "{index.table_name}" WHERE rowid=1')
    conn.commit()
    query_embedder = QueryEmbedder({"unmatched": (1.0, 0.0, 0.0)})
    expected = _retriever(conn, query_embedder, "brute").search("unmatched")

    with caplog.at_level(logging.WARNING, logger="enfold.sqlite_vec_index"):
        actual = _retriever(conn, query_embedder, "auto").search("unmatched")

    assert actual == expected
    assert "falling back to brute" in caplog.text


def test_auto_falls_back_when_sampled_vector_payload_is_stale(tmp_path, caplog):
    conn = _database(tmp_path)
    _fact(conn, 1, "first memory", (1.0, 0.0, 0.0))
    _fact(conn, 2, "second memory", (0.0, 1.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None
    stale = embedding_to_bytes(np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
    conn.execute(f'DELETE FROM "{index.table_name}" WHERE rowid=1')
    conn.execute(
        f'INSERT INTO "{index.table_name}"(rowid, embedding) VALUES (?, ?)',
        (1, stale),
    )
    conn.commit()
    query_embedder = QueryEmbedder({"unmatched": (1.0, 0.0, 0.0)})
    expected = _retriever(conn, query_embedder, "brute").search("unmatched")

    with caplog.at_level(logging.WARNING, logger="enfold.sqlite_vec_index"):
        actual = _retriever(conn, query_embedder, "auto").search("unmatched")

    assert actual == expected
    assert "payload does not match" in caplog.text


def test_query_time_index_problem_falls_back_with_identical_results(tmp_path, caplog):
    conn = _database(tmp_path)
    _fact(conn, 1, "zero vector is valid canonical data", (0.0, 0.0, 0.0))
    _fact(conn, 2, "ordinary vector", (1.0, 0.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    query_embedder = QueryEmbedder({"unmatched": (1.0, 0.0, 0.0)})
    expected = _retriever(conn, query_embedder, "brute").search("unmatched")

    with caplog.at_level(logging.WARNING, logger="enfold.hybrid_retrieval"):
        retriever = _retriever(conn, query_embedder, "sqlite-vec")
        actual = retriever.search("unmatched")

    assert actual == expected
    assert retriever.metadata["vector_backend"] == "brute"
    assert retriever.metadata["vector_fallback_active"] is True
    assert retriever.metadata["vector_fallback_count"] == 1
    assert retriever.metadata["vector_last_fallback_reason"] == "sqlite_vec_query_error"
    assert "falling back to brute" in caplog.text


def test_extension_loading_is_disabled_immediately_after_load(tmp_path):
    conn = _database(tmp_path)

    load_sqlite_vec(conn)

    with pytest.raises(Exception):
        conn.load_extension("definitely-not-an-extension")


def test_transactional_upsert_and_delete_hooks(tmp_path):
    conn = _database(tmp_path)
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None

    vector = embedding_to_bytes(np.asarray((1.0, 0.0, 0.0), dtype=np.float32))
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, trust_score, schema_version) "
        "VALUES (1, 'atomic', 'private', 0.8, 1)"
    )
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (1, ?, 3, ?)",
        (vector, IDENTITY),
    )
    index.upsert_in_transaction(1, vector)
    conn.rollback()
    assert index.count() == 0

    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, trust_score, schema_version) "
        "VALUES (1, 'atomic', 'private', 0.8, 1)"
    )
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (1, ?, 3, ?)",
        (vector, IDENTITY),
    )
    index.upsert_in_transaction(1, vector)
    conn.commit()
    assert index.count() == 1

    conn.execute("BEGIN IMMEDIATE")
    conn.execute("DELETE FROM fact_embeddings WHERE fact_id = 1")
    index.delete_in_transaction(1)
    conn.commit()
    assert index.count() == 0


def test_embed_store_write_and_delete_paths_keep_vec0_in_sync(tmp_path):
    conn = _database(tmp_path)
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, trust_score, schema_version) "
        "VALUES (1, 'legacy path', 'private', 0.8, 1)"
    )
    conn.commit()
    rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    store = EmbedStore(conn, embedding_identity=IDENTITY)

    store.upsert(1, np.asarray((1.0, 0.0, 0.0), dtype=np.float32))
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None and index.count() == 1

    store.delete(1)
    assert index.count() == 0


def test_rebuild_is_idempotent_and_replaces_corrupt_population(tmp_path):
    conn = _database(tmp_path)
    _fact(conn, 1, "one", (1.0, 0.0, 0.0))
    _fact(conn, 2, "two", (0.0, 1.0, 0.0))
    conn.commit()

    first = rebuild_sqlite_vec_index(conn, IDENTITY, 3)
    second = rebuild_sqlite_vec_index(conn, IDENTITY, 3)

    assert first.indexed_count == second.indexed_count == 2
    index = SQLiteVecIndex.open(conn, IDENTITY, 3)
    assert index is not None and index.count() == 2
