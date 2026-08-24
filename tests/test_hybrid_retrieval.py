from __future__ import annotations

from collections.abc import Sequence

from enfold.core_store import connect_database
import pytest

from enfold.hybrid_retrieval import (
    DeterministicFeatureHashEmbedder,
    HybridRetriever,
    RankingConfig,
    SQLiteVersionedEmbeddingBackend,
    SQLiteStoredEmbeddingWriter,
    StoredEmbeddingError,
    VectorFallbackTelemetry,
    VersionedStoredEmbeddingAdapter,
    deterministic_retriever_factory,
    named_anchor_tokens,
)
from enfold.embeddings import embedding_to_bytes
import numpy as np
from enfold.schema import migrate
from enfold.sqlite_vec_index import SQLiteVecIndex, rebuild_sqlite_vec_index


class TableEmbedder:
    identity = "test-table-v1"
    production_ready = False

    def __init__(self, table: dict[str, Sequence[float]]):
        self.table = table
        self.document_calls: list[tuple[str, ...]] = []

    def embed_query(self, text: str) -> Sequence[float]:
        return self.table[text]

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.document_calls.append(tuple(texts))
        return tuple(self.table[text] for text in texts)


def _store(tmp_path):
    conn = connect_database(tmp_path / "hybrid.db")
    migrate(conn)
    return conn


def _fact(conn, fact_id: int, content: str, **fields):
    values = {
        "category": "general",
        "tags": "",
        "trust_score": 0.8,
        "scope": "private",
        "sensitivity": "normal",
        "schema_version": 1,
        **fields,
    }
    columns = ("fact_id", "content", *values)
    conn.execute(
        f"INSERT INTO facts({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)})",
        (fact_id, content, *values.values()),
    )


def test_dense_signal_recovers_semantic_only_candidate(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "The vehicle is parked inside the garage")
    _fact(conn, 2, "A recipe calls for toasted walnuts")
    conn.commit()
    embedder = TableEmbedder(
        {
            "automobile location": (1.0, 0.0),
            "The vehicle is parked inside the garage": (1.0, 0.0),
            "A recipe calls for toasted walnuts": (0.0, 1.0),
        }
    )

    rows = HybridRetriever(conn, embedder).search("automobile location")

    assert rows[0]["fact_id"] == 1
    assert rows[0]["fts_score"] == 0.0
    assert rows[0]["jaccard_score"] == 0.0
    assert rows[0]["dense_score"] == 1.0
    conn.close()


def test_negative_dense_cosine_is_clamped_to_zero(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "A document pointing away from the lookup")
    conn.commit()
    embedder = TableEmbedder(
        {
            "needleterm": (1.0, 0.0),
            "A document pointing away from the lookup": (-1.0, 0.0),
        }
    )

    base_only = RankingConfig(
        trust_weight=0.0, memory_kind_weight=0.0, recency_weight=0.0
    )
    rows = HybridRetriever(
        conn, embedder, min_score=0.0, ranking_config=base_only
    ).search("needleterm")

    assert len(rows) == 1
    assert rows[0]["dense_score"] == 0.0
    assert rows[0]["score"] == 0.0
    conn.close()


def test_combined_score_orders_lexical_and_dense_signals_together(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "Unrelated embedding-only candidate")
    _fact(conn, 2, "ranking query lexical candidate")
    conn.commit()
    embedder = TableEmbedder(
        {
            "ranking query": (1.0, 0.0),
            "Unrelated embedding-only candidate": (1.0, 0.0),
            "ranking query lexical candidate": (0.5, 0.8660254),
        }
    )

    base_only = RankingConfig(
        trust_weight=0.0, memory_kind_weight=0.0, recency_weight=0.0
    )
    rows = HybridRetriever(conn, embedder, ranking_config=base_only).search(
        "ranking query"
    )

    assert [row["fact_id"] for row in rows] == [2, 1]
    assert rows[0]["score"] == pytest.approx(0.35 + 0.25 * 0.5 + 0.4 * 0.5)
    assert rows[1]["score"] == pytest.approx(0.4)
    conn.close()


def test_fts_score_rewards_distinct_query_coverage_over_term_repetition(tmp_path):
    conn = _store(tmp_path)
    repeated = "hermes " * 20
    complete = "hermes audit " + "context " * 50
    _fact(conn, 1, repeated)
    _fact(conn, 2, complete)
    conn.commit()
    embedder = TableEmbedder(
        {
            "hermes audit": (1.0, 0.0),
            repeated: (1.0, 0.0),
            complete: (1.0, 0.0),
        }
    )
    lexical_only = RankingConfig(
        fts_weight=1.0,
        jaccard_weight=0.0,
        dense_weight=0.0,
        trust_weight=0.0,
        memory_kind_weight=0.0,
        recency_weight=0.0,
        score_floor=0.0,
        ambiguity_margin=0.0,
    )

    rows = HybridRetriever(conn, embedder, ranking_config=lexical_only).search(
        "hermes audit"
    )

    assert [row["fact_id"] for row in rows] == [2, 1]
    assert [row["fts_score"] for row in rows] == [0.875, 0.625]
    conn.close()


def test_fts_tags_recall_candidates_without_inflating_content_coverage(tmp_path):
    conn = _store(tmp_path)
    content = "The operating procedure is documented"
    _fact(conn, 1, content, tags="hermes,audit")
    conn.commit()
    embedder = TableEmbedder({"hermes audit": (1.0, 0.0), content: (1.0, 0.0)})
    lexical_only = RankingConfig(
        fts_weight=1.0,
        jaccard_weight=0.0,
        dense_weight=0.0,
        trust_weight=0.0,
        memory_kind_weight=0.0,
        recency_weight=0.0,
        score_floor=0.0,
        ambiguity_margin=0.0,
    )

    row = HybridRetriever(conn, embedder, ranking_config=lexical_only).search(
        "hermes audit"
    )[0]

    assert row["fact_id"] == 1
    assert row["fts_score"] == 0.25
    conn.close()


def test_old_fts_hit_is_unioned_with_newest_candidate_window(tmp_path):
    conn = _store(tmp_path)
    _fact(
        conn, 1, "needleterm appears only in this old fact", hrr_vector=b"legacy-blob"
    )
    _fact(conn, 2, "newer unrelated fact")
    _fact(conn, 3, "newest unrelated fact")
    conn.commit()
    embedder = TableEmbedder(
        {
            "needleterm": (1.0, 0.0),
            "needleterm appears only in this old fact": (0.0, 1.0),
            "newer unrelated fact": (0.0, 1.0),
            "newest unrelated fact": (0.0, 1.0),
        }
    )
    lexical_only = RankingConfig(
        fts_weight=1.0,
        jaccard_weight=0.0,
        dense_weight=0.0,
        trust_weight=0.0,
        memory_kind_weight=0.0,
        recency_weight=0.0,
        score_floor=0.0,
    )

    rows = HybridRetriever(
        conn,
        embedder,
        candidate_limit=2,
        ranking_config=lexical_only,
    ).search("needleterm", limit=1)

    assert [row["fact_id"] for row in rows] == [1]
    assert "hrr_vector" not in rows[0]
    conn.close()


def test_nonindexed_retriever_reports_its_bounded_dense_candidate_window(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "old semantic candidate")
    _fact(conn, 2, "new unrelated candidate")
    conn.commit()

    retriever = HybridRetriever(
        conn,
        TableEmbedder(
            {
                "semantic query": (1.0, 0.0),
                "old semantic candidate": (1.0, 0.0),
                "new unrelated candidate": (0.0, 1.0),
            }
        ),
        candidate_limit=1,
    )

    assert retriever.metadata["dense_candidate_coverage"] == "bounded"
    assert retriever.metadata["candidate_generation"] == "recent-plus-lexical"
    conn.close()


def test_vector_fallback_telemetry_clears_active_degradation_after_recovery():
    telemetry = VectorFallbackTelemetry()
    telemetry.record("sqlite_vec_query_error")

    telemetry.recover()

    assert telemetry.snapshot() == {
        "vector_fallback_active": False,
        "vector_fallback_count": 1,
        "vector_fallback_recovery_count": 1,
        "vector_last_fallback_reason": None,
    }


def test_scope_current_conflict_and_trust_filters_run_before_dense_embedding(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "eligible current private memory")
    _fact(conn, 2, "forbidden team memory", scope="team")
    _fact(conn, 3, "invalid historical memory", invalid_at="2026-01-01T00:00:00Z")
    _fact(conn, 4, "superseded historical memory", superseded_by=1)
    _fact(conn, 5, "unsettled conflict memory", conflict_group="conflict-1")
    _fact(conn, 6, "low trust memory", trust_score=0.1)
    _fact(conn, 7, "sensitive private memory", sensitivity="sensitive")
    conn.commit()
    texts = [row[0] for row in conn.execute("SELECT content FROM facts")]
    embedder = TableEmbedder(
        {"memory lookup": (1.0, 0.0), **{text: (1.0, 0.0) for text in texts}}
    )

    rows = HybridRetriever(conn, embedder, allowed_scopes=("private",)).search(
        "memory lookup", min_trust=0.3
    )

    assert [row["fact_id"] for row in rows] == [1]
    assert embedder.document_calls == [("eligible current private memory",)]
    conn.close()


def test_named_anchor_abstains_before_calling_dense_embedder(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "The approved initiative budget is twelve thousand dollars")
    conn.commit()
    embedder = TableEmbedder({})

    rows = HybridRetriever(conn, embedder).search(
        "What budget was approved for Project Ember?"
    )

    assert rows == []
    assert embedder.document_calls == []
    conn.close()


def test_polite_sentence_opener_is_not_a_named_anchor(tmp_path):
    conn = _store(tmp_path)
    content = "Orchid status is ready for review"
    _fact(conn, 1, content)
    conn.commit()
    embedder = TableEmbedder(
        {
            "Please find Orchid status": (1.0, 0.0),
            content: (1.0, 0.0),
        }
    )

    rows = HybridRetriever(conn, embedder).search("Please find Orchid status")

    assert [row["fact_id"] for row in rows] == [1]
    assert embedder.document_calls == [(content,)]
    conn.close()


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("How should Atlas-5 work through Relay?", frozenset({"atlas", "5", "relay"})),
        ("I need the Nimbus monitor status", frozenset({"nimbus"})),
        ("How do I inspect the Nimbus Monitor cron?", frozenset({"nimbus", "monitor"})),
        ("What changed in July for Avery's persona?", frozenset({"july", "avery"})),
        ("What did the April 11 system audit find?", frozenset({"april"})),
        ("What did April decide?", frozenset({"april"})),
        (
            "What did the Avery-maintained Hermes note say?",
            frozenset({"avery", "hermes"}),
        ),
        ("Can you Show Orchid status?", frozenset({"orchid"})),
        ("Please Tell me about Orchid", frozenset({"orchid"})),
        ("Tell me about May", frozenset({"may"})),
        ("Please find Orchid status", frozenset({"orchid"})),
    ],
)
def test_named_anchors_share_candidate_tokenization_and_ignore_pronoun(query, expected):
    assert named_anchor_tokens(query) == expected


def test_named_anchor_matches_compound_written_as_separate_words(tmp_path):
    conn = _store(tmp_path)
    content = "The Atlas Deck worktree recovery procedure is documented"
    _fact(conn, 1, content)
    conn.commit()
    embedder = TableEmbedder(
        {
            "What is the Atlasdeck worktree gotcha?": (1.0, 0.0),
            content: (1.0, 0.0),
        }
    )

    rows = HybridRetriever(conn, embedder).search(
        "What is the Atlasdeck worktree gotcha?"
    )

    assert [row["fact_id"] for row in rows] == [1]
    conn.close()


def test_named_anchor_matches_three_word_compound(tmp_path):
    conn = _store(tmp_path)
    content = "The North Star Relay evaluation procedure is documented"
    query = "How does NorthStarRelay evaluation work?"
    _fact(conn, 1, content)
    conn.commit()
    embedder = TableEmbedder({query: (1.0, 0.0), content: (1.0, 0.0)})

    rows = HybridRetriever(conn, embedder).search(query)

    assert [row["fact_id"] for row in rows] == [1]
    conn.close()


def test_named_month_anchor_matches_standard_abbreviation(tmp_path):
    conn = _store(tmp_path)
    content = "Avery persona was updated on Jul 10"
    query = "What changed in July for Avery's persona?"
    _fact(conn, 1, content)
    conn.commit()
    embedder = TableEmbedder({query: (1.0, 0.0), content: (1.0, 0.0)})

    rows = HybridRetriever(conn, embedder).search(query)

    assert [row["fact_id"] for row in rows] == [1]
    conn.close()


def test_named_month_abbreviation_matches_full_month(tmp_path):
    conn = _store(tmp_path)
    content = "Avery persona was updated in July"
    query = "What changed in Jul for Avery's persona?"
    _fact(conn, 1, content)
    conn.commit()
    embedder = TableEmbedder({query: (1.0, 0.0), content: (1.0, 0.0)})

    rows = HybridRetriever(conn, embedder).search(query)

    assert [row["fact_id"] for row in rows] == [1]
    conn.close()


def test_ci_feature_hash_embedder_and_ranking_are_deterministic(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "Orchid backup runs every Tuesday")
    _fact(conn, 2, "Quartz deployment uses a blue environment")
    conn.commit()
    first = HybridRetriever(conn, DeterministicFeatureHashEmbedder()).search(
        "When does Orchid backup run?"
    )
    second = HybridRetriever(conn, DeterministicFeatureHashEmbedder()).search(
        "When does Orchid backup run?"
    )

    assert [(row["fact_id"], row["score"]) for row in first] == [
        (row["fact_id"], row["score"]) for row in second
    ]
    assert first[0]["fact_id"] == 1
    conn.close()


class StoredBackend:
    identity = "local-fastembed"
    embedding_version = "bge-small-en-v1.5@sha256:fixture"
    dimensions = 2

    def __init__(self):
        self.documents = []

    def embed_query(self, text):
        return (1.0, 0.0)

    def load_documents(self, documents):
        self.documents.append(tuple(documents))
        return tuple(
            (1.0, 0.0) if fact_id == 1 else (0.0, 1.0) for fact_id, _ in documents
        )


def test_versioned_backend_receives_only_eligible_candidate_ids(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "The vehicle is parked inside the garage")
    _fact(conn, 2, "forbidden team memory", scope="team")
    conn.commit()
    backend = StoredBackend()
    adapter = VersionedStoredEmbeddingAdapter(backend)

    rows = HybridRetriever(conn, adapter).search("automobile location")

    assert rows[0]["fact_id"] == 1
    assert backend.documents == [((1, "The vehicle is parked inside the garage"),)]
    assert adapter.identity == "local-fastembed@bge-small-en-v1.5@sha256:fixture:2"
    conn.close()


def test_versioned_backend_rejects_invalid_vectors(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "eligible memory")
    conn.commit()
    backend = StoredBackend()
    backend.dimensions = 3

    with pytest.raises(ValueError, match="dimensions"):
        HybridRetriever(conn, VersionedStoredEmbeddingAdapter(backend)).search("memory")
    conn.close()


def test_deterministic_factory_reports_nonproduction_metadata(tmp_path):
    conn = _store(tmp_path)
    retriever = deterministic_retriever_factory(dimensions=64)(conn, ("private",))

    assert retriever.metadata["embedder_identity"] == "ci-feature-hash-v1:64"
    assert retriever.metadata["embedder_production_ready"] is False
    assert retriever.metadata["filter_before_dense_ranking"] is True
    conn.close()


@pytest.mark.parametrize("coverage_weight", [0.0, 1.0])
def test_fts_query_coverage_boundaries_are_configurable_and_reported(
    tmp_path, coverage_weight
):
    conn = _store(tmp_path)
    retriever = HybridRetriever(
        conn,
        TableEmbedder({}),
        ranking_config=RankingConfig(fts_query_coverage_weight=coverage_weight),
    )

    assert retriever.metadata["fts_query_coverage_weight"] == coverage_weight
    assert (
        retriever.metadata["fts_score_formula"]
        == "(1-query_coverage_weight)*reciprocal_bm25_rank+"
        "query_coverage_weight*distinct_query_token_coverage"
    )
    conn.close()


class FakeQueryEmbedder:
    def __init__(self, vector=(1.0, 0.0)):
        self.vector = vector
        self.calls = []

    def embed(self, text):
        self.calls.append(text)
        return self.vector


def _embedding_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fact_embeddings(
            fact_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            dim INTEGER NOT NULL,
            embedding_identity TEXT NOT NULL,
            PRIMARY KEY(fact_id, embedding_identity)
        )
        """
    )


def _stored(conn, fact_id, vector, identity="fake:model:document:prefix:v1"):
    array = np.asarray(vector, dtype=np.float32)
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (fact_id, embedding_to_bytes(array), len(array), identity),
    )


def _sqlite_backend(conn, query_embedder):
    return SQLiteVersionedEmbeddingBackend(
        conn,
        query_embedder,
        query_identity="fake:model:query:prefix:v1",
        document_identity="fake:model:document:prefix:v1",
        embedding_version="v1",
        dimensions=2,
        query_prefix="Represent this query: ",
    )


def test_sqlite_backend_loads_only_requested_candidate_ids_in_input_order(tmp_path):
    conn = _store(tmp_path)
    _embedding_table(conn)
    _fact(conn, 1, "stored one")
    _fact(conn, 2, "stored two")
    _stored(conn, 1, (1.0, 0.0))
    _stored(conn, 2, (0.0, 1.0))
    conn.commit()
    query_embedder = FakeQueryEmbedder()
    backend = _sqlite_backend(conn, query_embedder)
    statements = []
    conn.set_trace_callback(statements.append)

    vectors = backend.load_documents(((2, "second"),))
    query = backend.embed_query("where")

    assert [tuple(vector) for vector in vectors] == [(0.0, 1.0)]
    assert query == (1.0, 0.0)
    assert query_embedder.calls == ["Represent this query: where"]
    candidate_selects = [sql for sql in statements if "FROM fact_embeddings" in sql]
    assert len(candidate_selects) == 1
    assert "fact_id IN (2)" in candidate_selects[0]
    assert backend.metadata["missing_embedding_behavior"] == "fail-closed"
    conn.close()


def test_stored_dense_scores_are_protocol_json_scalars(tmp_path):
    conn = _store(tmp_path)
    _embedding_table(conn)
    _fact(conn, 1, "Tuesday preference")
    _stored(conn, 1, (1.0, 0.0))
    conn.commit()
    retriever = HybridRetriever(
        conn,
        VersionedStoredEmbeddingAdapter(
            _sqlite_backend(conn, FakeQueryEmbedder((1.0, 0.0)))
        ),
        allowed_scopes=("private",),
    )

    row = retriever.search("Tuesday", limit=1)[0]

    assert type(row["dense_score"]) is float
    assert type(row["score"]) is float
    conn.close()


def test_sqlite_vec_prefilters_named_anchors_before_dense_scoring(
    tmp_path, monkeypatch
):
    conn = _store(tmp_path)
    _embedding_table(conn)
    matching = "Avery persona was updated in Jul"
    unrelated = "A generic deployment note"
    _fact(conn, 1, matching)
    _fact(conn, 2, unrelated)
    _stored(conn, 1, (0.5, 0.5))
    _stored(conn, 2, (1.0, 0.0))
    conn.commit()
    rebuild_sqlite_vec_index(conn, "fake:model:document:prefix:v1", 2)
    retriever = HybridRetriever(
        conn,
        VersionedStoredEmbeddingAdapter(
            _sqlite_backend(conn, FakeQueryEmbedder((1.0, 0.0)))
        ),
        allowed_scopes=("private",),
    )
    assert retriever._vector_index is not None
    scored_fact_ids = []

    def capture_scores(query_vector, fact_ids):
        scored_fact_ids.append(tuple(fact_ids))
        return {fact_id: 1.0 for fact_id in fact_ids}

    monkeypatch.setattr(retriever._vector_index, "scores", capture_scores)

    rows = retriever.search("What changed in July for Avery?")

    assert scored_fact_ids == [(1,)]
    assert [row["fact_id"] for row in rows] == [1]
    conn.close()


def test_sqlite_backend_fails_closed_on_missing_candidate_coverage(tmp_path):
    conn = _store(tmp_path)
    _embedding_table(conn)
    _fact(conn, 1, "stored one")
    _stored(conn, 1, (1.0, 0.0))
    conn.commit()
    backend = _sqlite_backend(conn, FakeQueryEmbedder())

    with pytest.raises(StoredEmbeddingError, match="missing 1 required"):
        backend.load_documents(((1, "present"), (2, "missing")))
    conn.close()


def test_sqlite_backend_validates_identity_dimension_and_query_availability(tmp_path):
    conn = _store(tmp_path)
    _embedding_table(conn)
    _fact(conn, 1, "stored one")
    _stored(conn, 1, (1.0, 0.0))
    conn.commit()

    with pytest.raises(ValueError, match="exactly match"):
        SQLiteVersionedEmbeddingBackend(
            conn,
            FakeQueryEmbedder(),
            query_identity="fake:model:query:prefix:v1",
            document_identity="fake:other:document:prefix:v1",
            embedding_version="v1",
            dimensions=2,
        )
    with pytest.raises(StoredEmbeddingError, match="unexpected dimension"):
        SQLiteVersionedEmbeddingBackend(
            conn,
            FakeQueryEmbedder(),
            query_identity="fake:model:query:prefix:v1",
            document_identity="fake:model:document:prefix:v1",
            embedding_version="v1",
            dimensions=3,
        )
    backend = _sqlite_backend(conn, FakeQueryEmbedder(None))
    with pytest.raises(StoredEmbeddingError, match="query embedding is unavailable"):
        backend.embed_query("test")
    conn.close()


def test_explicit_stored_embedding_writer_is_idempotent_and_not_service_wired(tmp_path):
    conn = _store(tmp_path)
    _embedding_table(conn)
    _fact(conn, 1, "document to embed")
    conn.commit()
    rebuild_sqlite_vec_index(conn, "fake:model:document:none:v1", 2)
    embedder = FakeQueryEmbedder((0.25, 0.75))
    writer = SQLiteStoredEmbeddingWriter(
        conn,
        embedder,
        document_identity="fake:model:document:none:v1",
        embedding_version="v1",
        model_fingerprint="v1",
        prefix_policy="none",
        dimensions=2,
    )

    assert writer.ensure_fact(1) is True
    assert writer.ensure_fact(1) is False
    assert embedder.calls == ["document to embed"]
    row = conn.execute(
        "SELECT dim, embedding_identity FROM fact_embeddings WHERE fact_id = 1"
    ).fetchone()
    assert tuple(row) == (2, "fake:model:document:none:v1")
    index = SQLiteVecIndex.open(conn, "fake:model:document:none:v1", 2)
    assert index is not None and index.count() == 1
    conn.close()
