from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
import hashlib

from enfold.core_store import connect_database
import pytest

from enfold.embeddings import embedding_to_bytes
from enfold.hybrid_retrieval import (
    HybridRetriever,
    RankingConfig,
    SQLiteVersionedEmbeddingBackend,
    VersionedStoredEmbeddingAdapter,
    leading_person_tokens,
    lexical_retriever_factory,
    named_anchor_tokens,
    named_subject_score,
)
from enfold.schema import migrate
import numpy as np


class TableEmbedder:
    identity = "ranking-quality-fixture"
    production_ready = False

    def __init__(self, table: dict[str, Sequence[float]]):
        self._table = table

    def embed_query(self, text: str) -> Sequence[float]:
        return self._table[text]

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._table[text] for text in texts)


def _store(tmp_path):
    conn = connect_database(tmp_path / "ranking-quality.db")
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


def test_fresh_high_trust_state_paraphrase_outranks_old_low_trust_match(tmp_path):
    conn = _store(tmp_path)
    old = "Archived replicas live in the west storage vault"
    fresh = "Current backup snapshots are retained in the western repository"
    _fact(
        conn,
        1,
        old,
        trust_score=0.31,
        created_at="2020-01-01 00:00:00",
        updated_at="2020-01-01 00:00:00",
        memory_kind="event",
    )
    _fact(conn, 2, fresh, trust_score=0.95, memory_kind="state")
    conn.commit()
    query = "Where are nightly copies kept?"
    embedder = TableEmbedder(
        {
            query: (1.0, 0.0),
            old: (0.98, 0.2),
            fresh: (0.95, 0.1),
        }
    )

    rows = HybridRetriever(
        conn,
        embedder,
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0)

    assert [row["fact_id"] for row in rows[:2]] == [2, 1]
    assert rows[0]["trust_score_component"] > rows[1]["trust_score_component"]
    assert rows[0]["recency_score"] > rows[1]["recency_score"]
    assert rows[0]["memory_kind_score"] > rows[1]["memory_kind_score"]
    conn.close()


def test_durable_state_outranks_fresh_event_at_equal_relevance(tmp_path):
    conn = _store(tmp_path)
    state = "The active editor remains vim"
    event = "The active editor remains vscode"
    _fact(
        conn,
        1,
        state,
        created_at="2024-01-01 00:00:00",
        updated_at="2024-01-01 00:00:00",
        memory_kind="state",
        subject_key="ada",
        predicate_key="editor",
        object_value="vim",
    )
    _fact(
        conn,
        2,
        event,
        created_at="2026-08-24 12:00:00",
        updated_at="2026-08-24 12:00:00",
        memory_kind="event",
    )
    conn.commit()
    query = "active editor remains"
    embedder = TableEmbedder({query: (1.0, 0.0), state: (1.0, 0.0), event: (1.0, 0.0)})

    rows = HybridRetriever(
        conn,
        embedder,
        now=datetime(2026, 8, 24, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0)

    assert [row["fact_id"] for row in rows[:2]] == [1, 2]
    assert rows[0]["memory_kind_score"] > rows[1]["memory_kind_score"]
    conn.close()


def test_state_kind_breaks_an_otherwise_identical_relevance_tie(tmp_path):
    conn = _store(tmp_path)
    event = "The deployment target is the cedar cluster"
    state = "The active deployment target remains the cedar cluster"
    _fact(conn, 1, event, memory_kind="event")
    _fact(conn, 2, state, memory_kind="state")
    conn.commit()
    query = "active target location"
    embedder = TableEmbedder({query: (1.0, 0.0), event: (1.0, 0.0), state: (1.0, 0.0)})

    rows = HybridRetriever(
        conn,
        embedder,
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0)

    assert [row["fact_id"] for row in rows[:2]] == [2, 1]
    conn.close()


def test_tiny_top_margin_returns_noncontradicting_tied_rows(tmp_path):
    conn = _store(tmp_path)
    first = "Archived replicas use the west vault"
    second = "Stored copies use the western vault"
    _fact(conn, 1, first)
    _fact(conn, 2, second)
    conn.commit()
    query = "nightly retention location"
    embedder = TableEmbedder({query: (1.0, 0.0), first: (1.0, 0.0), second: (1.0, 0.0)})

    rows = HybridRetriever(
        conn,
        embedder,
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0, limit=10)

    assert {row["fact_id"] for row in rows} == {1, 2}
    assert all(row.get("ambiguous") is not True for row in rows)
    conn.close()


def test_compatible_near_tie_does_not_drop_the_rest_of_the_list(tmp_path):
    conn = _store(tmp_path)
    query = "crash durability load memory"
    table = {query: (1.0, 0.0)}
    for fact_id in range(1, 5):
        content = f"Crash durability load memory {fact_id}"
        _fact(conn, fact_id, content)
        table[content] = (1.0, 0.0)
    conn.commit()

    rows = HybridRetriever(
        conn,
        TableEmbedder(table),
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0, limit=10)

    assert {row["fact_id"] for row in rows} == {1, 2, 3, 4}
    conn.close()


def test_top_candidate_below_score_floor_abstains(tmp_path):
    conn = _store(tmp_path)
    content = "Unrelated archived observation"
    _fact(
        conn,
        1,
        content,
        trust_score=0,
        created_at="2020-01-01 00:00:00",
        updated_at="2020-01-01 00:00:00",
        memory_kind="event",
    )
    conn.commit()
    query = "nightly retention location"
    embedder = TableEmbedder({query: (1.0, 0.0), content: (0.0, 1.0)})

    rows = HybridRetriever(
        conn,
        embedder,
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0)

    assert rows == []
    conn.close()


def test_person_name_query_prefers_subject_and_reviewed_facts(tmp_path):
    conn = _store(tmp_path)
    other_subject = (
        "Elena Voss (ID Manager, Northline Online, $91,240) reports to Mira Calder."
    )
    about_subject = "Mira Calder created the HarborKit committee channel."
    reviewed = (
        "Historical relationship map recorded 2026-05-27: Ada Morrow reported a "
        "stronger working relationship with Mira Calder than with Nia."
    )
    _fact(conn, 1, other_subject, trust_score=0.5)
    _fact(conn, 2, about_subject, trust_score=0.5)
    _fact(
        conn,
        3,
        reviewed,
        trust_score=1.0,
        correction_status="human_corrected",
    )
    conn.commit()
    query = "Mira Calder"
    embedder = TableEmbedder(
        {
            query: (1.0, 0.0),
            other_subject: (0.85, 0.1),
            about_subject: (0.8, 0.2),
            reviewed: (0.7, 0.3),
        }
    )

    rows = HybridRetriever(
        conn,
        embedder,
        now=datetime(2026, 8, 24, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0)

    assert rows[0]["fact_id"] != 1
    assert {row["fact_id"] for row in rows[:2]} == {2, 3}
    assert {row["fact_id"]: row["named_subject_score"] for row in rows}[2] == 1.0
    assert {row["fact_id"]: row["review_score"] for row in rows}[3] == 1.0
    assert {row["fact_id"]: row["named_subject_score"] for row in rows}[1] == 0.0
    conn.close()


def test_imported_topic_title_and_real_sentence_define_named_subject():
    query = "Is Northline Engine a current focus for Ada?"
    content = (
        "Claude Code memory topic `project-northline-engine` (project): "
        "Ada maintains Northline Engine as a current focus."
    )

    score = named_subject_score(named_anchor_tokens(query), content)

    assert score == 1.0


@pytest.mark.parametrize(
    "content",
    (
        "Claude Code memory topic `` (project): Ada keeps the note.",
        "Claude Code memory topic `project-`northline`-engine` (project): "
        "Ada keeps the note.",
    ),
)
def test_malformed_imported_topic_title_does_not_replace_real_subject(content):
    assert leading_person_tokens(content) == {"ada"}


def test_as_of_date_opener_does_not_replace_named_subject():
    query = "What embedding configuration is current for Enfold?"
    content = "As of 2026-07-04, Enfold uses embeddinggemma."

    score = named_subject_score(named_anchor_tokens(query), content)

    assert score == 1.0


def test_slash_compound_contributes_both_leading_subject_names():
    assert leading_person_tokens("Harbor/Mira keeps Sol as the front door") == {
        "harbor",
        "mira",
    }


def test_default_formula_components_sum_to_reported_score(tmp_path):
    conn = _store(tmp_path)
    content = "Atlas backups run Tuesday"
    _fact(conn, 1, content, trust_score=0.8, memory_kind="insight")
    conn.commit()
    query = "Atlas backups"
    row = HybridRetriever(
        conn,
        TableEmbedder({query: (1.0, 0.0), content: (0.6, 0.8)}),
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0)[0]

    ranking = RankingConfig()
    expected = (
        ranking.relevance_weight * row["fusion_score"]
        + ranking.trust_weight * row["trust_score_component"]
        + ranking.memory_kind_weight * row["memory_kind_score"]
        + ranking.recency_weight * row["recency_score"]
        + ranking.review_weight * row["review_score"]
        + ranking.named_subject_weight * row["named_subject_score"]
    )
    assert row["score"] == pytest.approx(expected)
    conn.close()


@pytest.mark.parametrize(
    "field", ["trust_weight", "score_floor", "recency_half_life_days"]
)
def test_ranking_config_rejects_non_finite_values(field):
    with pytest.raises(ValueError):
        RankingConfig(**{field: float("nan")})


@pytest.mark.parametrize("value", [float("nan"), -0.01, 1.01])
def test_ranking_config_rejects_invalid_fts_query_coverage_weight(value):
    with pytest.raises(ValueError):
        RankingConfig(fts_query_coverage_weight=value)


class _QueryEmbedder:
    def __init__(self, vector=(1.0, 0.0)):
        self.vector = vector

    def embed(self, text):
        return self.vector


def _store_vector(conn, fact_id, vector, identity="fake:model:document:prefix:v1"):
    array = np.asarray(vector, dtype=np.float32)
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (fact_id, embedding_to_bytes(array), len(array), identity),
    )


def _pending_embedding_job(conn, fact_id, content, identity="fake:model:document:prefix:v1"):
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    conn.execute(
        """
        INSERT INTO embedding_jobs(
            fact_id, document_identity, embedding_version, dimensions,
            content_sha256, status, attempts, available_at, created_at, updated_at
        ) VALUES (?, ?, 'v1', 2, ?, 'pending', 0, '2026-07-12 12:00:00',
                  '2026-07-12 12:00:00', '2026-07-12 12:00:00')
        """,
        (fact_id, identity, digest),
    )


def test_local_lexical_state_outranks_event_at_equal_token_overlap(tmp_path):
    conn = _store(tmp_path)
    state = "The active editor remains vim"
    event = "The active editor remains vscode"
    _fact(
        conn,
        1,
        state,
        created_at="2024-01-01 00:00:00",
        updated_at="2024-01-01 00:00:00",
        memory_kind="state",
        subject_key="ada",
        predicate_key="editor",
        object_value="vim",
    )
    _fact(
        conn,
        2,
        event,
        created_at="2026-08-24 12:00:00",
        updated_at="2026-08-24 12:00:00",
        memory_kind="event",
    )
    conn.commit()

    rows = lexical_retriever_factory(
        now=datetime(2026, 8, 24, 12, tzinfo=timezone.utc),
    )(conn, ("private",)).search("active editor remains", min_trust=0)

    assert [row["fact_id"] for row in rows[:2]] == [1, 2]
    assert rows[0]["memory_kind_score"] > rows[1]["memory_kind_score"]
    assert all(row["dense_score"] == 0.0 for row in rows)
    conn.close()


def test_queued_embedding_is_lexical_only_not_a_zero_dense_hit(tmp_path):
    conn = _store(tmp_path)
    gold = "Nightly copies live in the west vault"
    distractor = "The kitchen pantry holds dry goods"
    lexical_query = "Where are west vault copies kept?"
    unmatched_query = "Where are backup snapshots retained?"
    _fact(conn, 1, distractor, trust_score=0.5)
    _fact(conn, 2, gold, trust_score=0.5)
    _store_vector(conn, 1, (1.0, 0.0))
    _pending_embedding_job(conn, 2, gold)
    conn.commit()
    retriever = HybridRetriever(
        conn,
        VersionedStoredEmbeddingAdapter(
            SQLiteVersionedEmbeddingBackend(
                conn,
                _QueryEmbedder((1.0, 0.0)),
                query_identity="fake:model:query:prefix:v1",
                document_identity="fake:model:document:prefix:v1",
                embedding_version="v1",
                dimensions=2,
                query_prefix="Represent this query: ",
            )
        ),
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    )

    rows = retriever.search(lexical_query, min_trust=0)
    unmatched = retriever.search(unmatched_query, min_trust=0)

    assert 2 in {row["fact_id"] for row in rows}
    gold_row = next(row for row in rows if row["fact_id"] == 2)
    assert gold_row["dense_score"] == 0.0
    assert gold_row.get("fusion_score", 1.0) > 0.0
    assert 2 not in {row["fact_id"] for row in unmatched}
    conn.close()
