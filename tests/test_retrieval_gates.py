from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone

from enfold.core_store import connect_database

from enfold.hybrid_retrieval import (
    HybridRetriever,
    lexical_retriever_factory,
    named_anchor_tokens,
    select_named_anchor_matches,
)
from enfold.schema import migrate


class TableEmbedder:
    identity = "retrieval-gates-fixture"
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
    conn = connect_database(tmp_path / "retrieval-gates.db")
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


def test_local_lexical_named_project_absent_from_every_candidate_still_returns_empty(
    tmp_path,
):
    conn = _store(tmp_path)
    _fact(conn, 1, "A recipe calls for toasted walnuts")
    conn.commit()

    rows = lexical_retriever_factory()(conn, ("private",)).search(
        "What is the budget for Project Unicorn?"
    )

    assert rows == []
    conn.close()


def test_paraphrased_capitalized_query_returns_stored_preference(tmp_path):
    conn = _store(tmp_path)
    content = "The user prefers vim"
    query = "What Editor does the User prefer?"
    _fact(conn, 1, content)
    conn.commit()
    embedder = TableEmbedder({query: (1.0, 0.0), content: (1.0, 0.0)})

    rows = HybridRetriever(conn, embedder).search(query)

    assert [row["fact_id"] for row in rows] == [1]
    assert "vim" in rows[0]["content"]
    conn.close()


def test_single_shared_anchor_keeps_the_matching_cluster(tmp_path):
    conn = _store(tmp_path)
    mira = "Mira Calder works with Ada on CRU projects."
    other = "A recipe calls for toasted walnuts"
    _fact(conn, 1, mira, tags="people,mira")
    _fact(conn, 2, other)
    conn.commit()
    query = "Mira Claude"
    embedder = TableEmbedder(
        {
            query: (1.0, 0.0),
            mira: (1.0, 0.0),
            other: (0.0, 1.0),
        }
    )

    rows = HybridRetriever(conn, embedder).search(query)

    assert [row["fact_id"] for row in rows] == [1]
    conn.close()


def test_select_named_anchor_matches_keeps_best_single_anchor_cluster():
    kept = select_named_anchor_matches(
        ["Mira Calder works with Ada", "A recipe calls for walnuts"],
        frozenset({"mira", "claude"}),
        lambda item: item,
    )

    assert kept == ["Mira Calder works with Ada"]


def test_select_named_anchor_matches_fails_closed_when_no_anchor_hits():
    kept = select_named_anchor_matches(
        ["The approved initiative budget is twelve thousand dollars"],
        frozenset({"ember"}),
        lambda item: item,
    )

    assert kept == []


def test_calendar_month_words_are_not_required_named_anchors():
    assert named_anchor_tokens("Tell me about May") == frozenset()
    assert named_anchor_tokens("What changed in July for Avery?") == frozenset(
        {"avery"}
    )


def test_month_query_still_retrieves_matching_non_name_fact(tmp_path):
    conn = _store(tmp_path)
    content = "Garden planting notes for late spring"
    query = "Tell me about May"
    _fact(conn, 1, content)
    conn.commit()
    embedder = TableEmbedder({query: (1.0, 0.0), content: (1.0, 0.0)})

    rows = HybridRetriever(conn, embedder).search(query)

    assert [row["fact_id"] for row in rows] == [1]
    conn.close()


def test_generic_project_word_does_not_rescue_a_missing_project_name(tmp_path):
    conn = _store(tmp_path)
    orion = "Project Orion stores compiled release bundles in the Cedar registry."
    summary = "Sol now prefers concise weekly project summaries."
    _fact(conn, 1, orion)
    _fact(conn, 2, summary)
    conn.commit()
    query = "What budget was approved for Project Ember?"
    embedder = TableEmbedder(
        {query: (1.0, 0.0), orion: (1.0, 0.0), summary: (0.5, 0.5)}
    )

    rows = HybridRetriever(conn, embedder).search(query)

    assert rows == []
    conn.close()


def test_select_named_anchor_matches_ignores_generic_title_only_hits():
    kept = select_named_anchor_matches(
        [
            "Project Orion stores compiled release bundles",
            "Sol now prefers concise weekly project summaries",
        ],
        frozenset({"project", "ember"}),
        lambda item: item,
    )

    assert kept == []


def test_named_project_absent_from_every_candidate_still_returns_empty(tmp_path):
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


def test_near_tie_of_compatible_paraphrases_returns_both_rows(tmp_path):
    conn = _store(tmp_path)
    first = "The preferred editor is vim"
    second = "The preferred editor remains vim"
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


def test_near_tie_of_contradicting_state_is_visibly_ambiguous(tmp_path):
    conn = _store(tmp_path)
    conn.execute("DROP INDEX IF EXISTS uq_facts_current_state_slot")
    vim = "Current editor preference is vim"
    emacs = "Current editor preference is emacs"
    _fact(
        conn,
        1,
        vim,
        memory_kind="state",
        subject_key="user",
        predicate_key="editor",
        object_value="vim",
    )
    _fact(
        conn,
        2,
        emacs,
        memory_kind="state",
        subject_key="user",
        predicate_key="editor",
        object_value="emacs",
    )
    conn.commit()
    query = "nightly retention location"
    embedder = TableEmbedder({query: (1.0, 0.0), vim: (1.0, 0.0), emacs: (1.0, 0.0)})

    rows = HybridRetriever(
        conn,
        embedder,
        now=datetime(2026, 7, 12, 12, tzinfo=timezone.utc),
    ).search(query, min_trust=0, limit=10)

    assert {row["fact_id"] for row in rows} == {1, 2}
    assert all(row.get("ambiguous") is True for row in rows)
    conn.close()
