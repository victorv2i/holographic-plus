import sqlite3

import numpy as np

from enfold.embed_store import EmbedStore

_DOC = "test:model:document:none:v1"
_QUERY = "test:model:query:none:v1"


def _store():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return EmbedStore(conn, embedding_identity=_DOC)


def _count(es):
    return int(es._conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0])


def test_upsert_roundtrip_and_idempotence():
    es = _store()
    v = np.array([0.6, 0.8], dtype=np.float32)
    es.upsert(1, v)
    results = es.score_all(v, embedding_identity=_QUERY)
    assert results and results[0][0] == 1
    assert results[0][1] > 0.99  # the stored vector matches itself
    assert _count(es) == 1
    # upsert is idempotent on (fact_id, identity)
    es.upsert(1, v)
    assert _count(es) == 1


def test_score_all_ranks_most_similar_first():
    es = _store()
    es.upsert(1, np.array([1.0, 0.0], dtype=np.float32))
    es.upsert(2, np.array([0.0, 1.0], dtype=np.float32))
    results = es.score_all(np.array([1.0, 0.0], dtype=np.float32), embedding_identity=_QUERY)
    assert results[0][0] == 1  # fact 1 wins
    assert results[0][1] > results[1][1]


def test_ids_without_embeddings():
    es = _store()
    es.upsert(1, np.array([1.0, 0.0], dtype=np.float32))
    missing = es.ids_without_embeddings([1, 2, 3], embedding_identity=_DOC)
    assert set(missing) == {2, 3}


def test_delete_removes_and_invalidates_cache():
    es = _store()
    es.upsert(1, np.array([1.0, 0.0], dtype=np.float32))
    # prime the cache
    es.score_all(np.array([1.0, 0.0], dtype=np.float32), embedding_identity=_QUERY)
    es.delete(1)
    assert _count(es) == 0
    assert es.score_all(np.array([1.0, 0.0], dtype=np.float32), embedding_identity=_QUERY) == []


def test_score_all_empty_store():
    es = _store()
    assert es.score_all(np.array([1.0, 0.0], dtype=np.float32), embedding_identity=_QUERY) == []


def test_construction_preserves_callers_transaction_and_rollback():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE caller_rows(id INTEGER)")
    conn.commit()
    conn.execute("INSERT INTO caller_rows VALUES (1)")

    EmbedStore(conn, embedding_identity=_DOC)

    assert conn.in_transaction
    conn.rollback()
    assert conn.execute("SELECT * FROM caller_rows").fetchall() == []
    assert conn.execute(
        "SELECT 1 FROM sqlite_master WHERE name = 'fact_embeddings'"
    ).fetchone() is None


def test_mutators_preserve_callers_transaction_and_rollback():
    conn = sqlite3.connect(":memory:")
    es = EmbedStore(conn, embedding_identity=_DOC)
    es.upsert(1, np.array([1.0, 0.0], dtype=np.float32))
    conn.execute("CREATE TABLE caller_rows(id INTEGER)")
    conn.commit()
    conn.execute("INSERT INTO caller_rows VALUES (1)")

    es.upsert(2, np.array([0.0, 1.0], dtype=np.float32))
    es.delete(1)

    assert conn.in_transaction
    conn.rollback()
    assert conn.execute("SELECT * FROM caller_rows").fetchall() == []
    assert conn.execute(
        "SELECT fact_id FROM fact_embeddings ORDER BY fact_id"
    ).fetchall() == [(1,)]


def test_cached_matrix_refreshes_after_another_connection_commits(tmp_path):
    path = tmp_path / "embeddings.db"
    reader_conn = sqlite3.connect(path)
    writer_conn = sqlite3.connect(path)
    reader = EmbedStore(reader_conn, embedding_identity=_DOC)
    writer = EmbedStore(writer_conn, embedding_identity=_DOC)
    query = np.array([1.0, 0.0], dtype=np.float32)
    writer.upsert(1, query)
    assert [fact_id for fact_id, _ in reader.score_all(query, _QUERY)] == [1]

    writer.upsert(2, np.array([0.0, 1.0], dtype=np.float32))

    assert {fact_id for fact_id, _ in reader.score_all(query, _QUERY)} == {1, 2}
    reader_conn.close()
    writer_conn.close()
