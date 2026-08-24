import logging
import sqlite3

import numpy as np
import pytest

from enfold.embed_store import EmbedStore
from enfold.embeddings import embedding_to_bytes
from enfold.sqlite_vec_index import SQLiteVecIndex


_DOC = "test:model:document:none:v1"
_OLD = "test:old-model:document:none:v1"
_QUERY = "test:model:query:none:v1"


def _store():
    conn = sqlite3.connect(":memory:")
    return EmbedStore(conn, embedding_identity=_DOC)


def _vector(*values):
    return np.asarray(values, dtype=np.float32)


def _insert_canonical(conn, fact_id, vector, identity=_DOC):
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (fact_id, embedding_to_bytes(vector), len(vector), identity),
    )


def _install_vector_table(store, rows, *, identity=_DOC, fail_operation):
    conn = store._conn
    conn.execute("CREATE TABLE vector_rows(embedding BLOB NOT NULL)")
    conn.executemany(
        "INSERT INTO vector_rows(rowid, embedding) VALUES (?, ?)",
        [(fact_id, embedding_to_bytes(vector)) for fact_id, vector in rows],
    )
    conn.execute(
        f"""
        CREATE TRIGGER fail_vector_mutation
        BEFORE {fail_operation} ON vector_rows
        BEGIN
            SELECT RAISE(ABORT, 'forced vector failure');
        END
        """
    )
    conn.commit()
    store._vector_index = SQLiteVecIndex(conn, identity, 2, "vector_rows")


def test_failed_upsert_is_atomic_in_callers_transaction_and_invalidates_cache():
    store = _store()
    store.upsert(1, _vector(1.0, 0.0))
    _install_vector_table(store, [], fail_operation="INSERT")
    assert [fact_id for fact_id, _ in store.score_all(_vector(1.0, 0.0), _QUERY)] == [1]

    _insert_canonical(store._conn, 2, _vector(0.0, 1.0))
    with pytest.raises(sqlite3.IntegrityError, match="forced vector failure"):
        store.upsert(3, _vector(0.5, 0.5))

    assert store._conn.in_transaction
    assert store._conn.execute(
        "SELECT fact_id FROM fact_embeddings ORDER BY fact_id"
    ).fetchall() == [(1,), (2,)]
    assert store._conn.execute("SELECT rowid FROM vector_rows").fetchall() == []
    assert {
        fact_id for fact_id, _ in store.score_all(_vector(1.0, 0.0), _QUERY)
    } == {1, 2}


def test_failed_delete_is_atomic_in_callers_transaction_and_invalidates_cache():
    store = _store()
    store.upsert(1, _vector(1.0, 0.0))
    store.upsert(2, _vector(0.0, 1.0))
    _install_vector_table(
        store,
        [(1, _vector(1.0, 0.0)), (2, _vector(0.0, 1.0))],
        fail_operation="DELETE",
    )
    store.score_all(_vector(1.0, 0.0), _QUERY)

    _insert_canonical(store._conn, 3, _vector(0.5, 0.5))
    with pytest.raises(sqlite3.IntegrityError, match="forced vector failure"):
        store.delete(1)

    assert store._conn.in_transaction
    assert store._conn.execute(
        "SELECT fact_id FROM fact_embeddings ORDER BY fact_id"
    ).fetchall() == [(1,), (2,), (3,)]
    assert store._conn.execute(
        "SELECT rowid FROM vector_rows ORDER BY rowid"
    ).fetchall() == [(1,), (2,)]
    assert {
        fact_id for fact_id, _ in store.score_all(_vector(1.0, 0.0), _QUERY)
    } == {1, 2, 3}


def test_failed_prune_is_atomic_in_callers_transaction_and_invalidates_cache():
    store = _store()
    store.upsert(1, _vector(1.0, 0.0))
    store.upsert(2, _vector(0.0, 1.0), embedding_identity=_OLD)
    _install_vector_table(
        store,
        [(2, _vector(0.0, 1.0))],
        identity=_OLD,
        fail_operation="DELETE",
    )
    store.score_all(_vector(1.0, 0.0), _QUERY)

    _insert_canonical(store._conn, 3, _vector(0.5, 0.5))
    with pytest.raises(sqlite3.IntegrityError, match="forced vector failure"):
        store.prune_identities({_DOC})

    assert store._conn.in_transaction
    assert store.identity_counts() == {_DOC: 2, _OLD: 1}
    assert store._conn.execute("SELECT rowid FROM vector_rows").fetchall() == [(2,)]
    assert {
        fact_id for fact_id, _ in store.score_all(_vector(1.0, 0.0), _QUERY)
    } == {1, 3}


@pytest.mark.parametrize(
    "vector, message",
    [
        (np.asarray([], dtype=np.float32), "non-empty 1-D"),
        (np.asarray([[1.0, 0.0]], dtype=np.float32), "non-empty 1-D"),
        (np.asarray([np.nan, 0.0], dtype=np.float32), "finite"),
        (np.asarray([np.inf, 0.0], dtype=np.float32), "finite"),
    ],
)
def test_upsert_rejects_invalid_vectors(vector, message):
    store = _store()

    with pytest.raises(ValueError, match=message):
        store.upsert(1, vector)

    assert store.identity_counts() == {}
    assert not store._conn.in_transaction


def test_dense_scoring_skips_blobs_whose_length_disagrees_with_dim(caplog):
    store = _store()
    store.upsert(1, _vector(1.0, 0.0))
    store._conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (2, embedding_to_bytes(_vector(1.0)), 2, _DOC),
    )
    store._conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (3, b"x", 2, _DOC),
    )
    store._conn.commit()

    with caplog.at_level(logging.DEBUG, logger="enfold.embed_store"):
        results = store.score_all(_vector(1.0, 0.0), _QUERY)

    assert [fact_id for fact_id, _ in results] == [1]
    assert caplog.text.count("Skipping embedding for fact") == 2
