import sqlite3

import numpy as np

from enfold.embed_store import EmbedStore
from enfold.embeddings import embedding_to_bytes
from enfold.reflection import select_clusters


def test_cosine_cap_prioritizes_recent_facts_beyond_old_query_prefix():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE facts (fact_id INTEGER PRIMARY KEY, content TEXT, category TEXT)"
    )
    identity = "test:model:document:none:v1"
    embed_store = EmbedStore(conn, embedding_identity=identity)
    population = 2_002
    conn.executemany(
        "INSERT INTO facts (fact_id, content, category) VALUES (?, ?, 'general')",
        ((fact_id, f"Fact {fact_id}") for fact_id in range(1, population + 1)),
    )
    unrelated = embedding_to_bytes(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    recent_a = embedding_to_bytes(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    recent_b = embedding_to_bytes(np.array([0.8, 0.6, 0.0], dtype=np.float32))
    recent = {population - 1: recent_a, population: recent_b}
    conn.executemany(
        """
        INSERT INTO fact_embeddings (fact_id, embedding, dim, embedding_identity)
        VALUES (?, ?, 3, ?)
        """,
        (
            (
                fact_id,
                recent.get(fact_id, unrelated),
                identity,
            )
            for fact_id in range(1, population + 1)
        ),
    )
    conn.commit()

    clusters = select_clusters(
        conn,
        max_clusters=1,
        embed_store=embed_store,
        embedding_identity=identity,
        cosine_low=0.75,
        cosine_high=0.92,
    )

    assert clusters == [[population - 1, population]]
