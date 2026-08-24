"""PlusFactRetriever: encode-once hot path and parent equivalence."""

import fake_hermes
import pytest

CONTENTS = [
    ("The user prefers pnpm for all node projects", "tool", "pnpm,node"),
    ("The deploy target for web projects is vercel", "tool", "deploy,vercel"),
    ("The tracker app uses sqlite for the fact store", "project", "tracker,sqlite"),
    ("The gateway restarts are scheduled overnight", "general", ""),
    ("The user keeps projects under the home projects directory", "project", "projects"),
    ("Node version is managed with mise for projects", "tool", "node,mise"),
    ("The memory plugin stores facts in sqlite", "project", "memory,sqlite"),
    ("The user likes dark themed dashboards for projects", "user_pref", "ui,dark"),
]


@pytest.fixture()
def populated_store(tmp_path):
    store = fake_hermes.MemoryStore(db_path=tmp_path / "facts.db", hrr_dim=64)
    for content, category, tags in CONTENTS:
        store.add_fact(content, category=category, tags=tags)
    yield store
    store.close()


def _spy_encode_text(monkeypatch):
    calls = []
    original = fake_hermes.hrr.encode_text

    def spy(text, dim=1024):
        calls.append(text)
        return original(text, dim)

    monkeypatch.setattr(fake_hermes.hrr, "encode_text", spy)
    return calls


def _add_lifecycle_columns(store):
    store._conn.executescript(
        """
        ALTER TABLE facts ADD COLUMN invalid_at TEXT;
        ALTER TABLE facts ADD COLUMN superseded_by INTEGER;
        ALTER TABLE facts ADD COLUMN conflict_group TEXT;
        ALTER TABLE facts ADD COLUMN scope TEXT NOT NULL DEFAULT 'private';
        ALTER TABLE facts ADD COLUMN sensitivity TEXT NOT NULL DEFAULT 'normal';
        """
    )
    store._conn.commit()


def test_query_encoded_exactly_once_per_search(hp, populated_store, monkeypatch):
    retriever = hp.retrieval_plus.PlusFactRetriever(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    calls = _spy_encode_text(monkeypatch)
    results = retriever.search("projects", min_trust=0.0, limit=3)
    assert results, "expected FTS matches for 'projects'"
    assert calls == ["projects"], "query must be HRR-encoded exactly once"


def test_parent_encodes_per_candidate_baseline(populated_store, monkeypatch):
    # Baseline check that the parent really does re-encode per candidate,
    # so the test above is meaningful.
    retriever = fake_hermes.FactRetriever(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    calls = _spy_encode_text(monkeypatch)
    results = retriever.search("projects", min_trust=0.0, limit=3)
    assert results
    assert len(calls) > 1


def test_single_token_queries_match_parent_exactly(hp, populated_store):
    """Single-token queries: byte-identical to the parent.

    With one token there is no AND-vs-OR difference (the parent's raw
    ``facts_fts MATCH 'projects'`` and the sanitised ``MATCH '"projects"'``
    select the same rows), so the hot-path optimisations must reproduce the
    parent's ids and scores exactly.
    """
    kwargs = dict(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    parent = fake_hermes.FactRetriever(**kwargs)
    plus = hp.retrieval_plus.PlusFactRetriever(**kwargs)

    for query in ["projects", "user", "sqlite", "node"]:
        expected = parent.search(query, min_trust=0.0, limit=5)
        actual = plus.search(query, min_trust=0.0, limit=5)
        assert [f["fact_id"] for f in actual] == [f["fact_id"] for f in expected], query
        for a, e in zip(actual, expected):
            assert a["score"] == pytest.approx(e["score"])
            assert "hrr_vector" not in a


def test_multi_token_recall_is_a_parent_superset(hp, populated_store):
    """Multi-token queries: a strict recall improvement over the parent.

    The parent feeds the raw query to FTS5, which ANDs every token, so a
    fact must contain ALL tokens to be a candidate (and any hyphen/punctuation
    silently errors the whole match out). The sanitiser ORs the significant
    tokens, so PlusFactRetriever finds a SUPERSET of the parent's candidates:
    every fact the parent returned is still present, plus additional genuine
    lexical matches the parent's AND-semantics missed. The shared facts keep
    the parent's relative order.
    """
    kwargs = dict(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    parent = fake_hermes.FactRetriever(**kwargs)
    plus = hp.retrieval_plus.PlusFactRetriever(**kwargs)

    saw_strict_superset = False
    for query in ["sqlite store", "node deploy", "memory facts", "dark dashboards"]:
        expected = parent.search(query, min_trust=0.0, limit=10)
        actual = plus.search(query, min_trust=0.0, limit=10)
        exp_ids = [f["fact_id"] for f in expected]
        act_ids = [f["fact_id"] for f in actual]
        # Superset: every parent hit is still retrieved.
        assert set(exp_ids).issubset(set(act_ids)), f"{query}: {exp_ids} !subset {act_ids}"
        # Shared facts keep the parent's relative order.
        shared = [fid for fid in act_ids if fid in set(exp_ids)]
        assert shared == exp_ids, f"{query}: shared order {shared} != parent {exp_ids}"
        for a in actual:
            assert "hrr_vector" not in a
        if len(act_ids) > len(exp_ids):
            saw_strict_superset = True
    assert saw_strict_superset, (
        "expected at least one query where the sanitised OR recovers facts the "
        "parent's AND-semantics missed"
    )


def test_category_and_trust_filters_match_parent(hp, populated_store):
    kwargs = dict(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    parent = fake_hermes.FactRetriever(**kwargs)
    plus = hp.retrieval_plus.PlusFactRetriever(**kwargs)
    expected = parent.search("projects", category="project", min_trust=0.0, limit=5)
    actual = plus.search("projects", category="project", min_trust=0.0, limit=5)
    assert [f["fact_id"] for f in actual] == [f["fact_id"] for f in expected]
    assert all(f["category"] == "project" for f in actual)


def test_fts_candidates_exclude_hrr_blob(hp, populated_store):
    plus = hp.retrieval_plus.PlusFactRetriever(
        store=populated_store, hrr_dim=64,
        fts_weight=3 / 7, jaccard_weight=2 / 7, hrr_weight=2 / 7,
    )
    candidates = plus._fts_candidates("projects", None, 0.0, 30)
    assert candidates
    for fact in candidates:
        assert "hrr_vector" not in fact
        assert "fts_rank" in fact


def test_fts_candidates_filter_lifecycle_and_scope_before_limit(hp, populated_store):
    _add_lifecycle_columns(populated_store)
    rows = populated_store._conn.execute(
        "SELECT fact_id, content FROM facts WHERE content LIKE '%projects%'"
    ).fetchall()
    ids = {row["content"]: int(row["fact_id"]) for row in rows}
    eligible = ids["The user keeps projects under the home projects directory"]
    populated_store._conn.execute(
        "UPDATE facts SET invalid_at = '2026-01-01' WHERE fact_id = ?",
        (ids["The user prefers pnpm for all node projects"],),
    )
    populated_store._conn.execute(
        "UPDATE facts SET superseded_by = ? WHERE fact_id = ?",
        (eligible, ids["The deploy target for web projects is vercel"]),
    )
    populated_store._conn.execute(
        "UPDATE facts SET conflict_group = 'open' WHERE fact_id = ?",
        (ids["Node version is managed with mise for projects"],),
    )
    populated_store._conn.execute(
        "UPDATE facts SET scope = 'work' WHERE fact_id = ?",
        (ids["The user likes dark themed dashboards for projects"],),
    )
    populated_store._conn.commit()
    plus = hp.retrieval_plus.PlusFactRetriever(
        store=populated_store,
        hrr_dim=64,
        allowed_scopes=("private",),
    )

    candidates = plus._fts_candidates("projects", None, 0.0, 1)
    results = plus.search("projects", min_trust=0.0, limit=10)

    assert [fact["fact_id"] for fact in candidates] == [eligible]
    assert [fact["fact_id"] for fact in results] == [eligible]


def test_conflicts_stay_excluded_when_lifecycle_filter_is_disabled(
    hp, populated_store
):
    _add_lifecycle_columns(populated_store)
    conflicted = populated_store._conn.execute(
        "SELECT fact_id FROM facts WHERE content LIKE '%home projects%'"
    ).fetchone()["fact_id"]
    populated_store._conn.execute(
        "UPDATE facts SET conflict_group = 'open' WHERE fact_id = ?",
        (conflicted,),
    )
    populated_store._conn.commit()
    plus = hp.retrieval_plus.PlusFactRetriever(
        store=populated_store,
        hrr_dim=64,
        lifecycle_filter=False,
    )

    results = plus.search("home projects", min_trust=0.0, limit=10)

    assert conflicted not in {fact["fact_id"] for fact in results}


def test_sensitive_candidates_require_matching_capability(hp, populated_store):
    _add_lifecycle_columns(populated_store)
    sensitive = populated_store.add_fact("Sensitive projects launch note")
    populated_store._conn.execute(
        "UPDATE facts SET sensitivity = 'sensitive' WHERE fact_id = ?",
        (sensitive,),
    )
    populated_store._conn.commit()

    ordinary = hp.retrieval_plus.PlusFactRetriever(
        store=populated_store,
        hrr_dim=64,
        allowed_scopes=("private",),
    ).search("projects", min_trust=0.0, limit=20)
    privileged = hp.retrieval_plus.PlusFactRetriever(
        store=populated_store,
        hrr_dim=64,
        allowed_scopes=("private", "sensitive"),
    ).search("projects", min_trust=0.0, limit=20)

    assert sensitive not in {fact["fact_id"] for fact in ordinary}
    assert sensitive in {fact["fact_id"] for fact in privileged}


def test_explain_candidates_preserve_normal_filtered_window(hp, tmp_path):
    store = fake_hermes.MemoryStore(db_path=tmp_path / "window.db", hrr_dim=64)
    try:
        _add_lifecycle_columns(store)
        invalid_ids = []
        for i in range(4):
            fact_id = store.add_fact(f"Windowprobe retired candidate {i}")
            invalid_ids.append(fact_id)
            store._conn.execute(
                "UPDATE facts SET invalid_at = '2026-01-01' WHERE fact_id = ?",
                (fact_id,),
            )
        live_id = store.add_fact("Windowprobe live candidate")
        store._conn.commit()
        plus = hp.retrieval_plus.PlusFactRetriever(
            store=store,
            hrr_dim=64,
            hrr_weight=0.0,
        )

        normal = plus.search("windowprobe", min_trust=0.0, limit=1)
        explained = plus.search(
            "windowprobe", min_trust=0.0, limit=1, explain=True
        )

        assert [fact["fact_id"] for fact in normal] == [live_id]
        assert live_id in {fact["fact_id"] for fact in explained}
        assert set(invalid_ids) & {fact["fact_id"] for fact in explained}
    finally:
        store.close()


def test_entity_expansion_filters_lifecycle_and_scope(hp, tmp_path):
    store = fake_hermes.MemoryStore(db_path=tmp_path / "entities.db", hrr_dim=64)
    try:
        _add_lifecycle_columns(store)
        direct = store.add_fact("Orchid rollout is led by Alex Rivera")
        eligible = store.add_fact("Alex Rivera owns the private roadmap")
        invalid = store.add_fact("Alex Rivera owns an invalid roadmap")
        superseded = store.add_fact("Alex Rivera owns an obsolete roadmap")
        conflicted = store.add_fact("Alex Rivera owns a disputed roadmap")
        work = store.add_fact("Alex Rivera owns the work roadmap")
        store._conn.execute(
            "UPDATE facts SET invalid_at = '2026-01-01' WHERE fact_id = ?",
            (invalid,),
        )
        store._conn.execute(
            "UPDATE facts SET superseded_by = ? WHERE fact_id = ?",
            (eligible, superseded),
        )
        store._conn.execute(
            "UPDATE facts SET conflict_group = 'open' WHERE fact_id = ?",
            (conflicted,),
        )
        store._conn.execute(
            "UPDATE facts SET scope = 'work' WHERE fact_id = ?",
            (work,),
        )
        store._conn.commit()
        plus = hp.retrieval_plus.PlusFactRetriever(
            store=store,
            hrr_dim=64,
            entity_expansion=True,
            allowed_scopes=("private",),
        )

        results = plus.search("Orchid", min_trust=0.0, limit=10)

        assert [fact["fact_id"] for fact in results] == [direct, eligible]
        assert results[1]["expanded_from_entity"] == "Alex Rivera"
    finally:
        store.close()


@pytest.mark.parametrize("malformed_blob", [b"x", b"\0" * 16])
def test_malformed_hrr_vector_uses_neutral_score(
    hp, populated_store, malformed_blob
):
    target = populated_store._conn.execute(
        "SELECT fact_id FROM facts WHERE content LIKE '%home projects%'"
    ).fetchone()["fact_id"]
    populated_store._conn.execute(
        "UPDATE facts SET hrr_vector = ? WHERE fact_id = ?",
        (malformed_blob, target),
    )
    populated_store._conn.commit()
    plus = hp.retrieval_plus.PlusFactRetriever(
        store=populated_store,
        hrr_dim=64,
    )

    results = plus.search("home projects", min_trust=0.0, limit=10, explain=True)

    matched = next(fact for fact in results if fact["fact_id"] == target)
    assert matched["_breakdown"]["hrr_score"] == 0.5


def test_no_blob_loads_when_hrr_disabled(hp, populated_store, monkeypatch):
    plus = hp.retrieval_plus.PlusFactRetriever(
        store=populated_store, hrr_dim=64,
        fts_weight=0.6, jaccard_weight=0.4, hrr_weight=0.0,
    )
    calls = _spy_encode_text(monkeypatch)
    loads = []
    original = hp.retrieval_plus.PlusFactRetriever._load_hrr_vectors
    monkeypatch.setattr(
        hp.retrieval_plus.PlusFactRetriever,
        "_load_hrr_vectors",
        lambda self, ids: loads.append(ids) or original(self, ids),
    )
    results = plus.search("projects", min_trust=0.0, limit=3)
    assert results
    assert calls == []
    assert loads == []


def test_malformed_fts_query_returns_empty(hp, populated_store):
    plus = hp.retrieval_plus.PlusFactRetriever(store=populated_store, hrr_dim=64)
    assert plus.search('"unbalanced AND (', min_trust=0.0, limit=5) == []
