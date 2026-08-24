"""Offline near-duplicate cluster merge tool (cluster_merge.py).

Covers union-find clustering by embedding cosine, survivor selection (pre-
existing-fact bias for clusters spanning a flood cutoff, else
trust*retrieval/earliest), the merge plan, and the guard rails on execution
(live-path refusal, backup-file requirement, drop-count band, dry-run
default).
"""

import os
import sqlite3

import numpy as np
import pytest

import enfold.cluster_merge as cluster_merge
from enfold.cluster_merge import (
    build_clusters,
    choose_survivor,
    plan_merge,
    execute_merge,
    GuardRailError,
    main,
)

_SCHEMA = """
CREATE TABLE facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    category        TEXT DEFAULT 'general',
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    invalid_at      TIMESTAMP,
    superseded_by   INTEGER,
    conflict_group  TEXT,
    scope           TEXT NOT NULL DEFAULT 'private',
    hrr_vector      BLOB
);
CREATE TABLE fact_entities (
    fact_id   INTEGER,
    entity_id INTEGER,
    PRIMARY KEY (fact_id, entity_id)
);
CREATE VIRTUAL TABLE facts_fts USING fts5(content, tags, content=facts, content_rowid=fact_id);
CREATE TABLE fact_embeddings (
    fact_id    INTEGER NOT NULL,
    embedding  BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    embedding_identity TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (fact_id, embedding_identity)
);
"""

_ID = "test:model:document:none:v1"


def _vec(*xs):
    return np.array(xs, dtype=np.float32)


def _conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(_SCHEMA)
    c.commit()
    return c


def _add_fact(conn, content, created_at, trust=0.5, retrieval=0, helpful=0,
              tags="", category="general", invalid_at=None,
              superseded_by=None, conflict_group=None, scope="private"):
    cur = conn.execute(
        "INSERT INTO facts (content, category, tags, trust_score, retrieval_count, "
        "helpful_count, created_at, updated_at, invalid_at, superseded_by, "
        "conflict_group, scope) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            content,
            category,
            tags,
            trust,
            retrieval,
            helpful,
            created_at,
            created_at,
            invalid_at,
            superseded_by,
            conflict_group,
            scope,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def _embed(conn, fact_id, vec, identity=_ID):
    conn.execute(
        "INSERT INTO fact_embeddings (fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, ?, ?)",
        (fact_id, np.asarray(vec, dtype="<f4").tobytes(), len(vec), identity),
    )
    conn.commit()


def _backup_database(db_path, backup_path):
    source = sqlite3.connect(str(db_path))
    backup = sqlite3.connect(str(backup_path))
    source.backup(backup)
    backup.close()
    source.close()


# ---------------------------------------------------------------------------
# build_clusters
# ---------------------------------------------------------------------------

def test_build_clusters_groups_near_identical_vectors():
    conn = _conn()
    a = _add_fact(conn, "a", "2026-06-01 00:00:00")
    b = _add_fact(conn, "b", "2026-06-01 00:00:01")
    c = _add_fact(conn, "c", "2026-06-01 00:00:02")
    _embed(conn, a, _vec(1.0, 0.0))
    _embed(conn, b, _vec(0.999, 0.001))
    _embed(conn, c, _vec(0.0, 1.0))
    clusters = build_clusters(conn, threshold=0.92, embedding_identity=_ID)
    assert sorted(map(sorted, clusters)) == [sorted([a, b])]


def test_build_clusters_transitive_chain_merges():
    conn = _conn()
    a = _add_fact(conn, "a", "2026-06-01 00:00:00")
    b = _add_fact(conn, "b", "2026-06-01 00:00:01")
    c = _add_fact(conn, "c", "2026-06-01 00:00:02")
    # a~b close, b~c close, a~c not directly close enough alone -> still one cluster via union-find
    _embed(conn, a, _vec(1.0, 0.0, 0.0))
    _embed(conn, b, _vec(0.95, 0.31, 0.0))
    _embed(conn, c, _vec(0.85, 0.5, 0.17))
    clusters = build_clusters(conn, threshold=0.9, embedding_identity=_ID)
    assert len(clusters) == 1
    assert set(clusters[0]) == {a, b, c}


def test_build_clusters_requires_identity_when_multiple_are_present():
    conn = _conn()
    a = _add_fact(conn, "service endpoint", "2026-06-01 00:00:00")
    b = _add_fact(conn, "endpoint for the service", "2026-06-01 00:00:01")
    _embed(conn, a, _vec(1.0, 0.0), identity=_ID)
    _embed(conn, b, _vec(0.999, 0.001), identity=_ID)
    _embed(conn, a, _vec(0.0, 1.0), identity="other:model:document:none:v1")

    with pytest.raises(GuardRailError, match="multiple embedding identities") as exc_info:
        build_clusters(conn, threshold=0.92)
    assert _ID in str(exc_info.value)
    assert "other:model:document:none:v1" in str(exc_info.value)

    assert build_clusters(conn, threshold=0.92, embedding_identity=_ID) == [[a, b]]


@pytest.mark.parametrize(
    ("first", "second"),
    [
        ("dashboard listens on port 3000", "dashboard listens on port 3100"),
        ("dashboard is enabled", "dashboard is disabled"),
    ],
)
def test_build_clusters_blocks_semantically_changed_facts(first, second):
    conn = _conn()
    a = _add_fact(conn, first, "2026-06-01 00:00:00")
    b = _add_fact(conn, second, "2026-06-01 00:00:01")
    _embed(conn, a, _vec(1.0, 0.0))
    _embed(conn, b, _vec(1.0, 0.0))

    assert build_clusters(conn, threshold=0.92, embedding_identity=_ID) == []


def test_build_clusters_ignores_singletons():
    conn = _conn()
    a = _add_fact(conn, "a", "2026-06-01 00:00:00")
    b = _add_fact(conn, "b", "2026-06-01 00:00:01")
    _embed(conn, a, _vec(1.0, 0.0))
    _embed(conn, b, _vec(0.0, 1.0))
    assert build_clusters(conn, threshold=0.92, embedding_identity=_ID) == []


def test_build_clusters_excludes_structurally_invalid_and_legacy_superseded_facts():
    conn = _conn()
    active = _add_fact(conn, "current routing fact", "2026-06-01 00:00:00")
    invalid = _add_fact(
        conn,
        "old routing fact",
        "2026-05-01 00:00:00",
        invalid_at="2026-06-02 00:00:00",
    )
    legacy = _add_fact(
        conn,
        "SUPERSEDED 2026-06-02: older routing fact",
        "2026-04-01 00:00:00",
    )
    _embed(conn, active, _vec(1.0, 0.0))
    _embed(conn, invalid, _vec(0.999, 0.001))
    _embed(conn, legacy, _vec(0.999, 0.002))

    assert build_clusters(conn, threshold=0.92, embedding_identity=_ID) == []


def test_build_clusters_partitions_scope_and_excludes_conflict_state():
    conn = _conn()
    private = _add_fact(
        conn, "service endpoint location", "2026-06-01 00:00:00", scope="private"
    )
    team = _add_fact(
        conn, "location of the service endpoint", "2026-06-01 00:00:01", scope="team"
    )
    conflicted = _add_fact(
        conn,
        "service endpoint address",
        "2026-06-01 00:00:02",
        conflict_group="open-conflict",
    )
    superseded = _add_fact(
        conn,
        "address for the service endpoint",
        "2026-06-01 00:00:03",
        superseded_by=private,
    )
    for fact_id in (private, team, conflicted, superseded):
        _embed(conn, fact_id, _vec(1.0, 0.0))

    assert build_clusters(conn, threshold=0.92, embedding_identity=_ID) == []


# ---------------------------------------------------------------------------
# choose_survivor
# ---------------------------------------------------------------------------

def test_choose_survivor_prefers_pre_existing_when_cluster_spans_cutoff():
    conn = _conn()
    pre = _add_fact(conn, "pre-existing statement", "2026-06-20 00:00:00", trust=0.5, retrieval=1)
    flood1 = _add_fact(conn, "flood restatement one", "2026-06-30 05:01:00", trust=0.9, retrieval=9)
    flood2 = _add_fact(conn, "flood restatement two", "2026-06-30 05:01:04", trust=0.9, retrieval=9)
    survivor, losers = choose_survivor(
        conn, [pre, flood1, flood2], flood_cutoff="2026-06-29 18:00:00"
    )
    assert survivor == pre
    assert set(losers) == {flood1, flood2}


def test_choose_survivor_all_flood_picks_highest_trust_then_earliest():
    conn = _conn()
    a = _add_fact(conn, "a", "2026-06-30 05:01:00", trust=0.5, retrieval=2)
    b = _add_fact(conn, "b", "2026-06-30 05:01:04", trust=0.7, retrieval=2)
    c = _add_fact(conn, "c", "2026-06-30 05:01:08", trust=0.7, retrieval=2)
    survivor, losers = choose_survivor(conn, [a, b, c], flood_cutoff="2026-06-29 18:00:00")
    # b and c tie on trust*retrieval (1.4) and both beat a (1.0) -> earliest of the tie wins
    assert survivor == b
    assert set(losers) == {a, c}


def test_choose_survivor_all_pre_existing_picks_by_trust_times_retrieval():
    conn = _conn()
    a = _add_fact(conn, "a", "2026-06-01 00:00:00", trust=0.5, retrieval=10)
    b = _add_fact(conn, "b", "2026-06-02 00:00:00", trust=0.9, retrieval=10)
    survivor, losers = choose_survivor(conn, [a, b], flood_cutoff="2026-06-29 18:00:00")
    assert survivor == b
    assert losers == [a]


def test_plan_merge_does_not_delete_active_replacement_for_invalid_high_score_fact():
    conn = _conn()
    invalid = _add_fact(
        conn,
        "old dashboard port is 3000",
        "2026-06-01 00:00:00",
        trust=0.99,
        retrieval=100,
        invalid_at="2026-06-15 00:00:00",
    )
    active = _add_fact(
        conn,
        "dashboard port is 3100",
        "2026-06-20 00:00:00",
        trust=0.5,
        retrieval=1,
    )
    _embed(conn, invalid, _vec(1.0, 0.0))
    _embed(conn, active, _vec(0.999, 0.001))

    plan = plan_merge(
        conn,
        threshold=0.92,
        flood_cutoff="2026-06-29 18:00:00",
        embedding_identity=_ID,
    )

    assert plan.clusters == []


# ---------------------------------------------------------------------------
# plan_merge
# ---------------------------------------------------------------------------

def test_plan_merge_merges_counts_and_tags():
    conn = _conn()
    pre = _add_fact(conn, "pre-existing statement", "2026-06-20 00:00:00",
                     trust=0.5, retrieval=3, helpful=1, tags="alpha")
    flood = _add_fact(conn, "flood restatement", "2026-06-30 05:01:00",
                       trust=0.9, retrieval=5, helpful=2, tags="beta")
    _embed(conn, pre, _vec(1.0, 0.0))
    _embed(conn, flood, _vec(0.999, 0.001))
    plan = plan_merge(conn, threshold=0.92, flood_cutoff="2026-06-29 18:00:00",
                       embedding_identity=_ID)
    assert len(plan.clusters) == 1
    m = plan.clusters[0]
    assert m.survivor_id == pre
    assert m.loser_ids == [flood]
    assert m.merged_retrieval_count == 8
    assert m.merged_helpful_count == 3
    assert set(m.merged_tags.split(",")) == {"alpha", "beta"}
    assert plan.drop_count == 1
    assert plan.projected_final_count == 1


def test_plan_merge_flags_suspicious_oversized_cluster():
    conn = _conn()
    ids = []
    for i in range(30):
        suffix = chr(ord("a") + i // 26) + chr(ord("a") + i % 26)
        fid = _add_fact(conn, f"restatement {suffix}", f"2026-06-30 05:0{i%6}:0{i%9}")
        _embed(conn, fid, _vec(1.0, 0.0001 * i))
        ids.append(fid)
    plan = plan_merge(conn, threshold=0.9, flood_cutoff="2026-06-29 18:00:00",
                       embedding_identity=_ID, suspicious_cluster_size=20)
    assert plan.clusters[0].suspicious is True


# ---------------------------------------------------------------------------
# execute_merge guard rails
# ---------------------------------------------------------------------------

def _seed_two_dup_facts(conn):
    a = _add_fact(conn, "a restatement", "2026-06-30 05:01:00", trust=0.5, retrieval=1)
    b = _add_fact(conn, "a restatement again", "2026-06-30 05:01:04", trust=0.5, retrieval=1)
    _embed(conn, a, _vec(1.0, 0.0))
    _embed(conn, b, _vec(0.999, 0.001))
    return a, b


def test_execute_merge_defaults_to_dry_run(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    a, b = _seed_two_dup_facts(conn)
    conn.close()
    (tmp_path / "store.db.backup").write_bytes(b"x")

    result = execute_merge(str(db_path), threshold=0.92,
                            flood_cutoff="2026-06-29 18:00:00",
                            embedding_identity=_ID,
                            backup_path=str(tmp_path / "store.db.backup"),
                            expected_drop_min=0, expected_drop_max=10)
    assert result.dry_run is True
    assert result.drop_count == 1

    conn2 = sqlite3.connect(str(db_path))
    remaining = conn2.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert remaining == 2  # nothing actually deleted in dry-run


def test_execute_merge_refuses_live_hermes_path(tmp_path):
    fake_home = tmp_path / ".hermes"
    fake_home.mkdir()
    db_path = fake_home / "memory_store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()
    with pytest.raises(GuardRailError, match="hermes"):
        execute_merge(str(db_path), threshold=0.92,
                      flood_cutoff="2026-06-29 18:00:00",
                      embedding_identity=_ID,
                      backup_path=str(tmp_path / "nope.backup"),
                      dry_run=False,
                      expected_drop_min=0, expected_drop_max=10)


def test_execute_merge_refuses_live_hermes_path_via_symlink(tmp_path):
    # A symlink whose own literal path does NOT contain ".hermes" but which
    # resolves into a real .hermes directory must be refused just like a
    # direct literal path would be.
    real_hermes_dir = tmp_path / "real_store" / ".hermes"
    real_hermes_dir.mkdir(parents=True)
    real_db = real_hermes_dir / "memory_store.db"
    conn = sqlite3.connect(str(real_db))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    conn.close()

    sneaky_link = tmp_path / "sneaky_link"
    os.symlink(real_hermes_dir, sneaky_link)
    db_path = sneaky_link / "memory_store.db"
    assert ".hermes" not in str(db_path).split("/")

    with pytest.raises(GuardRailError, match="hermes"):
        execute_merge(str(db_path), threshold=0.92,
                      flood_cutoff="2026-06-29 18:00:00",
                      embedding_identity=_ID,
                      backup_path=str(tmp_path / "nope.backup"),
                      dry_run=False,
                      expected_drop_min=0, expected_drop_max=10)


def test_execute_merge_refuses_configured_live_hermes_home(tmp_path, monkeypatch):
    hermes_home = tmp_path / "custom-live-home"
    hermes_home.mkdir()
    db_path = hermes_home / "memory_store.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.close()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))

    with pytest.raises(GuardRailError, match="live hermes"):
        execute_merge(
            str(db_path), threshold=0.92,
            flood_cutoff="2026-06-29 18:00:00",
            embedding_identity=_ID,
            backup_path=str(tmp_path / "backup.db"),
            dry_run=False,
            expected_drop_min=0, expected_drop_max=10,
        )


def test_execute_merge_refuses_without_backup_file(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    _seed_two_dup_facts(conn)
    conn.close()

    with pytest.raises(GuardRailError, match="backup"):
        execute_merge(str(db_path), threshold=0.92,
                      flood_cutoff="2026-06-29 18:00:00",
                      embedding_identity=_ID,
                      backup_path=str(tmp_path / "missing.backup"),
                      dry_run=False,
                      expected_drop_min=0, expected_drop_max=10)


def test_execute_merge_requires_distinct_readable_sqlite_backup(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.close()
    backup_dir = tmp_path / "backup-dir"
    backup_dir.mkdir()
    garbage = tmp_path / "garbage.db"
    garbage.write_bytes(b"not sqlite")

    for backup_path, message in (
        (backup_dir, "readable file"),
        (db_path, "distinct"),
        (garbage, "SQLite"),
    ):
        with pytest.raises(GuardRailError, match=message):
            execute_merge(
                str(db_path), threshold=0.92,
                flood_cutoff="2026-06-29 18:00:00",
                embedding_identity=_ID,
                backup_path=str(backup_path),
                dry_run=False,
                expected_drop_min=0, expected_drop_max=10,
            )


def test_execute_merge_rejects_healthy_sqlite_from_another_database(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _seed_two_dup_facts(conn)
    conn.close()

    unrelated_path = tmp_path / "unrelated.db"
    unrelated = sqlite3.connect(str(unrelated_path))
    unrelated.row_factory = sqlite3.Row
    unrelated.executescript(_SCHEMA)
    _add_fact(
        unrelated,
        "unrelated healthy database",
        "2026-06-01 00:00:00",
    )
    unrelated.close()

    with pytest.raises(GuardRailError, match="target database"):
        execute_merge(
            str(db_path), threshold=0.92,
            flood_cutoff="2026-06-29 18:00:00",
            embedding_identity=_ID,
            backup_path=str(unrelated_path),
            dry_run=False,
            expected_drop_min=0, expected_drop_max=10,
        )


def test_execute_merge_dry_run_does_not_create_embedding_indexes(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    _seed_two_dup_facts(conn)
    before = conn.execute(
        "SELECT name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()
    conn.close()

    result = execute_merge(
        str(db_path), threshold=0.92,
        flood_cutoff="2026-06-29 18:00:00",
        embedding_identity=_ID,
        backup_path=str(tmp_path / "unused.backup"),
        expected_drop_min=0, expected_drop_max=10,
    )

    conn = sqlite3.connect(str(db_path))
    after = conn.execute(
        "SELECT name, sql FROM sqlite_master ORDER BY name"
    ).fetchall()
    conn.close()
    assert result.dry_run is True
    assert after == before


def test_execute_merge_refuses_when_drop_count_outside_band(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()
    _seed_two_dup_facts(conn)
    conn.close()
    _backup_database(db_path, tmp_path / "store.db.backup")

    with pytest.raises(GuardRailError, match="drop count"):
        execute_merge(str(db_path), threshold=0.92,
                      flood_cutoff="2026-06-29 18:00:00",
                      embedding_identity=_ID,
                      backup_path=str(tmp_path / "store.db.backup"),
                      dry_run=False,
                      expected_drop_min=5, expected_drop_max=10)


def test_execute_merge_refuses_when_drop_count_exceeds_relative_cap(tmp_path):
    # 10 active facts total: one cluster of 7 near-identical facts (6 losers,
    # 1 survivor) plus 3 standalone facts. 6 drops / 10 active facts = 60%,
    # under the absolute band [0, 10000] but over the default 0.5 relative
    # cap, so this must be refused even though the absolute check would pass.
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.commit()

    for i in range(7):
        fid = _add_fact(
            conn, f"restatement {chr(ord('a') + i)}", f"2026-06-30 05:0{i}:00"
        )
        _embed(conn, fid, _vec(1.0, 0.0001 * i), identity=_ID)
    for i in range(3):
        _add_fact(conn, f"standalone fact {i}", f"2026-06-30 05:1{i}:00")
    conn.close()
    _backup_database(db_path, tmp_path / "store.db.backup")

    with pytest.raises(GuardRailError, match="relative"):
        execute_merge(str(db_path), threshold=0.9,
                      flood_cutoff="2026-06-29 18:00:00",
                      embedding_identity=_ID,
                      backup_path=str(tmp_path / "store.db.backup"),
                      dry_run=False,
                      expected_drop_min=0, expected_drop_max=10_000)


def test_relative_cap_denominator_excludes_historical_facts(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    for i in range(3):
        fact_id = _add_fact(
            conn,
            f"active restatement {chr(ord('a') + i)}",
            f"2026-06-30 05:0{i}:00",
        )
        _embed(conn, fact_id, _vec(1.0, 0.0001 * i), identity=_ID)
    for i in range(10):
        _add_fact(
            conn,
            f"historical fact {i}",
            f"2026-05-{i + 1:02d} 00:00:00",
            invalid_at="2026-06-01 00:00:00",
        )
    conn.close()
    backup_path = tmp_path / "backup.db"
    _backup_database(db_path, backup_path)

    with pytest.raises(GuardRailError, match="3 active facts"):
        execute_merge(
            str(db_path), threshold=0.9,
            flood_cutoff="2026-06-29 18:00:00",
            embedding_identity=_ID,
            backup_path=str(backup_path),
            dry_run=False,
            expected_drop_min=0, expected_drop_max=10,
        )


def test_execute_merge_real_run_deletes_losers_and_checkpoints(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    conn.commit()
    a, b = _seed_two_dup_facts(conn)
    conn.close()
    _backup_database(db_path, tmp_path / "store.db.backup")

    result = execute_merge(str(db_path), threshold=0.92,
                            flood_cutoff="2026-06-29 18:00:00",
                            embedding_identity=_ID,
                            backup_path=str(tmp_path / "store.db.backup"),
                            dry_run=False,
                            expected_drop_min=0, expected_drop_max=10)
    assert result.dry_run is False
    assert result.drop_count == 1
    assert result.integrity_ok is True

    conn2 = sqlite3.connect(str(db_path))
    conn2.row_factory = sqlite3.Row
    remaining_ids = {r[0] for r in conn2.execute("SELECT fact_id FROM facts").fetchall()}
    assert remaining_ids == {a}
    embedded_ids = {r[0] for r in conn2.execute("SELECT fact_id FROM fact_embeddings").fetchall()}
    assert embedded_ids == {a}
    survivor = conn2.execute(
        "SELECT retrieval_count FROM facts WHERE fact_id = ?", (a,)
    ).fetchone()[0]
    assert survivor == 2  # 1 + 1 merged from the loser


def test_execute_merge_reports_projected_and_final_active_fact_counts(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    _seed_two_dup_facts(conn)
    _add_fact(
        conn,
        "Retired historical record",
        "2026-05-01 00:00:00",
        invalid_at="2026-06-01 00:00:00",
    )
    conn.close()
    backup_path = tmp_path / "backup.db"
    _backup_database(db_path, backup_path)

    result = execute_merge(
        str(db_path), threshold=0.92,
        flood_cutoff="2026-06-29 18:00:00",
        embedding_identity=_ID,
        backup_path=str(backup_path),
        dry_run=False,
        expected_drop_min=0, expected_drop_max=10,
    )

    assert result.projected_final_count == 1
    assert result.final_fact_count == result.projected_final_count
    with sqlite3.connect(str(db_path)) as check:
        assert check.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 2


def test_execute_merge_reassigns_or_deletes_every_fact_reference(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    conn.executescript(
        """
        CREATE TABLE fact_provenance (
            fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
            observation_id INTEGER NOT NULL,
            relation TEXT NOT NULL,
            PRIMARY KEY (fact_id, observation_id, relation)
        );
        CREATE TABLE memory_write_log (
            write_id TEXT PRIMARY KEY,
            fact_id INTEGER REFERENCES facts(fact_id),
            existing_fact_id INTEGER REFERENCES facts(fact_id)
        );
        CREATE TABLE privacy_erasure_log (
            erasure_id TEXT PRIMARY KEY,
            fact_id INTEGER NOT NULL REFERENCES facts(fact_id)
        );
        CREATE TABLE embedding_jobs (
            job_id INTEGER PRIMARY KEY,
            fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
            document_identity TEXT NOT NULL,
            UNIQUE (fact_id, document_identity)
        );
        CREATE TABLE fact_conflicts (
            conflict_id TEXT PRIMARY KEY,
            resolution_fact_id INTEGER REFERENCES facts(fact_id)
        );
        CREATE TABLE fact_conflict_members (
            conflict_id TEXT NOT NULL REFERENCES fact_conflicts(conflict_id),
            fact_id INTEGER NOT NULL REFERENCES facts(fact_id),
            PRIMARY KEY (conflict_id, fact_id)
        );
        CREATE TABLE fact_conflict_resolutions (
            conflict_id TEXT PRIMARY KEY REFERENCES fact_conflicts(conflict_id),
            resolution_fact_id INTEGER NOT NULL REFERENCES facts(fact_id)
        );
        """
    )
    survivor, loser = _seed_two_dup_facts(conn)
    referring = _add_fact(conn, "older dependent statement", "2026-05-01 00:00:00")
    conn.execute("UPDATE facts SET superseded_by = ? WHERE fact_id = ?", (loser, referring))
    conn.executemany(
        "INSERT INTO fact_entities (fact_id, entity_id) VALUES (?, 7)",
        ((survivor,), (loser,)),
    )
    conn.executemany(
        "INSERT INTO fact_provenance (fact_id, observation_id, relation) "
        "VALUES (?, 11, 'supports')",
        ((survivor,), (loser,)),
    )
    conn.execute(
        "INSERT INTO memory_write_log VALUES ('write', ?, ?)", (loser, loser)
    )
    conn.execute("INSERT INTO privacy_erasure_log VALUES ('erase', ?)", (loser,))
    conn.execute(
        "INSERT INTO embedding_jobs (fact_id, document_identity) VALUES (?, 'doc')",
        (loser,),
    )
    conn.execute("INSERT INTO fact_conflicts VALUES ('conflict', ?)", (loser,))
    conn.executemany(
        "INSERT INTO fact_conflict_members VALUES ('conflict', ?)",
        ((survivor,), (loser,)),
    )
    conn.execute(
        "INSERT INTO fact_conflict_resolutions VALUES ('conflict', ?)", (loser,)
    )
    conn.commit()
    conn.close()
    backup_path = tmp_path / "backup.db"
    _backup_database(db_path, backup_path)

    result = execute_merge(
        str(db_path), threshold=0.92,
        flood_cutoff="2026-06-29 18:00:00",
        embedding_identity=_ID,
        backup_path=str(backup_path),
        dry_run=False,
        expected_drop_min=0, expected_drop_max=10,
    )

    conn = sqlite3.connect(str(db_path))
    assert result.integrity_ok is True
    assert conn.execute(
        "SELECT superseded_by FROM facts WHERE fact_id = ?", (referring,)
    ).fetchone()[0] == survivor
    assert conn.execute("SELECT fact_id FROM fact_entities").fetchall() == [(survivor,)]
    assert conn.execute("SELECT fact_id FROM fact_provenance").fetchall() == [(survivor,)]
    assert conn.execute(
        "SELECT fact_id, existing_fact_id FROM memory_write_log"
    ).fetchone() == (survivor, survivor)
    assert conn.execute("SELECT fact_id FROM privacy_erasure_log").fetchone()[0] == survivor
    assert conn.execute("SELECT COUNT(*) FROM embedding_jobs").fetchone()[0] == 0
    assert conn.execute("SELECT fact_id FROM fact_embeddings").fetchall() == [(survivor,)]
    assert conn.execute("SELECT resolution_fact_id FROM fact_conflicts").fetchone()[0] == survivor
    assert conn.execute("SELECT fact_id FROM fact_conflict_members").fetchall() == [(survivor,)]
    assert conn.execute(
        "SELECT resolution_fact_id FROM fact_conflict_resolutions"
    ).fetchone()[0] == survivor
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()


def test_execute_merge_rewrites_active_reflection_source_tags(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    survivor, loser = _seed_two_dup_facts(conn)
    insight = _add_fact(
        conn,
        "The repeated statements support a stable preference.",
        "2026-07-01 00:00:00",
        category="insight",
        tags=f"reflection,source_facts:{survivor},{loser}",
    )
    conn.close()
    backup_path = tmp_path / "backup.db"
    _backup_database(db_path, backup_path)

    execute_merge(
        str(db_path), threshold=0.92,
        flood_cutoff="2026-06-29 18:00:00",
        embedding_identity=_ID,
        backup_path=str(backup_path),
        dry_run=False,
        expected_drop_min=0, expected_drop_max=10,
    )

    conn = sqlite3.connect(str(db_path))
    tags = conn.execute(
        "SELECT tags FROM facts WHERE fact_id = ?", (insight,)
    ).fetchone()[0]
    conn.close()
    assert tags == f"reflection,source_facts:{survivor}"


def test_execute_merge_locks_before_planning(tmp_path, monkeypatch):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    _seed_two_dup_facts(conn)
    conn.close()
    backup_path = tmp_path / "backup.db"
    _backup_database(db_path, backup_path)
    original = cluster_merge.plan_merge

    def assert_locked(conn, *args, **kwargs):
        assert conn.in_transaction
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        competing = sqlite3.connect(str(db_path), timeout=0)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            competing.execute("BEGIN IMMEDIATE")
        competing.close()
        return original(conn, *args, **kwargs)

    monkeypatch.setattr(cluster_merge, "plan_merge", assert_locked)
    result = execute_merge(
        str(db_path), threshold=0.92,
        flood_cutoff="2026-06-29 18:00:00",
        embedding_identity=_ID,
        backup_path=str(backup_path),
        dry_run=False,
        expected_drop_min=0, expected_drop_max=10,
    )
    assert result.drop_count == 1


def test_execute_merge_rolls_back_when_integrity_check_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    _seed_two_dup_facts(conn)
    conn.close()
    backup_path = tmp_path / "backup.db"
    _backup_database(db_path, backup_path)
    monkeypatch.setattr(cluster_merge, "_integrity_check", lambda conn: False)

    with pytest.raises(GuardRailError, match="integrity_check"):
        execute_merge(
            str(db_path), threshold=0.92,
            flood_cutoff="2026-06-29 18:00:00",
            embedding_identity=_ID,
            backup_path=str(backup_path),
            dry_run=False,
            expected_drop_min=0, expected_drop_max=10,
        )

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 2
    assert conn.execute("SELECT SUM(retrieval_count) FROM facts").fetchone()[0] == 2
    conn.close()


def test_execute_merge_orphan_check_fails_and_rolls_back(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.execute(
        "CREATE TABLE memory_write_log ("
        "write_id TEXT PRIMARY KEY, fact_id INTEGER REFERENCES facts(fact_id))"
    )
    _seed_two_dup_facts(conn)
    conn.execute("INSERT INTO memory_write_log VALUES ('orphan', 999)")
    conn.commit()
    conn.close()
    backup_path = tmp_path / "backup.db"
    _backup_database(db_path, backup_path)

    with pytest.raises(GuardRailError, match="orphaned fact reference"):
        execute_merge(
            str(db_path), threshold=0.92,
            flood_cutoff="2026-06-29 18:00:00",
            embedding_identity=_ID,
            backup_path=str(backup_path),
            dry_run=False,
            expected_drop_min=0, expected_drop_max=10,
        )

    conn = sqlite3.connect(str(db_path))
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == 2
    conn.close()


def test_cli_returns_nonzero_on_merge_failure(tmp_path):
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA)
    conn.close()

    assert main([
        str(db_path),
        "--flood-cutoff", "2026-06-29 18:00:00",
        "--embedding-identity", _ID,
        "--backup-path", str(tmp_path / "missing.db"),
        "--execute",
    ]) == 1
