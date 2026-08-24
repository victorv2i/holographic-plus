"""Offline near-duplicate cluster merge tool.

Standalone maintenance script for a ``enfold`` SQLite fact store.
Clusters facts by dense-embedding cosine similarity (union-find), picks one
survivor per cluster, merges retrieval/helpful counts and tags onto it, and
deletes the losers. Meant to run against a *copy* of a fact store, never the
live path, and defaults to a dry run.

Survivor selection:
    - If a cluster spans both a pre-existing fact (created before
      *flood_cutoff*) and a flood fact (created at/after it), the
      pre-existing fact survives: it already carries real trust/retrieval
      history, and the flood fact is a paraphrase restatement of it.
    - Otherwise, the fact with the highest ``trust_score * retrieval_count``
      survives; ties break on the earliest ``created_at`` (the original
      statement, not a later restatement).

Guard rails on ``execute_merge``:
    - ``dry_run`` defaults to True. A real run must be requested explicitly.
    - Refuses any *db_path* under a ``.hermes`` directory, live or not,
      checking both the literal path and its resolved (realpath) form so a
      symlink cannot bypass the check.
    - Requires *backup_path* to be a distinct, readable SQLite database.
    - Refuses unless the computed drop count falls within
      [*expected_drop_min*, *expected_drop_max*].
    - Refuses if the computed drop count exceeds *max_drop_fraction*
      (default 0.5) of the starting active fact count, even if it is
      within the absolute band above.
    - Plans and applies under ``BEGIN IMMEDIATE``, then runs orphan, SQLite,
      and FTS5 integrity checks before committing.
    - A successful real run ends with ``PRAGMA wal_checkpoint(TRUNCATE)``.
"""

from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .schema import SchemaError, schema_version
from .sqlite_vec_index import SQLiteVecIndex

_SUPERSEDED_PREFIXES = ("superseded", "stale/disabled", "historical/superseded")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_NEGATION_WORDS = frozenset(("not", "no", "never", "without"))
_DATE_WORDS = frozenset(
    "january february march april may june july august september october november december "
    "monday tuesday wednesday thursday friday saturday sunday today tomorrow yesterday"
    .split()
)
_STATE_WORD_GROUPS = (
    frozenset(("enabled", "disabled")),
    frozenset(("active", "inactive", "archived")),
    frozenset(("on", "off")),
    frozenset(("open", "closed")),
    frozenset(("paused", "resumed")),
    frozenset(("up", "down")),
    frozenset(("alive", "dead")),
    frozenset(("started", "stopped")),
    frozenset(("pending", "running", "completed", "failed", "succeeded")),
    frozenset(("deployed", "undeployed", "installed", "uninstalled")),
    frozenset(("available", "unavailable")),
    frozenset(("approved", "rejected", "accepted", "denied")),
)
_STATE_WORDS = frozenset().union(*_STATE_WORD_GROUPS)
_SOURCE_FACTS_TAG_RE = re.compile(r"source_facts:([0-9]+(?:,[0-9]+)*)")


class GuardRailError(Exception):
    """Raised when execute_merge refuses to run for safety reasons."""


@dataclass(frozen=True, slots=True)
class NearDuplicateCandidate:
    """An active FTS-prefiltered fact whose stored vector matches a write."""

    fact_id: int
    trust_score: float
    created_at: str
    cosine: float


def _tokens(content: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall((content or "").lower()))


def _value_tokens(content: str) -> tuple[str, ...]:
    """Concrete values, including dates, versions, ids, and plain numbers."""
    return tuple(
        token for token in _tokens(content)
        if any(char.isdigit() for char in token) or token in _DATE_WORDS
    )


def _state_tokens(content: str) -> frozenset[str]:
    tokens = _tokens(content)
    states = set(tokens) & _STATE_WORDS
    # ``on`` is commonly a preposition ("listens on port 3100"), whereas
    # ``is on`` and ``turned on`` are lifecycle assertions.  Do not let the
    # preposition turn an otherwise safe paraphrase into a false state change.
    for index, token in enumerate(tokens):
        if token in {"on", "off"} and (
            index == 0 or tokens[index - 1] not in {"is", "was", "turned", "set"}
        ):
            states.discard(token)
    return frozenset(states)


def safe_to_merge_near_duplicate(content: str, other: str) -> bool:
    """Return False for textual signals that can denote a changed fact.

    Dense embeddings blur numeric, temporal, polarity, and lifecycle changes.
    A difference in any of those signals is therefore an absolute block, not a
    score penalty.  The caller still applies its cosine threshold afterwards.
    """
    if _value_tokens(content) != _value_tokens(other):
        return False
    if bool(frozenset(_tokens(content)) & _NEGATION_WORDS) != bool(
        frozenset(_tokens(other)) & _NEGATION_WORDS
    ):
        return False
    return _state_tokens(content) == _state_tokens(other)


def find_write_near_duplicates(
    conn: sqlite3.Connection,
    *,
    content: str,
    scope: str,
    query_embedding: np.ndarray,
    threshold: float,
    candidate_limit: int,
    embedding_identity: Optional[str] = None,
) -> List[NearDuplicateCandidate]:
    """Find safe write-time near duplicates without scanning a whole scope.

    FTS supplies a bounded lexical candidate set before any vector is decoded.
    If FTS or stored embeddings are unavailable, this returns no candidates so
    the write path can retain its exact-match fallback rather than blocking a
    fact on incomplete embedding work.
    """
    terms = tuple(dict.fromkeys(_tokens(content)))
    vector = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
    if not terms or vector.size == 0 or not np.isfinite(vector).all():
        return []
    if candidate_limit <= 0 or not 0.0 <= threshold <= 1.0:
        raise ValueError("near-duplicate search configuration is invalid")
    match_query = " OR ".join(f'"{term}"' for term in terms)
    identity_clause = ""
    params: list[object] = [match_query, scope]
    if embedding_identity is not None:
        identity_clause = " AND e.embedding_identity = ?"
        params.append(embedding_identity)
    params.append(candidate_limit)
    try:
        rows = conn.execute(
            f"""
            SELECT f.fact_id, f.content, f.trust_score, f.created_at,
                   e.embedding, e.dim
            FROM facts_fts
            JOIN facts AS f ON f.fact_id = facts_fts.rowid
            JOIN fact_embeddings AS e ON e.fact_id = f.fact_id
            WHERE facts_fts MATCH ? AND f.scope = ?
              AND f.invalid_at IS NULL AND f.superseded_by IS NULL
              AND f.conflict_group IS NULL{identity_clause}
            ORDER BY bm25(facts_fts), f.fact_id
            LIMIT ?
            """,
            params,
        ).fetchall()
    except sqlite3.DatabaseError:
        return []

    vector_norm = float(np.linalg.norm(vector))
    if vector_norm == 0.0:
        return []
    matches: List[NearDuplicateCandidate] = []
    for row in rows:
        if int(row["dim"]) != vector.size or not safe_to_merge_near_duplicate(
            content, str(row["content"])
        ):
            continue
        stored = np.frombuffer(row["embedding"], dtype="<f4")
        if stored.size != vector.size:
            continue
        stored_norm = float(np.linalg.norm(stored))
        if stored_norm == 0.0:
            continue
        cosine = float(np.dot(vector, stored) / (vector_norm * stored_norm))
        if cosine >= threshold:
            matches.append(
                NearDuplicateCandidate(
                    fact_id=int(row["fact_id"]),
                    trust_score=float(row["trust_score"]),
                    created_at=str(row["created_at"]),
                    cosine=cosine,
                )
            )
    return matches


# ---------------------------------------------------------------------------
# Union-find clustering
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, ids: Sequence[int]) -> None:
        self._parent = {i: i for i in ids}

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def build_clusters(
    conn: sqlite3.Connection,
    threshold: float,
    embedding_identity: Optional[str] = None,
) -> List[List[int]]:
    """Return groups (size >= 2) of fact_ids whose embeddings are near-duplicates.

    Computes full pairwise cosine similarity over every embedded fact and
    union-finds pairs at or above *threshold*. Facts with no embedding, or
    whose only neighbor is themselves, are omitted (singletons aren't
    clusters).
    """
    identity = _embedding_identity(conn, embedding_identity)
    if identity is None:
        return []
    fact_ids_arr, matrix = _embedding_matrix(conn, identity, _dim(conn, identity))
    active_ids = _active_fact_ids(conn)
    fact_ids = [
        fid for fid in fact_ids_arr.astype(int).tolist()
        if fid in active_ids
    ]
    if len(fact_ids) != len(fact_ids_arr):
        keep = [
            i for i, fid in enumerate(fact_ids_arr.astype(int).tolist())
            if fid in active_ids
        ]
        matrix = matrix[keep]
    if len(fact_ids) < 2:
        return []

    uf = _UnionFind(fact_ids)
    contents = _fact_rows(conn, fact_ids)
    n = len(fact_ids)
    sims = matrix @ matrix.T
    for i in range(n):
        row = sims[i]
        for j in range(i + 1, n):
            if (
                contents[fact_ids[i]]["scope"] == contents[fact_ids[j]]["scope"]
                and row[j] >= threshold
                and safe_to_merge_near_duplicate(
                    str(contents[fact_ids[i]]["content"]),
                    str(contents[fact_ids[j]]["content"]),
                )
            ):
                uf.union(fact_ids[i], fact_ids[j])

    groups: Dict[int, List[int]] = {}
    for fid in fact_ids:
        groups.setdefault(uf.find(fid), []).append(fid)

    return [members for members in groups.values() if len(members) >= 2]


def _embedding_identity(
    conn: sqlite3.Connection, embedding_identity: Optional[str]
) -> Optional[str]:
    rows = conn.execute(
        "SELECT DISTINCT embedding_identity FROM fact_embeddings "
        "ORDER BY embedding_identity"
    ).fetchall()
    identities = [str(row[0]) for row in rows]
    if embedding_identity is not None:
        return embedding_identity
    if not identities:
        return None
    if len(identities) > 1:
        raise GuardRailError(
            "multiple embedding identities are present; pass --embedding-identity "
            f"with one of: {', '.join(identities)}"
        )
    return identities[0]


def _dim(conn: sqlite3.Connection, embedding_identity: str) -> int:
    rows = conn.execute(
        "SELECT DISTINCT dim FROM fact_embeddings "
        "WHERE embedding_identity = ? ORDER BY dim",
        (embedding_identity,),
    ).fetchall()
    if not rows:
        return 0
    if len(rows) > 1:
        dimensions = ", ".join(str(int(row[0])) for row in rows)
        raise GuardRailError(
            f"embedding identity {embedding_identity!r} has multiple dimensions: "
            f"{dimensions}"
        )
    return int(rows[0][0])


def _embedding_matrix(
    conn: sqlite3.Connection, embedding_identity: str, dim: int
) -> Tuple[np.ndarray, np.ndarray]:
    rows = conn.execute(
        "SELECT fact_id, embedding FROM fact_embeddings "
        "WHERE embedding_identity = ? AND dim = ? ORDER BY fact_id",
        (embedding_identity, dim),
    ).fetchall()
    if not rows:
        return np.array([], dtype=np.int64), np.empty((0, dim), dtype=np.float32)
    fact_ids = np.array([int(row[0]) for row in rows], dtype=np.int64)
    vectors = [np.frombuffer(row[1], dtype="<f4") for row in rows]
    if any(vector.size != dim for vector in vectors):
        raise GuardRailError(
            f"stored embedding size does not match dimension {dim} for "
            f"identity {embedding_identity!r}"
        )
    matrix = np.stack(vectors).astype(np.float32, copy=False)
    row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms = np.where(row_norms < 1e-9, 1.0, row_norms)
    return fact_ids, matrix / row_norms


def _is_legacy_superseded(content: str) -> bool:
    return (content or "").lstrip().lower().startswith(_SUPERSEDED_PREFIXES)


def _fact_table_cols(conn: sqlite3.Connection) -> set:
    return {row[1] for row in conn.execute("PRAGMA table_info(facts)").fetchall()}


def _active_fact_ids(conn: sqlite3.Connection) -> set:
    cols = _fact_table_cols(conn)
    predicates = [
        f"{column} IS NULL"
        for column in ("invalid_at", "superseded_by", "conflict_group")
        if column in cols
    ]
    where = " WHERE " + " AND ".join(predicates) if predicates else ""
    rows = conn.execute(f"SELECT fact_id, content FROM facts{where}").fetchall()
    return {
        int(row["fact_id"])
        for row in rows
        if not _is_legacy_superseded(row["content"])
    }


# ---------------------------------------------------------------------------
# Survivor selection
# ---------------------------------------------------------------------------

def _fact_rows(conn: sqlite3.Connection, fact_ids: Sequence[int]) -> Dict[int, sqlite3.Row]:
    placeholders = ",".join("?" * len(fact_ids))
    columns = _fact_table_cols(conn)
    invalid_at_select = "invalid_at" if "invalid_at" in columns else "NULL AS invalid_at"
    scope_select = "scope" if "scope" in columns else "'private' AS scope"
    rows = conn.execute(
        f"SELECT fact_id, content, tags, trust_score, retrieval_count, "
        f"helpful_count, created_at, {invalid_at_select}, {scope_select} "
        f"FROM facts WHERE fact_id IN ({placeholders})",
        list(fact_ids),
    ).fetchall()
    return {int(r["fact_id"]): r for r in rows}


def choose_survivor(
    conn: sqlite3.Connection,
    fact_ids: Sequence[int],
    flood_cutoff: str,
) -> Tuple[int, List[int]]:
    """Pick the survivor fact_id for a cluster; return (survivor, losers).

    See module docstring for the selection rule.
    """
    rows = _fact_rows(conn, fact_ids)
    active_ids = [
        fid for fid in fact_ids
        if rows[fid]["invalid_at"] is None
        and not _is_legacy_superseded(rows[fid]["content"])
    ]
    if active_ids:
        fact_ids = active_ids

    pre_existing = [fid for fid in fact_ids if rows[fid]["created_at"] < flood_cutoff]
    flood = [fid for fid in fact_ids if rows[fid]["created_at"] >= flood_cutoff]

    if pre_existing and flood:
        candidates = pre_existing
    else:
        candidates = list(fact_ids)

    def _key(fid: int):
        r = rows[fid]
        score = float(r["trust_score"]) * float(r["retrieval_count"])
        # Higher score first, then earliest created_at first.
        return (-score, r["created_at"])

    survivor = min(candidates, key=_key)
    losers = [fid for fid in fact_ids if fid != survivor]
    return survivor, losers


# ---------------------------------------------------------------------------
# Merge plan
# ---------------------------------------------------------------------------

@dataclass
class ClusterMerge:
    survivor_id: int
    loser_ids: List[int]
    merged_retrieval_count: int
    merged_helpful_count: int
    merged_tags: str
    suspicious: bool = False


@dataclass
class MergePlan:
    """Planned changes and counts over the active-fact population."""

    clusters: List[ClusterMerge] = field(default_factory=list)
    starting_fact_count: int = 0

    @property
    def drop_count(self) -> int:
        return sum(len(c.loser_ids) for c in self.clusters)

    @property
    def projected_final_count(self) -> int:
        return self.starting_fact_count - self.drop_count


def _merge_tags(*tag_strings: str) -> str:
    seen: List[str] = []
    for tags in tag_strings:
        for tag in (tags or "").split(","):
            tag = tag.strip()
            if tag and tag not in seen:
                seen.append(tag)
    return ",".join(seen)


def plan_merge(
    conn: sqlite3.Connection,
    threshold: float,
    flood_cutoff: str,
    embedding_identity: Optional[str] = None,
    suspicious_cluster_size: int = 25,
) -> MergePlan:
    """Build a MergePlan: one ClusterMerge per near-duplicate cluster.

    A cluster is flagged *suspicious* when it has more members than
    *suspicious_cluster_size* -- large enough that it's worth a human
    spot-check before trusting the merge, rather than assuming every member
    is a genuine paraphrase of the same statement.
    """
    starting = len(_active_fact_ids(conn))
    clusters = build_clusters(conn, threshold, embedding_identity=embedding_identity)

    plan = MergePlan(starting_fact_count=starting)
    for members in clusters:
        survivor, _ = choose_survivor(conn, members, flood_cutoff)
        rows = _fact_rows(conn, members)
        members = [
            fact_id for fact_id in members
            if fact_id == survivor or safe_to_merge_near_duplicate(
                str(rows[survivor]["content"]), str(rows[fact_id]["content"])
            )
        ]
        if len(members) < 2:
            continue
        survivor, losers = choose_survivor(conn, members, flood_cutoff)
        merged_retrieval = sum(int(rows[fid]["retrieval_count"]) for fid in members)
        merged_helpful = sum(int(rows[fid]["helpful_count"]) for fid in members)
        merged_tags = _merge_tags(*(rows[fid]["tags"] for fid in members))
        plan.clusters.append(
            ClusterMerge(
                survivor_id=survivor,
                loser_ids=losers,
                merged_retrieval_count=merged_retrieval,
                merged_helpful_count=merged_helpful,
                merged_tags=merged_tags,
                suspicious=len(members) > suspicious_cluster_size,
            )
        )
    return plan


# ---------------------------------------------------------------------------
# Execution (guarded)
# ---------------------------------------------------------------------------

@dataclass
class MergeResult:
    """Merge outcome whose projected and final counts cover active facts only."""

    dry_run: bool
    drop_count: int
    projected_final_count: int
    final_fact_count: Optional[int] = None
    integrity_ok: Optional[bool] = None
    fts_integrity_ok: Optional[bool] = None


def _refuse_if_live_path(db_path: str) -> None:
    literal = Path(db_path).expanduser()
    resolved = literal.resolve()
    hermes_home = Path(
        os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))
    ).expanduser().resolve()
    under_hermes = (
        ".hermes" in literal.parts
        or ".hermes" in resolved.parts
        or resolved == hermes_home
        or hermes_home in resolved.parents
    )
    live_database = hermes_home / "memory_store.db"
    same_as_live = False
    if resolved.exists() and live_database.exists():
        same_as_live = os.path.samefile(resolved, live_database)
    if under_hermes or same_as_live:
        raise GuardRailError(
            f"refusing to run against a live hermes path ({db_path}); "
            "this tool only ever runs against a copy"
        )


def _read_only_connection(path: str) -> sqlite3.Connection:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise GuardRailError(f"database file does not exist ({resolved})")
    return sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)


def _read_write_connection(path: str) -> sqlite3.Connection:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise GuardRailError(f"database file does not exist ({resolved})")
    return sqlite3.connect(f"{resolved.as_uri()}?mode=rw", uri=True)


def _validate_backup(db_path: str, backup_path: str) -> None:
    database = Path(db_path).expanduser().resolve()
    backup = Path(backup_path).expanduser().resolve()
    if not backup.is_file() or not os.access(backup, os.R_OK):
        raise GuardRailError(
            f"refusing to run: backup is not a readable file ({backup_path})"
        )
    if backup == database or (
        database.exists() and os.path.samefile(database, backup)
    ):
        raise GuardRailError(
            "refusing to run: backup must be distinct from the target database"
        )
    try:
        with backup.open("rb") as backup_file:
            if backup_file.read(16) != b"SQLite format 3\x00":
                raise GuardRailError(
                    f"refusing to run: backup is not a readable SQLite file "
                    f"({backup_path})"
                )
        with closing(_read_only_connection(str(database))) as database_conn, closing(
            _read_only_connection(str(backup))
        ) as backup_conn:
            rows = backup_conn.execute("PRAGMA quick_check").fetchall()
            if schema_version(database_conn) != schema_version(backup_conn):
                raise GuardRailError(
                    "refusing to run: backup schema version does not match "
                    "the target database"
                )
            target_rows = database_conn.execute(
                "SELECT fact_id, content FROM facts ORDER BY fact_id LIMIT 16"
            ).fetchall()
            if target_rows:
                placeholders = ",".join("?" * len(target_rows))
                backup_rows = backup_conn.execute(
                    f"SELECT fact_id, content FROM facts "
                    f"WHERE fact_id IN ({placeholders})",
                    [int(row[0]) for row in target_rows],
                ).fetchall()
                if dict(backup_rows) != dict(target_rows):
                    raise GuardRailError(
                        "refusing to run: backup facts do not match the target database"
                    )
            else:
                target_schema = database_conn.execute(
                    "SELECT type, name, sql FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                ).fetchall()
                backup_schema = backup_conn.execute(
                    "SELECT type, name, sql FROM sqlite_master "
                    "WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
                ).fetchall()
                if backup_schema != target_schema:
                    raise GuardRailError(
                        "refusing to run: backup schema does not match the target database"
                    )
    except GuardRailError:
        raise
    except (SchemaError, sqlite3.DatabaseError) as exc:
        raise GuardRailError(
            f"refusing to run: backup is not a readable SQLite file ({backup_path})"
        ) from exc
    if not rows or any(str(row[0]).lower() != "ok" for row in rows):
        raise GuardRailError(
            f"refusing to run: backup SQLite integrity check failed ({backup_path})"
        )


def execute_merge(
    db_path: str,
    threshold: float,
    flood_cutoff: str,
    embedding_identity: Optional[str],
    backup_path: str,
    expected_drop_min: int,
    expected_drop_max: int,
    dry_run: bool = True,
    suspicious_cluster_size: int = 25,
    max_drop_fraction: float = 0.5,
) -> MergeResult:
    """Plan and (optionally) execute the merge against *db_path*.

    Always refuses a live Hermes path. A non-dry-run additionally requires
    *backup_path* to be a distinct, readable SQLite file and the drop count to fall
    inside [*expected_drop_min*, *expected_drop_max*] AND to not exceed
    *max_drop_fraction* of the starting active fact count (default 0.5, i.e.
    refuse a plan that would drop more than half the store even if it is
    still within the absolute band), then deletes losers, merges their
    counts/tags onto each survivor, drops their embeddings, and runs an
    integrity check + wal_checkpoint.
    """
    _refuse_if_live_path(db_path)

    if not dry_run:
        _validate_backup(db_path, backup_path)

    conn = (
        _read_only_connection(db_path)
        if dry_run
        else _read_write_connection(db_path)
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys=ON")
        if not dry_run and not bool(conn.execute("PRAGMA foreign_keys").fetchone()[0]):
            raise GuardRailError("refusing to run: SQLite foreign keys are disabled")
        if not dry_run:
            conn.execute("BEGIN IMMEDIATE")
        plan = plan_merge(
            conn, threshold, flood_cutoff,
            embedding_identity=embedding_identity,
            suspicious_cluster_size=suspicious_cluster_size,
        )

        if not dry_run:
            if not (expected_drop_min <= plan.drop_count <= expected_drop_max):
                raise GuardRailError(
                    f"refusing to run: drop count {plan.drop_count} is outside "
                    f"the expected band [{expected_drop_min}, {expected_drop_max}]"
                )
            if plan.starting_fact_count > 0 and (
                plan.drop_count > max_drop_fraction * plan.starting_fact_count
            ):
                raise GuardRailError(
                    f"refusing to run: drop count {plan.drop_count} exceeds the "
                    f"relative cap of {max_drop_fraction:.0%} of the "
                    f"{plan.starting_fact_count} active facts"
                )
            _apply_merge(conn, plan)
            _check_fact_reference_orphans(conn)
            integrity_ok = _integrity_check(conn)
            fts_ok = _fts_integrity_check(conn)
            if not integrity_ok or not fts_ok:
                failed = []
                if not integrity_ok:
                    failed.append("SQLite integrity_check")
                if not fts_ok:
                    failed.append("FTS integrity check")
                raise GuardRailError(
                    "refusing to commit: " + " and ".join(failed) + " failed"
                )
            final_active_count = len(_active_fact_ids(conn))
            conn.commit()
            _wal_checkpoint(conn)
            return MergeResult(
                dry_run=False,
                drop_count=plan.drop_count,
                projected_final_count=plan.projected_final_count,
                final_fact_count=final_active_count,
                integrity_ok=integrity_ok,
                fts_integrity_ok=fts_ok,
            )

        return MergeResult(
            dry_run=True,
            drop_count=plan.drop_count,
            projected_final_count=plan.projected_final_count,
        )
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


def _apply_merge(conn: sqlite3.Connection, plan: MergePlan) -> None:
    vector_index = SQLiteVecIndex.open_configured(conn, warn=False)
    for cluster in plan.clusters:
        conn.execute(
            "UPDATE facts SET retrieval_count = ?, helpful_count = ?, tags = ? "
            "WHERE fact_id = ?",
            (
                cluster.merged_retrieval_count,
                cluster.merged_helpful_count,
                cluster.merged_tags,
                cluster.survivor_id,
            ),
        )
        for loser_id in cluster.loser_ids:
            _merge_fact_references(conn, loser_id, cluster.survivor_id)
            if vector_index is not None:
                vector_index.delete_in_transaction(loser_id)
            conn.execute("DELETE FROM facts WHERE fact_id = ?", (loser_id,))


_FACT_REFERENCE_ACTIONS = (
    ("facts", "superseded_by", "update"),
    ("fact_entities", "fact_id", "deduplicate"),
    ("fact_provenance", "fact_id", "deduplicate"),
    ("memory_write_log", "fact_id", "update"),
    ("memory_write_log", "existing_fact_id", "update"),
    ("privacy_erasure_log", "fact_id", "update"),
    ("embedding_jobs", "fact_id", "delete"),
    ("fact_embeddings", "fact_id", "delete"),
    ("fact_conflicts", "resolution_fact_id", "update"),
    ("fact_conflict_members", "fact_id", "deduplicate"),
    ("fact_conflict_resolutions", "resolution_fact_id", "update"),
)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    quoted_table = _quote_identifier(table)
    return {
        str(row[1])
        for row in conn.execute(f"PRAGMA table_info({quoted_table})").fetchall()
    }


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _merge_fact_references(
    conn: sqlite3.Connection, loser_id: int, survivor_id: int
) -> None:
    _rewrite_reflection_source_tags(conn, loser_id, survivor_id)
    for table, column, action in _FACT_REFERENCE_ACTIONS:
        if column not in _table_columns(conn, table):
            continue
        if table == "facts" and column == "superseded_by":
            conn.execute(
                "UPDATE facts SET superseded_by = NULL "
                "WHERE fact_id = ? AND superseded_by = ?",
                (survivor_id, loser_id),
            )
        if action == "delete":
            conn.execute(
                f'DELETE FROM "{table}" WHERE "{column}" = ?', (loser_id,)
            )
            continue
        prefix = "UPDATE OR IGNORE" if action == "deduplicate" else "UPDATE"
        conn.execute(
            f'{prefix} "{table}" SET "{column}" = ? WHERE "{column}" = ?',
            (survivor_id, loser_id),
        )
        if action == "deduplicate":
            conn.execute(
                f'DELETE FROM "{table}" WHERE "{column}" = ?', (loser_id,)
            )


def _rewrite_reflection_source_tags(
    conn: sqlite3.Connection, loser_id: int, survivor_id: int
) -> None:
    columns = _table_columns(conn, "facts")
    if not {"category", "tags"} <= columns:
        return
    active_clause = "".join(
        f" AND {column} IS NULL"
        for column in ("invalid_at", "superseded_by", "conflict_group")
        if column in columns
    )
    rows = conn.execute(
        "SELECT fact_id, tags FROM facts "
        f"WHERE category = 'insight'{active_clause} AND tags LIKE ?",
        (f"%source_facts:%{loser_id}%",),
    ).fetchall()
    for row in rows:
        tags = str(row["tags"] or "")

        def replace(match: re.Match[str]) -> str:
            cited = [int(value) for value in match.group(1).split(",")]
            rewritten = []
            for fact_id in cited:
                fact_id = survivor_id if fact_id == loser_id else fact_id
                if fact_id not in rewritten:
                    rewritten.append(fact_id)
            return "source_facts:" + ",".join(str(fact_id) for fact_id in rewritten)

        rewritten_tags = _SOURCE_FACTS_TAG_RE.sub(replace, tags)
        if rewritten_tags != tags:
            conn.execute(
                "UPDATE facts SET tags = ? WHERE fact_id = ?",
                (rewritten_tags, int(row["fact_id"])),
            )


def _check_fact_reference_orphans(conn: sqlite3.Connection) -> None:
    references = {(table, column) for table, column, _ in _FACT_REFERENCE_ACTIONS}
    tables = [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
    ]
    for table in tables:
        quoted_table = _quote_identifier(table)
        for row in conn.execute(
            f"PRAGMA foreign_key_list({quoted_table})"
        ).fetchall():
            if str(row[2]) == "facts" and str(row[4]) == "fact_id":
                references.add((table, str(row[3])))
    for table, column in sorted(references):
        if column not in _table_columns(conn, table):
            continue
        quoted_table = _quote_identifier(table)
        quoted_column = _quote_identifier(column)
        orphan = conn.execute(
            f"SELECT child.{quoted_column} FROM {quoted_table} AS child "
            f"LEFT JOIN facts AS parent ON parent.fact_id = child.{quoted_column} "
            f"WHERE child.{quoted_column} IS NOT NULL "
            "AND parent.fact_id IS NULL LIMIT 1"
        ).fetchone()
        if orphan is not None:
            raise GuardRailError(
                f"refusing to commit: orphaned fact reference in {table}.{column} "
                f"({orphan[0]})"
            )


def _integrity_check(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("PRAGMA integrity_check").fetchall()
    return bool(rows) and all(row[0] == "ok" for row in rows)


def _fts_integrity_check(conn: sqlite3.Connection) -> bool:
    tables = {
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'facts_fts'"
        ).fetchall()
    }
    if "facts_fts" not in tables:
        return True
    try:
        conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('integrity-check')")
        return True
    except sqlite3.DatabaseError:
        return False


def _wal_checkpoint(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.DatabaseError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", help="path to the fact store copy to clean up")
    parser.add_argument("--threshold", type=float, default=0.92,
                         help="cosine similarity threshold for clustering (default 0.92)")
    parser.add_argument("--flood-cutoff", required=True,
                         help="created_at cutoff (YYYY-MM-DD HH:MM:SS) marking pre-existing vs flood facts")
    parser.add_argument("--embedding-identity", default=None,
                         help="embedding_identity to cluster (required when multiple exist)")
    parser.add_argument("--backup-path", required=True,
                         help="path to a pre-existing backup of db_path (required for --execute)")
    parser.add_argument("--expected-drop-min", type=int, default=0)
    parser.add_argument("--expected-drop-max", type=int, default=10_000)
    parser.add_argument("--max-drop-fraction", type=float, default=0.5,
                         help="refuse if drop count exceeds this fraction of active facts (default 0.5)")
    parser.add_argument("--execute", action="store_true",
                         help="actually perform the merge (default is dry-run)")
    args = parser.parse_args(argv)

    try:
        result = execute_merge(
            args.db_path,
            threshold=args.threshold,
            flood_cutoff=args.flood_cutoff,
            embedding_identity=args.embedding_identity,
            backup_path=args.backup_path,
            expected_drop_min=args.expected_drop_min,
            expected_drop_max=args.expected_drop_max,
            max_drop_fraction=args.max_drop_fraction,
            dry_run=not args.execute,
        )
    except (GuardRailError, OSError, sqlite3.DatabaseError, ValueError) as exc:
        print(f"cluster merge failed: {exc}", file=sys.stderr)
        return 1
    if result.integrity_ok is False or result.fts_integrity_ok is False:
        print("cluster merge failed: post-merge integrity check failed", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
