"""Standalone, scope-safe hybrid retrieval for Enfold v1 stores.

Candidate authorization and current-truth filtering happen before dense
embedding.  The embedder is deliberately pluggable; this module does not load
models, use the network, or depend on Hermes.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import sqlite3
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

from .core_store import build_visibility_predicate
from .embeddings import bytes_to_embedding, embedding_to_bytes
from .sqlite_vec_index import SQLiteVecIndex

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]*|\d+")
_MONTH_ALIASES = {
    "jan": "january",
    "feb": "february",
    "mar": "march",
    "apr": "april",
    "jun": "june",
    "jul": "july",
    "aug": "august",
    "sep": "september",
    "sept": "september",
    "oct": "october",
    "nov": "november",
    "dec": "december",
}
_MONTH_WORDS = frozenset(
    {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        *_MONTH_ALIASES.keys(),
        *_MONTH_ALIASES.values(),
    }
)
_SENTENCE_OPENERS = frozenset(
    {
        "A",
        "An",
        "Are",
        "Can",
        "Could",
        "Did",
        "Do",
        "Does",
        "Find",
        "Give",
        "How",
        "I",
        "Is",
        "Kindly",
        "May",
        "Me",
        "My",
        "Our",
        "Please",
        "Should",
        "Show",
        "Tell",
        "The",
        "Us",
        "Was",
        "Were",
        "What",
        "When",
        "Where",
        "Which",
        "Who",
        "Whom",
        "Whose",
        "Why",
        "Will",
        "Would",
        "You",
        "Your",
    }
)
_LEADING_REQUEST_VERBS = frozenset({"Find", "Give", "Show", "Tell"})
_GENERIC_TITLE_WORDS = frozenset({"project"})
_RRF_K = 60.0
_DENSE_LIST_MIN = 0.1
_ENTITY_GRAPH_SEED_LIMIT = 5
_ENTITY_HUB_DEGREE_LIMIT = 25
_ENTITY_MAX_HOPS = 2
_ENTITY_HOP_PRIOR = 0.85
_CANDIDATE_COLUMNS = (
    "fact_id",
    "content",
    "category",
    "tags",
    "trust_score",
    "retrieval_count",
    "helpful_count",
    "created_at",
    "updated_at",
    "valid_from",
    "invalid_at",
    "superseded_by",
    "memory_kind",
    "subject_key",
    "predicate_key",
    "object_value",
    "object_entity_id",
    "confidence",
    "source_authority",
    "scope",
    "sensitivity",
    "correction_status",
    "schema_version",
    "conflict_group",
)
LOGGER = logging.getLogger(__name__)


class VectorFallbackTelemetry:
    """Process-local, non-sensitive sqlite-vec fallback telemetry."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._count = 0
        self._recovery_count = 0
        self._active = False
        self._last_reason: str | None = None

    def record(self, reason: str) -> None:
        with self._lock:
            self._count += 1
            self._active = True
            self._last_reason = reason

    def recover(self) -> None:
        """Clear a transient active fallback after sqlite-vec is usable again."""

        with self._lock:
            if self._active:
                self._active = False
                self._recovery_count += 1
                self._last_reason = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "vector_fallback_active": self._active,
                "vector_fallback_count": self._count,
                "vector_fallback_recovery_count": self._recovery_count,
                "vector_last_fallback_reason": self._last_reason,
            }


@dataclass(frozen=True, slots=True)
class RankingConfig:
    """All score weights and confidence gates for hybrid retrieval.

    With the defaults, lexical and dense lists are joined by weighted
    reciprocal rank fusion (``k=60``), then ``score = 0.76 * fusion +
    0.05*trust + 0.02*memory_kind + 0.03*recency + 0.08*review +
    0.06*named_subject``.  Trust is in ``[0, 1]``; state, insight, untyped,
    and event kind priors are respectively 1.0, 0.75, 0.5, and 0.25.
    Recency decays exponentially from 1.0 with the configured half-life
    for events and untyped rows; state and insight use ``valid_from`` or
    stay current so a later ``updated_at`` touch cannot bury them.
    Review and named-subject priors keep person-name queries from ranking
    another subject's short mention above a reviewed fact about the named
    person.
    """

    fts_weight: float = 0.35
    jaccard_weight: float = 0.25
    dense_weight: float = 0.40
    fts_query_coverage_weight: float = 0.75
    trust_weight: float = 0.05
    memory_kind_weight: float = 0.02
    recency_weight: float = 0.03
    review_weight: float = 0.08
    named_subject_weight: float = 0.06
    recency_half_life_days: float = 365.0
    state_kind_score: float = 1.0
    insight_kind_score: float = 0.75
    untyped_kind_score: float = 0.5
    event_kind_score: float = 0.25
    score_floor: float = 0.12
    ambiguity_margin: float = 0.005

    def __post_init__(self) -> None:
        relevance = (self.fts_weight, self.jaccard_weight, self.dense_weight)
        priors = (
            self.trust_weight,
            self.memory_kind_weight,
            self.recency_weight,
            self.review_weight,
            self.named_subject_weight,
        )
        kind_scores = (
            self.state_kind_score,
            self.insight_kind_score,
            self.untyped_kind_score,
            self.event_kind_score,
        )
        numeric_values = (
            *relevance,
            self.fts_query_coverage_weight,
            *priors,
            *kind_scores,
        )
        if any(not math.isfinite(value) for value in numeric_values):
            raise ValueError("ranking weights and scores must be finite")
        if any(weight < 0 for weight in (*relevance, *priors)):
            raise ValueError("ranking weights must be non-negative")
        if not math.isclose(sum(relevance), 1.0):
            raise ValueError("relevance weights must sum to 1")
        if not 0.0 <= self.fts_query_coverage_weight <= 1.0:
            raise ValueError("FTS query-coverage weight must be between 0 and 1")
        if sum(priors) >= 1.0:
            raise ValueError("ranking prior weights must sum to less than 1")
        if any(not 0.0 <= score <= 1.0 for score in kind_scores):
            raise ValueError("memory kind scores must be between 0 and 1")
        if (
            not math.isfinite(self.recency_half_life_days)
            or self.recency_half_life_days <= 0
        ):
            raise ValueError("recency half-life must be positive")
        if (
            not math.isfinite(self.score_floor)
            or not math.isfinite(self.ambiguity_margin)
            or self.score_floor < 0
            or self.score_floor > 1
            or self.ambiguity_margin < 0
            or self.ambiguity_margin > 1
        ):
            raise ValueError(
                "ranking confidence gates must be finite and between 0 and 1"
            )

    @property
    def relevance_weight(self) -> float:
        return (
            1.0
            - self.trust_weight
            - self.memory_kind_weight
            - self.recency_weight
            - self.review_weight
            - self.named_subject_weight
        )


DEFAULT_RANKING_CONFIG = RankingConfig()
LEXICAL_RANKING_CONFIG = RankingConfig(
    fts_weight=0.35 / 0.60,
    jaccard_weight=0.25 / 0.60,
    dense_weight=0.0,
)


class DenseEmbedder(Protocol):
    """Minimal dense embedding interface used by the standalone retriever."""

    identity: str

    def embed_query(self, text: str) -> Sequence[float]: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...


class VersionedEmbeddingBackend(Protocol):
    """Backend contract for production, versioned document embeddings.

    Implementations may read vectors from SQLite or another local index and
    backfill missing vectors outside the synchronous search path.  Enfold only
    passes candidates that already survived scope/current/conflict filtering.
    """

    identity: str
    embedding_version: str
    dimensions: int

    def embed_query(self, text: str) -> Sequence[float]: ...

    def load_documents(
        self, documents: Sequence[tuple[int, str]]
    ) -> Sequence[Sequence[float]]: ...


class QueryEmbedder(Protocol):
    """Small adapter boundary shared by Ollama, FastEmbed, and test doubles."""

    def embed(self, text: str) -> Sequence[float] | None: ...


class StoredEmbeddingError(RuntimeError):
    """Stored dense retrieval cannot safely serve the requested candidates."""


class SQLiteVersionedEmbeddingBackend:
    """Load exact-version vectors for authorized candidates from SQLite.

    The query/document identities are deliberately separate configuration
    values.  Their roles must map exactly, preventing a query vector from
    being compared with a different model, prefix policy, or vector version.
    Missing, duplicate, malformed, or dimension-mismatched candidate vectors
    fail the whole search; production never silently substitutes freshly
    embedded documents or a deterministic CI vector.
    """

    required_columns = frozenset(
        {
            "fact_id",
            "embedding",
            "dim",
            "embedding_identity",
        }
    )

    def __init__(
        self,
        conn: sqlite3.Connection,
        query_embedder: QueryEmbedder,
        *,
        query_identity: str,
        document_identity: str,
        embedding_version: str,
        dimensions: int,
        query_prefix: str = "",
        sql_batch_size: int = 500,
    ):
        if query_identity.count(":query:") != 1:
            raise ValueError("query_identity must contain exactly one ':query:' role")
        expected_document = query_identity.replace(":query:", ":document:")
        if document_identity != expected_document:
            raise ValueError(
                "document_identity must exactly match query_identity with its role "
                "changed from query to document"
            )
        if not embedding_version.strip():
            raise ValueError("embedding_version must be non-empty")
        if not query_identity.endswith(f":{embedding_version}"):
            raise ValueError(
                "embedding_version must be the final component of both stored "
                "embedding identities"
            )
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if sql_batch_size < 1:
            raise ValueError("sql_batch_size must be positive")
        columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(fact_embeddings)").fetchall()
        }
        if not self.required_columns <= columns:
            raise StoredEmbeddingError(
                "fact_embeddings is absent or incompatible; backfill the exact "
                "production identity before activation"
            )
        population = conn.execute(
            """
            SELECT COUNT(*), MIN(dim), MAX(dim)
            FROM fact_embeddings
            WHERE embedding_identity = ?
            """,
            (document_identity,),
        ).fetchone()
        if (
            population is not None
            and int(population[0]) > 0
            and (int(population[1]) != dimensions or int(population[2]) != dimensions)
        ):
            raise StoredEmbeddingError(
                "configured document identity contains vectors with an unexpected dimension"
            )
        missing_active = conn.execute(
            """
            SELECT COUNT(*)
            FROM facts AS f
            LEFT JOIN fact_embeddings AS e
              ON e.fact_id = f.fact_id
             AND e.embedding_identity = ?
             AND e.dim = ?
            LEFT JOIN embedding_jobs AS j
              ON j.fact_id = f.fact_id
             AND j.document_identity = ?
             AND j.embedding_version = ?
             AND j.dimensions = ?
             AND j.status IN ('pending', 'processing')
            WHERE f.invalid_at IS NULL
              AND f.superseded_by IS NULL
              AND f.conflict_group IS NULL
              AND e.fact_id IS NULL
              AND j.job_id IS NULL
            """,
            (
                document_identity,
                dimensions,
                document_identity,
                embedding_version,
                dimensions,
            ),
        ).fetchone()
        if missing_active is not None and int(missing_active[0]) > 0:
            raise StoredEmbeddingError(
                f"{int(missing_active[0])} active fact(s) lack the configured stored "
                "embedding identity and dimension"
            )
        self._conn = conn
        self._query_embedder = query_embedder
        self.query_identity = query_identity
        self.document_identity = document_identity
        self.identity = query_identity
        self.embedding_version = embedding_version
        self.dimensions = dimensions
        self.query_prefix = query_prefix
        self._sql_batch_size = sql_batch_size

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "query_embedding_identity": self.query_identity,
            "document_embedding_identity": self.document_identity,
            "embedding_version": self.embedding_version,
            "embedding_dimensions": self.dimensions,
            "stored_embedding_coverage": "strict-all-candidates",
            "missing_embedding_behavior": "fail-closed",
            "queued_embedding_behavior": "lexical-only-until-processed",
            "candidate_vector_source": "sqlite-fact_embeddings",
        }

    def embed_query(self, text: str) -> Sequence[float]:
        vector = self._query_embedder.embed(f"{self.query_prefix}{text}")
        if vector is None:
            raise StoredEmbeddingError("production query embedding is unavailable")
        return vector

    def load_documents(
        self, documents: Sequence[tuple[int, str]]
    ) -> Sequence[Sequence[float]]:
        if not documents:
            return ()
        fact_ids = [fact_id for fact_id, _content in documents]
        if len(set(fact_ids)) != len(fact_ids):
            raise StoredEmbeddingError("candidate fact ids must be unique")

        loaded: dict[int, Sequence[float]] = {}
        for offset in range(0, len(fact_ids), self._sql_batch_size):
            batch = fact_ids[offset : offset + self._sql_batch_size]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"""
                SELECT fact_id, embedding, dim
                FROM fact_embeddings
                WHERE embedding_identity = ?
                  AND fact_id IN ({placeholders})
                """,
                (self.document_identity, *batch),
            ).fetchall()
            for row in rows:
                fact_id = int(row[0])
                if fact_id in loaded:
                    raise StoredEmbeddingError(
                        f"duplicate stored embedding for candidate fact {fact_id}"
                    )
                if int(row[2]) != self.dimensions:
                    raise StoredEmbeddingError(
                        f"stored embedding dimension mismatch for candidate fact {fact_id}"
                    )
                blob = row[1]
                if not isinstance(blob, bytes | bytearray | memoryview):
                    raise StoredEmbeddingError(
                        f"malformed stored embedding for candidate fact {fact_id}"
                    )
                vector = bytes_to_embedding(bytes(blob))
                if len(vector) != self.dimensions:
                    raise StoredEmbeddingError(
                        f"stored embedding byte length mismatch for candidate fact {fact_id}"
                    )
                loaded[fact_id] = vector

        missing = [fact_id for fact_id in fact_ids if fact_id not in loaded]
        if missing:
            placeholders = ",".join("?" for _ in missing)
            queued = {
                int(row[0])
                for row in self._conn.execute(
                    f"""
                    SELECT fact_id FROM embedding_jobs
                    WHERE document_identity = ? AND embedding_version = ?
                      AND dimensions = ? AND status IN ('pending', 'processing')
                      AND fact_id IN ({placeholders})
                    """,
                    (
                        self.document_identity,
                        self.embedding_version,
                        self.dimensions,
                        *missing,
                    ),
                ).fetchall()
            }
            uncovered = [fact_id for fact_id in missing if fact_id not in queued]
            if uncovered:
                preview = ", ".join(str(fact_id) for fact_id in uncovered[:10])
                suffix = "..." if len(uncovered) > 10 else ""
                raise StoredEmbeddingError(
                    f"missing {len(uncovered)} required stored candidate embedding(s) "
                    f"without a viable exact-identity job: {preview}{suffix}"
                )
            loaded.update((fact_id, None) for fact_id in missing)
        return tuple(loaded[fact_id] for fact_id in fact_ids)


class SQLiteStoredEmbeddingWriter:
    """Explicit idempotent maintenance/backfill API for document vectors.

    The packaged request service does not call this API: model work is
    forbidden on the synchronous memory-write path.  A future durable outbox
    processor may use it after claiming a job outside the write transaction.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        document_embedder: QueryEmbedder,
        *,
        document_identity: str,
        embedding_version: str,
        model_fingerprint: str,
        prefix_policy: str,
        dimensions: int,
        document_prefix: str = "",
        query_prefix: str = "",
    ):
        if document_identity.count(":document:") != 1:
            raise ValueError(
                "document_identity must contain exactly one ':document:' role"
            )
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if embedding_version != model_fingerprint:
            raise ValueError("model fingerprint must equal embedding version")
        if prefix_policy == "none":
            if query_prefix or document_prefix:
                raise ValueError(
                    "none prefix policy requires empty query/document prefixes"
                )
        elif prefix_policy.startswith("sha256-"):
            digest = hashlib.sha256(
                f"{query_prefix}\0{document_prefix}".encode("utf-8")
            ).hexdigest()
            if prefix_policy != f"sha256-{digest}":
                raise ValueError("document prefix does not match prefix policy")
        else:
            raise ValueError("prefix policy must be none or sha256-<full digest>")
        if not document_identity.endswith(
            f":document:{prefix_policy}:{embedding_version}"
        ):
            raise ValueError("document identity is not bound to writer configuration")
        self._conn = conn
        self._document_embedder = document_embedder
        self.document_identity = document_identity
        self.embedding_version = embedding_version
        self.model_fingerprint = model_fingerprint
        self.prefix_policy = prefix_policy
        self.dimensions = dimensions
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self._vector_index = SQLiteVecIndex.open(
            conn, document_identity, dimensions, warn=False
        )

    def embed_document(self, content: str) -> bytes:
        """Run the configured model and return a validated portable vector blob."""

        vector = self._document_embedder.embed(f"{self.document_prefix}{content}")
        if vector is None:
            raise StoredEmbeddingError("production document embedding is unavailable")
        if len(vector) != self.dimensions or any(
            not math.isfinite(float(value)) for value in vector
        ):
            raise StoredEmbeddingError("production document embedding is invalid")
        return embedding_to_bytes(np.asarray(vector, dtype=np.float32))

    def upsert_in_transaction(self, fact_id: int, embedding: bytes) -> None:
        """Store a prepared vector inside a caller-owned validation transaction."""

        if not self._conn.in_transaction:
            raise RuntimeError("prepared embedding upsert requires a transaction")
        if len(bytes_to_embedding(embedding)) != self.dimensions:
            raise StoredEmbeddingError("prepared document embedding is invalid")
        self._conn.execute(
            """
            INSERT INTO fact_embeddings(
                fact_id, embedding, dim, embedding_identity
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(fact_id, embedding_identity) DO UPDATE SET
                embedding = excluded.embedding, dim = excluded.dim
            """,
            (fact_id, embedding, self.dimensions, self.document_identity),
        )
        if self._vector_index is not None:
            self._vector_index.upsert_in_transaction(fact_id, embedding)

    def ensure_fact(self, fact_id: int, *, force: bool = False) -> bool:
        """Ensure one committed fact has its exact production vector.

        Returns ``False`` when a valid vector already exists and ``True`` after
        writing one. Callers must invoke this only from an explicit maintenance
        flow or an asynchronous durable-job processor.
        """

        if self._conn.in_transaction:
            raise RuntimeError("stored embedding write-through must run after commit")
        row = self._conn.execute(
            "SELECT content FROM facts WHERE fact_id = ?", (fact_id,)
        ).fetchone()
        if row is None:
            raise StoredEmbeddingError("committed fact disappeared before embedding")
        existing = self._conn.execute(
            """
            SELECT embedding, dim FROM fact_embeddings
            WHERE fact_id = ? AND embedding_identity = ?
            """,
            (fact_id, self.document_identity),
        ).fetchone()
        if existing is not None and not force:
            vector = bytes_to_embedding(bytes(existing[0]))
            if int(existing[1]) != self.dimensions or len(vector) != self.dimensions:
                raise StoredEmbeddingError(
                    "existing stored embedding has the wrong dimension"
                )
            return False
        content = str(row[0])
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        embedding = self.embed_document(content)
        # Serialize explicitly as portable little-endian float32 bytes without
        # requiring callers to know the storage format.
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            current = self._conn.execute(
                "SELECT content, invalid_at, superseded_by, conflict_group FROM facts "
                "WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()
            if current is None or any(value is not None for value in current[1:]):
                raise StoredEmbeddingError("fact became ineligible during embedding")
            current_hash = hashlib.sha256(str(current[0]).encode("utf-8")).hexdigest()
            if current_hash != content_hash:
                raise StoredEmbeddingError("fact content changed during embedding")
            self.upsert_in_transaction(fact_id, embedding)
            self._conn.commit()
        except BaseException:
            if self._conn.in_transaction:
                self._conn.rollback()
            raise
        return True


@dataclass(frozen=True, slots=True)
class VersionedStoredEmbeddingAdapter:
    """Adapt a versioned production backend to Enfold's retriever contract."""

    backend: VersionedEmbeddingBackend
    production_ready: bool = True

    @property
    def identity(self) -> str:
        return (
            f"{self.backend.identity}@{self.backend.embedding_version}"
            f":{self.backend.dimensions}"
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(getattr(self.backend, "metadata", {}))

    def embed_query(self, text: str) -> Sequence[float]:
        vector = self.backend.embed_query(text)
        self._validate_vector(vector, "query")
        return vector

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise RuntimeError(
            "versioned stored embeddings require candidate ids; use "
            "embed_candidates through HybridRetriever"
        )

    def embed_candidates(
        self, candidates: Sequence[Mapping[str, Any]]
    ) -> Sequence[Sequence[float]]:
        documents = tuple(
            (int(row["fact_id"]), str(row["content"])) for row in candidates
        )
        vectors = self.backend.load_documents(documents)
        if len(vectors) != len(documents):
            raise ValueError("embedding backend returned the wrong number of vectors")
        for vector in vectors:
            if vector is None:
                continue
            self._validate_vector(vector, "document")
        return vectors

    def _validate_vector(self, vector: Sequence[float], kind: str) -> None:
        if len(vector) != self.backend.dimensions:
            raise ValueError(
                f"{kind} embedding dimensions do not match backend metadata"
            )
        if any(not math.isfinite(float(value)) for value in vector):
            raise ValueError(f"{kind} embedding contains a non-finite value")


RetrieverFactory = Callable[[sqlite3.Connection, Sequence[str]], "HybridRetriever"]


def deterministic_retriever_factory(
    *,
    dimensions: int = 256,
    fts_weight: float = DEFAULT_RANKING_CONFIG.fts_weight,
    jaccard_weight: float = DEFAULT_RANKING_CONFIG.jaccard_weight,
    dense_weight: float = DEFAULT_RANKING_CONFIG.dense_weight,
    min_score: float = DEFAULT_RANKING_CONFIG.score_floor,
    vector_backend: str = "auto",
    vector_fallback_telemetry: VectorFallbackTelemetry | None = None,
) -> RetrieverFactory:
    """Return an offline factory suitable for tests and explicit CI config."""

    def build(conn: sqlite3.Connection, scopes: Sequence[str]) -> HybridRetriever:
        return HybridRetriever(
            conn,
            DeterministicFeatureHashEmbedder(dimensions),
            allowed_scopes=scopes,
            fts_weight=fts_weight,
            jaccard_weight=jaccard_weight,
            dense_weight=dense_weight,
            min_score=min_score,
            vector_backend=vector_backend,
            vector_fallback_telemetry=vector_fallback_telemetry,
        )

    return build


def lexical_retriever_factory(
    *,
    ranking_config: RankingConfig = LEXICAL_RANKING_CONFIG,
    now: datetime | None = None,
    vector_fallback_telemetry: VectorFallbackTelemetry | None = None,
    entity_expansion: bool = False,
) -> RetrieverFactory:
    """Return the production-honest offline lexical default retriever."""

    def build(conn: sqlite3.Connection, scopes: Sequence[str]) -> HybridRetriever:
        return HybridRetriever(
            conn,
            DisabledDenseEmbedder(),
            allowed_scopes=scopes,
            ranking_config=ranking_config,
            now=now,
            vector_backend="brute",
            vector_fallback_telemetry=vector_fallback_telemetry,
            entity_expansion=entity_expansion,
        )

    return build


class DisabledDenseEmbedder:
    """Refuse dense scoring so lexical mode cannot fake embeddings."""

    identity = "local-lexical-v1"
    production_ready = True
    metadata = {
        "dense_scoring": "disabled",
        "paraphrase_matching": "lexical-overlap-only",
    }

    def embed_query(self, text: str) -> Sequence[float]:
        raise RuntimeError("local-lexical retrieval disables dense scoring")

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        raise RuntimeError("local-lexical retrieval disables dense scoring")


class DeterministicFeatureHashEmbedder:
    """Offline deterministic CI embedder, explicitly not a semantic model.

    It hashes word tokens and character trigrams into a fixed vector.  This is
    useful for exercising dense plumbing reproducibly, but production should
    inject a real versioned local embedding model.
    """

    production_ready = False

    def __init__(self, dimensions: int = 256):
        if dimensions < 16:
            raise ValueError("dimensions must be at least 16")
        self.dimensions = dimensions
        self.identity = f"ci-feature-hash-v1:{dimensions}"

    def _embed(self, text: str) -> tuple[float, ...]:
        normalized = " ".join(_TOKEN_RE.findall(text.lower()))
        features = [f"w:{token}" for token in normalized.split()]
        compact = normalized.replace(" ", "_")
        features.extend(
            f"c3:{compact[i : i + 3]}" for i in range(max(0, len(compact) - 2))
        )
        vector = [0.0] * self.dimensions
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return tuple(vector)

    def embed_query(self, text: str) -> Sequence[float]:
        return self._embed(text)

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return tuple(self._embed(text) for text in texts)


def named_anchor_tokens(text: str) -> frozenset[str]:
    """Return normalized explicit names that must occur in a result.

    Anchor matching uses the same tokenizer as candidate matching.  Keeping a
    hyphenated surface form such as ``GPT-5`` as one anchor could never match
    candidate tokens (``gpt``, ``5``), causing a false fail-closed abstention.
    The first-person pronoun is grammatical capitalization, not a name.
    """

    words = _WORD_RE.findall(text)
    anchors = []
    for index, word in enumerate(words):
        if word.isdigit() or not word[0].isupper() or word == "I":
            continue
        if word.lower() in _MONTH_WORDS:
            continue
        if index == 0 and word in _SENTENCE_OPENERS:
            continue
        if not anchors and word in _LEADING_REQUEST_VERBS:
            continue
        parts = re.split(r"[-_]", word)
        anchors.extend(
            token
            for part_index, part in enumerate(parts)
            if part and (part_index == 0 or part[0].isupper() or part.isdigit())
            for token in _tokens(part)
        )
    return frozenset(anchors)


def select_named_anchor_matches(
    items: Sequence[Any],
    anchors: frozenset[str],
    text_of: Callable[[Any], str],
) -> list[Any]:
    """Keep exact named-anchor hits, or the best cluster of at least one.

    A query that names one project must still fail closed when that name is
    absent. A query that piles extra proper nouns on top of a real name
    should not erase the facts that match the strongest shared cluster.
    """

    if not anchors:
        return list(items)
    scored = [
        (len(anchors & _anchor_match_tokens(text_of(item))), item) for item in items
    ]
    exact = [item for hit, item in scored if hit == len(anchors)]
    if exact:
        return exact
    distinctive = anchors - _GENERIC_TITLE_WORDS
    if distinctive and distinctive != anchors:
        distinctive_scored = [
            (len(distinctive & _anchor_match_tokens(text_of(item))), item)
            for item in items
        ]
        distinctive_max = max((hit for hit, _item in distinctive_scored), default=0)
        if distinctive_max == 0:
            return []
        return [item for hit, item in distinctive_scored if hit == distinctive_max]
    max_hit = max((hit for hit, _item in scored), default=0)
    if max_hit >= 1:
        return [item for hit, item in scored if hit == max_hit]
    return []


def _tokens(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _anchor_match_tokens(text: str) -> frozenset[str]:
    """Return exact tokens plus normalized adjacent compound spellings."""

    tokens = _TOKEN_RE.findall(text.lower())
    matches = set(tokens)
    for short, full in _MONTH_ALIASES.items():
        if short in matches:
            matches.add(full)
        if full in matches:
            matches.add(short)
    for width in (2, 3):
        matches.update(
            "".join(tokens[offset : offset + width])
            for offset in range(0, len(tokens) - width + 1)
        )
    return frozenset(matches)


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _weighted_rrf(
    lexical_rank: int | None,
    dense_rank: int | None,
    lexical_weight: float,
    dense_weight: float,
    k: float = _RRF_K,
) -> float:
    """Normalize weighted RRF so rank 1 on both lists scores 1.0."""

    score = 0.0
    if lexical_rank is not None:
        score += lexical_weight / (k + lexical_rank)
    if dense_rank is not None:
        score += dense_weight / (k + dense_rank)
    return score * (k + 1.0)


def _query_coverage(query: frozenset[str], document: frozenset[str]) -> float:
    if not query:
        return 0.0
    return len(query & document) / len(query)


def _is_absent_vector(vector: Sequence[float] | None) -> bool:
    """A zero vector is missing coverage, not a valid embedding at cosine 0."""

    if vector is None:
        return True
    return not any(float(value) != 0.0 for value in vector)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("query and document embedding dimensions differ")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    # NumPy-backed stored vectors yield NumPy scalar products.  Normalize the
    # public score to a built-in float so protocol JSON serialization cannot
    # fail only after dense coverage becomes complete.
    return float(
        sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)
    )


def _fts_query(text: str) -> str:
    return " OR ".join(f'"{token}"' for token in sorted(_tokens(text)))


def _as_utc(value: Any) -> datetime | None:
    if not isinstance(value, (str, datetime)):
        return None
    try:
        parsed = (
            value
            if isinstance(value, datetime)
            else datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        )
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recency_score(
    row: Mapping[str, Any], now: datetime, half_life_days: float
) -> float:
    kind = row.get("memory_kind")
    if kind in {"state", "insight"}:
        timestamp = _as_utc(row.get("valid_from"))
        if timestamp is None:
            return 1.0
    else:
        timestamp = _as_utc(row.get("updated_at")) or _as_utc(row.get("created_at"))
        if timestamp is None:
            return 0.0
    age_days = max(0.0, (now - timestamp).total_seconds() / 86_400.0)
    return math.exp(-math.log(2.0) * age_days / half_life_days)


_LEADING_PERSON_RE = re.compile(
    r"^\s*((?:[A-Z][A-Za-z0-9'-]+(?:/[A-Z][A-Za-z0-9'-]+)*)"
    r"(?:\s+[A-Z][A-Za-z0-9'-]+(?:/[A-Z][A-Za-z0-9'-]+)*){0,3})"
)
_IMPORTED_TOPIC_RE = re.compile(
    r"^\s*Claude Code memory topic `(?P<title>[^\r\n]*)` \([^)]*\):\s*"
)
_AS_OF_DATE_RE = re.compile(r"^\s*As of \d{4}-\d{2}-\d{2},\s*")
_REVIEWED_STATUSES = frozenset({"human_confirmed", "human_corrected"})


def leading_person_tokens(text: str) -> frozenset[str]:
    """Return the opening proper-name run, skipping sentence openers."""

    remaining = text.strip()
    imported_topic = _IMPORTED_TOPIC_RE.match(remaining)
    if imported_topic is not None:
        remaining = remaining[imported_topic.end() :]
    as_of_date = _AS_OF_DATE_RE.match(remaining)
    if as_of_date is not None:
        remaining = remaining[as_of_date.end() :]
    while remaining:
        match = _LEADING_PERSON_RE.match(remaining)
        if match is None:
            return frozenset()
        words = match.group(1).split()
        if words[0] in _SENTENCE_OPENERS or words[0] in _LEADING_REQUEST_VERBS:
            remaining = remaining[match.end() :].lstrip()
            continue
        return frozenset(token for word in words for token in _tokens(word))
    return frozenset()


def named_subject_score(anchors: frozenset[str], content: str) -> float:
    """Score whether the query names the fact's subject, not a later mention."""

    if not anchors:
        return 0.0
    subjects = [leading_person_tokens(content)]
    imported_topic = _IMPORTED_TOPIC_RE.match(content)
    if imported_topic is not None:
        subjects.append(
            _tokens(imported_topic.group("title")) - _GENERIC_TITLE_WORDS
        )
    scores = []
    for subject in subjects:
        if not subject:
            scores.append(0.5)
        elif subject <= anchors or anchors <= subject:
            scores.append(1.0)
        elif anchors.isdisjoint(subject) and len(subject) >= 2:
            scores.append(0.0)
        else:
            scores.append(0.5)
    return max(scores)


def _contradicting_state_slots(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    if left.get("memory_kind") != "state" or right.get("memory_kind") != "state":
        return False
    subject = left.get("subject_key")
    predicate = left.get("predicate_key")
    if not subject or not predicate:
        return False
    return (
        left.get("scope") == right.get("scope")
        and right.get("subject_key") == subject
        and right.get("predicate_key") == predicate
        and left.get("object_value") != right.get("object_value")
    )


def _review_score(status: Any) -> float:
    return 1.0 if status in _REVIEWED_STATUSES else 0.0


def _memory_kind_score(kind: Any, config: RankingConfig) -> float:
    return {
        "state": config.state_kind_score,
        "insight": config.insight_kind_score,
        "event": config.event_kind_score,
    }.get(kind, config.untyped_kind_score)


class HybridRetriever:
    """Blend FTS, Jaccard, and pluggable dense scores over eligible facts."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        embedder: DenseEmbedder,
        *,
        allowed_scopes: Sequence[str] = ("private",),
        fts_weight: float | None = None,
        jaccard_weight: float | None = None,
        dense_weight: float | None = None,
        min_score: float | None = None,
        candidate_limit: int = 10_000,
        vector_backend: str = "auto",
        ranking_config: RankingConfig = DEFAULT_RANKING_CONFIG,
        now: datetime | None = None,
        vector_fallback_telemetry: VectorFallbackTelemetry | None = None,
        entity_expansion: bool = False,
    ):
        relevance_weights = (
            ranking_config.fts_weight if fts_weight is None else fts_weight,
            ranking_config.jaccard_weight if jaccard_weight is None else jaccard_weight,
            ranking_config.dense_weight if dense_weight is None else dense_weight,
        )
        min_score = ranking_config.score_floor if min_score is None else min_score
        if any(weight < 0 for weight in relevance_weights) or not math.isclose(
            sum(relevance_weights), 1.0
        ):
            raise ValueError("retrieval weights must be non-negative and sum to 1")
        if (
            not math.isfinite(min_score)
            or not 0 <= min_score <= 1
            or candidate_limit < 1
        ):
            raise ValueError(
                "min_score must be between 0 and 1 and candidate_limit positive"
            )
        if vector_backend not in {"auto", "sqlite-vec", "brute"}:
            raise ValueError("vector_backend must be auto, sqlite-vec, or brute")
        self._conn = conn
        self._embedder = embedder
        self._allowed_scopes = tuple(allowed_scopes)
        self._weights = relevance_weights
        self._ranking = ranking_config
        self._min_score = min_score
        self._candidate_limit = candidate_limit
        self._entity_expansion = entity_expansion
        self._vector_fallback_telemetry = (
            vector_fallback_telemetry or VectorFallbackTelemetry()
        )
        self._vector_index: SQLiteVecIndex | None = None
        dense_disabled = relevance_weights[2] == 0.0
        if (
            not dense_disabled
            and vector_backend != "brute"
            and isinstance(embedder, VersionedStoredEmbeddingAdapter)
        ):
            backend = embedder.backend
            document_identity = getattr(backend, "document_identity", None)
            if isinstance(document_identity, str):
                self._vector_index = SQLiteVecIndex.open(
                    conn,
                    document_identity,
                    backend.dimensions,
                    warn=True,
                )
        if self._vector_index is not None:
            self._vector_fallback_telemetry.recover()
        clock = now or datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        if not isinstance(clock, datetime):
            raise ValueError("now must be a datetime")
        self._now = _as_utc(clock)
        assert self._now is not None
        if dense_disabled:
            retrieval_stack = "standalone_core_fts+jaccard+priors"
            vector_backend_name = "disabled"
            dense_coverage = "disabled"
            candidate_generation = "lexical"
            score_formula = (
                "relevance_weight*rrf(lexical_list)+"
                "trust_weight*trust+memory_kind_weight*memory_kind+"
                "recency_weight*recency+review_weight*review+"
                "named_subject_weight*named_subject"
            )
        else:
            retrieval_stack = "standalone_core_fts+jaccard+pluggable_dense"
            vector_backend_name = "sqlite-vec" if self._vector_index else "brute"
            dense_coverage = (
                "global" if self._vector_index is not None else "bounded"
            )
            candidate_generation = (
                "global-index-plus-lexical"
                if self._vector_index is not None
                else "recent-plus-lexical"
            )
            score_formula = (
                "relevance_weight*rrf(lexical_list, dense_list)+"
                "trust_weight*trust+memory_kind_weight*memory_kind+"
                "recency_weight*recency+review_weight*review+"
                "named_subject_weight*named_subject"
            )
        self.metadata = {
            "retrieval_stack": retrieval_stack,
            "embedder_identity": str(embedder.identity),
            "embedder_production_ready": bool(
                getattr(embedder, "production_ready", True)
            ),
            "dense_scoring": "disabled" if dense_disabled else "enabled",
            "filter_before_dense_ranking": True,
            "explicit_named_anchor_abstention": True,
            "natural_language_query_parser": "quoted_token_or_v1",
            "stored_embedding_contract": "versioned-candidate-id-v1",
            "score_formula": score_formula,
            "fts_score_formula": (
                "(1-query_coverage_weight)*reciprocal_bm25_rank+"
                "query_coverage_weight*distinct_query_token_coverage"
            ),
            "fts_query_coverage_weight": ranking_config.fts_query_coverage_weight,
            "weights": {
                "relevance": ranking_config.relevance_weight,
                "fts": relevance_weights[0],
                "jaccard": relevance_weights[1],
                "dense": relevance_weights[2],
                "trust": ranking_config.trust_weight,
                "memory_kind": ranking_config.memory_kind_weight,
                "recency": ranking_config.recency_weight,
                "review": ranking_config.review_weight,
                "named_subject": ranking_config.named_subject_weight,
            },
            "score_floor": min_score,
            "ambiguity_margin": ranking_config.ambiguity_margin,
            "recency_half_life_days": ranking_config.recency_half_life_days,
            "vector_backend_config": "disabled" if dense_disabled else vector_backend,
            "vector_backend": vector_backend_name,
            "dense_candidate_coverage": dense_coverage,
            "candidate_generation": candidate_generation,
            **self._vector_fallback_telemetry.snapshot(),
        }
        self.metadata.update(dict(getattr(embedder, "metadata", {})))

    def _stored_mmr_embeddings(
        self, fact_ids: Sequence[int]
    ) -> dict[int, tuple[float, ...]]:
        """Decode canonical vectors for the bounded vec0 result set only."""

        if not fact_ids or not isinstance(
            self._embedder, VersionedStoredEmbeddingAdapter
        ):
            return {}
        backend = self._embedder.backend
        identity = getattr(backend, "document_identity", None)
        dimensions = getattr(backend, "dimensions", None)
        if not isinstance(identity, str) or not isinstance(dimensions, int):
            return {}
        loaded: dict[int, tuple[float, ...]] = {}
        for offset in range(0, len(fact_ids), 500):
            batch = fact_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"""
                SELECT fact_id, embedding, dim
                FROM fact_embeddings
                WHERE embedding_identity = ?
                  AND fact_id IN ({placeholders})
                """,
                (identity, *batch),
            ).fetchall()
            for row in rows:
                blob = row[1]
                if int(row[2]) != dimensions or not isinstance(
                    blob, bytes | bytearray | memoryview
                ):
                    continue
                try:
                    vector = bytes_to_embedding(bytes(blob))
                except ValueError:
                    continue
                if len(vector) == dimensions:
                    loaded[int(row[0])] = tuple(float(value) for value in vector)
        if any(fact_id not in loaded for fact_id in fact_ids):
            return {}
        return loaded

    def _expand_via_entities(
        self,
        seed_fact_ids: Sequence[int],
        excluded_fact_ids: set[int],
        selected_columns: Sequence[str],
        active_predicates: Sequence[str],
        active_params: Sequence[Any],
    ) -> list[dict[str, Any]]:
        """Return eligible facts within two non-hub entity hops of the seeds."""

        frontier = list(dict.fromkeys(int(fact_id) for fact_id in seed_fact_ids))
        if not frontier:
            return []
        visited = set(frontier)
        expanded: list[dict[str, Any]] = []
        destination_rows_loaded = 0
        selected_sql = ", ".join(f"f.{name}" for name in selected_columns)
        eligibility_sql = " AND ".join(active_predicates)

        try:
            for hop_distance in range(1, _ENTITY_MAX_HOPS + 1):
                entity_ids: set[int] = set()
                for offset in range(0, len(frontier), 400):
                    batch = frontier[offset : offset + 400]
                    placeholders = ",".join("?" for _ in batch)
                    rows = self._conn.execute(
                        f"SELECT DISTINCT fe.entity_id FROM fact_entities AS fe "
                        "JOIN facts AS f ON f.fact_id = fe.fact_id "
                        f"WHERE fe.fact_id IN ({placeholders}) AND {eligibility_sql}",
                        (*batch, *active_params),
                    ).fetchall()
                    entity_ids.update(int(row[0]) for row in rows)
                if not entity_ids:
                    break

                expandable: list[int] = []
                ordered_entity_ids = sorted(entity_ids)
                for offset in range(0, len(ordered_entity_ids), 400):
                    batch = ordered_entity_ids[offset : offset + 400]
                    placeholders = ",".join("?" for _ in batch)
                    rows = self._conn.execute(
                        "SELECT fe.entity_id, COUNT(*) AS degree "
                        "FROM fact_entities AS fe "
                        "JOIN facts AS f ON f.fact_id = fe.fact_id "
                        f"WHERE fe.entity_id IN ({placeholders}) "
                        f"AND {eligibility_sql} GROUP BY fe.entity_id",
                        (*batch, *active_params),
                    ).fetchall()
                    expandable.extend(
                        int(row[0])
                        for row in rows
                        if int(row[1]) <= _ENTITY_HUB_DEGREE_LIMIT
                    )
                if not expandable:
                    break

                next_frontier: list[int] = []
                for offset in range(0, len(expandable), 400):
                    remaining_load = self._candidate_limit - destination_rows_loaded
                    if remaining_load <= 0:
                        return expanded
                    batch = expandable[offset : offset + 400]
                    placeholders = ",".join("?" for _ in batch)
                    rows = self._conn.execute(
                        f"SELECT {selected_sql}, e.name AS via_entity "
                        "FROM fact_entities AS fe "
                        "JOIN entities AS e ON e.entity_id = fe.entity_id "
                        "JOIN facts AS f ON f.fact_id = fe.fact_id "
                        f"WHERE fe.entity_id IN ({placeholders}) "
                        f"AND {eligibility_sql} ORDER BY f.fact_id, fe.entity_id "
                        "LIMIT ?",
                        (*batch, *active_params, remaining_load),
                    ).fetchall()
                    destination_rows_loaded += len(rows)
                    for row in rows:
                        fact_id = int(row["fact_id"])
                        if fact_id in visited:
                            continue
                        via_entity = str(row["via_entity"])
                        via_tokens = _tokens(via_entity) - _GENERIC_TITLE_WORDS
                        subject_tokens = leading_person_tokens(str(row["content"]))
                        if not via_tokens or not subject_tokens or not (
                            via_tokens <= subject_tokens
                            or subject_tokens <= via_tokens
                        ):
                            continue
                        visited.add(fact_id)
                        next_frontier.append(fact_id)
                        if fact_id in excluded_fact_ids:
                            continue
                        fact = dict(row)
                        fact["expanded_from_entity"] = fact.pop("via_entity")
                        fact["hop_distance"] = hop_distance
                        expanded.append(fact)
                        if len(expanded) >= self._candidate_limit:
                            return expanded
                frontier = next_frontier
                if not frontier:
                    break
        except sqlite3.OperationalError:
            return []
        return expanded

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        bump: bool = False,
        include_unreviewed: bool = False,
    ) -> list[dict[str, Any]]:
        del bump
        query = query.strip()
        if not query or limit <= 0:
            return []

        columns = frozenset(
            str(row[1])
            for row in self._conn.execute("PRAGMA table_info(facts)").fetchall()
        )
        active_predicates = [
            f"{name} IS NULL"
            for name in ("invalid_at", "superseded_by", "conflict_group")
            if name in columns
        ]
        scope_sql, scope_params = build_visibility_predicate(
            self._allowed_scopes,
            scope_column_available="scope" in columns,
            sensitivity_column_available="sensitivity" in columns,
        )
        active_predicates.append(scope_sql)
        active_params: list[Any] = list(scope_params)
        if category is not None:
            active_predicates.append("category = ?")
            active_params.append(category)
        active_predicates.append("trust_score >= ?")
        active_params.append(min_trust)
        if "correction_status" in columns and not include_unreviewed:
            # IS NOT keeps NULL agent writes; != would silently drop them.
            active_predicates.append("correction_status IS NOT ?")
            active_params.append("unreviewed")

        fts_scores: dict[int, float] = {}
        fts_ids: list[int] = []
        fts_query = _fts_query(query)
        global_index_mode = self._vector_index is not None
        if fts_query:
            fts_predicates = ["facts_fts MATCH ?"]
            fts_predicates.extend(
                f"f.{name} IS NULL"
                for name in ("invalid_at", "superseded_by", "conflict_group")
                if name in columns
            )
            fts_scope_sql, fts_scope_params = build_visibility_predicate(
                self._allowed_scopes,
                scope_column="f.scope",
                sensitivity_column="f.sensitivity",
                scope_column_available="scope" in columns,
                sensitivity_column_available="sensitivity" in columns,
            )
            fts_predicates.append(fts_scope_sql)
            fts_params: list[Any] = [fts_query, *fts_scope_params]
            if category is not None:
                fts_predicates.append("f.category = ?")
                fts_params.append(category)
            fts_predicates.append("f.trust_score >= ?")
            fts_params.append(min_trust)
            if "correction_status" in columns and not include_unreviewed:
                fts_predicates.append("f.correction_status IS NOT ?")
                fts_params.append("unreviewed")
            fts_limit_sql = ""
            if not global_index_mode:
                fts_limit_sql = "LIMIT ?"
                fts_params.append(self._candidate_limit)
            rows = self._conn.execute(
                f"""
                SELECT f.fact_id
                FROM facts_fts
                JOIN facts AS f ON f.fact_id = facts_fts.rowid
                WHERE {" AND ".join(fts_predicates)}
                ORDER BY bm25(facts_fts), f.fact_id DESC
                {fts_limit_sql}
                """,
                fts_params,
            ).fetchall()
            for rank, row in enumerate(rows):
                fact_id = int(row[0])
                fts_ids.append(fact_id)
                fts_scores[fact_id] = 1.0 / (rank + 1.0)

        dense_disabled = self._weights[2] == 0.0
        query_vector: Sequence[float] | None = None
        dense_scores: dict[int, float] | None = None
        dense_ids: list[int] = []
        if not dense_disabled and self._vector_index is not None:
            # sqlite-vec holds every canonical vector, while authorization and
            # lifecycle state live in ``facts``. Score the complete eligible
            # set exactly, then take the global dense window. This removes the
            # former newest-N gate that made old semantic-only memories
            # unreachable once the store grew past ``candidate_limit``.
            eligible_rows = self._conn.execute(
                f"SELECT f.fact_id, f.content, f.tags FROM facts AS f "
                "JOIN fact_embeddings AS e ON e.fact_id = f.fact_id "
                "AND e.embedding_identity = ? AND e.dim = ? WHERE "
                f"{' AND '.join(active_predicates)}",
                (
                    self._vector_index.identity,
                    self._vector_index.dimensions,
                    *active_params,
                ),
            ).fetchall()
            anchors = named_anchor_tokens(query)
            eligible_rows = select_named_anchor_matches(
                eligible_rows,
                anchors,
                lambda row: f"{row['content']} {row['tags']}",
            )
            eligible_ids = [int(row["fact_id"]) for row in eligible_rows]
            if eligible_ids:
                query_vector = self._embedder.embed_query(query)
                try:
                    dense_scores = self._vector_index.scores(query_vector, eligible_ids)
                    dense_ids = [
                        fact_id
                        for fact_id, _ in sorted(
                            dense_scores.items(), key=lambda item: (-item[1], item[0])
                        )
                    ]
                except Exception as exc:
                    LOGGER.warning(
                        "sqlite-vec health warning: %s; falling back to brute", exc
                    )
                    self._vector_index = None
                    self.metadata["vector_backend"] = "brute"
                    self.metadata["dense_candidate_coverage"] = "bounded"
                    self.metadata["candidate_generation"] = "recent-plus-lexical"
                    self._vector_fallback_telemetry.record("sqlite_vec_query_error")
                    self.metadata.update(self._vector_fallback_telemetry.snapshot())
                    dense_scores = None
                    fts_ids = fts_ids[: self._candidate_limit]
                    fts_scores = {fact_id: fts_scores[fact_id] for fact_id in fts_ids}
        if not dense_disabled and dense_scores is None:
            dense_ids = [
                int(row[0])
                for row in self._conn.execute(
                    f"SELECT fact_id FROM facts WHERE {' AND '.join(active_predicates)} "
                    "ORDER BY fact_id DESC LIMIT ?",
                    (*active_params, self._candidate_limit),
                ).fetchall()
            ]

        candidate_ids = tuple(dict.fromkeys((*dense_ids, *fts_ids)))
        if not candidate_ids:
            return []
        selected_columns = tuple(name for name in _CANDIDATE_COLUMNS if name in columns)
        loaded: dict[int, dict[str, Any]] = {}
        for offset in range(0, len(candidate_ids), 500):
            batch = candidate_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in batch)
            rows = self._conn.execute(
                f"SELECT {', '.join(selected_columns)} FROM facts "
                f"WHERE fact_id IN ({placeholders}) "
                f"AND {' AND '.join(active_predicates)}",
                (*batch, *active_params),
            ).fetchall()
            loaded.update((int(row["fact_id"]), dict(row)) for row in rows)
        candidates = [loaded[fact_id] for fact_id in candidate_ids if fact_id in loaded]

        # Trust, lifecycle, and scope filtering happened before document text
        # reaches the dense embedder. Named anchors fail closed at the same boundary.
        anchors = named_anchor_tokens(query)
        candidates = select_named_anchor_matches(
            candidates,
            anchors,
            lambda row: f"{row.get('content', '')} {row.get('tags', '')}",
        )
        if not candidates:
            return []

        if not dense_disabled and query_vector is None:
            query_vector = self._embedder.embed_query(query)
        if not dense_disabled and dense_scores is None and self._vector_index is not None:
            try:
                dense_scores = self._vector_index.scores(
                    query_vector, tuple(int(row["fact_id"]) for row in candidates)
                )
            # The index is optional derived state.  Treat every failure inside
            # its scoring boundary (including unexpected extension result
            # types such as NULL cosine distances for zero vectors) as an
            # index health problem, then use the canonical embedding blobs.
            except Exception as exc:
                LOGGER.warning(
                    "sqlite-vec health warning: %s; falling back to brute", exc
                )
                self._vector_index = None
                self.metadata["vector_backend"] = "brute"
                self.metadata["dense_candidate_coverage"] = "bounded"
                self.metadata["candidate_generation"] = "recent-plus-lexical"
                self._vector_fallback_telemetry.record("sqlite_vec_query_error")
                self.metadata.update(self._vector_fallback_telemetry.snapshot())
        document_vectors: Sequence[Sequence[float]] = ()
        if not dense_disabled and dense_scores is None:
            candidate_embedder = getattr(self._embedder, "embed_candidates", None)
            if callable(candidate_embedder):
                document_vectors = candidate_embedder(candidates)
            else:
                document_vectors = self._embedder.embed_documents(
                    tuple(str(row["content"]) for row in candidates)
                )
            if len(document_vectors) != len(candidates):
                raise ValueError(
                    "embedder returned the wrong number of document vectors"
                )

        query_tokens = _tokens(query)
        fts_weight, jaccard_weight, dense_weight = self._weights
        lexical_weight = fts_weight + jaccard_weight
        ranking = self._ranking

        def prepare_rows(
            rows: Sequence[dict[str, Any]],
            vectors: Sequence[Sequence[float]] = (),
        ) -> list[dict[str, Any]]:
            prepared_rows: list[dict[str, Any]] = []
            for position, row in enumerate(rows):
                fact_id = int(row["fact_id"])
                # Tags can broaden FTS recall, but only content terms count as
                # evidence that the question itself is covered.
                document_tokens = _tokens(str(row["content"]))
                fts_score = (
                    (
                        (1.0 - ranking.fts_query_coverage_weight)
                        * fts_scores[fact_id]
                        + ranking.fts_query_coverage_weight
                        * _query_coverage(query_tokens, document_tokens)
                    )
                    if fact_id in fts_scores
                    else 0.0
                )
                jaccard_score = _jaccard(query_tokens, document_tokens)
                document_vector = vectors[position] if vectors else None
                if dense_disabled:
                    has_dense = False
                    dense_score = 0.0
                elif dense_scores is not None:
                    has_dense = fact_id in dense_scores
                    dense_score = dense_scores[fact_id] if has_dense else 0.0
                elif _is_absent_vector(document_vector):
                    has_dense = False
                    dense_score = 0.0
                else:
                    has_dense = True
                    dense_score = max(
                        0.0, _cosine(query_vector, document_vector)
                    )
                vector = (
                    tuple(float(value) for value in document_vector)
                    if document_vector is not None
                    and not _is_absent_vector(document_vector)
                    else None
                )
                prepared_rows.append(
                    {
                        "row": row,
                        "fact_id": fact_id,
                        "fts_score": fts_score,
                        "jaccard_score": jaccard_score,
                        "dense_score": dense_score,
                        "lexical_score": (
                            fts_weight * fts_score + jaccard_weight * jaccard_score
                        ),
                        "has_dense": has_dense and dense_score >= _DENSE_LIST_MIN,
                        "vector": vector,
                        "hop_distance": int(row.get("hop_distance") or 0),
                    }
                )
            return prepared_rows

        prepared = prepare_rows(candidates, document_vectors)

        def rank_prepared(
            items: Sequence[dict[str, Any]],
        ) -> list[tuple[float, int, dict[str, Any]]]:
            lexical_order = sorted(
                (item for item in items if item["lexical_score"] > 0.0),
                key=lambda item: (-item["lexical_score"], item["fact_id"]),
            )
            dense_order = sorted(
                (item for item in items if item["has_dense"]),
                key=lambda item: (-item["dense_score"], item["fact_id"]),
            )
            lexical_ranks = {
                item["fact_id"]: rank
                for rank, item in enumerate(lexical_order, start=1)
            }
            dense_ranks = {
                item["fact_id"]: rank
                for rank, item in enumerate(dense_order, start=1)
            }
            ranked_rows: list[tuple[float, int, dict[str, Any]]] = []
            for item in items:
                fact_id = item["fact_id"]
                row = item["row"]
                fusion_score = _weighted_rrf(
                    lexical_ranks.get(fact_id),
                    dense_ranks.get(fact_id),
                    lexical_weight,
                    dense_weight,
                )
                trust_score = min(
                    1.0, max(0.0, float(row.get("trust_score") or 0.0))
                )
                memory_kind_score = _memory_kind_score(
                    row.get("memory_kind"), ranking
                )
                recency_score = _recency_score(
                    row, self._now, ranking.recency_half_life_days
                )
                review_score = _review_score(row.get("correction_status"))
                subject_score = named_subject_score(
                    anchors, str(row.get("content") or "")
                )
                hop_distance = int(item["hop_distance"])
                hop_prior = _ENTITY_HOP_PRIOR**hop_distance
                score = hop_prior * (
                    ranking.relevance_weight * fusion_score
                    + ranking.trust_weight * trust_score
                    + ranking.memory_kind_weight * memory_kind_score
                    + ranking.recency_weight * recency_score
                    + ranking.review_weight * review_score
                    + ranking.named_subject_weight * subject_score
                )
                if score < self._min_score:
                    continue
                result = dict(row)
                result.update(
                    {
                        "score": score,
                        "_mmr_embedding": item["vector"],
                        "fts_score": item["fts_score"],
                        "jaccard_score": item["jaccard_score"],
                        "dense_score": item["dense_score"],
                        "fusion_score": fusion_score,
                        "trust_score_component": trust_score,
                        "memory_kind_score": memory_kind_score,
                        "recency_score": recency_score,
                        "review_score": review_score,
                        "named_subject_score": subject_score,
                    }
                )
                if hop_distance:
                    result["hop_distance"] = hop_distance
                    result["hop_prior"] = hop_prior
                ranked_rows.append((score, fact_id, result))
            ranked_rows.sort(
                key=lambda item: (
                    -item[0],
                    -float(item[2].get("trust_score") or 0.0),
                    item[1],
                )
            )
            return ranked_rows

        ranked = rank_prepared(prepared)
        graph_predicates = list(active_predicates)
        graph_params = list(active_params)
        if "correction_status" in columns and include_unreviewed:
            graph_predicates.append("correction_status IS NOT ?")
            graph_params.append("unreviewed")
        expanded = []
        if self._entity_expansion:
            expanded = self._expand_via_entities(
                [fact_id for _, fact_id, _ in ranked[:_ENTITY_GRAPH_SEED_LIMIT]],
                {int(item["fact_id"]) for item in prepared},
                selected_columns,
                graph_predicates,
                graph_params,
            )
        expanded_vectors: Sequence[Sequence[float]] = ()
        if expanded and not dense_disabled:
            expanded_ids = tuple(int(row["fact_id"]) for row in expanded)
            if dense_scores is not None and self._vector_index is not None:
                try:
                    dense_scores.update(
                        self._vector_index.scores(query_vector, expanded_ids)
                    )
                except Exception as exc:
                    LOGGER.warning(
                        "sqlite-vec graph expansion warning: %s; using stored vectors",
                        exc,
                    )
                    candidate_embedder = getattr(
                        self._embedder, "embed_candidates", None
                    )
                    if callable(candidate_embedder):
                        expanded_vectors = candidate_embedder(expanded)
                    else:
                        expanded_vectors = self._embedder.embed_documents(
                            tuple(str(row["content"]) for row in expanded)
                        )
                    for row, vector in zip(expanded, expanded_vectors, strict=True):
                        if not _is_absent_vector(vector):
                            dense_scores[int(row["fact_id"])] = max(
                                0.0, _cosine(query_vector, vector)
                            )
            elif dense_scores is None:
                candidate_embedder = getattr(self._embedder, "embed_candidates", None)
                if callable(candidate_embedder):
                    expanded_vectors = candidate_embedder(expanded)
                else:
                    expanded_vectors = self._embedder.embed_documents(
                        tuple(str(row["content"]) for row in expanded)
                    )
            if expanded_vectors and len(expanded_vectors) != len(expanded):
                raise ValueError(
                    "embedder returned the wrong number of document vectors"
                )

        if expanded:
            prepared.extend(prepare_rows(expanded, expanded_vectors))
            ranked = rank_prepared(prepared)
        # A near-tie between compatible rows is not a reason to discard them
        # or the rest of the list. A near-tie between contradicting current
        # state slots stays fail-closed and is marked so the caller can see
        # the disagreement.
        if len(ranked) > 1 and ranked[0][0] - ranked[1][0] < ranking.ambiguity_margin:
            cluster = [
                item
                for item in ranked
                if ranked[0][0] - item[0] < ranking.ambiguity_margin
            ]
            if len(cluster) > 1 and _contradicting_state_slots(
                cluster[0][2], cluster[1][2]
            ):
                for _, _, row in cluster:
                    row["ambiguous"] = True
                return [row for _, _, row in cluster]
        survivors = ranked[:limit]
        if dense_disabled:
            for _, _, row in survivors:
                row["_mmr_embedding"] = None
        elif dense_scores is not None:
            mmr_vectors = self._stored_mmr_embeddings(
                [fact_id for _, fact_id, _ in survivors]
            )
            for _, fact_id, row in survivors:
                row["_mmr_embedding"] = mmr_vectors.get(fact_id)
        elif any(row["_mmr_embedding"] is None for _, _, row in survivors):
            for _, _, row in survivors:
                row["_mmr_embedding"] = None
        return [row for _, _, row in survivors]
