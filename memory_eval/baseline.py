from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from enfold.hybrid_retrieval import (
    DEFAULT_RANKING_CONFIG,
    DeterministicFeatureHashEmbedder,
    HybridRetriever,
    RankingConfig,
    SQLiteVersionedEmbeddingBackend,
    StoredEmbeddingError,
    VectorFallbackTelemetry,
    VersionedStoredEmbeddingAdapter,
)

from .cases import generate_paraphrase_cases, load_cases, write_json_report
from .runner import EvalCase, run_retrieval_cases, summarize_results
from .schema_checks import inspect_memory_schema
from .sqlite_utils import BackupResult, TERMINAL_EXTRACT_QUEUE_STATUSES, backup_sqlite_db, quick_check


@dataclass(frozen=True)
class PreparedEvalDb:
    path: Path
    backup: BackupResult


def production_like_config(db_path: str | Path) -> dict[str, Any]:
    """Return HybridRetriever knobs that match the daemon's default RankingConfig."""
    ranking = DEFAULT_RANKING_CONFIG
    return {
        "db_path": str(Path(db_path)),
        "retriever_mode": "stored",
        "fts_weight": ranking.fts_weight,
        "jaccard_weight": ranking.jaccard_weight,
        "dense_weight": ranking.dense_weight,
        "fts_query_coverage_weight": ranking.fts_query_coverage_weight,
        "recency_half_life_days": ranking.recency_half_life_days,
        "score_floor": ranking.score_floor,
        "ambiguity_margin": ranking.ambiguity_margin,
        "vector_backend": "auto",
        "allowed_scopes": ["private", "work", "public"],
        "embed_on_add": False,
        "dedup_on_add": False,
        "min_trust_threshold": 0.3,
    }


def ranking_config_from_eval(config: dict[str, Any]) -> RankingConfig:
    """Map eval knobs onto RankingConfig fields that HybridRetriever actually uses."""
    defaults = DEFAULT_RANKING_CONFIG
    return RankingConfig(
        fts_weight=float(config.get("fts_weight", defaults.fts_weight)),
        jaccard_weight=float(config.get("jaccard_weight", defaults.jaccard_weight)),
        dense_weight=float(config.get("dense_weight", defaults.dense_weight)),
        fts_query_coverage_weight=float(
            config.get("fts_query_coverage_weight", defaults.fts_query_coverage_weight)
        ),
        trust_weight=float(config.get("trust_weight", defaults.trust_weight)),
        memory_kind_weight=float(config.get("memory_kind_weight", defaults.memory_kind_weight)),
        recency_weight=float(config.get("recency_weight", defaults.recency_weight)),
        review_weight=float(config.get("review_weight", defaults.review_weight)),
        named_subject_weight=float(config.get("named_subject_weight", defaults.named_subject_weight)),
        recency_half_life_days=float(
            config.get("recency_half_life_days", defaults.recency_half_life_days)
        ),
        state_kind_score=float(config.get("state_kind_score", defaults.state_kind_score)),
        insight_kind_score=float(config.get("insight_kind_score", defaults.insight_kind_score)),
        untyped_kind_score=float(config.get("untyped_kind_score", defaults.untyped_kind_score)),
        event_kind_score=float(config.get("event_kind_score", defaults.event_kind_score)),
        score_floor=float(config.get("score_floor", defaults.score_floor)),
        ambiguity_margin=float(config.get("ambiguity_margin", defaults.ambiguity_margin)),
    )


def resolve_cases(
    *,
    db_path: str | Path,
    cases_path: str | Path | None,
    sample: int,
    min_trust: float,
) -> list[EvalCase]:
    if cases_path is not None:
        return load_cases(cases_path)
    return generate_paraphrase_cases(db_path, limit=sample, min_trust=min_trust)


def prepare_eval_db(db_path: str | Path, scratch_db: str | Path) -> PreparedEvalDb:
    """Create the writable DB snapshot used by HybridRetriever during eval.

    HybridRetriever opens a normal SQLite connection. Even with `bump=False`,
    the safest boundary is therefore: never hand the retriever the
    operator-supplied path. Always run on a fresh backup-API copy.
    """
    backup = backup_sqlite_db(db_path, scratch_db, overwrite=True)
    return PreparedEvalDb(path=backup.destination, backup=backup)


def clear_pending_extract_queue_for_eval(db_path: str | Path) -> int:
    """Delete pending extraction rows from the scratch DB before loading the retriever.

    The eval runner snapshots first, then measures schema state. HybridRetriever
    does not start an extraction worker, but the scratch copy is still cleared
    so eval retrieval cannot observe pending-queue side effects.
    """
    db = Path(db_path)
    with closing(sqlite3.connect(db)) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "extract_queue" not in tables:
            return 0
        cols = {row[1] for row in conn.execute("PRAGMA table_info(extract_queue)")}
        if "status" in cols:
            placeholders = ", ".join("?" for _ in TERMINAL_EXTRACT_QUEUE_STATUSES)
            where = f"status NOT IN ({placeholders})"
            count = int(conn.execute(
                f"SELECT COUNT(*) FROM extract_queue WHERE {where}",
                TERMINAL_EXTRACT_QUEUE_STATUSES,
            ).fetchone()[0])
            conn.execute(
                f"DELETE FROM extract_queue WHERE {where}",
                TERMINAL_EXTRACT_QUEUE_STATUSES,
            )
        else:
            count = int(conn.execute("SELECT COUNT(*) FROM extract_queue").fetchone()[0])
            conn.execute("DELETE FROM extract_queue")
        conn.commit()
        return count


class HybridEvalProvider:
    """SearchProvider adapter over the daemon's HybridRetriever."""

    def __init__(self, retriever: HybridRetriever, conn: sqlite3.Connection):
        self._retriever = retriever
        self._conn = conn

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        min_trust: float = 0.3,
        limit: int = 10,
        bump: bool = False,
    ) -> list[dict[str, Any]]:
        return self._retriever.search(
            query,
            category=category,
            min_trust=min_trust,
            limit=limit,
            bump=bump,
        )

    def shutdown(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _stored_adapter(conn: sqlite3.Connection, config: dict[str, Any]) -> VersionedStoredEmbeddingAdapter:
    """Build the stored embedder adapter the same way the daemon does.

    Mirrors enfold/server.py:911-926. Missing production dependencies raise
    instead of falling back to EnfoldProvider.
    """
    from enfold.embeddings import FastEmbedder, OllamaEmbedder

    required = (
        "provider",
        "model",
        "dimensions",
        "query_identity",
        "document_identity",
        "embedding_version",
    )
    missing = [key for key in required if config.get(key) in (None, "")]
    if missing:
        raise RuntimeError(
            "stored retrieval is not ready; missing config fields "
            f"{missing}. The eval harness will not fall back to EnfoldProvider."
        )

    provider_name = str(config["provider"])
    if provider_name == "ollama":
        kwargs: dict[str, Any] = {"model": config["model"]}
        if config.get("timeout") is not None:
            kwargs["timeout"] = config["timeout"]
        if config.get("keep_alive") is not None:
            kwargs["keep_alive"] = config["keep_alive"]
        if config.get("base_url") is not None:
            kwargs["base_url"] = config["base_url"]
        query_embedder = OllamaEmbedder(**kwargs)
    elif provider_name == "fastembed":
        query_embedder = FastEmbedder(
            model=str(config["model"]),
            cache_dir=config.get("cache_dir"),
        )
    else:
        raise RuntimeError(
            f"stored retrieval provider {provider_name!r} is unsupported. "
            "The eval harness will not fall back to EnfoldProvider."
        )

    try:
        backend = SQLiteVersionedEmbeddingBackend(
            conn,
            query_embedder,
            query_identity=str(config["query_identity"]),
            document_identity=str(config["document_identity"]),
            embedding_version=str(config["embedding_version"]),
            dimensions=int(config["dimensions"]),
            query_prefix=str(config.get("query_prefix") or ""),
        )
    except (StoredEmbeddingError, ValueError) as exc:
        raise RuntimeError(
            f"stored retrieval is not ready: {exc}. "
            "The eval harness will not fall back to EnfoldProvider."
        ) from exc
    return VersionedStoredEmbeddingAdapter(backend)


def load_eval_retriever(db_path: str | Path, config: dict[str, Any]) -> HybridEvalProvider:
    """Construct the same HybridRetriever class the daemon serves.

    Stored mode follows enfold/server.py:928-943. CI mode uses the same
    HybridRetriever class with DeterministicFeatureHashEmbedder, matching
    the daemon's explicit non-production CI factory. There is no
    EnfoldProvider fallback.
    """
    mode = config.get("retriever_mode")
    if mode not in {"ci", "stored"}:
        raise RuntimeError(
            "eval retriever_mode must be 'stored' or 'ci'. "
            "The eval harness will not fall back to EnfoldProvider."
        )

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    ranking = ranking_config_from_eval(config)
    scopes = tuple(config.get("allowed_scopes") or ("private", "work", "public"))
    telemetry = VectorFallbackTelemetry()
    vector_backend = str(config.get("vector_backend") or "auto")

    try:
        if mode == "ci":
            if not config.get("allow_nonproduction"):
                raise RuntimeError(
                    "CI retrieval is non-production and requires allow_nonproduction=true. "
                    "The eval harness will not fall back to EnfoldProvider."
                )
            embedder: Any = DeterministicFeatureHashEmbedder(int(config.get("dimensions", 256)))
        else:
            embedder = _stored_adapter(conn, config)
        retriever = HybridRetriever(
            conn,
            embedder,
            allowed_scopes=scopes,
            vector_backend=vector_backend,
            vector_fallback_telemetry=telemetry,
            ranking_config=ranking,
        )
    except Exception:
        conn.close()
        raise
    return HybridEvalProvider(retriever, conn)


def load_provider(repo_root: Path, config: dict[str, Any], *, hermes_src: Path | None, test_stubs: bool):
    """Load HybridRetriever for eval. Hermes stubs are unused and ignored."""
    del repo_root, hermes_src, test_stubs
    return load_eval_retriever(config["db_path"], config)


def _metadata(
    db_path: Path,
    cases: list[EvalCase],
    config: dict[str, Any],
    *,
    quick: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    # Keep only non-secret operational fields; this function intentionally never
    # reports URLs with credentials or environment-derived provider/API settings.
    allowed = {
        "retriever_mode",
        "provider",
        "model",
        "dimensions",
        "fts_weight",
        "jaccard_weight",
        "dense_weight",
        "fts_query_coverage_weight",
        "recency_half_life_days",
        "score_floor",
        "ambiguity_margin",
        "vector_backend",
        "allowed_scopes",
        "allow_nonproduction",
        "embed_on_add",
        "dedup_on_add",
        "min_trust_threshold",
    }
    redacted_config = {k: v for k, v in config.items() if k in allowed}
    return {
        "db_path": str(db_path),
        "quick_check": quick,
        "case_count": len(cases),
        "case_source": (
            "exact-fact-smoke"
            if cases and all("exact-fact-smoke" in c.tags for c in cases)
            else "template-paraphrase"
            if cases and all("template-paraphrase" in c.tags for c in cases)
            else "file"
        ),
        "config": redacted_config,
        "schema": schema,
    }


def run_baseline(
    *,
    db_path: str | Path,
    out_path: str | Path,
    cases_path: str | Path | None = None,
    sample: int = 50,
    limit: int = 10,
    min_trust: float = 0.3,
    repo_root: str | Path = ".",
    hermes_src: str | Path | None = None,
    test_stubs: bool = False,
    include_text: bool = False,
    scratch_db: str | Path | None = None,
    retriever_mode: str | None = None,
) -> dict[str, Any]:
    out = Path(out_path)
    scratch = Path(scratch_db) if scratch_db is not None else out.with_suffix(".db")
    prepared = prepare_eval_db(db_path, scratch)
    db = prepared.path
    quick = quick_check(db)
    cases = resolve_cases(db_path=db, cases_path=cases_path, sample=sample, min_trust=min_trust)
    config = production_like_config(db)
    if retriever_mode is not None:
        config["retriever_mode"] = retriever_mode
        if retriever_mode == "ci":
            config["allow_nonproduction"] = True
    schema = inspect_memory_schema(
        db,
        current_embedding_identity="ollama:embeddinggemma:document:auto:v1",
    )
    cleared_queue_rows = clear_pending_extract_queue_for_eval(db)
    provider = load_provider(
        Path(repo_root),
        config,
        hermes_src=Path(hermes_src) if hermes_src else None,
        test_stubs=test_stubs,
    )
    try:
        results = run_retrieval_cases(provider, cases, limit=limit)
    finally:
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()
    summary = summarize_results(results)
    metadata = _metadata(db, cases, config, quick=quick, schema=schema)
    metadata["input_db_path"] = str(Path(db_path))
    metadata["snapshot"] = {
        "source": str(prepared.backup.source),
        "destination": str(prepared.backup.destination),
        "quick_check": prepared.backup.quick_check,
        "bytes": prepared.backup.bytes,
    }
    metadata["eval_safety"] = {
        "cleared_extract_queue_rows": cleared_queue_rows,
    }
    write_json_report(out_path, summary=summary, results=results, metadata=metadata, include_text=include_text)
    return {"metadata": metadata, "summary": summary, "out_path": str(out_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Enfold read-only memory baseline on a SQLite DB snapshot.")
    parser.add_argument("--db", required=True, help="Input SQLite memory_store.db; copied to a scratch DB before provider use")
    parser.add_argument("--out", required=True, help="Path to write JSON report")
    parser.add_argument("--scratch-db", help="Writable snapshot path; defaults to OUT with .db suffix")
    parser.add_argument("--cases", help="Optional JSON eval case file")
    parser.add_argument("--sample", type=int, default=50, help="Number of template-paraphrase cases when --cases is omitted")
    parser.add_argument("--limit", type=int, default=10, help="Search result limit")
    parser.add_argument("--min-trust", type=float, default=0.3)
    parser.add_argument("--repo-root", default=str(Path.cwd()))
    parser.add_argument("--hermes-src", help="Hermes source root for real parent provider imports")
    parser.add_argument("--test-stubs", action="store_true", help="Ignored; HybridRetriever does not use Hermes stubs")
    parser.add_argument("--include-text", action="store_true", help="Include public-tier query/result content in the local JSON report")
    parser.add_argument(
        "--retriever-mode",
        choices=("stored", "ci"),
        default="stored",
        help="stored matches the daemon; ci uses HybridRetriever with the deterministic embedder",
    )
    args = parser.parse_args(argv)

    result = run_baseline(
        db_path=args.db,
        out_path=args.out,
        scratch_db=args.scratch_db,
        cases_path=args.cases,
        sample=args.sample,
        limit=args.limit,
        min_trust=args.min_trust,
        repo_root=args.repo_root,
        hermes_src=args.hermes_src,
        test_stubs=args.test_stubs,
        include_text=args.include_text,
        retriever_mode=args.retriever_mode,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
