"""Private, offline retrieval benchmark for a snapshot of an Enfold database.

This module deliberately uses :class:`enfold.hybrid_retrieval.HybridRetriever`
instead of a test-only ranking implementation.  The default Python API still
uses the deterministic feature-hash embedder so CI stays offline.  Published
retrieval claims must call ``embedder_mode='production'`` (the CLI default),
which loads the daemon's stored-embedding path and fails closed when the
production identity is missing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re
import sqlite3
import tempfile
from typing import Any, Iterable
from urllib.parse import quote

from enfold.hybrid_retrieval import DeterministicFeatureHashEmbedder, HybridRetriever


DEFAULT_DB_PATH = Path.home() / ".hermes" / "memory_store.db"
DEFAULT_CASES_PATH = Path.home() / ".config" / "enfold" / "private-arena" / "cases-v0.jsonl"


@dataclass(frozen=True, slots=True)
class PersonalCase:
    id: str
    query: str
    expected_fact_ids: tuple[int, ...]
    expected_content_regexes: tuple[str, ...]
    forbidden_content_regexes: tuple[str, ...]
    category: str
    asof: str | None = None

    @property
    def should_abstain(self) -> bool:
        return not self.expected_fact_ids and not self.expected_content_regexes


@dataclass(frozen=True, slots=True)
class PersonalResult:
    case: PersonalCase
    ranked_fact_ids: tuple[int, ...]
    expected_rank: int | None
    forbidden_rank: int | None
    abstained: bool


@dataclass(frozen=True, slots=True)
class PersonalArenaRun:
    results: tuple[PersonalResult, ...]
    summary: dict[str, Any]
    metadata: dict[str, Any]


def _readonly_uri(path: str | Path) -> str:
    return f"file:{quote(str(Path(path).expanduser().resolve()), safe='/')}?mode=ro"


def snapshot_database(source: str | Path, destination: str | Path) -> None:
    """Create a consistent SQLite backup without opening the source for writes."""

    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_readonly_uri(source), uri=True) as live:
        with sqlite3.connect(destination_path) as snapshot:
            live.backup(snapshot)


def _strings(value: Any, label: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if value is None and allow_empty:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{label} must be a list of non-empty strings")
    return tuple(item.strip() for item in value)


def _fact_ids(value: Any, label: str) -> tuple[int, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value):
        raise ValueError(f"{label} must be a list of positive integers")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates")
    return tuple(value)


def _case(raw: Any, line_number: int) -> PersonalCase:
    label = f"line {line_number}"
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a JSON object")
    case_id = raw.get("id")
    query = raw.get("query")
    category = raw.get("category")
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError(f"{label}.id must be a non-empty string")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{label}.query must be a non-empty string")
    if not isinstance(category, str) or not category.strip():
        raise ValueError(f"{label}.category must be a non-empty string")
    fact_ids = _fact_ids(raw.get("expected_fact_ids"), f"{label}.expected_fact_ids")
    expected_regexes = _strings(raw.get("expected_content_regexes"), f"{label}.expected_content_regexes")
    forbidden_regexes = _strings(raw.get("forbidden_content_regexes"), f"{label}.forbidden_content_regexes")
    asof = raw.get("asof")
    if asof is not None and (not isinstance(asof, str) or not asof.strip()):
        raise ValueError(f"{label}.asof must be a non-empty string when supplied")
    for expression in (*expected_regexes, *forbidden_regexes):
        try:
            re.compile(expression)
        except re.error as exc:
            raise ValueError(f"{label} has an invalid content regex: {exc}") from exc
    return PersonalCase(
        case_id.strip(), query.strip(), fact_ids, expected_regexes, forbidden_regexes,
        category.strip(), asof.strip() if asof else None,
    )


def load_personal_cases(path: str | Path) -> tuple[PersonalCase, ...]:
    """Load a JSONL case bank, keeping all real content outside the repository."""

    source = Path(path).expanduser()
    cases: list[PersonalCase] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"could not read case bank {source}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            cases.append(_case(json.loads(line), line_number))
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number} is not valid JSON: {exc}") from exc
    if not cases:
        raise ValueError("case bank contains no cases")
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("case ids must be unique")
    return tuple(cases)


def validate_personal_cases(
    conn: sqlite3.Connection, cases: Iterable[PersonalCase]
) -> None:
    """Ensure each positive expectation remains eligible in this DB snapshot."""

    for case in cases:
        eligible = conn.execute(
            """
            SELECT fact_id, content
            FROM facts
            WHERE scope = 'private'
              AND category = ?
              AND invalid_at IS NULL
              AND superseded_by IS NULL
              AND conflict_group IS NULL
              AND trust_score >= 0.3
            """,
            (case.category,),
        ).fetchall()
        by_id = {int(row[0]): str(row[1]) for row in eligible}
        if case.should_abstain:
            continue
        satisfiable = False
        forbidden_match = False
        for fact_id in case.expected_fact_ids:
            content = by_id.get(fact_id)
            if content is None:
                continue
            if any(
                re.search(expression, content, re.IGNORECASE)
                for expression in case.forbidden_content_regexes
            ):
                forbidden_match = True
            else:
                satisfiable = True
        contents = list(by_id.values())
        for expression in case.expected_content_regexes:
            matches = [
                content
                for content in contents
                if re.search(expression, content, re.IGNORECASE)
            ]
            for content in matches:
                if any(
                    re.search(forbidden, content, re.IGNORECASE)
                    for forbidden in case.forbidden_content_regexes
                ):
                    forbidden_match = True
                else:
                    satisfiable = True
        if satisfiable:
            continue
        if forbidden_match:
            raise ValueError(
                f"case {case.id!r} expected alternative also matches a forbidden "
                "content regex"
            )
        raise ValueError(
            f"case {case.id!r} expected alternatives are not an active private fact "
            f"in category {case.category!r}"
        )


def _expected_rank(case: PersonalCase, rows: Iterable[dict[str, Any]]) -> int | None:
    for rank, row in enumerate(rows, start=1):
        if int(row["fact_id"]) in case.expected_fact_ids:
            return rank
        content = str(row.get("content", ""))
        if any(re.search(expression, content, re.IGNORECASE) for expression in case.expected_content_regexes):
            return rank
    return None


def _forbidden_rank(case: PersonalCase, rows: Iterable[dict[str, Any]]) -> int | None:
    for rank, row in enumerate(rows, start=1):
        content = str(row.get("content", ""))
        if any(re.search(expression, content, re.IGNORECASE) for expression in case.forbidden_content_regexes):
            return rank
    return None


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _summarize(results: tuple[PersonalResult, ...]) -> dict[str, Any]:
    retrieval = [result for result in results if not result.case.should_abstain]
    abstentions = [result for result in results if result.case.should_abstain]
    stale_sensitive = [result for result in results if result.case.forbidden_content_regexes]

    def one(group: list[PersonalResult]) -> dict[str, Any]:
        positive = [result for result in group if not result.case.should_abstain]
        negative = [result for result in group if result.case.should_abstain]
        protected = [result for result in group if result.case.forbidden_content_regexes]
        return {
            "cases": len(group),
            "recall_at_1": _rate(sum(not result.abstained and result.expected_rank == 1 for result in positive), len(positive)),
            "recall_at_3": _rate(sum(not result.abstained and result.expected_rank is not None and result.expected_rank <= 3 for result in positive), len(positive)),
            "stale_leak_rate": _rate(sum(result.forbidden_rank is not None and result.forbidden_rank <= 3 for result in protected), len(protected)),
            "abstention_correctness": _rate(sum(result.abstained for result in negative), len(negative)),
            "false_abstentions": sum(result.abstained for result in positive),
        }

    by_category: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[PersonalResult]] = defaultdict(list)
    for result in results:
        grouped[result.case.category].append(result)
    for category in sorted(grouped):
        by_category[category] = one(grouped[category])
    summary = one(list(results))
    summary.update({
        "retrieval_cases": len(retrieval),
        "abstention_cases": len(abstentions),
        "stale_sensitive_cases": len(stale_sensitive),
        "by_category": by_category,
    })
    return summary


def _search_personal_cases(
    search,
    cases: list[PersonalCase],
    *,
    abstention_min_score: float,
) -> list[PersonalResult]:
    results: list[PersonalResult] = []
    for case in cases:
        rows = search(case.query, category=case.category, limit=3)
        expected_rank = _expected_rank(case, rows)
        forbidden_rank = _forbidden_rank(case, rows)
        confidently_answered = bool(rows) and float(rows[0]["score"]) >= abstention_min_score
        results.append(PersonalResult(
            case=case,
            ranked_fact_ids=tuple(int(row["fact_id"]) for row in rows),
            expected_rank=expected_rank,
            forbidden_rank=forbidden_rank,
            abstained=not confidently_answered,
        ))
    return results


def run_personal_arena(
    cases_path: str | Path = DEFAULT_CASES_PATH,
    database_path: str | Path = DEFAULT_DB_PATH,
    *,
    dimensions: int = 256,
    seed: int = 0,
    abstention_min_score: float = 0.35,
    embedder_mode: str = "hash",
    embedder_config: dict[str, Any] | None = None,
) -> PersonalArenaRun:
    """Run cases through a temporary snapshot and the production hybrid scorer.

    ``embedder_mode='hash'`` is the offline CI plumbing path. It is not a
    published retrieval claim. ``embedder_mode='production'`` uses the same
    stored-embedding HybridRetriever the daemon serves and fails closed when
    identities or coverage are missing.
    """

    if embedder_mode not in {"hash", "production"}:
        raise ValueError("embedder_mode must be 'hash' or 'production'")
    if not 0.0 <= abstention_min_score <= 1.0:
        raise ValueError("abstention_min_score must be between 0 and 1")
    cases = list(load_personal_cases(cases_path))
    random.Random(seed).shuffle(cases)
    with tempfile.TemporaryDirectory(prefix="enfold-personal-arena-") as directory:
        snapshot_path = Path(directory) / "memory_store.snapshot.db"
        snapshot_database(database_path, snapshot_path)
        conn = sqlite3.connect(snapshot_path)
        conn.row_factory = sqlite3.Row
        retriever_metadata: dict[str, Any]
        try:
            validate_personal_cases(conn, cases)
            if embedder_mode == "production":
                from .baseline import load_eval_retriever, production_like_config

                config = production_like_config(snapshot_path)
                config["retriever_mode"] = "stored"
                config["allowed_scopes"] = ["private"]
                if embedder_config:
                    config.update(embedder_config)
                provider = load_eval_retriever(snapshot_path, config)
                try:
                    results = _search_personal_cases(
                        provider.search, cases, abstention_min_score=abstention_min_score,
                    )
                    retriever_metadata = dict(provider._retriever.metadata)
                finally:
                    provider.shutdown()
            else:
                retriever = HybridRetriever(
                    conn,
                    DeterministicFeatureHashEmbedder(dimensions),
                    allowed_scopes=("private",),
                )
                results = _search_personal_cases(
                    retriever.search, cases, abstention_min_score=abstention_min_score,
                )
                retriever_metadata = dict(retriever.metadata)
        finally:
            conn.close()
    final_results = tuple(results)
    production = embedder_mode == "production"
    return PersonalArenaRun(
        final_results,
        _summarize(final_results),
        {
            "snapshot_copy": True,
            "seed": seed,
            "dimensions": dimensions,
            "abstention_min_score": abstention_min_score,
            "embedder_mode": embedder_mode,
            "claimable_retrieval": production and bool(
                retriever_metadata.get("embedder_production_ready")
            ),
            "retrieval": retriever_metadata,
            "exercised": (
                "active filters, FTS, Jaccard, hybrid weighting, production stored embeddings"
                if production
                else "active filters, FTS, Jaccard, hybrid weighting, deterministic dense scoring"
            ),
            "not_exercised": (
                "MCP transport"
                if production
                else "production embedding model quality, stored production vectors, MCP transport"
            ),
        },
    )


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _print_scorecard(run: PersonalArenaRun) -> None:
    summary = run.summary
    print("PersonalArena scorecard")
    print(f"- Cases: {summary['cases']} ({summary['retrieval_cases']} retrieval, {summary['abstention_cases']} abstention)")
    print(f"- Recall@1: {_percent(summary['recall_at_1'])}")
    print(f"- Recall@3: {_percent(summary['recall_at_3'])}")
    print(f"- Stale-leak rate@3: {_percent(summary['stale_leak_rate'])} ({summary['stale_sensitive_cases']} protected cases)")
    print(f"- Abstention correctness: {_percent(summary['abstention_correctness'])}")
    print("- Per category:")
    for category, metrics in summary["by_category"].items():
        print(
            f"  - {category}: n={metrics['cases']}, R@1={_percent(metrics['recall_at_1'])}, "
            f"R@3={_percent(metrics['recall_at_3'])}, stale={_percent(metrics['stale_leak_rate'])}, "
            f"abstain={_percent(metrics['abstention_correctness'])}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Enfold's private retrieval benchmark offline.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--dimensions", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--abstention-min-score", type=float, default=0.35)
    parser.add_argument(
        "--embedder",
        choices=("hash", "production"),
        default="production",
        help="production is the published path; hash is offline CI plumbing",
    )
    parser.add_argument(
        "--embedder-config",
        type=Path,
        help="JSON file with stored-embedding identities for --embedder production",
    )
    args = parser.parse_args(argv)
    embedder_config = None
    if args.embedder_config is not None:
        embedder_config = json.loads(args.embedder_config.read_text(encoding="utf-8"))
        if not isinstance(embedder_config, dict):
            raise ValueError("embedder config must be a JSON object")
    _print_scorecard(run_personal_arena(
        args.cases, args.db, dimensions=args.dimensions, seed=args.seed,
        abstention_min_score=args.abstention_min_score,
        embedder_mode=args.embedder,
        embedder_config=embedder_config,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
