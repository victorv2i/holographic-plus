from __future__ import annotations

from collections import Counter
import json
import random
from pathlib import Path
from typing import Any

from .runner import EvalCase, EvalResult, STALE_CONTENT_PREFIXES
from .sqlite_utils import connect_readonly

_SUPERSEDED_PREFIXES = tuple(f"{prefix}%" for prefix in STALE_CONTENT_PREFIXES)
_PARAPHRASE_TEMPLATES = (
    "What is known about {cue}?",
    "Which memory covers {cue}?",
    "Recall the recorded fact regarding {cue}",
)


def _case_from_mapping(data: dict[str, Any]) -> EvalCase:
    return EvalCase(
        id=str(data["id"]),
        query=str(data["query"]),
        gold_fact_id=int(data["gold_fact_id"]),
        category=data.get("category"),
        min_trust=float(data.get("min_trust", 0.3)),
        stale_fact_ids=[int(x) for x in data.get("stale_fact_ids", [])],
        tags=[str(x) for x in data.get("tags", [])],
        case_type=str(data.get("case_type", "exact_fact")),
        expected_current_fact_ids=[int(x) for x in data.get("expected_current_fact_ids", [])],
        entity_refs=[str(x) for x in data.get("entity_refs", [])],
        provenance=data.get("provenance"),
        answer_rubric=data.get("answer_rubric"),
        difficulty=str(data.get("difficulty", "easy")),
        generation=str(data.get("generation", "auto")),
        privacy_tier=str(data.get("privacy_tier", "private")),
        should_abstain=bool(data.get("should_abstain", False)),
    )


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load eval cases from either a list or {"cases": [...]} JSON file."""
    payload = json.loads(Path(path).read_text())
    rows = payload["cases"] if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("case file must contain a list or a {'cases': [...]} object")
    return [_case_from_mapping(row) for row in rows]


def _active_fact_rows(
    db_path: str | Path,
    *,
    limit: int,
    min_trust: float,
    category: str | None,
) -> list[Any]:
    if limit <= 0:
        raise ValueError("limit must be positive")
    where = ["trust_score >= ?", "content IS NOT NULL", "trim(content) != ''"]
    params: list[Any] = [min_trust]
    for prefix in _SUPERSEDED_PREFIXES:
        where.append("trim(lower(content)) NOT LIKE ?")
        params.append(prefix)
    if category is not None:
        where.append("category = ?")
        params.append(category)
    params.append(limit)
    sql = f"""
        SELECT fact_id, content, category, trust_score
        FROM facts
        WHERE {' AND '.join(where)}
        ORDER BY fact_id
        LIMIT ?
    """
    with connect_readonly(db_path) as conn:
        return conn.execute(sql, params).fetchall()


def _cue_from_content(content: str) -> str:
    words = [part for part in content.split() if part]
    if len(words) <= 1:
        return f"the topic {content.strip()}"
    return " ".join(words[1:] + words[:1])


def template_paraphrase(content: str, fact_id: int) -> str:
    """Model-free question form that is never the fact's own content."""
    query = _PARAPHRASE_TEMPLATES[fact_id % len(_PARAPHRASE_TEMPLATES)].format(
        cue=_cue_from_content(content),
    )
    if query.strip() == content.strip():
        query = f"What is recorded about the topic {fact_id}?"
    return query


def generate_exact_fact_cases(
    db_path: str | Path,
    *,
    limit: int = 50,
    min_trust: float = 0.3,
    category: str | None = None,
) -> list[EvalCase]:
    """Generate a smoke eval where each fact's own content is its query.

    This is not a semantic benchmark; it is a read-only baseline that proves the
    production retrieval stack can recover known facts from a copied database.
    """
    rows = _active_fact_rows(db_path, limit=limit, min_trust=min_trust, category=category)
    return [
        EvalCase(
            id=f"fact-{int(row['fact_id'])}",
            query=str(row["content"]),
            gold_fact_id=int(row["fact_id"]),
            category=row["category"],
            min_trust=min_trust,
            tags=["exact-fact-smoke"],
            case_type="exact_fact",
            generation="auto",
            privacy_tier="private",
        )
        for row in rows
    ]


def generate_paraphrase_cases(
    db_path: str | Path,
    *,
    limit: int = 50,
    min_trust: float = 0.3,
    category: str | None = None,
) -> list[EvalCase]:
    """Generate template paraphrases so default queries are not the gold content.

    These are model-free rotations plus a question wrapper. They are not a
    semantic benchmark; they exist so FTS cannot win by exact-string identity.
    """
    rows = _active_fact_rows(db_path, limit=limit, min_trust=min_trust, category=category)
    return [
        EvalCase(
            id=f"paraphrase-{int(row['fact_id'])}",
            query=template_paraphrase(str(row["content"]), int(row["fact_id"])),
            gold_fact_id=int(row["fact_id"]),
            category=row["category"],
            min_trust=min_trust,
            tags=["template-paraphrase"],
            case_type="paraphrase",
            generation="template-paraphrase",
            privacy_tier="private",
        )
        for row in rows
    ]


def split_tune_and_holdout(
    cases: list[EvalCase],
    *,
    holdout_fraction: float = 0.4,
    seed: int = 1701,
) -> tuple[list[EvalCase], list[EvalCase]]:
    """Split cases so the tuner and the reported score do not share a set."""
    if holdout_fraction <= 0 or holdout_fraction >= 1:
        raise ValueError("holdout_fraction must be between 0 and 1 exclusive")
    if len(cases) < 2:
        return list(cases), []
    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    n_holdout = max(1, int(round(len(shuffled) * holdout_fraction)))
    if n_holdout >= len(shuffled):
        n_holdout = len(shuffled) - 1
    return shuffled[n_holdout:], shuffled[:n_holdout]


def is_self_referential_case(case: EvalCase, gold_content: str | None = None) -> bool:
    if case.case_type == "exact_fact":
        return True
    if gold_content is None:
        return False
    return case.query.strip() == gold_content.strip()


def count_self_referential_cases(cases: list[EvalCase], db_path: str | Path | None = None) -> int:
    contents: dict[int, str] = {}
    if db_path is not None:
        fact_ids = [case.gold_fact_id for case in cases]
        if fact_ids:
            placeholders = ", ".join("?" for _ in fact_ids)
            with connect_readonly(db_path) as conn:
                rows = conn.execute(
                    f"SELECT fact_id, content FROM facts WHERE fact_id IN ({placeholders})",
                    fact_ids,
                ).fetchall()
            contents = {int(row["fact_id"]): str(row["content"]) for row in rows}
    return sum(
        1
        for case in cases
        if is_self_referential_case(case, contents.get(case.gold_fact_id))
    )


def describe_case_run(
    cases: list[EvalCase],
    *,
    tune: list[EvalCase],
    holdout: list[EvalCase],
    self_referential: int,
) -> list[str]:
    types = dict(sorted(Counter(case.case_type for case in cases).items()))
    generations = dict(sorted(Counter(case.generation for case in cases).items()))
    return [
        f"Case types actually run: {types}.",
        f"Case generation: {generations}.",
        f"Tune cases: {len(tune)}; holdout cases: {len(holdout)}.",
        f"Self-referential queries: {self_referential} of {len(cases)}.",
    ]


def write_json_report(
    path: str | Path,
    *,
    summary: dict[str, Any],
    results: list[EvalResult],
    metadata: dict[str, Any],
    include_text: bool = False,
) -> None:
    """Write a local JSON report. Private text is omitted unless explicitly public.

    ``include_text=True`` is intended only for deliberately public case banks;
    public cases may include raw retrieved rows, so private MemoryArena reports
    should leave it disabled. Private case tags are also omitted because
    hand-authored tags can carry sensitive context.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    serialised = []
    for result in results:
        row: dict[str, Any] = {
            "case_id": result.case.id,
            "case_type": result.case.case_type,
            "privacy_tier": result.case.privacy_tier,
            "gold_fact_id": result.case.gold_fact_id,
            "gold_rank": result.gold_rank,
            "top_fact_ids": result.ranked_fact_ids,
            "top_scores": result.scores,
            "top_score": result.top_score,
            "score_margin": result.score_margin,
            "expected_current_fact_ids": result.case.expected_current_fact_ids,
            "stale_leak_ranks": result.stale_leak_ranks,
            "should_abstain": result.case.should_abstain,
            "latency_ms": result.latency_ms,
            "tags": result.case.tags if result.case.privacy_tier == "public" else [],
        }
        if include_text and result.case.privacy_tier == "public":
            row["query"] = result.case.query
            row["top_results"] = result.results
        serialised.append(row)
    out.write_text(json.dumps({
        "metadata": metadata,
        "summary": summary,
        "results": serialised,
    }, indent=2, sort_keys=True))
