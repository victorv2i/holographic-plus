"""Offline LongMemEval-S adapter. Oracle is never labeled as S."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


FIXTURES_DIR = Path(__file__).with_name("fixtures")
LME_SMOKE_PATH = FIXTURES_DIR / "longmemeval_smoke.json"

LME_ACQUISITION = """Acquire LongMemEval-S outside git, then hash it:

  mkdir -p ~/.config/enfold/benchmark/data
  # Download longmemeval_s_cleaned.json from HuggingFace
  # xiaowu0162/longmemeval-cleaned into that directory.
  sha256sum ~/.config/enfold/benchmark/data/longmemeval_s_cleaned.json
  # expected d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442

Do not report oracle as S. Do not commit the dataset. Run with:

  PYTHONPATH=$PWD python -m memory_eval.longmemeval_adapter \\
    --data ~/.config/enfold/benchmark/data/longmemeval_s_cleaned.json --split S
"""


@dataclass(frozen=True, slots=True)
class LmeSession:
    question_id: str
    session_index: int
    observed_at: str
    transcript: str


@dataclass(frozen=True, slots=True)
class LmeQuestion:
    question_id: str
    question: str
    answer: str
    question_type: str
    reference_time: str
    evidence_session_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LmeDataset:
    rows: tuple[dict[str, Any], ...]
    source_path: Path
    split: str


def load_longmemeval(path: str | Path, *, split: str = "S") -> LmeDataset:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"longmemeval dataset not found at {source}. {LME_ACQUISITION}")
    if split == "S" and "oracle" in source.name.lower():
        raise ValueError("never report oracle as LongMemEval-S")
    if split not in {"S", "M", "oracle"}:
        raise ValueError("split must be S, M, or oracle")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError("LongMemEval payload must be a list or {data: [...]}")
    return LmeDataset(tuple(rows), source.resolve(), split)


def _turn_text(turn: Any) -> str:
    if isinstance(turn, str):
        return turn.strip()
    if not isinstance(turn, dict):
        return ""
    content = turn.get("content") or turn.get("text") or ""
    role = str(turn.get("role") or turn.get("speaker") or "").strip()
    text = str(content).strip()
    if role and text:
        return f"{role}: {text}"
    return text


def lme_sessions(dataset: LmeDataset) -> list[LmeSession]:
    sessions: list[LmeSession] = []
    for row in dataset.rows:
        question_id = str(row.get("question_id") or "").strip() or "unknown"
        haystacks = row.get("haystack_sessions") or []
        dates = row.get("haystack_dates") or []
        if not isinstance(haystacks, list):
            raise ValueError(f"{question_id} haystack_sessions must be a list")
        if not isinstance(dates, list):
            raise ValueError(f"{question_id} haystack_dates must be a list")
        for index, session in enumerate(haystacks):
            observed = dates[index] if index < len(dates) else ""
            if not isinstance(observed, str) or not observed.strip():
                raise ValueError(f"{question_id} session {index} is missing haystack_dates")
            turns = session if isinstance(session, list) else [session]
            transcript = "\n".join(text for text in (_turn_text(turn) for turn in turns) if text)
            sessions.append(LmeSession(question_id, index, observed.strip(), transcript))
    return sessions


def lme_questions(dataset: LmeDataset) -> list[LmeQuestion]:
    questions: list[LmeQuestion] = []
    for row in dataset.rows:
        question_id = str(row.get("question_id") or "").strip()
        question = row.get("question")
        answer = row.get("answer")
        question_type = str(row.get("question_type") or "").strip()
        reference_time = str(row.get("question_date") or row.get("reference_time") or "").strip()
        if not question_id or not isinstance(question, str) or not question.strip():
            raise ValueError("LongMemEval row is missing question_id or question")
        if not reference_time:
            raise ValueError(f"{question_id} is missing question_date")
        evidence = row.get("answer_session_ids") or row.get("evidence") or []
        if not isinstance(evidence, list):
            raise ValueError(f"{question_id} evidence session ids must be a list")
        questions.append(LmeQuestion(
            question_id,
            question.strip(),
            "" if answer is None else str(answer),
            question_type,
            reference_time,
            tuple(str(item) for item in evidence),
        ))
    return questions


def run_longmemeval(path: str | Path, *, split: str = "S") -> dict[str, Any]:
    """Parse or block. This function never fabricates a published score."""

    source = Path(path)
    try:
        dataset = load_longmemeval(source, split=split)
    except FileNotFoundError:
        return {
            "status": "blocked",
            "scores": None,
            "acquisition": LME_ACQUISITION,
            "reason": f"dataset missing at {source}",
        }
    questions = lme_questions(dataset)
    sessions = lme_sessions(dataset)
    return {
        "status": "parsed",
        "scores": None,
        "split": dataset.split,
        "comparable_to_paper": dataset.split == "S" and "oracle" not in dataset.source_path.name.lower(),
        "questions": len(questions),
        "sessions": len(sessions),
        "by_type": {
            question_type: sum(1 for item in questions if item.question_type == question_type)
            for question_type in sorted({item.question_type for item in questions})
        },
        "reason": "dataset parsed; retrieval/QA scores require a local reader run",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Parse LongMemEval offline. Does not invent scores.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=("S", "M", "oracle"), default="S")
    args = parser.parse_args(argv)
    print(json.dumps(run_longmemeval(args.data, split=args.split), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
