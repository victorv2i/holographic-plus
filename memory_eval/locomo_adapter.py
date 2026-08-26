"""Offline LOCOMO adapter. No scores are invented when the dataset is absent."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from .protocol import LOCOMO_DATASET, RETRIEVAL_K


FIXTURES_DIR = Path(__file__).with_name("fixtures")
LOCOMO_SMOKE_PATH = FIXTURES_DIR / "locomo_smoke.json"
_SESSION_KEY = re.compile(r"^session_(\d+)$")

LOCOMO_ACQUISITION = f"""Acquire LOCOMO outside git, then hash it:

  mkdir -p ~/.config/enfold/benchmark/data
  curl -L -o ~/.config/enfold/benchmark/data/locomo10.json \\
    {LOCOMO_DATASET["source_url"]}
  sha256sum ~/.config/enfold/benchmark/data/locomo10.json
  # expected {LOCOMO_DATASET["sha256"]}

Alternatively copy competitors/AgenticMemory/data/locomo10.json if that tree
is present. Do not commit the dataset. Run with:

  PYTHONPATH=$PWD python -m memory_eval.locomo_adapter \\
    --data ~/.config/enfold/benchmark/data/locomo10.json
"""


@dataclass(frozen=True, slots=True)
class LocomoSession:
    conversation_id: str
    session_key: str
    observed_at: str
    transcript: str
    dialog_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocomoQuestion:
    conversation_id: str
    question: str
    answer: str | None
    evidence: tuple[str, ...]
    category: int
    adversarial_answer: str | None = None


@dataclass(frozen=True, slots=True)
class LocomoDataset:
    conversations: tuple[Any, ...]
    source_path: Path
    sha256: str


def load_locomo(path: str | Path, *, require_published_hash: bool = False) -> LocomoDataset:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"locomo10.json not found at {source}. {LOCOMO_ACQUISITION}")
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if require_published_hash and digest != LOCOMO_DATASET["sha256"]:
        raise ValueError(
            f"locomo sha256 mismatch: got {digest} expected {LOCOMO_DATASET['sha256']}"
        )
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("LOCOMO payload must be a list of conversations")
    return LocomoDataset(tuple(payload), source.resolve(), digest)


def _conversation_id(row: dict[str, Any], index: int) -> str:
    conversation = row.get("conversation")
    if isinstance(conversation, dict):
        speaker_a = str(conversation.get("speaker_a") or "").strip()
        speaker_b = str(conversation.get("speaker_b") or "").strip()
        if speaker_a and speaker_b:
            return f"{index}:{speaker_a}-{speaker_b}"
    return f"locomo-{index}"


def _turn_text(turn: dict[str, Any]) -> str:
    parts = []
    text = turn.get("text")
    if isinstance(text, str) and text.strip():
        parts.append(text.strip())
    caption = turn.get("blip_caption")
    if isinstance(caption, str) and caption.strip():
        parts.append(f"[image: {caption.strip()}]")
    return " ".join(parts)


def locomo_sessions(dataset: LocomoDataset) -> list[LocomoSession]:
    sessions: list[LocomoSession] = []
    for index, row in enumerate(dataset.conversations):
        if not isinstance(row, dict):
            raise ValueError(f"conversation {index} must be an object")
        conversation = row.get("conversation")
        if not isinstance(conversation, dict):
            raise ValueError(f"conversation {index} is missing conversation")
        conv_id = _conversation_id(row, index)
        numbered: list[tuple[int, str, str, list[dict[str, Any]]]] = []
        for key, value in conversation.items():
            match = _SESSION_KEY.match(str(key))
            if not match or not isinstance(value, list):
                continue
            date_key = f"{key}_date_time"
            observed = conversation.get(date_key)
            if not isinstance(observed, str) or not observed.strip():
                raise ValueError(f"{conv_id} {key} is missing {date_key}")
            numbered.append((int(match.group(1)), key, observed.strip(), value))
        numbered.sort(key=lambda item: item[0])
        for _number, key, observed, turns in numbered:
            lines: list[str] = []
            dialog_ids: list[str] = []
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("speaker") or "").strip() or "unknown"
                text = _turn_text(turn)
                if not text:
                    continue
                lines.append(f"{speaker}: {text}")
                dia_id = turn.get("dia_id")
                if isinstance(dia_id, str) and dia_id.strip():
                    dialog_ids.append(dia_id.strip())
            sessions.append(LocomoSession(
                conv_id, key, observed, "\n".join(lines), tuple(dialog_ids),
            ))
    return sessions


def locomo_questions(
    dataset: LocomoDataset,
    *,
    drop_category_5: bool = False,
) -> list[LocomoQuestion]:
    if drop_category_5:
        raise ValueError("never drop category 5")
    questions: list[LocomoQuestion] = []
    for index, row in enumerate(dataset.conversations):
        if not isinstance(row, dict):
            raise ValueError(f"conversation {index} must be an object")
        conv_id = _conversation_id(row, index)
        qa_rows = row.get("qa")
        if not isinstance(qa_rows, list):
            raise ValueError(f"{conv_id} qa must be a list")
        for qa in qa_rows:
            if not isinstance(qa, dict):
                raise ValueError(f"{conv_id} qa item must be an object")
            category = qa.get("category")
            if not isinstance(category, int) or isinstance(category, bool):
                raise ValueError(f"{conv_id} qa category must be an integer")
            evidence = qa.get("evidence") or []
            if not isinstance(evidence, list) or any(not isinstance(item, str) for item in evidence):
                raise ValueError(f"{conv_id} evidence must be a list of strings")
            question = qa.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"{conv_id} question must be a non-empty string")
            answer = qa.get("answer")
            if answer is not None:
                answer = str(answer)
            adversarial = qa.get("adversarial_answer")
            questions.append(LocomoQuestion(
                conv_id,
                question.strip(),
                answer,
                tuple(item.strip() for item in evidence if item.strip()),
                category,
                str(adversarial) if adversarial is not None else None,
            ))
    return questions


def score_locomo_retrieval(
    questions: Sequence[LocomoQuestion],
    retrieved_ids: Sequence[Sequence[str]],
    *,
    k_values: Sequence[int] = RETRIEVAL_K,
) -> dict[str, Any]:
    if len(questions) != len(retrieved_ids):
        raise ValueError("retrieved_ids must align with questions")
    card: dict[str, Any] = {
        "kind": "retrieval_only",
        "reader_used": False,
        "cases": len(questions),
    }
    for k in k_values:
        hits = 0
        for question, ranked in zip(questions, retrieved_ids, strict=True):
            top = list(ranked)[:k]
            if any(evidence_id in top for evidence_id in question.evidence):
                hits += 1
        card[f"recall@{k}"] = 0.0 if not questions else hits / len(questions)
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({question.category for question in questions}):
        indexes = [i for i, question in enumerate(questions) if question.category == category]
        by_category[str(category)] = {
            "id": category,
            "cases": len(indexes),
            "recall@1": (
                sum(
                    1
                    for i in indexes
                    if any(evidence_id in list(retrieved_ids[i])[:1] for evidence_id in questions[i].evidence)
                )
                / len(indexes)
            ),
        }
    card["by_category"] = by_category
    return card


def run_locomo(
    path: str | Path,
    *,
    require_published_hash: bool = False,
) -> dict[str, Any]:
    """Parse or block. This function never fabricates a published score."""

    source = Path(path)
    try:
        dataset = load_locomo(source, require_published_hash=require_published_hash)
    except FileNotFoundError:
        return {
            "status": "blocked",
            "scores": None,
            "acquisition": LOCOMO_ACQUISITION,
            "reason": f"dataset missing at {source}",
        }
    questions = locomo_questions(dataset)
    sessions = locomo_sessions(dataset)
    return {
        "status": "parsed",
        "scores": None,
        "dataset_sha256": dataset.sha256,
        "conversations": len(dataset.conversations),
        "questions": len(questions),
        "sessions": len(sessions),
        "categories": sorted({question.category for question in questions}),
        "reason": "dataset parsed; retrieval/QA scores require a local reader run",
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Parse LOCOMO offline. Does not invent scores.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--require-published-hash", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run_locomo(args.data, require_published_hash=args.require_published_hash), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
