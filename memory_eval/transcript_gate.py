"""Real-transcript capture gate.

This is the instrument that decides whether automatic session capture may ship.
It scores saved, provider-neutral extraction outputs against a bank of
role-structured transcripts. It never opens the live Enfold database, never
imports the extraction runtime, and never makes a network request.

The bundled seven-case synthetic seed is not this gate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


CASE_SCHEMA_VERSION = "enfold-transcript-gate-case-v1"
OUTPUT_SCHEMA_VERSION = "enfold-transcript-gate-output-v1"
ROLES = ("user", "assistant", "tool")
FACT_KINDS = ("typed_slot", "untyped_durable")
FORBIDDEN_CLASSES = ("assistant_authored", "tool_banner", "assistant_monologue")
DECISIONS = ("add", "dedup", "supersede", "conflict", "reject", "abstain")

FIXTURES_DIR = Path(__file__).with_name("fixtures")
DEFAULT_CASES_PATH = FIXTURES_DIR / "transcript_gate_cases.jsonl"
ROLE_GOLD_OUTPUTS_PATH = FIXTURES_DIR / "transcript_gate_gold.jsonl"
PRODUCTION_JUNK_OUTPUTS_PATH = FIXTURES_DIR / "transcript_gate_production_junk.jsonl"
PRODUCTION_FAILURE_CASE_ID = "prod-autoextract-junk-replay"

# False assistant/tool writes corrupt memory. False misses are friction.
# Same split as the verifier gate: 1.6% false-verify was tolerated only
# because a false needs_review costs review, while a false verified
# writes fiction. Here the corrupting class must be zero.
GATE_THRESHOLDS: dict[str, float] = {
    "forbidden_assistant_tool_rate": 0.0,
    "speaker_misattribution_rate": 0.0,
    "typed_slot_precision_min": 1.0,
    "typed_slot_completeness_min": 0.90,
    "silent_demotion_rate": 0.0,
    "incidental_durable_recall_min": 0.70,
    "incidental_durable_precision_min": 0.90,
}


@dataclass(frozen=True, slots=True)
class Turn:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class ExpectedFact:
    key: str
    kind: str
    speaker: str
    content_patterns: tuple[str, ...]
    state: Mapping[str, Any] | None
    evidence_excerpt: str | None

    def content_matches(self, content: str) -> bool:
        return any(re.search(pattern, content) is not None for pattern in self.content_patterns)


@dataclass(frozen=True, slots=True)
class ForbiddenSpan:
    key: str
    speaker: str
    class_name: str
    content_patterns: tuple[str, ...]

    def matches(self, content: str) -> bool:
        return any(re.search(pattern, content) is not None for pattern in self.content_patterns)


@dataclass(frozen=True, slots=True)
class TranscriptCase:
    case_id: str
    turns: tuple[Turn, ...]
    expected_decision: str
    expected_facts: tuple[ExpectedFact, ...]
    forbidden_spans: tuple[ForbiddenSpan, ...]
    tags: tuple[str, ...]
    source: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class TranscriptBank:
    cases: tuple[TranscriptCase, ...]
    source_path: Path
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PredictedFact:
    content: str
    asserted_by: str
    evidence_role: str | None
    evidence_excerpt: str | None
    state: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class PredictedOutput:
    case_id: str
    decision: str
    facts: tuple[PredictedFact, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CaseScore:
    case_id: str
    expected_decision: str
    actual_decision: str | None
    speaker_correct: int
    speaker_predicted: int
    speaker_misattributions: int
    typed_expected: int
    typed_complete: int
    typed_predicted: int
    typed_correct: int
    silent_demotions: int
    incidental_expected: int
    incidental_matched: int
    incidental_predicted: int
    forbidden_leaks: tuple[str, ...]
    assistant_authored_facts: int
    tool_banner_facts: int
    missing_expectations: tuple[str, ...]
    failures: tuple[str, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class TranscriptGateScore:
    bank: TranscriptBank
    cases: tuple[CaseScore, ...]
    metrics: Mapping[str, Any]

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, label: str, *, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    values = tuple(_text(item, f"{label}[]") for item in value)
    if not allow_empty and not values:
        raise ValueError(f"{label} must not be empty")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} must not contain duplicates")
    return values


def _objects(value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError(f"{label}[{index}] must be an object")
        rows.append(item)
    return rows


def _strict_keys(
    value: Mapping[str, Any],
    label: str,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    missing = required - set(value)
    extra = set(value) - required - optional
    if missing:
        raise ValueError(f"{label} is missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"{label} has unknown fields: {sorted(extra)}")


def _patterns(value: Any, label: str) -> tuple[str, ...]:
    patterns = _string_list(value, label, allow_empty=False)
    for index, pattern in enumerate(patterns):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"{label}[{index}] is not a valid regular expression: {exc}") from exc
    return patterns


def _state(value: Any, label: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be an object or null")
    allowed = {"kind", "subject", "predicate", "object", "value", "valid_from", "confidence"}
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"{label} has unknown fields: {sorted(extra)}")
    return dict(value)


def _turn(value: Mapping[str, Any], label: str) -> Turn:
    _strict_keys(value, label, required={"role", "content"})
    role = _text(value["role"], f"{label}.role")
    if role not in ROLES:
        raise ValueError(f"{label}.role must be one of {list(ROLES)}")
    return Turn(role, _text(value["content"], f"{label}.content"))


def _expected_fact(value: Mapping[str, Any], label: str) -> ExpectedFact:
    _strict_keys(
        value,
        label,
        required={"key", "kind", "speaker", "content_patterns"},
        optional={"state", "evidence_excerpt"},
    )
    kind = _text(value["kind"], f"{label}.kind")
    if kind not in FACT_KINDS:
        raise ValueError(f"{label}.kind must be one of {list(FACT_KINDS)}")
    speaker = _text(value["speaker"], f"{label}.speaker")
    if speaker != "user":
        raise ValueError(f"{label}.speaker must be 'user'")
    excerpt = value.get("evidence_excerpt")
    if excerpt is not None:
        excerpt = _text(excerpt, f"{label}.evidence_excerpt")
    state = _state(value.get("state"), f"{label}.state")
    if kind == "typed_slot" and state is None:
        raise ValueError(f"{label} typed_slot facts must include a state group")
    if kind == "untyped_durable" and state is not None:
        raise ValueError(f"{label} untyped_durable facts must not include state")
    return ExpectedFact(
        _text(value["key"], f"{label}.key"),
        kind,
        speaker,
        _patterns(value["content_patterns"], f"{label}.content_patterns"),
        state,
        excerpt,
    )


def _forbidden(value: Mapping[str, Any], label: str) -> ForbiddenSpan:
    _strict_keys(value, label, required={"key", "speaker", "class", "content_patterns"})
    speaker = _text(value["speaker"], f"{label}.speaker")
    if speaker not in {"assistant", "tool"}:
        raise ValueError(f"{label}.speaker must be assistant or tool")
    class_name = _text(value["class"], f"{label}.class")
    if class_name not in FORBIDDEN_CLASSES:
        raise ValueError(f"{label}.class must be one of {list(FORBIDDEN_CLASSES)}")
    return ForbiddenSpan(
        _text(value["key"], f"{label}.key"),
        speaker,
        class_name,
        _patterns(value["content_patterns"], f"{label}.content_patterns"),
    )


def _case(value: Mapping[str, Any], index: int) -> TranscriptCase:
    label = f"cases[{index}]"
    _strict_keys(
        value,
        label,
        required={
            "schema_version",
            "case_id",
            "turns",
            "expected_decision",
            "expected_facts",
            "forbidden_spans",
        },
        optional={"tags", "source"},
    )
    if value["schema_version"] != CASE_SCHEMA_VERSION:
        raise ValueError(f"{label}.schema_version must be {CASE_SCHEMA_VERSION!r}")
    turns = tuple(
        _turn(row, f"{label}.turns[{turn_index}]")
        for turn_index, row in enumerate(_objects(value["turns"], f"{label}.turns"))
    )
    if not turns:
        raise ValueError(f"{label}.turns must not be empty")
    decision = _text(value["expected_decision"], f"{label}.expected_decision")
    if decision not in DECISIONS:
        raise ValueError(f"{label}.expected_decision must be one of {list(DECISIONS)}")
    expected = tuple(
        _expected_fact(row, f"{label}.expected_facts[{fact_index}]")
        for fact_index, row in enumerate(
            _objects(value["expected_facts"], f"{label}.expected_facts")
        )
    )
    forbidden = tuple(
        _forbidden(row, f"{label}.forbidden_spans[{span_index}]")
        for span_index, row in enumerate(
            _objects(value["forbidden_spans"], f"{label}.forbidden_spans")
        )
    )
    if decision in {"reject", "abstain"} and expected:
        raise ValueError(f"{label} reject/abstain cases cannot require facts")
    if decision not in {"reject", "abstain"} and not expected:
        raise ValueError(f"{label} {decision} cases must require at least one fact")
    keys = [fact.key for fact in expected] + [span.key for span in forbidden]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label} keys must be unique")
    source = value.get("source", {})
    if not isinstance(source, dict):
        raise ValueError(f"{label}.source must be an object")
    return TranscriptCase(
        _text(value["case_id"], f"{label}.case_id"),
        turns,
        decision,
        expected,
        forbidden,
        _string_list(value.get("tags", []), f"{label}.tags"),
        source,
    )


def _load_rows(path: str | Path, wrapper_key: str) -> tuple[list[dict[str, Any]], Path]:
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"could not read {source}: {exc}") from exc
    if not text.strip():
        raise ValueError(f"{source} is empty")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source} line {line_number} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{source} line {line_number} must be a JSON object")
            rows.append(row)
    else:
        if isinstance(payload, list):
            raw_rows = payload
        elif isinstance(payload, dict) and wrapper_key in payload:
            raw_rows = payload[wrapper_key]
        elif isinstance(payload, dict):
            raw_rows = [payload]
        else:
            raise ValueError(f"{source} must contain objects, a list, or a {wrapper_key!r} wrapper")
        rows = _objects(raw_rows, wrapper_key)
    if not rows:
        raise ValueError(f"{source} contains no {wrapper_key}")
    return rows, source.resolve()


def load_transcript_gate(path: str | Path = DEFAULT_CASES_PATH) -> TranscriptBank:
    """Load the real-transcript capture bank."""

    rows, source = _load_rows(path, "cases")
    header_tags: tuple[str, ...] = ()
    if rows and rows[0].get("schema_version") == "enfold-transcript-gate-bank-v1":
        header = rows.pop(0)
        header_tags = _string_list(header.get("tags", []), "bank.tags")
    cases = tuple(_case(row, index) for index, row in enumerate(rows))
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("transcript gate case ids must be unique")
    return TranscriptBank(cases, source, header_tags)


def _predicted_fact(value: Mapping[str, Any], label: str) -> PredictedFact:
    _strict_keys(
        value,
        label,
        required={"content", "asserted_by"},
        optional={"evidence_role", "evidence_excerpt", "state"},
    )
    role = value.get("evidence_role")
    if role is not None:
        role = _text(role, f"{label}.evidence_role")
        if role not in ROLES:
            raise ValueError(f"{label}.evidence_role must be one of {list(ROLES)}")
    excerpt = value.get("evidence_excerpt")
    if excerpt is not None:
        excerpt = _text(excerpt, f"{label}.evidence_excerpt")
    return PredictedFact(
        _text(value["content"], f"{label}.content"),
        _text(value["asserted_by"], f"{label}.asserted_by"),
        role,
        excerpt,
        _state(value.get("state"), f"{label}.state"),
    )


def _predicted_output(value: Mapping[str, Any], index: int) -> PredictedOutput:
    label = f"outputs[{index}]"
    _strict_keys(
        value,
        label,
        required={"schema_version", "case_id", "decision", "facts"},
        optional={"metadata"},
    )
    if value["schema_version"] != OUTPUT_SCHEMA_VERSION:
        raise ValueError(f"{label}.schema_version must be {OUTPUT_SCHEMA_VERSION!r}")
    decision = _text(value["decision"], f"{label}.decision")
    if decision not in DECISIONS:
        raise ValueError(f"{label}.decision must be one of {list(DECISIONS)}")
    facts = tuple(
        _predicted_fact(row, f"{label}.facts[{fact_index}]")
        for fact_index, row in enumerate(_objects(value["facts"], f"{label}.facts"))
    )
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{label}.metadata must be an object")
    return PredictedOutput(
        _text(value["case_id"], f"{label}.case_id"),
        decision,
        facts,
        metadata,
    )


def load_predicted_outputs(path: str | Path) -> tuple[PredictedOutput, ...]:
    """Load saved gate outputs without invoking any provider or model."""

    rows, _source = _load_rows(path, "outputs")
    outputs = tuple(_predicted_output(row, index) for index, row in enumerate(rows))
    ids = [output.case_id for output in outputs]
    if len(ids) != len(set(ids)):
        raise ValueError("predicted output case ids must be unique")
    return outputs


def _complete_state(value: Mapping[str, Any] | None) -> Mapping[str, str] | None:
    if not value:
        return None
    kind = value.get("kind")
    subject = value.get("subject")
    predicate = value.get("predicate")
    object_value = value.get("value", value.get("object"))
    if not all(
        isinstance(item, str) and item.strip()
        for item in (kind, subject, predicate, object_value)
    ):
        return None
    return {
        "kind": str(kind).strip(),
        "subject": str(subject).strip().casefold(),
        "predicate": str(predicate).strip().casefold(),
        "value": str(object_value).strip().casefold(),
    }


def _state_matches(expected: Mapping[str, Any] | None, actual: Mapping[str, Any] | None) -> bool:
    want = _complete_state(expected)
    got = _complete_state(actual)
    if want is None or got is None:
        return False
    return want == got


def _locate_evidence(case: TranscriptCase, fact: PredictedFact) -> Turn | None:
    excerpt = fact.evidence_excerpt
    if excerpt:
        hits = [turn for turn in case.turns if excerpt in turn.content]
        if len(hits) == 1:
            return hits[0]
        if fact.evidence_role is not None:
            role_hits = [turn for turn in hits if turn.role == fact.evidence_role]
            if len(role_hits) == 1:
                return role_hits[0]
        return None
    if fact.evidence_role is not None:
        role_hits = [turn for turn in case.turns if turn.role == fact.evidence_role]
        if len(role_hits) == 1:
            return role_hits[0]
    return None


def _forbidden_hits(case: TranscriptCase, fact: PredictedFact, turn: Turn | None) -> tuple[str, ...]:
    hits: list[str] = []
    for span in case.forbidden_spans:
        if span.matches(fact.content) or (
            fact.evidence_excerpt is not None and span.matches(fact.evidence_excerpt)
        ):
            hits.append(span.key)
    if turn is not None and turn.role in {"assistant", "tool"}:
        class_name = "tool_banner" if turn.role == "tool" else "assistant_authored"
        key = f"{class_name}:{turn.role}"
        if key not in hits:
            hits.append(key)
    if fact.evidence_role in {"assistant", "tool"}:
        class_name = "tool_banner" if fact.evidence_role == "tool" else "assistant_authored"
        key = f"{class_name}:{fact.evidence_role}"
        if key not in hits:
            hits.append(key)
    return tuple(hits)


def _maximum_match(
    expected: Sequence[ExpectedFact],
    facts: Sequence[PredictedFact],
) -> dict[int, int]:
    neighbors: dict[int, list[int]] = {}
    for expected_index, item in enumerate(expected):
        eligible = [
            actual_index
            for actual_index, actual in enumerate(facts)
            if item.content_matches(actual.content)
        ]
        neighbors[expected_index] = eligible
    assigned: dict[int, int] = {}

    def assign(expected_index: int, seen: set[int]) -> bool:
        for actual_index in neighbors[expected_index]:
            if actual_index in seen:
                continue
            seen.add(actual_index)
            previous = assigned.get(actual_index)
            if previous is None or assign(previous, seen):
                assigned[actual_index] = expected_index
                return True
        return False

    for expected_index in sorted(neighbors, key=lambda item: (len(neighbors[item]), item)):
        assign(expected_index, set())
    return {expected_index: actual_index for actual_index, expected_index in assigned.items()}


def _score_case(case: TranscriptCase, output: PredictedOutput | None) -> CaseScore:
    typed_expected = tuple(fact for fact in case.expected_facts if fact.kind == "typed_slot")
    incidental_expected = tuple(
        fact for fact in case.expected_facts if fact.kind == "untyped_durable"
    )
    if output is None:
        return CaseScore(
            case_id=case.case_id,
            expected_decision=case.expected_decision,
            actual_decision=None,
            speaker_correct=0,
            speaker_predicted=0,
            speaker_misattributions=0,
            typed_expected=len(typed_expected),
            typed_complete=0,
            typed_predicted=0,
            typed_correct=0,
            silent_demotions=0,
            incidental_expected=len(incidental_expected),
            incidental_matched=0,
            incidental_predicted=0,
            forbidden_leaks=(),
            assistant_authored_facts=0,
            tool_banner_facts=0,
            missing_expectations=tuple(fact.key for fact in case.expected_facts),
            failures=("missing saved output",),
            passed=False,
        )

    matching = _maximum_match(case.expected_facts, output.facts)
    missing = tuple(
        expected.key
        for index, expected in enumerate(case.expected_facts)
        if index not in matching
    )
    speaker_correct = 0
    misattributions = 0
    forbidden: list[str] = []
    assistant_facts = 0
    tool_facts = 0
    for actual in output.facts:
        turn = _locate_evidence(case, actual)
        leaks = _forbidden_hits(case, actual, turn)
        if leaks:
            forbidden.extend(leaks)
        if (turn is not None and turn.role == "assistant") or actual.evidence_role == "assistant":
            assistant_facts += 1
        if (turn is not None and turn.role == "tool") or actual.evidence_role == "tool":
            tool_facts += 1
        if turn is None:
            continue
        attributed_user = actual.asserted_by == "user"
        if attributed_user and turn.role == "user" and (actual.evidence_role in {None, "user"}):
            speaker_correct += 1
        elif attributed_user and turn.role != "user":
            misattributions += 1
        elif actual.evidence_role is not None and actual.evidence_role != turn.role:
            misattributions += 1
        elif turn.role == "user" and actual.asserted_by == "user":
            speaker_correct += 1

    typed_complete = 0
    typed_correct = 0
    silent = 0
    incidental_matched = 0
    for expected_index, actual_index in matching.items():
        expected = case.expected_facts[expected_index]
        actual = output.facts[actual_index]
        if expected.kind == "typed_slot":
            if _state_matches(expected.state, actual.state):
                typed_complete += 1
                typed_correct += 1
            else:
                silent += 1
        else:
            incidental_matched += 1

    typed_predicted = sum(
        _complete_state(fact.state) is not None for fact in output.facts
    )
    incidental_predicted = sum(
        _complete_state(fact.state) is None for fact in output.facts
    )
    failures: list[str] = []
    if output.decision != case.expected_decision:
        failures.append(
            f"decision mismatch: expected {case.expected_decision}, got {output.decision}"
        )
    if missing:
        failures.append(f"missing required facts: {', '.join(missing)}")
    if forbidden:
        failures.append(f"forbidden facts leaked: {', '.join(sorted(set(forbidden)))}")
    if misattributions:
        failures.append(f"speaker misattribution: {misattributions}")
    if silent:
        failures.append(f"typed state silent demotion: {silent}")
    if case.expected_decision in {"reject", "abstain"} and output.facts:
        failures.append(f"{case.expected_decision} must return no facts")
    return CaseScore(
        case_id=case.case_id,
        expected_decision=case.expected_decision,
        actual_decision=output.decision,
        speaker_correct=speaker_correct,
        speaker_predicted=len(output.facts),
        speaker_misattributions=misattributions,
        typed_expected=len(typed_expected),
        typed_complete=typed_complete,
        typed_predicted=typed_predicted,
        typed_correct=typed_correct,
        silent_demotions=silent,
        incidental_expected=len(incidental_expected),
        incidental_matched=incidental_matched,
        incidental_predicted=incidental_predicted,
        forbidden_leaks=tuple(dict.fromkeys(forbidden)),
        assistant_authored_facts=assistant_facts,
        tool_banner_facts=tool_facts,
        missing_expectations=missing,
        failures=tuple(failures),
        passed=not failures,
    )


def _ratio(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return empty if denominator == 0 else numerator / denominator


def _metrics(scores: Sequence[CaseScore]) -> dict[str, Any]:
    speaker_predicted = sum(score.speaker_predicted for score in scores)
    speaker_correct = sum(score.speaker_correct for score in scores)
    misattributions = sum(score.speaker_misattributions for score in scores)
    typed_expected = sum(score.typed_expected for score in scores)
    typed_complete = sum(score.typed_complete for score in scores)
    typed_predicted = sum(score.typed_predicted for score in scores)
    typed_correct = sum(score.typed_correct for score in scores)
    silent = sum(score.silent_demotions for score in scores)
    incidental_expected = sum(score.incidental_expected for score in scores)
    incidental_matched = sum(score.incidental_matched for score in scores)
    incidental_predicted = sum(score.incidental_predicted for score in scores)
    leaks = sum(len(score.forbidden_leaks) for score in scores)
    leak_cases = sum(bool(score.forbidden_leaks) for score in scores)
    protected = sum(1 for score in scores if True)
    return {
        "speaker_attribution": {
            "correct": speaker_correct,
            "predicted": speaker_predicted,
            "misattributions": misattributions,
            "accuracy": _ratio(speaker_correct, speaker_predicted, empty=1.0),
            "misattribution_rate": _ratio(misattributions, speaker_predicted, empty=0.0),
        },
        "typed_slot_completeness": {
            "expected": typed_expected,
            "complete": typed_complete,
            "predicted": typed_predicted,
            "correct": typed_correct,
            "silent_demotions": silent,
            "recall": _ratio(typed_complete, typed_expected, empty=1.0),
            "precision": _ratio(typed_correct, typed_predicted, empty=1.0 if typed_expected == 0 else 0.0),
            "silent_demotion_rate": _ratio(silent, typed_expected, empty=0.0),
        },
        "incidental_durable_recall": {
            "expected": incidental_expected,
            "matched": incidental_matched,
            "predicted": incidental_predicted,
            "recall": _ratio(incidental_matched, incidental_expected, empty=1.0),
            "precision": _ratio(
                incidental_matched,
                incidental_predicted,
                empty=1.0 if incidental_expected == 0 else 0.0,
            ),
        },
        "forbidden_assistant_tool": {
            "leaks": leaks,
            "leak_cases": leak_cases,
            "protected_cases": protected,
            "rate": _ratio(leak_cases, protected, empty=0.0),
        },
    }


def score_transcript_gate(
    bank: TranscriptBank,
    outputs: Iterable[PredictedOutput],
) -> TranscriptGateScore:
    """Score saved outputs against the real-transcript bank."""

    output_rows = tuple(outputs)
    by_id = {output.case_id: output for output in output_rows}
    if len(by_id) != len(output_rows):
        raise ValueError("predicted output case ids must be unique")
    known = {case.case_id for case in bank.cases}
    unknown = sorted(set(by_id) - known)
    if unknown:
        raise ValueError(f"predicted outputs reference unknown cases: {unknown}")
    scores = tuple(_score_case(case, by_id.get(case.case_id)) for case in bank.cases)
    return TranscriptGateScore(bank, scores, _metrics(scores))


def evaluate_capture_gate(score: TranscriptGateScore) -> dict[str, Any]:
    """Decide whether automatic capture may ship."""

    metrics = score.metrics
    failed: list[str] = []
    if metrics["forbidden_assistant_tool"]["rate"] > GATE_THRESHOLDS["forbidden_assistant_tool_rate"]:
        failed.append("forbidden_assistant_tool")
    if metrics["forbidden_assistant_tool"]["leaks"] > 0:
        if "forbidden_assistant_tool" not in failed:
            failed.append("forbidden_assistant_tool")
    if (
        metrics["speaker_attribution"]["misattribution_rate"]
        > GATE_THRESHOLDS["speaker_misattribution_rate"]
    ):
        failed.append("speaker_attribution")
    if metrics["typed_slot_completeness"]["precision"] < GATE_THRESHOLDS["typed_slot_precision_min"]:
        failed.append("typed_slot_precision")
    if (
        metrics["typed_slot_completeness"]["silent_demotion_rate"]
        > GATE_THRESHOLDS["silent_demotion_rate"]
    ):
        failed.append("silent_demotion")
    if (
        metrics["typed_slot_completeness"]["recall"]
        < GATE_THRESHOLDS["typed_slot_completeness_min"]
    ):
        failed.append("typed_slot_completeness")
    if (
        metrics["incidental_durable_recall"]["recall"]
        < GATE_THRESHOLDS["incidental_durable_recall_min"]
    ):
        failed.append("incidental_durable_recall")
    if (
        metrics["incidental_durable_recall"]["precision"]
        < GATE_THRESHOLDS["incidental_durable_precision_min"]
    ):
        failed.append("incidental_durable_precision")
    replay = next(
        (case for case in score.cases if case.case_id == PRODUCTION_FAILURE_CASE_ID),
        None,
    )
    if replay is None:
        failed.append("production_replay_missing")
    elif replay.assistant_authored_facts or replay.tool_banner_facts or replay.forbidden_leaks:
        failed.append("production_replay")
    return {
        "ship": not failed,
        "failed_checks": failed,
        "thresholds": dict(GATE_THRESHOLDS),
        "seed_is_not_the_gate": True,
    }


def write_score_report(path: str | Path, score: TranscriptGateScore) -> None:
    gate = evaluate_capture_gate(score)
    report = {
        "metadata": {
            "arena": "enfold-transcript-gate",
            "case_schema_version": CASE_SCHEMA_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "fixture": score.bank.source_path.name,
            "offline": True,
            "provider_calls": 0,
            "model_calls": 0,
            "network_calls": 0,
            "seed_is_not_the_gate": True,
        },
        "metrics": score.metrics,
        "gate": gate,
        "cases": [asdict(case) for case in score.cases],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score saved extraction outputs against the real-transcript "
            "capture gate without provider or network calls."
        ),
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--require-ship",
        action="store_true",
        help="exit 1 unless the published capture-ship thresholds all pass",
    )
    args = parser.parse_args(argv)
    try:
        bank = load_transcript_gate(args.cases)
        score = score_transcript_gate(bank, load_predicted_outputs(args.outputs))
    except ValueError as exc:
        parser.error(str(exc))
    gate = evaluate_capture_gate(score)
    payload = {
        "metrics": score.metrics,
        "gate": gate,
        "cases": len(score.cases),
        "case_pass_rate": (
            0.0
            if not score.cases
            else sum(case.passed for case in score.cases) / len(score.cases)
        ),
    }
    if args.report is not None:
        write_score_report(args.report, score)
    print(json.dumps(payload, sort_keys=True))
    if args.require_ship and not gate["ship"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
