"""Provider-neutral, offline scoring for saved extraction outputs.

The Arena deliberately has no provider protocol: callers run a model or an
extractor elsewhere, normalize its saved output to the v1 output schema, and
then invoke this module.  Scoring reads fixture/output files only.  It never
opens an Enfold database, imports an extraction runtime, or makes a network
request.

Evidence offsets are zero-based, half-open Python string indices.  An output
may instead provide an exact ``evidence_excerpt``; it resolves only when that
excerpt occurs exactly once in the transcript.  Ambiguous or paraphrased
evidence is intentionally ungrounded.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from enfold.extraction_spans import transcript_spans
from enfold.state_slots import (
    StateCandidate,
    normalize_predicate_key,
    normalize_subject_key,
)


CASE_SCHEMA_VERSION = "enfold-extraction-arena-case-v1"
OUTPUT_SCHEMA_VERSION = "enfold-extraction-arena-output-v1"
DECISIONS = ("add", "dedup", "supersede", "conflict", "reject", "abstain")
SENSITIVITIES = ("normal", "sensitive")
STATE_KINDS = ("state", "preference", "commitment", "event")

FIXTURES_DIR = Path(__file__).with_name("fixtures")
DEFAULT_CASES_PATH = FIXTURES_DIR / "extraction_arena_seed.jsonl"
DEFAULT_OUTPUTS_PATH = FIXTURES_DIR / "extraction_arena_seed_outputs.jsonl"
CASE_SCHEMA_PATH = FIXTURES_DIR / "extraction_arena.schema.json"
OUTPUT_SCHEMA_PATH = FIXTURES_DIR / "extraction_arena_output.schema.json"

_STATE_STRING_FIELDS = (
    "kind",
    "subject",
    "predicate",
    "object",
    "value",
    "occurred_at",
    "valid_from",
)
_STATE_ACTUAL_FIELDS = set(_STATE_STRING_FIELDS) | {"negation", "confidence"}
_STATE_EXPECTED_FIELDS = _STATE_ACTUAL_FIELDS | {"confidence_min", "confidence_max"}


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    start: int
    end: int
    text: str


@dataclass(frozen=True, slots=True)
class FactExpectation:
    key: str
    content_patterns: tuple[str, ...]
    evidence_spans: tuple[EvidenceSpan, ...]
    categories: tuple[str, ...]
    sensitivity: str
    state: Mapping[str, Any] | None

    def content_matches(self, content: str) -> bool:
        return any(re.search(pattern, content) is not None for pattern in self.content_patterns)


@dataclass(frozen=True, slots=True)
class ForbiddenFact:
    key: str
    content_patterns: tuple[str, ...]

    def matches(self, content: str) -> bool:
        return any(re.search(pattern, content) is not None for pattern in self.content_patterns)


@dataclass(frozen=True, slots=True)
class PriorFact:
    key: str
    content: str
    category: str
    sensitivity: str
    state: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class ExtractionCase:
    case_id: str
    transcript: str
    expected_decision: str
    required_facts: tuple[FactExpectation, ...]
    forbidden_facts: tuple[ForbiddenFact, ...]
    memory_before: tuple[PriorFact, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExtractionArena:
    cases: tuple[ExtractionCase, ...]
    source_path: Path


@dataclass(frozen=True, slots=True)
class CandidateFact:
    content: str
    category: str
    sensitivity: str
    evidence_span: tuple[int, int] | None
    evidence_excerpt: str | None
    state: Mapping[str, Any] | None
    expectation_key: str | None


@dataclass(frozen=True, slots=True)
class CandidateOutput:
    case_id: str
    decision: str
    facts: tuple[CandidateFact, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CaseScore:
    case_id: str
    expected_decision: str
    actual_decision: str | None
    decision_correct: bool
    required_total: int
    required_matched: int
    predicted_total: int
    unexpected_predictions: int
    forbidden_expected: int
    forbidden_leaks: tuple[str, ...]
    grounded_predictions: int
    evidence_compatible: int
    evidence_expected_exact: int
    evidence_runtime_compatible: int
    category_correct: int
    sensitivity_correct: int
    typed_expected: int
    typed_predicted: int
    typed_state_correct: int
    matched_expectations: tuple[str, ...]
    missing_expectations: tuple[str, ...]
    failures: tuple[str, ...]
    passed: bool


@dataclass(frozen=True, slots=True)
class ExtractionArenaScore:
    arena: ExtractionArena
    cases: tuple[CaseScore, ...]
    summary: Mapping[str, Any]

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


def _number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not minimum <= number <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")
    return number


def _validate_state(value: Any, label: str, *, expected: bool) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object or null")
    allowed = _STATE_EXPECTED_FIELDS if expected else _STATE_ACTUAL_FIELDS
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"{label} has unknown fields: {sorted(extra)}")
    if not value:
        raise ValueError(f"{label} must not be empty")
    normalized: dict[str, Any] = {}
    for field in _STATE_STRING_FIELDS:
        if field in value:
            normalized[field] = _text(value[field], f"{label}.{field}")
    if normalized.get("kind") not in STATE_KINDS and "kind" in normalized:
        raise ValueError(f"{label}.kind must be one of {list(STATE_KINDS)}")
    if "negation" in value:
        if not isinstance(value["negation"], bool):
            raise ValueError(f"{label}.negation must be a boolean")
        normalized["negation"] = value["negation"]
    if "confidence" in value:
        normalized["confidence"] = _number(
            value["confidence"], f"{label}.confidence", minimum=0.0, maximum=1.0
        )
    if expected:
        for field in ("confidence_min", "confidence_max"):
            if field in value:
                normalized[field] = _number(
                    value[field], f"{label}.{field}", minimum=0.0, maximum=1.0
                )
        if (
            "confidence_min" in normalized
            and "confidence_max" in normalized
            and normalized["confidence_min"] > normalized["confidence_max"]
        ):
            raise ValueError(f"{label} confidence range is inverted")
    return normalized


def _find_occurrences(text: str, excerpt: str) -> tuple[int, ...]:
    positions: list[int] = []
    offset = 0
    while True:
        position = text.find(excerpt, offset)
        if position < 0:
            return tuple(positions)
        positions.append(position)
        offset = position + 1


def _explicit_span(value: Any, label: str, transcript: str) -> EvidenceSpan:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _strict_keys(
        value,
        label,
        required={"start", "end"},
        optional={"text"},
    )
    start = value["start"]
    end = value["end"]
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
        or end > len(transcript)
    ):
        raise ValueError(f"{label} must be a valid non-empty half-open transcript span")
    resolved = transcript[start:end]
    if "text" in value and value["text"] != resolved:
        raise ValueError(f"{label}.text does not equal transcript[start:end]")
    return EvidenceSpan(start, end, resolved)


def _expected_evidence(value: Any, label: str, transcript: str) -> tuple[EvidenceSpan, ...]:
    rows = _objects(value, label)
    if not rows:
        raise ValueError(f"{label} must not be empty")
    spans: list[EvidenceSpan] = []
    for index, row in enumerate(rows):
        item_label = f"{label}[{index}]"
        if set(row) == {"text"}:
            excerpt = _text(row["text"], f"{item_label}.text")
            occurrences = _find_occurrences(transcript, excerpt)
            if len(occurrences) != 1:
                raise ValueError(
                    f"{item_label}.text must occur exactly once in the transcript; "
                    f"found {len(occurrences)}"
                )
            start = occurrences[0]
            spans.append(EvidenceSpan(start, start + len(excerpt), excerpt))
        else:
            spans.append(_explicit_span(row, item_label, transcript))
    coordinates = [(span.start, span.end) for span in spans]
    if len(coordinates) != len(set(coordinates)):
        raise ValueError(f"{label} must not contain duplicate spans")
    return tuple(spans)


def _patterns(value: Any, label: str) -> tuple[str, ...]:
    patterns = _string_list(value, label, allow_empty=False)
    for index, pattern in enumerate(patterns):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"{label}[{index}] is not a valid regular expression: {exc}") from exc
    return patterns


def _required_fact(value: Mapping[str, Any], label: str, transcript: str) -> FactExpectation:
    _strict_keys(
        value,
        label,
        required={
            "key",
            "content_patterns",
            "evidence",
            "categories",
            "sensitivity",
            "state",
        },
    )
    sensitivity = _text(value["sensitivity"], f"{label}.sensitivity")
    if sensitivity not in SENSITIVITIES:
        raise ValueError(f"{label}.sensitivity must be one of {list(SENSITIVITIES)}")
    return FactExpectation(
        key=_text(value["key"], f"{label}.key"),
        content_patterns=_patterns(value["content_patterns"], f"{label}.content_patterns"),
        evidence_spans=_expected_evidence(value["evidence"], f"{label}.evidence", transcript),
        categories=_string_list(value["categories"], f"{label}.categories", allow_empty=False),
        sensitivity=sensitivity,
        state=_validate_state(value["state"], f"{label}.state", expected=True),
    )


def _forbidden_fact(value: Mapping[str, Any], label: str) -> ForbiddenFact:
    _strict_keys(value, label, required={"key", "content_patterns"})
    return ForbiddenFact(
        _text(value["key"], f"{label}.key"),
        _patterns(value["content_patterns"], f"{label}.content_patterns"),
    )


def _prior_fact(value: Mapping[str, Any], label: str) -> PriorFact:
    _strict_keys(
        value,
        label,
        required={"key", "content", "category", "sensitivity", "state"},
    )
    sensitivity = _text(value["sensitivity"], f"{label}.sensitivity")
    if sensitivity not in SENSITIVITIES:
        raise ValueError(f"{label}.sensitivity must be one of {list(SENSITIVITIES)}")
    return PriorFact(
        _text(value["key"], f"{label}.key"),
        _text(value["content"], f"{label}.content"),
        _text(value["category"], f"{label}.category"),
        sensitivity,
        _validate_state(value["state"], f"{label}.state", expected=False),
    )


def _case(value: Mapping[str, Any], index: int) -> ExtractionCase:
    label = f"cases[{index}]"
    _strict_keys(
        value,
        label,
        required={
            "schema_version",
            "case_id",
            "transcript",
            "expected_decision",
            "required_facts",
            "forbidden_facts",
        },
        optional={"memory_before", "tags"},
    )
    if value["schema_version"] != CASE_SCHEMA_VERSION:
        raise ValueError(f"{label}.schema_version must be {CASE_SCHEMA_VERSION!r}")
    transcript = _text(value["transcript"], f"{label}.transcript")
    decision = _text(value["expected_decision"], f"{label}.expected_decision")
    if decision not in DECISIONS:
        raise ValueError(f"{label}.expected_decision must be one of {list(DECISIONS)}")
    required = tuple(
        _required_fact(row, f"{label}.required_facts[{fact_index}]", transcript)
        for fact_index, row in enumerate(_objects(value["required_facts"], f"{label}.required_facts"))
    )
    forbidden = tuple(
        _forbidden_fact(row, f"{label}.forbidden_facts[{fact_index}]")
        for fact_index, row in enumerate(_objects(value["forbidden_facts"], f"{label}.forbidden_facts"))
    )
    memory = tuple(
        _prior_fact(row, f"{label}.memory_before[{fact_index}]")
        for fact_index, row in enumerate(
            _objects(value.get("memory_before", []), f"{label}.memory_before")
        )
    )
    if decision in {"reject", "abstain"} and required:
        raise ValueError(f"{label} reject/abstain cases cannot require facts")
    if decision not in {"reject", "abstain"} and not required:
        raise ValueError(f"{label} {decision} cases must require at least one fact")
    keys = [fact.key for fact in required]
    forbidden_keys = [fact.key for fact in forbidden]
    memory_keys = [fact.key for fact in memory]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{label}.required_facts keys must be unique")
    if len(forbidden_keys) != len(set(forbidden_keys)):
        raise ValueError(f"{label}.forbidden_facts keys must be unique")
    if len(memory_keys) != len(set(memory_keys)):
        raise ValueError(f"{label}.memory_before keys must be unique")
    return ExtractionCase(
        _text(value["case_id"], f"{label}.case_id"),
        transcript,
        decision,
        required,
        forbidden,
        memory,
        _string_list(value.get("tags", []), f"{label}.tags"),
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


def load_extraction_arena(path: str | Path = DEFAULT_CASES_PATH) -> ExtractionArena:
    """Load strict case rows from JSONL, a JSON list, or ``{"cases": [...]}``."""

    rows, source = _load_rows(path, "cases")
    cases = tuple(_case(row, index) for index, row in enumerate(rows))
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("extraction Arena case ids must be unique")
    return ExtractionArena(cases, source)


def _candidate_fact(value: Mapping[str, Any], label: str) -> CandidateFact:
    _strict_keys(
        value,
        label,
        required={"content", "category", "sensitivity"},
        optional={
            "evidence_span",
            "evidence_excerpt",
            "state",
            "expectation_key",
            "tags",
        },
    )
    sensitivity = _text(value["sensitivity"], f"{label}.sensitivity")
    if sensitivity not in SENSITIVITIES:
        raise ValueError(f"{label}.sensitivity must be one of {list(SENSITIVITIES)}")
    raw_span = value.get("evidence_span")
    span: tuple[int, int] | None = None
    if raw_span is not None:
        if not isinstance(raw_span, dict) or set(raw_span) != {"start", "end"}:
            raise ValueError(f"{label}.evidence_span must contain exactly start and end")
        start = raw_span["start"]
        end = raw_span["end"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise ValueError(f"{label}.evidence_span start/end must be integers")
        span = (start, end)
    excerpt = value.get("evidence_excerpt")
    if excerpt is not None:
        excerpt = _text(excerpt, f"{label}.evidence_excerpt")
    expectation_key = value.get("expectation_key")
    if expectation_key is not None:
        expectation_key = _text(expectation_key, f"{label}.expectation_key")
    tags = value.get("tags")
    if tags is not None and not isinstance(tags, (str, list)):
        raise ValueError(f"{label}.tags must be a string or list")
    if isinstance(tags, list):
        _string_list(tags, f"{label}.tags")
    return CandidateFact(
        _text(value["content"], f"{label}.content"),
        _text(value["category"], f"{label}.category"),
        sensitivity,
        span,
        excerpt,
        _validate_state(value.get("state"), f"{label}.state", expected=False),
        expectation_key,
    )


def _candidate_output(value: Mapping[str, Any], index: int) -> CandidateOutput:
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
        _candidate_fact(row, f"{label}.facts[{fact_index}]")
        for fact_index, row in enumerate(_objects(value["facts"], f"{label}.facts"))
    )
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError(f"{label}.metadata must be an object")
    return CandidateOutput(
        _text(value["case_id"], f"{label}.case_id"),
        decision,
        facts,
        metadata,
    )


def load_candidate_outputs(path: str | Path) -> tuple[CandidateOutput, ...]:
    """Load normalized saved outputs without invoking any provider or model."""

    rows, _source = _load_rows(path, "outputs")
    outputs = tuple(_candidate_output(row, index) for index, row in enumerate(rows))
    ids = [output.case_id for output in outputs]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate output case ids must be unique")
    return outputs


def resolve_candidate_evidence(
    transcript: str,
    fact: CandidateFact,
) -> tuple[EvidenceSpan | None, str | None]:
    """Resolve a candidate's explicit span or unique exact excerpt."""

    if fact.evidence_span is not None:
        start, end = fact.evidence_span
        if start < 0 or end <= start or end > len(transcript):
            return None, "invalid_span"
        resolved = transcript[start:end]
        if fact.evidence_excerpt is not None and fact.evidence_excerpt != resolved:
            return None, "span_excerpt_mismatch"
        return EvidenceSpan(start, end, resolved), None
    if fact.evidence_excerpt is None:
        return None, "missing_evidence"
    occurrences = _find_occurrences(transcript, fact.evidence_excerpt)
    if not occurrences:
        return None, "excerpt_not_found"
    if len(occurrences) != 1:
        return None, "ambiguous_excerpt"
    start = occurrences[0]
    return EvidenceSpan(start, start + len(fact.evidence_excerpt), fact.evidence_excerpt), None


def _normalized(value: str) -> str:
    return " ".join(value.split()).casefold()


def _runtime_state(
    value: Mapping[str, Any] | None,
    content: str,
) -> Mapping[str, Any] | None:
    """Mirror the processor's fail-soft typed-state normalization."""

    if value is None or not value or set(value) - _STATE_ACTUAL_FIELDS:
        return None
    kind = value.get("kind")
    confidence = value.get("confidence")
    negation = value.get("negation", False)
    if (
        kind not in STATE_KINDS
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.8 <= float(confidence) <= 1.0
        or not isinstance(negation, bool)
        or ("object" in value and "value" in value)
        or ("occurred_at" in value and "valid_from" in value)
    ):
        return None
    object_value = value.get("object", value.get("value"))
    if negation:
        if object_value is not None:
            return None
        object_value = None
    elif not isinstance(object_value, str) or not object_value.strip():
        return None
    else:
        object_value = object_value.strip()
    valid_from = value.get("valid_from", value.get("occurred_at"))
    if valid_from is not None and (
        not isinstance(valid_from, str) or not valid_from.strip()
    ):
        return None
    try:
        subject_key = normalize_subject_key(value.get("subject"))
        predicate_key = normalize_predicate_key(value.get("predicate"))
        StateCandidate(
            content=content,
            subject_key=subject_key,
            predicate_key=predicate_key,
            object_value=object_value,
            valid_from=valid_from,
        )
    except (TypeError, ValueError):
        return None
    return {
        "kind": kind,
        "subject": subject_key,
        "predicate": predicate_key,
        "object_value": object_value,
        "valid_from": valid_from.strip() if valid_from is not None else None,
        "negation": negation,
        "confidence": float(confidence),
    }


def _expected_runtime_state(
    expected: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if expected is None:
        return None
    confidence = expected.get("confidence")
    if confidence is None:
        confidence = max(0.8, float(expected.get("confidence_min", 0.8)))
    candidate = {
        field: value
        for field, value in expected.items()
        if field not in {"confidence_min", "confidence_max"}
    }
    candidate["confidence"] = confidence
    normalized = _runtime_state(candidate, "expected typed extraction")
    if normalized is None:
        return None
    if "confidence_min" in expected and confidence < expected["confidence_min"]:
        return None
    if "confidence_max" in expected and confidence > expected["confidence_max"]:
        return None
    return normalized


def _state_matches(
    expected: Mapping[str, Any] | None,
    actual: Mapping[str, Any] | None,
    content: str,
) -> bool:
    if expected is None:
        return _runtime_state(actual, content) is None
    normalized_expected = _expected_runtime_state(expected)
    normalized_actual = _runtime_state(actual, content)
    if normalized_expected is None or normalized_actual is None:
        return False
    for field, expected_value in normalized_expected.items():
        if field == "confidence":
            continue
        actual_value = normalized_actual.get(field)
        if isinstance(expected_value, str):
            if not isinstance(actual_value, str) or _normalized(actual_value) != _normalized(expected_value):
                return False
        elif actual_value != expected_value:
            return False
    confidence = normalized_actual["confidence"]
    if "confidence_min" in expected:
        if not isinstance(confidence, (int, float)) or confidence < expected["confidence_min"]:
            return False
    if "confidence_max" in expected:
        if not isinstance(confidence, (int, float)) or confidence > expected["confidence_max"]:
            return False
    return True


def _pair_checks(
    case: ExtractionCase,
    expected: FactExpectation,
    actual: CandidateFact,
) -> tuple[bool, bool, bool, bool, bool]:
    grounded, evidence, _expected_exact, _runtime_compatible = (
        _evidence_compatibility(case, expected, actual)
    )
    return (
        grounded,
        evidence,
        actual.category in expected.categories,
        actual.sensitivity == expected.sensitivity,
        _state_matches(expected.state, actual.state, actual.content),
    )


def _evidence_compatibility(
    case: ExtractionCase,
    expected: FactExpectation,
    actual: CandidateFact,
) -> tuple[bool, bool, bool, bool]:
    span, _reason = resolve_candidate_evidence(case.transcript, actual)
    grounded = span is not None
    expected_exact = False
    runtime_compatible = False
    if span is not None:
        actual_coordinates = (span.start, span.end)
        expected_coordinates = {
            (allowed.start, allowed.end) for allowed in expected.evidence_spans
        }
        runtime_coordinates = {
            (runtime.start, runtime.end)
            for runtime in transcript_spans(case.transcript)
            if any(
                runtime.start <= allowed.start and allowed.end <= runtime.end
                for allowed in expected.evidence_spans
            )
        }
        # Provider-neutral saved outputs may identify the smallest expected
        # supporting excerpt directly. Bundled v2 adapters instead return the
        # exact deterministic runtime span selected by ID. Both are valid only
        # when their coordinates are contract-derived; arbitrary containers do
        # not pass merely because they contain the expected words.
        expected_exact = actual_coordinates in expected_coordinates
        runtime_compatible = actual_coordinates in runtime_coordinates
    return (
        grounded,
        expected_exact or runtime_compatible,
        expected_exact,
        runtime_compatible,
    )


def _maximum_fact_matching(
    case: ExtractionCase,
    facts: Sequence[CandidateFact],
) -> dict[int, int]:
    neighbors: dict[int, list[int]] = {}
    for expected_index, expected in enumerate(case.required_facts):
        eligible = []
        for actual_index, actual in enumerate(facts):
            if actual.expectation_key is not None and actual.expectation_key != expected.key:
                continue
            if expected.content_matches(actual.content):
                quality = sum(_pair_checks(case, expected, actual))
                eligible.append((quality, actual_index))
        neighbors[expected_index] = [
            actual_index
            for _quality, actual_index in sorted(eligible, key=lambda item: (-item[0], item[1]))
        ]

    actual_to_expected: dict[int, int] = {}

    def assign(expected_index: int, seen: set[int]) -> bool:
        for actual_index in neighbors[expected_index]:
            if actual_index in seen:
                continue
            seen.add(actual_index)
            previous = actual_to_expected.get(actual_index)
            if previous is None or assign(previous, seen):
                actual_to_expected[actual_index] = expected_index
                return True
        return False

    for expected_index in sorted(neighbors, key=lambda item: (len(neighbors[item]), item)):
        assign(expected_index, set())
    return {expected: actual for actual, expected in actual_to_expected.items()}


def _score_case(case: ExtractionCase, output: CandidateOutput | None) -> CaseScore:
    if output is None:
        return CaseScore(
            case_id=case.case_id,
            expected_decision=case.expected_decision,
            actual_decision=None,
            decision_correct=False,
            required_total=len(case.required_facts),
            required_matched=0,
            predicted_total=0,
            unexpected_predictions=0,
            forbidden_expected=len(case.forbidden_facts),
            forbidden_leaks=(),
            grounded_predictions=0,
            evidence_compatible=0,
            evidence_expected_exact=0,
            evidence_runtime_compatible=0,
            category_correct=0,
            sensitivity_correct=0,
            typed_expected=sum(
                fact.state is not None for fact in case.required_facts
            ),
            typed_predicted=0,
            typed_state_correct=0,
            matched_expectations=(),
            missing_expectations=tuple(
                fact.key for fact in case.required_facts
            ),
            failures=("missing saved output",),
            passed=False,
        )

    matching = _maximum_fact_matching(case, output.facts)
    matched_actual = set(matching.values())
    missing = tuple(
        expected.key
        for index, expected in enumerate(case.required_facts)
        if index not in matching
    )
    matched_keys: list[str] = []
    evidence_compatible = 0
    evidence_expected_exact = 0
    evidence_runtime_compatible = 0
    category_correct = 0
    sensitivity_correct = 0
    typed_expected = sum(fact.state is not None for fact in case.required_facts)
    typed_predicted = sum(
        _runtime_state(fact.state, fact.content) is not None for fact in output.facts
    )
    typed_correct = 0
    failures: list[str] = []
    for expected_index, actual_index in sorted(matching.items()):
        expected = case.required_facts[expected_index]
        actual = output.facts[actual_index]
        matched_keys.append(expected.key)
        _grounded, evidence, category, sensitivity, state = _pair_checks(case, expected, actual)
        (
            _grounded,
            _evidence,
            expected_exact,
            runtime_compatible,
        ) = _evidence_compatibility(case, expected, actual)
        evidence_compatible += int(evidence)
        evidence_expected_exact += int(expected_exact)
        evidence_runtime_compatible += int(runtime_compatible)
        category_correct += int(category)
        sensitivity_correct += int(sensitivity)
        if expected.state is not None:
            typed_correct += int(state)
        if not evidence:
            failures.append(f"evidence span mismatch: {expected.key}")
        if not category:
            failures.append(f"category mismatch: {expected.key}")
        if not sensitivity:
            failures.append(f"sensitivity mismatch: {expected.key}")
        if not state:
            failures.append(f"typed state mismatch: {expected.key}")

    grounded = sum(
        resolve_candidate_evidence(case.transcript, fact)[0] is not None
        for fact in output.facts
    )
    leaked = tuple(
        forbidden.key
        for forbidden in case.forbidden_facts
        if any(forbidden.matches(fact.content) for fact in output.facts)
    )
    decision_correct = output.decision == case.expected_decision
    unexpected = len(output.facts) - len(matched_actual)
    if not decision_correct:
        failures.append(
            f"decision mismatch: expected {case.expected_decision}, got {output.decision}"
        )
    if missing:
        failures.append(f"missing required facts: {', '.join(missing)}")
    if unexpected:
        failures.append(f"unexpected facts: {unexpected}")
    if leaked:
        failures.append(f"forbidden facts leaked: {', '.join(leaked)}")
    if grounded != len(output.facts):
        failures.append(f"ungrounded facts: {len(output.facts) - grounded}")
    if case.expected_decision in {"reject", "abstain"} and output.facts:
        failures.append(f"{case.expected_decision} must return no facts")
    return CaseScore(
        case_id=case.case_id,
        expected_decision=case.expected_decision,
        actual_decision=output.decision,
        decision_correct=decision_correct,
        required_total=len(case.required_facts),
        required_matched=len(matching),
        predicted_total=len(output.facts),
        unexpected_predictions=unexpected,
        forbidden_expected=len(case.forbidden_facts),
        forbidden_leaks=leaked,
        grounded_predictions=grounded,
        evidence_compatible=evidence_compatible,
        evidence_expected_exact=evidence_expected_exact,
        evidence_runtime_compatible=evidence_runtime_compatible,
        category_correct=category_correct,
        sensitivity_correct=sensitivity_correct,
        typed_expected=typed_expected,
        typed_predicted=typed_predicted,
        typed_state_correct=typed_correct,
        matched_expectations=tuple(matched_keys),
        missing_expectations=missing,
        failures=tuple(failures),
        passed=not failures,
    )


def _ratio(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return empty if denominator == 0 else numerator / denominator


def _summary(scores: Sequence[CaseScore]) -> Mapping[str, Any]:
    required = sum(score.required_total for score in scores)
    matched = sum(score.required_matched for score in scores)
    predicted = sum(score.predicted_total for score in scores)
    precision = _ratio(matched, predicted, empty=1.0 if required == 0 else 0.0)
    recall = _ratio(matched, required, empty=1.0 if predicted == 0 else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    by_decision: dict[str, dict[str, Any]] = {}
    expected_counts = Counter(score.expected_decision for score in scores)
    correct_counts = Counter(
        score.expected_decision for score in scores if score.decision_correct
    )
    for decision in DECISIONS:
        expected = expected_counts[decision]
        correct = correct_counts[decision]
        by_decision[decision] = {
            "cases": expected,
            "correct": correct,
            "accuracy": _ratio(correct, expected, empty=1.0),
        }
    protected = sum(score.forbidden_expected > 0 for score in scores)
    leak_cases = sum(bool(score.forbidden_leaks) for score in scores)
    typed_expected = sum(score.typed_expected for score in scores)
    typed_predicted = sum(score.typed_predicted for score in scores)
    typed_correct = sum(score.typed_state_correct for score in scores)
    typed_precision = _ratio(
        typed_correct,
        typed_predicted,
        empty=1.0 if typed_expected == 0 else 0.0,
    )
    typed_recall = _ratio(
        typed_correct,
        typed_expected,
        empty=1.0 if typed_predicted == 0 else 0.0,
    )
    typed_f1 = (
        0.0
        if typed_precision + typed_recall == 0
        else 2 * typed_precision * typed_recall / (typed_precision + typed_recall)
    )
    return {
        "cases": len(scores),
        "passed": sum(score.passed for score in scores),
        "case_pass_rate": _ratio(sum(score.passed for score in scores), len(scores)),
        "decision_accuracy": _ratio(sum(score.decision_correct for score in scores), len(scores)),
        "by_decision": by_decision,
        "facts": {
            "required": required,
            "matched": matched,
            "predicted": predicted,
            "unexpected": sum(score.unexpected_predictions for score in scores),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        },
        "evidence": {
            "grounded_predictions": sum(score.grounded_predictions for score in scores),
            "grounding_accuracy": _ratio(
                sum(score.grounded_predictions for score in scores), predicted, empty=1.0
            ),
            "contract_compatible_spans": sum(
                score.evidence_compatible for score in scores
            ),
            "contract_compatible_accuracy": _ratio(
                sum(score.evidence_compatible for score in scores),
                required,
                empty=1.0,
            ),
            "exact_expected_spans": sum(
                score.evidence_expected_exact for score in scores
            ),
            "exact_expected_accuracy": _ratio(
                sum(score.evidence_expected_exact for score in scores),
                required,
                empty=1.0,
            ),
            "runtime_compatible_spans": sum(
                score.evidence_runtime_compatible for score in scores
            ),
            "runtime_compatible_accuracy": _ratio(
                sum(score.evidence_runtime_compatible for score in scores),
                required,
                empty=1.0,
            ),
        },
        "category_accuracy": _ratio(
            sum(score.category_correct for score in scores), required, empty=1.0
        ),
        "sensitivity_accuracy": _ratio(
            sum(score.sensitivity_correct for score in scores), required, empty=1.0
        ),
        # Compatibility gate: F1 ensures both missed typed proposals and
        # runtime-valid over-typing lower the top-level score.
        "typed_state_accuracy": typed_f1,
        "typed_state": {
            "expected": typed_expected,
            "predicted": typed_predicted,
            "correct": typed_correct,
            "false_positives": typed_predicted - typed_correct,
            "precision": typed_precision,
            "recall": typed_recall,
            "f1": typed_f1,
        },
        "forbidden": {
            "leaks": sum(len(score.forbidden_leaks) for score in scores),
            "leak_cases": leak_cases,
            "protected_cases": protected,
            "case_leak_rate": _ratio(leak_cases, protected),
        },
    }


def score_extraction_arena(
    arena: ExtractionArena,
    outputs: Iterable[CandidateOutput],
) -> ExtractionArenaScore:
    """Score normalized saved outputs against all Arena expectations."""

    output_rows = tuple(outputs)
    by_id = {output.case_id: output for output in output_rows}
    if len(by_id) != len(output_rows):
        raise ValueError("candidate output case ids must be unique")
    known = {case.case_id for case in arena.cases}
    unknown = sorted(set(by_id) - known)
    if unknown:
        raise ValueError(f"candidate outputs reference unknown cases: {unknown}")
    scores = tuple(_score_case(case, by_id.get(case.case_id)) for case in arena.cases)
    return ExtractionArenaScore(arena, scores, _summary(scores))


def score_saved_outputs(
    cases_path: str | Path = DEFAULT_CASES_PATH,
    outputs_path: str | Path = DEFAULT_OUTPUTS_PATH,
) -> ExtractionArenaScore:
    """Convenience entry point that performs file parsing and scoring only."""

    arena = load_extraction_arena(cases_path)
    outputs = load_candidate_outputs(outputs_path)
    return score_extraction_arena(arena, outputs)


def write_score_report(path: str | Path, score: ExtractionArenaScore) -> None:
    report = {
        "metadata": {
            "arena": "enfold-extraction-arena",
            "case_schema_version": CASE_SCHEMA_VERSION,
            "output_schema_version": OUTPUT_SCHEMA_VERSION,
            "fixture": score.arena.source_path.name,
            "offline": True,
            "provider_calls": 0,
            "model_calls": 0,
            "network_calls": 0,
        },
        "summary": score.summary,
        "cases": [asdict(case) for case in score.cases],
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score saved Enfold extraction outputs without provider or network calls.",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--require-perfect",
        action="store_true",
        help="exit 1 unless every case passes all exact checks",
    )
    args = parser.parse_args(argv)
    try:
        score = score_saved_outputs(args.cases, args.outputs)
    except ValueError as exc:
        parser.error(str(exc))
    if args.report is not None:
        write_score_report(args.report, score)
    print(json.dumps(score.summary, sort_keys=True))
    return 1 if args.require_perfect and not score.passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
