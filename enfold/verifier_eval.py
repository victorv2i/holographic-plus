"""Offline-first evaluation of evidence-verifier configurations.

The labeled fixture is scored without a model for the deterministic prefilter.
Live local models are optional: when Ollama is absent they are skipped, not
failed, so CI stays green.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from .extraction_processor import ExtractedMemory, ExtractionEnvelope
from .protocol import ClientContext

_CASES_PATH = (
    Path(__file__).resolve().parents[1]
    / "memory_eval"
    / "fixtures"
    / "verifier_cases.jsonl"
)


def load_verifier_cases(path: Path | None = None) -> list[dict[str, str]]:
    target = path or _CASES_PATH
    cases: list[dict[str, str]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        cases.append(json.loads(line))
    return cases


def _envelope(transcript: str) -> ExtractionEnvelope:
    return ExtractionEnvelope(
        transcript=transcript,
        source="verifier_eval",
        scope="private",
        context=ClientContext(
            client_id="hermes-install",
            surface="hermes",
            agent_id="verifier-eval",
            session_id="verifier-eval",
            access_scopes=("private",),
        ),
    )


def _metrics(
    *,
    name: str,
    labels: Sequence[str],
    predictions: Sequence[str],
    elapsed_seconds: float,
    categories: Sequence[str] | None = None,
) -> dict[str, Any]:
    n = len(labels)
    true_supported = sum(1 for label in labels if label == "supported")
    verified = 0
    false_verify = 0
    true_verify = 0
    by_category: dict[str, dict[str, int]] = {}
    rows = zip(labels, predictions, strict=True)
    if categories is not None:
        rows = zip(labels, predictions, categories, strict=True)
    for item in rows:
        if categories is None:
            label, prediction = item
            category = None
        else:
            label, prediction, category = item
        if category is not None:
            bucket = by_category.setdefault(
                category, {"n": 0, "verified": 0, "false_verify": 0}
            )
            bucket["n"] += 1
        if prediction != "verified":
            continue
        verified += 1
        if category is not None:
            by_category[category]["verified"] += 1
        if label == "supported":
            true_verify += 1
        else:
            false_verify += 1
            if category is not None:
                by_category[category]["false_verify"] += 1
    unsupported = n - true_supported
    precision = (true_verify / verified) if verified else None
    recall = (true_verify / true_supported) if true_supported else None
    false_verify_rate = (false_verify / unsupported) if unsupported else 0.0
    report = {
        "name": name,
        "skipped": False,
        "n": n,
        "verified": verified,
        "false_verify": false_verify,
        "false_verify_rate": false_verify_rate,
        "precision": precision,
        "recall": recall,
        "latency_ms_per_call": (elapsed_seconds / n) * 1000.0 if n else 0.0,
    }
    if categories is not None:
        report["by_category"] = by_category
    return report


def _skipped(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name,
        "skipped": True,
        "n": None,
        "verified": None,
        "false_verify": None,
        "false_verify_rate": None,
        "precision": None,
        "recall": None,
        "latency_ms_per_call": None,
        "skip_reason": reason,
    }


def score_prefilter(cases: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    from .evidence_verifier import LocalOllamaEvidenceVerifier

    verifier = LocalOllamaEvidenceVerifier(
        client=_UnreachableClient(),
        prefilter=True,
    )
    labels: list[str] = []
    predictions: list[str] = []
    started = time.perf_counter()
    for case in cases:
        labels.append(case["label"])
        claim = case["claim"]
        excerpt = case["excerpt"]
        if verifier._must_review(claim, excerpt, _envelope(excerpt)):
            predictions.append("needs_review")
            continue
        predictions.append("needs_review")
    return _metrics(
        name="prefilter",
        labels=labels,
        predictions=predictions,
        elapsed_seconds=time.perf_counter() - started,
        categories=[case["category"] for case in cases],
    )


def evaluate_configurations(
    cases: Sequence[Mapping[str, str]],
    *,
    models: Iterable[str],
    probe: Callable[[str], bool] | None = None,
    client_factory: Callable[[str], Any] | None = None,
    timeout_seconds: float | None = None,
) -> list[dict[str, Any]]:
    rows = [score_prefilter(cases)]
    check = probe if probe is not None else ollama_model_reachable
    for model in models:
        if not check(model):
            rows.append(_skipped(model, "ollama model is not reachable"))
            continue
        rows.append(
            score_model(
                cases,
                model=model,
                client_factory=client_factory,
                timeout_seconds=timeout_seconds,
            )
        )
    return rows


def score_model(
    cases: Sequence[Mapping[str, str]],
    *,
    model: str,
    client_factory: Callable[[str], Any] | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    from .evidence_verifier import DEFAULT_TIMEOUT_SECONDS, LocalOllamaEvidenceVerifier

    client = client_factory(model) if client_factory is not None else None
    kwargs: dict[str, Any] = {"model": model, "client": client}
    if timeout_seconds is not None:
        kwargs["timeout_seconds"] = timeout_seconds
    else:
        kwargs["timeout_seconds"] = DEFAULT_TIMEOUT_SECONDS
    verifier = LocalOllamaEvidenceVerifier(**kwargs)
    labels: list[str] = []
    predictions: list[str] = []
    started = time.perf_counter()
    for case in cases:
        labels.append(case["label"])
        result = verifier.verify(
            ExtractedMemory(case["claim"]),
            evidence_excerpt=case["excerpt"],
            envelope=_envelope(case["excerpt"]),
        )
        predictions.append(result.status)
    return _metrics(
        name=model,
        labels=labels,
        predictions=predictions,
        elapsed_seconds=time.perf_counter() - started,
        categories=[case["category"] for case in cases],
    )


def ollama_model_reachable(model: str) -> bool:
    from .evidence_verifier import probe_verifier_model

    try:
        probe_verifier_model(model)
    except Exception:
        return False
    return True


class _UnreachableClient:
    def complete(self, messages, *, timeout_seconds):
        raise RuntimeError("prefilter-only evaluation must not call a model")
