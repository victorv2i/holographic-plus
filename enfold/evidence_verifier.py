"""Independent local evidence verifier for automatic extraction.

The extractor proposes claims. This module decides whether an excerpt supports
the whole claim. Lexical containment is never treated as entailment. The cheap
pre-filter may only reject. Any error, timeout, or unparseable model output
fails closed to ``needs_review``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
import threading
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib import error, parse, request

from .extraction_contract import (
    VERIFIER_OUTPUT_SCHEMA,
    VERIFIER_SYSTEM_PROMPT,
    parse_verification_verdict,
)
from .extraction_processor import EvidenceVerification, ExtractedMemory, ExtractionEnvelope
from .extraction_spans import MAX_EVIDENCE_CHARS
from .prompt_safety import instruction_shaped, normalized_prompt_text

logger = logging.getLogger(__name__)

DEFAULT_VERIFIER_MODEL = "qwen3.8:27b"
RECOMMENDED_VERIFIER_MODEL = DEFAULT_VERIFIER_MODEL
LOCAL_VERIFIER_IMPORT = "enfold.evidence_verifier:LocalOllamaEvidenceVerifier"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RESPONSE_BYTES = 4096
TAGS_MAX_RESPONSE_BYTES = 262144
EXTRACTOR_MODEL_PREFIX = "qwen3:30b"
_IDENTITY_VERSION = "v1"
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "with",
        "now",
    }
)


class EvidenceModelClient(Protocol):
    """Narrow chat client used so unit tests never touch a live model."""

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        timeout_seconds: float,
    ) -> str:
        """Return model text or raise ``TimeoutError`` / ``OSError``."""


def is_extractor_model(model: str) -> bool:
    if not isinstance(model, str) or not model.strip():
        return False
    return model.strip().lower().startswith(EXTRACTOR_MODEL_PREFIX)


def validate_verifier_model(model: object) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("evidence verifier model must be a non-empty string")
    normalized = model.strip()
    if is_extractor_model(normalized):
        raise ValueError("evidence verifier model must not be the extractor model")
    return normalized


def validate_verifier_endpoint(endpoint: object) -> str:
    if not isinstance(endpoint, str):
        raise ValueError("evidence verifier endpoint must be a loopback /api/chat URL")
    parsed = parse.urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/api/chat"
        or parsed.hostname is None
    ):
        raise ValueError("evidence verifier endpoint must be a loopback /api/chat URL")
    try:
        if not ipaddress.ip_address(parsed.hostname).is_loopback:
            raise ValueError("evidence verifier endpoint must be loopback")
    except ValueError as exc:
        raise ValueError("evidence verifier endpoint must be loopback") from exc
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("evidence verifier endpoint port is invalid") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("evidence verifier endpoint port is invalid")
    return endpoint


def validate_verifier_timeout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError("evidence verifier timeout must be a positive finite number")
    return float(value)


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


@dataclass(frozen=True, slots=True)
class OllamaChatClient:
    """Loopback Ollama client with a hard timeout and a bounded response."""

    model: str
    endpoint: str = DEFAULT_ENDPOINT
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        timeout_seconds: float,
    ) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": VERIFIER_OUTPUT_SCHEMA,
                "messages": list(messages),
                "options": {"temperature": 0},
                "think": False,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        http = request.build_opener(request.ProxyHandler({}), _NoRedirect())
        req = request.Request(
            self.endpoint,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with http.open(req, timeout=float(timeout_seconds)) as response:
                if getattr(response, "status", 200) != 200:
                    raise OSError("evidence verifier endpoint returned a non-200 status")
                body = _read_bounded(response, self.max_response_bytes)
        except (
            error.HTTPError,
            error.URLError,
            TimeoutError,
            socket.timeout,
            OSError,
        ) as exc:
            raise TimeoutError("evidence verifier model call failed") from exc
        try:
            document = json.loads(body.decode("utf-8"))
            content = document["message"]["content"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError("evidence verifier model output is unparseable") from exc
        if not isinstance(content, str):
            raise ValueError("evidence verifier model output is unparseable")
        return content


def _read_bounded(response: Any, limit: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                raise ValueError("evidence verifier response is too large")
        except ValueError:
            raise
    body = response.read(limit + 1)
    if len(body) > limit:
        raise ValueError("evidence verifier response is too large")
    return body


def _content_tokens(text: str) -> frozenset[str]:
    folded = normalized_prompt_text(text).lower()
    tokens: list[str] = []
    current: list[str] = []
    for character in folded:
        if character.isalnum():
            current.append(character)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return frozenset(token for token in tokens if token not in _STOPWORDS)


class LocalOllamaEvidenceVerifier:
    """Local-first NLI verifier. Disabled until an operator installs it."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_VERIFIER_MODEL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        endpoint: str = DEFAULT_ENDPOINT,
        prefilter: bool = True,
        client: EvidenceModelClient | None = None,
    ) -> None:
        self._model = validate_verifier_model(model)
        self._timeout = validate_verifier_timeout(timeout_seconds)
        self._endpoint = validate_verifier_endpoint(endpoint)
        if not isinstance(prefilter, bool):
            raise ValueError("evidence verifier prefilter must be a boolean")
        self._prefilter = prefilter
        self._client = client or OllamaChatClient(
            model=self._model, endpoint=self._endpoint
        )
        self.identity = f"enfold-local-nli:{self._model}:{_IDENTITY_VERSION}"

    def verify(
        self,
        proposal: ExtractedMemory,
        *,
        evidence_excerpt: str,
        envelope: ExtractionEnvelope,
    ) -> EvidenceVerification:
        try:
            return self._verify(proposal, evidence_excerpt, envelope)
        except Exception as exc:
            logger.warning("evidence verifier failed closed: %s", type(exc).__name__)
            return EvidenceVerification("needs_review", self.identity)

    def _verify(
        self,
        proposal: ExtractedMemory,
        evidence_excerpt: str,
        envelope: ExtractionEnvelope,
    ) -> EvidenceVerification:
        claim = proposal.content if isinstance(getattr(proposal, "content", None), str) else ""
        if self._must_review(claim, evidence_excerpt, envelope):
            return EvidenceVerification("needs_review", self.identity)
        raw = self._complete(
            (
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"claim": claim, "excerpt": evidence_excerpt},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ),
                },
            )
        )
        if parse_verification_verdict(raw) == "supported":
            return EvidenceVerification("verified", self.identity)
        return EvidenceVerification("needs_review", self.identity)

    def _must_review(
        self,
        claim: str,
        evidence_excerpt: str,
        envelope: ExtractionEnvelope,
    ) -> bool:
        if not isinstance(evidence_excerpt, str) or not evidence_excerpt.strip():
            return True
        if len(evidence_excerpt) > MAX_EVIDENCE_CHARS:
            return True
        if not claim.strip() or not _content_tokens(claim):
            return True
        if instruction_shaped(evidence_excerpt) or instruction_shaped(claim):
            return True
        transcript = getattr(envelope, "transcript", "")
        if isinstance(transcript, str) and evidence_excerpt not in transcript:
            return True
        if self._prefilter and not (_content_tokens(claim) & _content_tokens(evidence_excerpt)):
            return True
        return False

    def _complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        box: dict[str, Any] = {}

        def invoke() -> None:
            try:
                box["value"] = self._client.complete(
                    messages, timeout_seconds=self._timeout
                )
            except BaseException as exc:
                box["error"] = exc

        thread = threading.Thread(
            target=invoke, name="enfold-evidence-verify", daemon=True
        )
        thread.start()
        thread.join(self._timeout)
        if thread.is_alive():
            raise TimeoutError("evidence verifier timed out")
        error = box.get("error")
        if error is not None:
            raise error
        value = box.get("value")
        if not isinstance(value, str):
            raise TypeError("evidence verifier produced no text")
        return value


class VerifierEnableError(RuntimeError):
    """Enabling the local evidence verifier failed before configuration changed."""


def _tags_url(chat_endpoint: str) -> str:
    parsed = parse.urlsplit(chat_endpoint)
    return parse.urlunsplit((parsed.scheme, parsed.netloc, "/api/tags", "", ""))


def _model_aliases(model: str) -> frozenset[str]:
    aliases = {model}
    final_segment = model.rsplit("/", 1)[-1]
    if ":" not in final_segment:
        aliases.add(f"{model}:latest")
    return frozenset(aliases)


def _listed_model_names(payload: object) -> set[str]:
    models = payload.get("models") if isinstance(payload, Mapping) else None
    names: set[str] = set()
    if not isinstance(models, list):
        return names
    for record in models:
        if not isinstance(record, Mapping):
            continue
        for key in ("name", "model"):
            value = record.get(key)
            if isinstance(value, str) and value.strip():
                names.add(value.strip())
    return names


def probe_verifier_model(
    model: str,
    *,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout_seconds: float = 5.0,
    opener: Callable[..., Any] | None = None,
) -> str:
    """Return ``model`` when the loopback registry lists it, else fail closed."""

    try:
        normalized = validate_verifier_model(model)
        chat_endpoint = validate_verifier_endpoint(endpoint)
        timeout = validate_verifier_timeout(timeout_seconds)
    except ValueError as exc:
        raise VerifierEnableError(str(exc)) from exc
    tags_url = _tags_url(chat_endpoint)
    http = opener or request.build_opener(request.ProxyHandler({}), _NoRedirect()).open
    req = request.Request(tags_url, method="GET")
    try:
            with http(req, timeout=timeout) as response:
                body = response.read(TAGS_MAX_RESPONSE_BYTES + 1)
                if len(body) > TAGS_MAX_RESPONSE_BYTES:
                    raise ValueError("evidence verifier registry response is too large")
            payload = json.loads(body.decode("utf-8"))
    except Exception as exc:
        raise VerifierEnableError("evidence verifier model is not reachable") from exc
    names = _listed_model_names(payload)
    if not names.intersection(_model_aliases(normalized)):
        raise VerifierEnableError("evidence verifier model is not reachable")
    return normalized


def enable_verifier(
    config_path: str | Path,
    *,
    model: str | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    probe: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """Write an opt-in local verifier after the chosen model answers locally."""

    try:
        chosen = validate_verifier_model(model or RECOMMENDED_VERIFIER_MODEL)
        chat_endpoint = validate_verifier_endpoint(endpoint)
        timeout = validate_verifier_timeout(timeout_seconds)
    except ValueError as exc:
        raise VerifierEnableError(str(exc)) from exc
    if probe is None:
        probe_verifier_model(
            chosen, endpoint=chat_endpoint, timeout_seconds=min(timeout, 5.0)
        )
    else:
        probe(chosen)
    path = Path(config_path)
    try:
        raw = path.read_text(encoding="utf-8")
        document = json.loads(raw)
    except FileNotFoundError as exc:
        raise VerifierEnableError("server configuration does not exist") from exc
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise VerifierEnableError("server configuration is not readable JSON") from exc
    if not isinstance(document, dict):
        raise VerifierEnableError("server configuration must be a JSON object")
    extraction = document.get("extraction")
    if extraction is None:
        extraction = {"mode": "disabled"}
        document["extraction"] = extraction
    if not isinstance(extraction, dict):
        raise VerifierEnableError("extraction configuration must be an object")
    if "mode" not in extraction:
        extraction["mode"] = "disabled"
    extraction["evidence_verifier"] = {
        "import": LOCAL_VERIFIER_IMPORT,
        "model": chosen,
        "timeout_seconds": timeout,
        "prefilter": True,
        "endpoint": chat_endpoint,
    }
    encoded = json.dumps(
        document, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    staging = path.with_name(path.name + ".verifier.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(staging, flags, 0o600)
    except OSError as exc:
        raise VerifierEnableError("cannot stage verifier configuration") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    except Exception:
        try:
            os.unlink(staging)
        except OSError:
            pass
        raise
    return {
        "config_path": str(path),
        "model": chosen,
        "endpoint": chat_endpoint,
        "timeout_seconds": timeout,
        "extraction_mode": extraction.get("mode"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="enfold-evidence-verifier",
        description="Evaluate or opt in to the local evidence verifier.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    enable = commands.add_parser(
        "enable",
        help="probe a local model, then write extraction.evidence_verifier",
    )
    enable.add_argument(
        "--config",
        type=Path,
        required=True,
        help="existing server.json to update",
    )
    enable.add_argument(
        "--model",
        default=RECOMMENDED_VERIFIER_MODEL,
        help="independent local Ollama model tag",
    )
    enable.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    enable.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    evaluate = commands.add_parser(
        "eval",
        help="score the labeled fixture; skip live models when Ollama is absent",
    )
    evaluate.add_argument(
        "--models",
        default="qwen2.5:3b-instruct,qwen3.8:27b",
        help="comma-separated local model tags",
    )
    evaluate.add_argument(
        "--timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
    )
    args = parser.parse_args(argv)
    if args.command == "enable":
        try:
            report = enable_verifier(
                args.config,
                model=args.model,
                endpoint=args.endpoint,
                timeout_seconds=args.timeout_seconds,
            )
        except VerifierEnableError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(
            f"wrote {report['config_path']} with model {report['model']}; "
            f"extraction mode stays {report['extraction_mode']}"
        )
        return 0
    from .verifier_eval import evaluate_configurations, load_verifier_cases

    models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    rows = evaluate_configurations(
        load_verifier_cases(),
        models=models,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
