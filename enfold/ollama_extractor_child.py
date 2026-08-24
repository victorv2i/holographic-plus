"""Secret-free local Ollama child for :class:`SubprocessHostExtractor`.

One bounded v1 request arrives on stdin and one v1 proposal document leaves on
stdout. Model and transcript data are never written to stderr.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import ipaddress
import json
import math
import os
import re
import socket
import sys
from typing import Any, Mapping, Sequence
from urllib import error, parse, request

from .extraction_contract import (
    ExtractionContractError,
    PROMPT_IDENTITY,
    SYSTEM_PROMPT,
    decode_supervisor_request,
    model_input,
    normalize_proposal_document,
    proposal_schema,
)
from .extraction_spans import TranscriptSpan, transcript_spans


DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3:30b"
MAX_STDIN_BYTES = 32 * 1024
DEFAULT_MAX_HTTP_BYTES = 64 * 1024

EXIT_CONFIG = 64
EXIT_INVALID_DATA = 65
EXIT_UNAVAILABLE = 69
EXIT_INTERNAL = 70
EXIT_INVALID_MODEL_OUTPUT = 76

_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,95}$")
_ENV_ENDPOINT = "ENFOLD_OLLAMA_ENDPOINT"
_ENV_MODEL = "ENFOLD_OLLAMA_MODEL"
_ENV_TIMEOUT = "ENFOLD_OLLAMA_TIMEOUT_SECONDS"
_ENV_MAX_HTTP = "ENFOLD_OLLAMA_MAX_RESPONSE_BYTES"


class ChildError(RuntimeError):
    """An intentionally detail-free failure with a stable process status."""

    def __init__(self, exit_code: int) -> None:
        super().__init__("ollama_extractor_failed")
        self.exit_code = exit_code


class _QuietParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise ChildError(EXIT_CONFIG)


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


@dataclass(frozen=True, slots=True)
class OllamaChildConfig:
    endpoint: str
    model: str
    model_identity: str
    prompt_identity: str = PROMPT_IDENTITY
    timeout_seconds: float = 120.0
    max_response_bytes: int = DEFAULT_MAX_HTTP_BYTES

    def __post_init__(self) -> None:
        _validate_endpoint(self.endpoint)
        for value in (self.model, self.model_identity, self.prompt_identity):
            if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
                raise ChildError(EXIT_CONFIG)
        if self.prompt_identity != PROMPT_IDENTITY:
            # This child embeds one fixed prompt/schema contract. Accepting a
            # caller-supplied identity for different semantics would make its
            # provenance unverifiable.
            raise ChildError(EXIT_CONFIG)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ChildError(EXIT_CONFIG)
        if (
            isinstance(self.max_response_bytes, bool)
            or not isinstance(self.max_response_bytes, int)
            or self.max_response_bytes <= 0
            or self.max_response_bytes > 1024 * 1024
        ):
            raise ChildError(EXIT_CONFIG)


def _validate_endpoint(endpoint: str) -> None:
    if not isinstance(endpoint, str):
        raise ChildError(EXIT_CONFIG)
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
        raise ChildError(EXIT_CONFIG)
    try:
        if not ipaddress.ip_address(parsed.hostname).is_loopback:
            raise ChildError(EXIT_CONFIG)
    except ValueError as exc:
        # Numeric loopback addresses avoid DNS rebinding and hosts-file drift.
        raise ChildError(EXIT_CONFIG) from exc
    try:
        port = parsed.port
    except ValueError as exc:
        raise ChildError(EXIT_CONFIG) from exc
    if port is not None and not (1 <= port <= 65535):
        raise ChildError(EXIT_CONFIG)


def _decode_input(raw: bytes, config: OllamaChildConfig) -> Mapping[str, Any]:
    try:
        return decode_supervisor_request(
            raw,
            model_identity=config.model_identity,
            prompt_identity=config.prompt_identity,
        )
    except ExtractionContractError as exc:
        raise ChildError(EXIT_INVALID_DATA) from exc


def _proposal_schema(spans: Sequence[TranscriptSpan]) -> Mapping[str, Any]:
    return proposal_schema(spans)


def _ollama_payload(
    envelope: Mapping[str, Any],
    config: OllamaChildConfig,
    spans: Sequence[TranscriptSpan],
) -> bytes:
    user_payload = model_input(envelope, spans)
    value = {
        "model": config.model,
        "stream": False,
        "format": _proposal_schema(spans),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
        "options": {"temperature": 0},
        "think": False,
    }
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _read_bounded(response: Any, limit: int) -> bytes:
    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                raise ChildError(EXIT_INVALID_MODEL_OUTPUT)
        except ValueError as exc:
            raise ChildError(EXIT_INVALID_MODEL_OUTPUT) from exc
    body = response.read(limit + 1)
    if len(body) > limit:
        raise ChildError(EXIT_INVALID_MODEL_OUTPUT)
    return body


def _call_ollama(
    payload: bytes,
    config: OllamaChildConfig,
    *,
    opener: Any | None = None,
) -> bytes:
    # Disable ambient proxy discovery. This child only talks to loopback and
    # deliberately has no authentication feature.
    http = opener or request.build_opener(request.ProxyHandler({}), _NoRedirect())
    req = request.Request(
        config.endpoint,
        data=payload,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with http.open(req, timeout=float(config.timeout_seconds)) as response:
            if getattr(response, "status", 200) != 200:
                raise ChildError(EXIT_UNAVAILABLE)
            return _read_bounded(response, config.max_response_bytes)
    except ChildError:
        raise
    except (
        error.HTTPError,
        error.URLError,
        TimeoutError,
        socket.timeout,
        OSError,
    ) as exc:
        raise ChildError(EXIT_UNAVAILABLE) from exc


def _validate_proposals(
    raw: bytes, spans: Sequence[TranscriptSpan]
) -> Mapping[str, Any]:
    try:
        response = json.loads(raw.decode("utf-8"))
        content = response["message"]["content"]
        proposals_doc = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ChildError(EXIT_INVALID_MODEL_OUTPUT) from exc
    if not isinstance(response, dict) or not isinstance(response.get("message"), dict):
        raise ChildError(EXIT_INVALID_MODEL_OUTPUT)
    if not isinstance(content, str) or not isinstance(proposals_doc, dict):
        raise ChildError(EXIT_INVALID_MODEL_OUTPUT)
    try:
        return normalize_proposal_document(proposals_doc, spans)
    except ExtractionContractError as exc:
        raise ChildError(EXIT_INVALID_MODEL_OUTPUT) from exc


def transform(
    raw: bytes, config: OllamaChildConfig, *, opener: Any | None = None
) -> bytes:
    """Transform one supervisor request into one canonical proposal response."""

    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_STDIN_BYTES:
        raise ChildError(EXIT_INVALID_DATA)
    envelope = _decode_input(raw, config)
    spans = transcript_spans(envelope["transcript"])
    if not spans:
        raise ChildError(EXIT_INVALID_DATA)
    response = _call_ollama(
        _ollama_payload(envelope, config, spans), config, opener=opener
    )
    proposals = _validate_proposals(response, spans)
    return json.dumps(
        proposals,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = _QuietParser(description="Enfold local Ollama extraction child")
    parser.add_argument(
        "--endpoint", default=os.environ.get(_ENV_ENDPOINT, DEFAULT_ENDPOINT)
    )
    parser.add_argument("--model", default=os.environ.get(_ENV_MODEL, DEFAULT_MODEL))
    parser.add_argument("--model-identity", required=True)
    parser.add_argument("--prompt-identity", default=PROMPT_IDENTITY)
    parser.add_argument(
        "--timeout-seconds", type=float, default=os.environ.get(_ENV_TIMEOUT, "120")
    )
    parser.add_argument(
        "--max-response-bytes",
        type=int,
        default=os.environ.get(_ENV_MAX_HTTP, str(DEFAULT_MAX_HTTP_BYTES)),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        config = OllamaChildConfig(
            endpoint=args.endpoint,
            model=args.model,
            model_identity=args.model_identity,
            prompt_identity=args.prompt_identity,
            timeout_seconds=args.timeout_seconds,
            max_response_bytes=args.max_response_bytes,
        )
        raw = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
        output = transform(raw, config)
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        return 0
    except ChildError as exc:
        return exc.exit_code
    except (BrokenPipeError, OSError):
        return EXIT_UNAVAILABLE
    except Exception:
        # Never echo exception text: model output or transcript fragments may
        # be attached to an exception.
        return EXIT_INTERNAL


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
