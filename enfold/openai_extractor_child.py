"""Bounded OpenAI Responses API child for :class:`SubprocessHostExtractor`.

The child accepts one supervisor request, makes one non-streaming
``store:false`` request with no tools, and emits one canonical proposal
document. Credentials, transcripts, model output, and API error bodies are
never written to stdout or stderr. Rate-limit failures may emit only a bounded
numeric retry hint on stderr for the supervisor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
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


DEFAULT_ENDPOINT = "https://api.openai.com/v1/responses"
DEFAULT_MODEL = "gpt-5.6-luna"
MAX_STDIN_BYTES = 32 * 1024
DEFAULT_MAX_HTTP_BYTES = 64 * 1024
DEFAULT_MAX_OUTPUT_TOKENS = 4_096

EXIT_CONFIG = 64
EXIT_INVALID_DATA = 65
EXIT_UNAVAILABLE = 69
EXIT_INTERNAL = 70
EXIT_RATE_LIMITED = 75
EXIT_INVALID_MODEL_OUTPUT = 76

_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,95}$")
_HEADER_VALUE = re.compile(r"^[\x21-\x7e]{1,256}$")
_ENV_API_KEY = "OPENAI_API_KEY"
_ENV_ORGANIZATION = "OPENAI_ORG_ID"
_ENV_PROJECT = "OPENAI_PROJECT_ID"
_ENV_ENDPOINT = "ENFOLD_OPENAI_ENDPOINT"
_ENV_MODEL = "ENFOLD_OPENAI_MODEL"
_ENV_TIMEOUT = "ENFOLD_OPENAI_TIMEOUT_SECONDS"
_ENV_MAX_HTTP = "ENFOLD_OPENAI_MAX_RESPONSE_BYTES"
_ENV_MAX_TOKENS = "ENFOLD_OPENAI_MAX_OUTPUT_TOKENS"
_TRANSIENT_HTTP_STATUSES = frozenset({408, 409, 425, 429})
_PERMANENT_HTTP_STATUSES = frozenset({400, 401, 403, 404, 405, 413, 422})


class ChildError(RuntimeError):
    """An intentionally detail-free failure with a stable process status."""

    def __init__(
        self, exit_code: int, *, retry_after_seconds: float | None = None
    ) -> None:
        super().__init__("openai_extractor_failed")
        self.exit_code = exit_code
        self.retry_after_seconds = retry_after_seconds


class _QuietParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        raise ChildError(EXIT_CONFIG)


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        del req, fp, code, msg, headers, newurl
        return None


@dataclass(frozen=True, slots=True)
class OpenAIChildConfig:
    model_identity: str
    api_key: str = field(repr=False, compare=False)
    endpoint: str = DEFAULT_ENDPOINT
    model: str = DEFAULT_MODEL
    prompt_identity: str = PROMPT_IDENTITY
    timeout_seconds: float = 120.0
    max_response_bytes: int = DEFAULT_MAX_HTTP_BYTES
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    reasoning_effort: str = "none"
    organization: str | None = None
    project: str | None = None

    def __post_init__(self) -> None:
        _validate_endpoint(self.endpoint)
        for value in (self.model, self.model_identity, self.prompt_identity):
            if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
                raise ChildError(EXIT_CONFIG)
        if self.prompt_identity != PROMPT_IDENTITY:
            raise ChildError(EXIT_CONFIG)
        if (
            not isinstance(self.api_key, str)
            or not self.api_key
            or self.api_key != self.api_key.strip()
            or len(self.api_key) > 512
            or any(ord(character) < 33 or ord(character) > 126 for character in self.api_key)
        ):
            raise ChildError(EXIT_CONFIG)
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(float(self.timeout_seconds))
            or self.timeout_seconds <= 0
        ):
            raise ChildError(EXIT_CONFIG)
        for value in (self.max_response_bytes, self.max_output_tokens):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ChildError(EXIT_CONFIG)
        if self.max_response_bytes > 1024 * 1024 or self.max_output_tokens > 16_384:
            raise ChildError(EXIT_CONFIG)
        if self.reasoning_effort not in {"none", "low"}:
            raise ChildError(EXIT_CONFIG)
        for value in (self.organization, self.project):
            if value is not None and (
                not isinstance(value, str) or not _HEADER_VALUE.fullmatch(value)
            ):
                raise ChildError(EXIT_CONFIG)


def _validate_endpoint(endpoint: str) -> None:
    if not isinstance(endpoint, str):
        raise ChildError(EXIT_CONFIG)
    parsed = parse.urlsplit(endpoint)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ChildError(EXIT_CONFIG) from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.openai.com"
        or port not in {None, 443}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1/responses"
    ):
        raise ChildError(EXIT_CONFIG)


def _decode_input(raw: bytes, config: OpenAIChildConfig) -> Mapping[str, Any]:
    try:
        return decode_supervisor_request(
            raw,
            model_identity=config.model_identity,
            prompt_identity=config.prompt_identity,
        )
    except ExtractionContractError as exc:
        raise ChildError(EXIT_INVALID_DATA) from exc


def _safety_identifier(envelope: Mapping[str, Any]) -> str:
    context = envelope["context"]
    client_id = context.get("client_id", "enfold")
    if not isinstance(client_id, str):
        client_id = "enfold"
    digest = hashlib.sha256(client_id.encode("utf-8")).hexdigest()[:32]
    return f"enfold_{digest}"


def _openai_payload(
    envelope: Mapping[str, Any],
    config: OpenAIChildConfig,
    spans: Sequence[TranscriptSpan],
) -> bytes:
    value = {
        "input": model_input(envelope, spans),
        "instructions": SYSTEM_PROMPT,
        "max_output_tokens": config.max_output_tokens,
        "model": config.model,
        "reasoning": {"effort": config.reasoning_effort},
        "safety_identifier": _safety_identifier(envelope),
        "store": False,
        "text": {
            "format": {
                "name": "enfold_memory_proposals",
                "schema": proposal_schema(spans, strict_nullable=True),
                "strict": True,
                "type": "json_schema",
            }
        },
        "tools": [],
        "truncation": "disabled",
    }
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
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


def _headers(config: OpenAIChildConfig) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "User-Agent": "enfold-openai-extractor/1",
    }
    if config.organization is not None:
        headers["OpenAI-Organization"] = config.organization
    if config.project is not None:
        headers["OpenAI-Project"] = config.project
    return headers


def _retry_after_seconds(headers: Any) -> float | None:
    value = headers.get("Retry-After") if headers is not None else None
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    if not math.isfinite(seconds) or seconds < 0:
        return None
    return seconds


def _http_failure(status: int, headers: Any = None) -> ChildError:
    if status in _PERMANENT_HTTP_STATUSES:
        return ChildError(EXIT_CONFIG)
    if status == 429:
        return ChildError(
            EXIT_RATE_LIMITED,
            retry_after_seconds=_retry_after_seconds(headers),
        )
    if status in _TRANSIENT_HTTP_STATUSES or status >= 500:
        return ChildError(EXIT_UNAVAILABLE)
    return ChildError(EXIT_UNAVAILABLE)


def _call_openai(
    payload: bytes,
    config: OpenAIChildConfig,
    *,
    opener: Any | None = None,
) -> bytes:
    http = opener or request.build_opener(request.ProxyHandler({}), _NoRedirect())
    req = request.Request(
        config.endpoint,
        data=payload,
        headers=_headers(config),
        method="POST",
    )
    try:
        with http.open(req, timeout=float(config.timeout_seconds)) as response:
            status = int(getattr(response, "status", 200))
            if status != 200:
                raise _http_failure(status, response.headers)
            return _read_bounded(response, config.max_response_bytes)
    except ChildError:
        raise
    except error.HTTPError as exc:
        raise _http_failure(int(exc.code), exc.headers) from exc
    except (error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        raise ChildError(EXIT_UNAVAILABLE) from exc


def _proposal_document(raw: bytes) -> Mapping[str, Any]:
    try:
        response = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChildError(EXIT_INVALID_MODEL_OUTPUT) from exc
    if (
        not isinstance(response, dict)
        or response.get("status") != "completed"
        or not isinstance(response.get("output"), list)
    ):
        raise ChildError(EXIT_INVALID_MODEL_OUTPUT)

    texts: list[str] = []
    for item in response["output"]:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        if item.get("role") != "assistant" or not isinstance(item.get("content"), list):
            raise ChildError(EXIT_INVALID_MODEL_OUTPUT)
        for part in item["content"]:
            if not isinstance(part, dict):
                raise ChildError(EXIT_INVALID_MODEL_OUTPUT)
            if part.get("type") == "refusal":
                raise ChildError(EXIT_INVALID_MODEL_OUTPUT)
            if part.get("type") != "output_text" or not isinstance(part.get("text"), str):
                raise ChildError(EXIT_INVALID_MODEL_OUTPUT)
            texts.append(part["text"])
    if len(texts) != 1:
        raise ChildError(EXIT_INVALID_MODEL_OUTPUT)
    try:
        document = json.loads(texts[0])
    except json.JSONDecodeError as exc:
        raise ChildError(EXIT_INVALID_MODEL_OUTPUT) from exc
    if not isinstance(document, dict):
        raise ChildError(EXIT_INVALID_MODEL_OUTPUT)
    return document


def transform(
    raw: bytes, config: OpenAIChildConfig, *, opener: Any | None = None
) -> bytes:
    """Transform one supervisor request into one canonical proposal response."""

    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_STDIN_BYTES:
        raise ChildError(EXIT_INVALID_DATA)
    envelope = _decode_input(raw, config)
    spans = transcript_spans(envelope["transcript"])
    if not spans:
        raise ChildError(EXIT_INVALID_DATA)
    response = _call_openai(
        _openai_payload(envelope, config, spans), config, opener=opener
    )
    try:
        proposals = normalize_proposal_document(_proposal_document(response), spans)
    except ExtractionContractError as exc:
        raise ChildError(EXIT_INVALID_MODEL_OUTPUT) from exc
    return json.dumps(
        proposals,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = _QuietParser(description="Enfold OpenAI extraction child")
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
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=os.environ.get(_ENV_MAX_TOKENS, str(DEFAULT_MAX_OUTPUT_TOKENS)),
    )
    parser.add_argument("--reasoning-effort", choices=("none", "low"), default="none")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        config = OpenAIChildConfig(
            endpoint=args.endpoint,
            model=args.model,
            model_identity=args.model_identity,
            api_key=os.environ.get(_ENV_API_KEY, ""),
            prompt_identity=args.prompt_identity,
            timeout_seconds=args.timeout_seconds,
            max_response_bytes=args.max_response_bytes,
            max_output_tokens=args.max_output_tokens,
            reasoning_effort=args.reasoning_effort,
            organization=os.environ.get(_ENV_ORGANIZATION) or None,
            project=os.environ.get(_ENV_PROJECT) or None,
        )
        raw = sys.stdin.buffer.read(MAX_STDIN_BYTES + 1)
        output = transform(raw, config)
        sys.stdout.buffer.write(output)
        sys.stdout.buffer.flush()
        return 0
    except ChildError as exc:
        if exc.retry_after_seconds is not None:
            hint = json.dumps(
                {"retry_after_seconds": exc.retry_after_seconds},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
            try:
                sys.stderr.buffer.write(hint)
                sys.stderr.buffer.flush()
            except (BrokenPipeError, OSError):
                return EXIT_UNAVAILABLE
        return exc.exit_code
    except (BrokenPipeError, OSError):
        return EXIT_UNAVAILABLE
    except Exception:
        return EXIT_INTERNAL


if __name__ == "__main__":  # pragma: no cover - exercised as a subprocess
    raise SystemExit(main())
