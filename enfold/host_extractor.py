"""Strict subprocess boundary for portable host-model extraction.

The daemon can select this adapter through explicit supervised extraction
configuration without taking a vendor SDK, credentials, or a shell dependency.
The child receives one bounded JSON request on stdin and returns one bounded
JSON response on stdout.

On Windows, descendant cleanup is best-effort through ``taskkill /T``. Pipe
handles are also closed and reader threads are joined within the configured
grace period, but the platform cannot provide POSIX process-group guarantees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import os
import re
import selectors
import signal
import subprocess
import threading
import time
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from .extractor_artifact import (
    ArtifactComponent,
    ExtractorArtifactError,
    InferenceRecipe,
)
from .extraction_processor import ExtractedMemory, ExtractionEnvelope


_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,95}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRETISH = re.compile(r"(?i)(?:api[_-]?key|secret|password|token|bearer|sk-)")
_PROPOSAL_FIELDS = frozenset(
    {
        "content",
        "category",
        "tags",
        "trust_score",
        "source_authority",
        "evidence_excerpt",
        "scope",
        "sensitivity",
        "state",
        "metadata",
    }
)


class HostExtractorError(RuntimeError):
    """Stable, redacted adapter failure suitable for durable queue status."""

    def __init__(
        self,
        error_code: str,
        *,
        retryable: bool = True,
        retry_after_seconds: float | None = None,
        consumes_attempt: bool = True,
    ) -> None:
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = bool(retryable)
        self.retry_after_seconds = retry_after_seconds
        self.consumes_attempt = bool(consumes_attempt)


@dataclass(frozen=True, slots=True)
class HostExtractorConfig:
    """Static, secret-free configuration for a local host-model child."""

    argv: tuple[str, ...]
    model_identity: str
    prompt_identity: str
    timeout_seconds: float = 180.0
    terminate_grace_seconds: float = 2.0
    max_input_bytes: int = 16 * 1024
    max_output_bytes: int = 64 * 1024
    max_error_bytes: int = 16 * 1024
    environment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        argv = tuple(self.argv)
        if not argv or not all(isinstance(value, str) and value and "\0" not in value for value in argv):
            raise ValueError("argv must be a non-empty sequence of non-empty strings")
        object.__setattr__(self, "argv", argv)
        for name, value in (
            ("model_identity", self.model_identity),
            ("prompt_identity", self.prompt_identity),
        ):
            if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
                raise ValueError(f"{name} must be a non-secret protocol identity")
            if _SECRETISH.search(value):
                raise ValueError(f"{name} must not contain secret-shaped text")
        if len(f"subprocess:{self.model_identity}:{self.prompt_identity}") > 128:
            raise ValueError("combined model/prompt identity is too long")
        for name, value in (
            ("timeout_seconds", self.timeout_seconds),
            ("terminate_grace_seconds", self.terminate_grace_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name, value in (
            ("max_input_bytes", self.max_input_bytes),
            ("max_output_bytes", self.max_output_bytes),
            ("max_error_bytes", self.max_error_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        environment = dict(self.environment)
        for key, value in environment.items():
            if not isinstance(key, str) or not _ENV_NAME.fullmatch(key):
                raise ValueError("environment names must be shell-independent identifiers")
            if not isinstance(value, str) or "\0" in value:
                raise ValueError("environment values must be NUL-free strings")
        # Do not inherit the daemon's ambient environment.  A caller must
        # deliberately allowlist every variable required by the host child.
        object.__setattr__(self, "environment", MappingProxyType(environment))

    def inference_recipe(
        self,
        *,
        model_artifact_digest: str,
        prompt: ArtifactComponent,
        schema: ArtifactComponent,
        source: ArtifactComponent,
    ) -> InferenceRecipe:
        """Build a deterministic recipe without inspecting environment values."""

        if prompt.identity != self.prompt_identity:
            raise ExtractorArtifactError(
                "prompt artifact identity does not match host configuration"
            )
        return InferenceRecipe(
            adapter_command=self.argv,
            timeout_seconds=float(self.timeout_seconds),
            terminate_grace_seconds=float(self.terminate_grace_seconds),
            max_input_bytes=self.max_input_bytes,
            max_output_bytes=self.max_output_bytes,
            max_error_bytes=self.max_error_bytes,
            environment_names=tuple(self.environment.keys()),
            model_identity=self.model_identity,
            model_artifact_digest=model_artifact_digest,
            prompt=prompt,
            schema=schema,
            source=source,
        )


class SubprocessHostExtractor:
    """Run a host-provided extractor command without a shell or inherited env."""

    def __init__(
        self,
        config: HostExtractorConfig,
        *,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        if not isinstance(config, HostExtractorConfig):
            raise TypeError("config must be HostExtractorConfig")
        self._config = config
        self._popen = popen_factory
        self._process_lock = threading.Lock()
        self._active_processes: dict[int, tuple[object, Any, int | None]] = {}

    @property
    def identity(self) -> str:
        return f"subprocess:{self._config.model_identity}:{self._config.prompt_identity}"

    def extract(
        self,
        envelope: ExtractionEnvelope,
        *,
        register_invocation: Callable[[object], None] | None = None,
    ) -> Sequence[ExtractedMemory]:
        payload = self._request_bytes(envelope)
        popen_kwargs: dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "bufsize": 0,
            "shell": False,
            "env": dict(self._config.environment),
        }
        if os.name == "posix":
            # The child becomes both a new session and process-group leader.
            # Cleanup can therefore signal every descendant without touching
            # this process's group.
            popen_kwargs["start_new_session"] = True
        try:
            process = self._popen(list(self._config.argv), **popen_kwargs)
        except OSError as exc:
            raise HostExtractorError("adapter_unavailable") from exc
        group_id = self._process_group_id(process)
        invocation = object()
        with self._process_lock:
            self._active_processes[id(invocation)] = (invocation, process, group_id)
        try:
            if register_invocation is not None:
                try:
                    register_invocation(invocation)
                except BaseException:
                    self._terminate_then_kill(process, group_id)
                    raise
            try:
                stdout, stderr = self._stream_process(process, payload)
            except HostExtractorError:
                self._terminate_then_kill(process, group_id)
                raise
            except OSError as exc:
                self._terminate_then_kill(process, group_id)
                raise HostExtractorError("adapter_unavailable") from exc
            if process.returncode != 0:
                self._terminate_then_kill(process, group_id)
                raise self._exit_error(process.returncode, stderr)
            if group_id is not None and self._group_exists(group_id):
                self._terminate_then_kill(process, group_id)
                raise HostExtractorError("adapter_cleanup_failed")
            return self._parse_response(stdout)
        finally:
            with self._process_lock:
                self._active_processes.pop(id(invocation), None)

    def cancel(self, invocation: object) -> None:
        """Best-effort cancellation of one active invocation."""

        with self._process_lock:
            active = self._active_processes.get(id(invocation))
        if active is not None and active[0] is invocation:
            self._terminate_then_kill(*active[1:])

    def _exit_error(self, returncode: object, stderr: bytes = b"") -> HostExtractorError:
        """Classify documented bundled-child statuses without guessing generics."""

        if self._is_bundled_extractor_child():
            if returncode == 64:
                return HostExtractorError("adapter_invalid_config", retryable=False)
            if returncode == 65:
                return HostExtractorError("adapter_invalid_input", retryable=False)
            if returncode == 69:
                return HostExtractorError("adapter_unavailable", retryable=True)
            if returncode == 75:
                return HostExtractorError(
                    "adapter_rate_limited",
                    retry_after_seconds=self._retry_hint(stderr),
                    consumes_attempt=False,
                )
            if returncode == 76:
                return HostExtractorError("adapter_invalid_output", retryable=True)
        return HostExtractorError("adapter_exit", retryable=True)

    @staticmethod
    def _retry_hint(stderr: bytes) -> float | None:
        try:
            value = json.loads(stderr.decode("ascii"))
            seconds = value["retry_after_seconds"]
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
            return None
        if (
            isinstance(seconds, bool)
            or not isinstance(seconds, (int, float))
            or not math.isfinite(float(seconds))
            or seconds < 0
        ):
            return None
        return float(seconds)

    def _is_bundled_extractor_child(self) -> bool:
        argv = self._config.argv
        command_name = os.path.basename(argv[0]).removesuffix(".exe")
        if command_name in {
            "enfold-ollama-extractor",
            "enfold-openai-extractor",
        }:
            return True
        return any(
            option == "-m"
            and module
            in {
                "enfold.ollama_extractor_child",
                "enfold.openai_extractor_child",
            }
            for option, module in zip(argv, argv[1:])
        )

    def _request_bytes(self, envelope: ExtractionEnvelope) -> bytes:
        if not isinstance(envelope, ExtractionEnvelope):
            raise TypeError("envelope must be ExtractionEnvelope")
        request = {
            "envelope": {
                "context": envelope.context.to_dict(),
                "scope": envelope.scope,
                "source": envelope.source,
                "transcript": envelope.transcript,
            },
            "model_identity": self._config.model_identity,
            "prompt_identity": self._config.prompt_identity,
            "version": 1,
        }
        try:
            encoded = json.dumps(
                request,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        except (TypeError, ValueError, RecursionError) as exc:
            raise HostExtractorError(
                "adapter_input_too_large", retryable=False
            ) from exc
        if len(encoded) > self._config.max_input_bytes:
            raise HostExtractorError("adapter_input_too_large", retryable=False)
        return encoded

    def _stream_process(self, process: Any, payload: bytes) -> tuple[bytes, bytes]:
        """Write and drain bounded pipes without ``communicate`` buffering them."""

        if os.name == "posix":
            return self._stream_posix(process, payload)
        return self._stream_threaded(process, payload)

    def _stream_posix(self, process: Any, payload: bytes) -> tuple[bytes, bytes]:
        streams = {
            "stdin": process.stdin,
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        if any(stream is None for stream in streams.values()):
            raise OSError("subprocess pipes are unavailable")
        stdout = bytearray()
        stderr = bytearray()
        input_offset = 0
        deadline = time.monotonic() + float(self._config.timeout_seconds)
        selector = selectors.DefaultSelector()

        def unregister(name: str) -> None:
            stream = streams.get(name)
            if stream is None:
                return
            try:
                selector.unregister(stream)
            except KeyError:
                pass
            try:
                stream.close()
            except OSError:
                pass
            streams[name] = None

        try:
            for name, event in (
                ("stdin", selectors.EVENT_WRITE),
                ("stdout", selectors.EVENT_READ),
                ("stderr", selectors.EVENT_READ),
            ):
                stream = streams[name]
                assert stream is not None
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, event, name)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HostExtractorError("adapter_timeout")
                events = selector.select(remaining)
                if not events:
                    raise HostExtractorError("adapter_timeout")
                for key, _mask in events:
                    name = str(key.data)
                    stream = streams[name]
                    if stream is None:
                        continue
                    if name == "stdin":
                        try:
                            written = os.write(stream.fileno(), payload[input_offset:])
                        except (BlockingIOError, InterruptedError):
                            continue
                        except BrokenPipeError:
                            unregister(name)
                            continue
                        input_offset += written
                        if input_offset == len(payload):
                            unregister(name)
                        continue
                    try:
                        chunk = os.read(stream.fileno(), 8192)
                    except (BlockingIOError, InterruptedError):
                        continue
                    if not chunk:
                        unregister(name)
                        continue
                    target, limit = (
                        (stdout, self._config.max_output_bytes)
                        if name == "stdout"
                        else (stderr, self._config.max_error_bytes)
                    )
                    if len(target) + len(chunk) > limit:
                        raise HostExtractorError("adapter_output_too_large")
                    target.extend(chunk)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HostExtractorError("adapter_timeout")
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise HostExtractorError("adapter_timeout") from exc
            return bytes(stdout), bytes(stderr)
        finally:
            for name in tuple(streams):
                unregister(name)
            selector.close()

    def _stream_threaded(self, process: Any, payload: bytes) -> tuple[bytes, bytes]:
        """Portable bounded fallback when selector cannot watch process pipes."""

        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise OSError("subprocess pipes are unavailable")
        stdout = bytearray()
        stderr = bytearray()
        overflow: list[str] = []
        readers_done = threading.Event()
        completed = 0
        completed_lock = threading.Lock()

        def finish_reader() -> None:
            nonlocal completed
            with completed_lock:
                completed += 1
                if completed == 2:
                    readers_done.set()

        def reader(stream: Any, target: bytearray, limit: int) -> None:
            try:
                while True:
                    try:
                        chunk = stream.read(8192)
                    except (OSError, ValueError):
                        return
                    if not chunk:
                        return
                    if len(target) + len(chunk) > limit:
                        overflow.append("adapter_output_too_large")
                        return
                    target.extend(chunk)
            finally:
                finish_reader()

        def writer() -> None:
            try:
                process.stdin.write(payload)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                pass
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        threads: list[threading.Thread] = []
        for stream, target, limit in (
            (process.stdout, stdout, self._config.max_output_bytes),
            (process.stderr, stderr, self._config.max_error_bytes),
        ):
            thread = threading.Thread(
                target=reader, args=(stream, target, limit), daemon=True
            )
            thread.start()
            threads.append(thread)
        writer_thread = threading.Thread(target=writer, daemon=True)
        writer_thread.start()
        threads.append(writer_thread)
        deadline = time.monotonic() + float(self._config.timeout_seconds)
        try:
            while True:
                if overflow:
                    raise HostExtractorError(overflow[0])
                if readers_done.is_set() and process.poll() is not None:
                    process.wait(timeout=0)
                    return bytes(stdout), bytes(stderr)
                if time.monotonic() >= deadline:
                    raise HostExtractorError("adapter_timeout")
                time.sleep(0.01)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
            join_deadline = time.monotonic() + float(
                self._config.terminate_grace_seconds
            )
            for thread in threads:
                thread.join(max(0.0, join_deadline - time.monotonic()))

    @staticmethod
    def _process_group_id(process: Any) -> int | None:
        if os.name != "posix" or not isinstance(getattr(process, "pid", None), int):
            return None
        return int(process.pid)

    @staticmethod
    def _group_exists(group_id: int) -> bool:
        try:
            os.killpg(group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _terminate_then_kill(self, process: Any, group_id: int | None) -> None:
        """Reap the direct child and its POSIX process group without logging text."""

        self._signal(process, group_id, signal.SIGTERM)
        if not self._wait_for_exit(process, group_id):
            self._signal(process, group_id, signal.SIGKILL)
            self._wait_for_exit(process, group_id)
        try:
            process.wait(timeout=float(self._config.terminate_grace_seconds))
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=float(self._config.terminate_grace_seconds))
            except (OSError, subprocess.TimeoutExpired):
                pass

    def _signal(self, process: Any, group_id: int | None, sig: int) -> None:
        try:
            if group_id is not None:
                os.killpg(group_id, sig)
            elif os.name == "nt" and isinstance(getattr(process, "pid", None), int):
                subprocess.run(
                    (
                        "taskkill",
                        "/PID",
                        str(process.pid),
                        "/T",
                        "/F",
                    ),
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=float(self._config.terminate_grace_seconds),
                )
            elif sig == signal.SIGTERM:
                process.terminate()
            else:
                process.kill()
        except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
            pass

    def _wait_for_exit(self, process: Any, group_id: int | None) -> bool:
        deadline = time.monotonic() + float(self._config.terminate_grace_seconds)
        while True:
            process.poll()
            group_gone = group_id is None or not self._group_exists(group_id)
            if group_gone:
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(0.01, remaining))

    @staticmethod
    def _parse_response(stdout: bytes) -> tuple[ExtractedMemory, ...]:
        try:
            response = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HostExtractorError("adapter_invalid_output") from exc
        if not isinstance(response, dict) or set(response) != {"proposals", "version"}:
            raise HostExtractorError("adapter_invalid_output")
        if response.get("version") != 1 or not isinstance(response.get("proposals"), list):
            raise HostExtractorError("adapter_invalid_output")
        proposals: list[ExtractedMemory] = []
        for value in response["proposals"]:
            if not isinstance(value, dict) or "content" not in value:
                raise HostExtractorError("adapter_invalid_output")
            if set(value) - _PROPOSAL_FIELDS:
                raise HostExtractorError("adapter_invalid_output")
            if not isinstance(value["content"], str):
                raise HostExtractorError("adapter_invalid_output")
            if "metadata" in value and not isinstance(value["metadata"], dict):
                raise HostExtractorError("adapter_invalid_output")
            if "state" in value and value["state"] is not None and not isinstance(value["state"], dict):
                raise HostExtractorError("adapter_invalid_output")
            proposals.append(ExtractedMemory(**value))
        return tuple(proposals)
