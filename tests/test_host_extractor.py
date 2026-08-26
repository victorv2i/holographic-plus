from __future__ import annotations

import os
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from enfold.extraction_processor import ExtractionEnvelope
from enfold.host_extractor import (
    HostExtractorConfig,
    HostExtractorError,
    SubprocessHostExtractor,
)
from enfold.protocol import ClientContext


def _envelope() -> ExtractionEnvelope:
    return ExtractionEnvelope(
        transcript="Avery prefers local tools.",
        source="session_end",
        scope="private",
        context=ClientContext(
            client_id="client-a-install",
            surface="client-a",
            agent_id="client-a",
            session_id="thread-1",
            access_scopes=("private",),
        ),
        turns=({"role": "user", "content": "Avery prefers local tools."},),
    )


def _config(script: str, **changes) -> HostExtractorConfig:
    values = {
        "argv": (sys.executable, "-c", script),
        "model_identity": "local-extractor:v1",
        "prompt_identity": "extract-v1",
        "timeout_seconds": 1.0,
        "terminate_grace_seconds": 0.05,
        "environment": {},
    }
    values.update(changes)
    return HostExtractorConfig(**values)


def test_subprocess_adapter_uses_argv_json_and_allowlisted_environment(monkeypatch):
    monkeypatch.setenv("ENFOLD_SHOULD_NOT_LEAK", "secret")
    script = """
import json, os, sys
request = json.load(sys.stdin)
assert request[\"version\"] == 1
assert request[\"envelope\"][\"scope\"] == \"private\"
assert os.environ[\"ONLY_THIS\"] == \"allowed\"
assert \"ENFOLD_SHOULD_NOT_LEAK\" not in os.environ
json.dump({\"version\": 1, \"proposals\": [{\"content\": \"Avery prefers local tools.\"}]}, sys.stdout)
"""
    extractor = SubprocessHostExtractor(
        _config(script, environment={"ONLY_THIS": "allowed"})
    )

    result = extractor.extract(_envelope())

    assert extractor.identity == "subprocess:local-extractor:v1:extract-v1"
    assert [proposal.content for proposal in result] == ["Avery prefers local tools."]


def test_subprocess_adapter_streams_and_limits_stdout_without_waiting_for_timeout():
    oversized = """
import sys, time
sys.stdout.buffer.write(b'x' * 65536)
sys.stdout.buffer.flush()
time.sleep(20)
"""
    started = time.monotonic()
    with pytest.raises(HostExtractorError, match="adapter_output_too_large"):
        SubprocessHostExtractor(
            _config(oversized, max_output_bytes=1024, timeout_seconds=2.0)
        ).extract(_envelope())
    assert time.monotonic() - started < 1.0


def test_subprocess_adapter_streams_and_limits_stderr_independently():
    oversized = """
import sys, time
sys.stderr.buffer.write(b'x' * 65536)
sys.stderr.buffer.flush()
time.sleep(20)
"""
    started = time.monotonic()
    with pytest.raises(HostExtractorError, match="adapter_output_too_large"):
        SubprocessHostExtractor(
            _config(oversized, max_error_bytes=1024, timeout_seconds=2.0)
        ).extract(_envelope())
    assert time.monotonic() - started < 1.0

    malformed = "import sys; sys.stdout.write('not json')"
    with pytest.raises(HostExtractorError, match="adapter_invalid_output") as raised:
        SubprocessHostExtractor(_config(malformed)).extract(_envelope())
    assert raised.value.retryable is True


def test_subprocess_adapter_timeout_terminates_and_reaps_without_error_text():
    stubborn = """
import signal, time
signal.signal(signal.SIGTERM, lambda _signum, _frame: None)
time.sleep(20)
"""
    started = time.monotonic()
    with pytest.raises(HostExtractorError, match="adapter_timeout") as raised:
        SubprocessHostExtractor(
            _config(stubborn, timeout_seconds=0.05, terminate_grace_seconds=0.05),
        ).extract(_envelope())
    assert raised.value.retryable is True
    assert time.monotonic() - started < 1.0


def test_cancel_terminates_only_the_selected_concurrent_invocation():
    extractor = SubprocessHostExtractor(
        _config("import time; time.sleep(20)", timeout_seconds=2.0)
    )
    errors = {}
    handles = {}

    def extract(name):
        try:
            extractor.extract(
                _envelope(),
                register_invocation=lambda handle: handles.setdefault(name, handle),
            )
        except HostExtractorError as exc:
            errors[name] = exc.error_code

    threads = {
        name: threading.Thread(target=extract, args=(name,))
        for name in ("cancelled", "survivor")
    }
    for thread in threads.values():
        thread.start()
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        with extractor._process_lock:
            if len(extractor._active_processes) == 2 and len(handles) == 2:
                break
        time.sleep(0.005)

    extractor.cancel(handles["cancelled"])
    threads["cancelled"].join(0.5)

    assert not threads["cancelled"].is_alive()
    assert threads["survivor"].is_alive()
    assert errors == {"cancelled": "adapter_exit"}

    extractor.cancel(handles["survivor"])
    threads["survivor"].join(0.5)
    assert not threads["survivor"].is_alive()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_timeout_kills_descendants_in_the_dedicated_process_group(tmp_path):
    pid_path = tmp_path / "descendant.pid"
    script = f"""
import pathlib, subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
pathlib.Path({str(pid_path)!r}).write_text(str(child.pid), encoding="utf-8")
time.sleep(20)
"""
    with pytest.raises(HostExtractorError, match="adapter_timeout"):
        SubprocessHostExtractor(
            _config(script, timeout_seconds=0.2, terminate_grace_seconds=0.05)
        ).extract(_envelope())
    assert pid_path.exists()
    descendant_pid = int(pid_path.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 1.0
    while os.path.exists(f"/proc/{descendant_pid}") and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not os.path.exists(f"/proc/{descendant_pid}")


def test_host_identity_rejects_secret_shaped_values():
    with pytest.raises(ValueError, match="must not contain secret"):
        _config("pass", model_identity="sk-live-secret")
    with pytest.raises(ValueError, match="must not contain secret"):
        _config("pass", prompt_identity="token-v1")


def test_oversized_adapter_input_is_permanent():
    with pytest.raises(HostExtractorError, match="adapter_input_too_large") as raised:
        SubprocessHostExtractor(_config("pass", max_input_bytes=1)).extract(
            _envelope()
        )

    assert raised.value.retryable is False


def _bundled_fixture(
    tmp_path, exit_code: int, *, stderr: str = ""
) -> HostExtractorConfig:
    command = tmp_path / "enfold-ollama-extractor"
    command.write_text(
        f"#!{sys.executable}\nimport sys\nsys.stdin.buffer.read()\n"
        f"sys.stderr.write({stderr!r})\nraise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    command.chmod(0o700)
    return HostExtractorConfig(
        argv=(str(command),),
        model_identity="ollama:qwen3-30b",
        prompt_identity="durable-memory-v1",
        timeout_seconds=1,
        terminate_grace_seconds=0.05,
    )


@pytest.mark.parametrize(
    ("exit_code", "error_code", "retryable"),
    [
        (64, "adapter_invalid_config", False),
        (65, "adapter_invalid_input", False),
        (69, "adapter_unavailable", True),
        (70, "adapter_exit", True),
        (75, "adapter_rate_limited", True),
        (76, "adapter_invalid_output", True),
    ],
)
def test_bundled_child_exit_status_preserves_failure_classification(
    tmp_path, exit_code, error_code, retryable
):
    with pytest.raises(HostExtractorError, match=error_code) as raised:
        SubprocessHostExtractor(_bundled_fixture(tmp_path, exit_code)).extract(
            _envelope()
        )

    assert raised.value.error_code == error_code
    assert raised.value.retryable is retryable


def test_bundled_rate_limit_exit_parses_only_numeric_retry_hint(tmp_path):
    config = _bundled_fixture(
        tmp_path,
        75,
        stderr='{"retry_after_seconds":120}',
    )

    with pytest.raises(HostExtractorError) as raised:
        SubprocessHostExtractor(config).extract(_envelope())

    assert raised.value.error_code == "adapter_rate_limited"
    assert raised.value.retry_after_seconds == 120.0
    assert raised.value.consumes_attempt is False


def test_windows_cleanup_uses_taskkill_for_the_process_tree(monkeypatch):
    calls = []
    extractor = SubprocessHostExtractor(_config("pass"))
    monkeypatch.setattr(
        "enfold.host_extractor.os",
        SimpleNamespace(name="nt"),
    )
    monkeypatch.setattr(
        "enfold.host_extractor.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    extractor._signal(SimpleNamespace(pid=4321), None, 15)

    assert calls[0][0] == ("taskkill", "/PID", "4321", "/T", "/F")
    assert calls[0][1]["timeout"] == 0.05


def test_generic_adapter_exit_65_retains_legacy_retryable_adapter_exit():
    with pytest.raises(HostExtractorError, match="adapter_exit") as raised:
        SubprocessHostExtractor(_config("raise SystemExit(65)")).extract(_envelope())

    assert raised.value.error_code == "adapter_exit"
    assert raised.value.retryable is True


@pytest.mark.parametrize(
    "module",
    [
        "enfold.ollama_extractor_child",
        "enfold.openai_extractor_child",
    ],
)
def test_bundled_module_detection_accepts_isolated_python_invocation(module):
    config = _config(
        "pass",
        argv=(
            sys.executable,
            "-I",
            "-m",
            module,
        ),
    )

    assert SubprocessHostExtractor(config)._is_bundled_extractor_child() is True
