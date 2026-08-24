from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat

import pytest

from enfold.activation import (
    ACTIVATION_CONFIG_VERSION,
    ActivationPreparationError,
    _load_base,
    stage_activation_config,
)
from enfold.extraction_contract import PROMPT_IDENTITY
from enfold.server import load_config


MODEL_DIGEST = "sha256:" + "a" * 64


def _candidate_executable(tmp_path: Path) -> Path:
    prefix = tmp_path / "candidate-installation"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    interpreter = bin_dir / "python"
    interpreter.write_bytes(b"candidate python interpreter")
    package = prefix / "lib" / "python3.13" / "site-packages" / "enfold"
    package.mkdir(parents=True, exist_ok=True)
    for module in ("ollama_extractor_child", "host_extractor", "extraction_spans"):
        (package / f"{module}.py").write_text(
            f"# candidate {module}\n", encoding="utf-8"
        )
    from enfold import extraction_contract as _contract
    from enfold import extraction_processor as _processor

    shutil.copyfile(
        _contract.__file__, package / "extraction_contract.py"
    )
    shutil.copyfile(
        _processor.__file__, package / "extraction_processor.py"
    )
    path = bin_dir / "enfold-ollama-extractor"
    path.write_text(f"#!{interpreter}\n", encoding="utf-8")
    path.chmod(0o700)
    return path


def _base_config(tmp_path: Path) -> Path:
    path = tmp_path / "base.json"
    value = {
        "database_path": str(tmp_path / "memory.db"),
        "socket_path": str(tmp_path / "live.sock"),
        "grants": {"client-a": ["private"]},
        "retrieval": {
            "mode": "ci",
            "allow_nonproduction": True,
            "dimensions": 64,
        },
        "extraction": {
            "mode": "daemon-supervised",
            "host": {
                "type": "subprocess",
                "argv": [
                    "/old/enfold-ollama-extractor",
                    "--endpoint",
                    "http://127.0.0.1:11434/api/chat",
                    "--model",
                    "qwen3:30b",
                    "--model-identity",
                    "ollama:qwen3-30b",
                ],
                "model_identity": "ollama:qwen3-30b",
                "prompt_identity": "durable-memory-v1",
                "environment": {},
            },
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)
    return path


def test_stage_config_pins_candidate_recipe_without_replacing_base(tmp_path):
    base = _base_config(tmp_path)
    original = base.read_bytes()
    output = tmp_path / "private" / "candidate.json"

    report = stage_activation_config(
        base,
        output,
        candidate_executable=_candidate_executable(tmp_path),
        model_digest=MODEL_DIGEST,
        database_path=tmp_path / "rehearsal.db",
        socket_path=tmp_path / "rehearsal.sock",
    )

    assert base.read_bytes() == original
    assert report.schema_version == ACTIVATION_CONFIG_VERSION
    assert report.status == "staged-not-activated"
    assert report.prompt_identity == PROMPT_IDENTITY
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    value = json.loads(output.read_text())
    assert value["extraction"]["host"]["argv"][0] == str(
        _candidate_executable(tmp_path)
    )
    assert value["extraction"]["host"]["prompt_identity"] == PROMPT_IDENTITY
    assert value["extraction"]["host"]["environment"] == {}
    assert value["extraction"]["artifact"] == {
        "provider": "ollama",
        "model": "qwen3:30b",
        "model_digest": MODEL_DIGEST,
        "recipe_digest": value["extraction"]["artifact"]["recipe_digest"],
    }
    assert value["extraction"]["artifact"]["recipe_digest"].startswith("sha256:")
    assert value["extraction"]["artifact_recheck_seconds"] == 60
    load_config(output)


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        (
            '"model_identity": "ollama:qwen3-30b", "prompt_identity"',
            '"model_identity": "operator-reviewed", '
            '"model_identity": "ollama:qwen3-30b", "prompt_identity"',
            "duplicate JSON object key: model_identity",
        ),
        (
            '"prompt_identity": "durable-memory-v1"',
            '"prompt_identity": NaN',
            "non-finite JSON number is not allowed: NaN",
        ),
    ],
)
def test_stage_config_rejects_ambiguous_json_before_transform(
    tmp_path, original, replacement, message
):
    base = _base_config(tmp_path)
    serialized = base.read_text(encoding="utf-8")
    assert original in serialized
    base.write_text(serialized.replace(original, replacement), encoding="utf-8")
    output = tmp_path / "candidate.json"

    with pytest.raises(ActivationPreparationError, match=message):
        stage_activation_config(
            base,
            output,
            candidate_executable=_candidate_executable(tmp_path),
            model_digest=MODEL_DIGEST,
        )

    assert not output.exists()


def test_stage_config_reads_and_validates_one_open_descriptor(tmp_path, monkeypatch):
    base = _base_config(tmp_path)
    real_fdopen = os.fdopen
    real_read_text = Path.read_text
    replaced = False

    def replace_path():
        nonlocal replaced
        if replaced:
            return
        replaced = True
        base.unlink()
        base.write_text("{not valid JSON", encoding="utf-8")
        base.chmod(0o600)

    def racing_fdopen(descriptor, *args, **kwargs):
        replace_path()
        return real_fdopen(descriptor, *args, **kwargs)

    def racing_read_text(candidate, *args, **kwargs):
        if candidate == base:
            replace_path()
        return real_read_text(candidate, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", racing_fdopen)
    monkeypatch.setattr(Path, "read_text", racing_read_text)

    value = _load_base(base)

    assert replaced is True
    assert value["grants"] == {"client-a": ["private"]}


def test_stage_config_wraps_oversized_json_integer(tmp_path):
    base = _base_config(tmp_path)
    serialized = base.read_text(encoding="utf-8").replace(
        '"prompt_identity": "durable-memory-v1"',
        f'"prompt_identity": {"9" * 5000}',
    )
    base.write_text(serialized, encoding="utf-8")

    with pytest.raises(ActivationPreparationError, match="not valid JSON"):
        stage_activation_config(
            base,
            tmp_path / "candidate.json",
            candidate_executable=_candidate_executable(tmp_path),
            model_digest=MODEL_DIGEST,
        )


def test_stage_config_refuses_overwrite_symlink_and_insecure_inputs(tmp_path):
    base = _base_config(tmp_path)
    output = tmp_path / "candidate.json"
    target = tmp_path / "unrelated.json"
    target.write_text("unchanged")
    output.symlink_to(target)

    with pytest.raises(ActivationPreparationError, match="already exists"):
        stage_activation_config(
            base,
            output,
            candidate_executable=_candidate_executable(tmp_path),
            model_digest=MODEL_DIGEST,
        )
    assert target.read_text() == "unchanged"

    output.unlink()
    base.chmod(0o666)
    with pytest.raises(ActivationPreparationError, match="not group/world writable"):
        stage_activation_config(
            base,
            output,
            candidate_executable=_candidate_executable(tmp_path),
            model_digest=MODEL_DIGEST,
        )


def test_stage_config_refuses_symlinked_base_and_executable(tmp_path):
    base = _base_config(tmp_path)
    base_link = tmp_path / "base-link.json"
    base_link.symlink_to(base)
    executable = _candidate_executable(tmp_path)
    executable_link = tmp_path / "extractor-link"
    executable_link.symlink_to(executable)

    with pytest.raises(ActivationPreparationError, match="non-symlink"):
        stage_activation_config(
            base_link,
            tmp_path / "candidate-a.json",
            candidate_executable=executable,
            model_digest=MODEL_DIGEST,
        )
    with pytest.raises(ActivationPreparationError, match="non-symlink"):
        stage_activation_config(
            base,
            tmp_path / "candidate-b.json",
            candidate_executable=executable_link,
            model_digest=MODEL_DIGEST,
        )


def test_stage_config_refuses_symlinked_output_directory(tmp_path):
    base = _base_config(tmp_path)
    real_directory = tmp_path / "real-output"
    real_directory.mkdir()
    linked_directory = tmp_path / "linked-output"
    linked_directory.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(ActivationPreparationError, match="non-symlink directory"):
        stage_activation_config(
            base,
            linked_directory / "candidate.json",
            candidate_executable=_candidate_executable(tmp_path),
            model_digest=MODEL_DIGEST,
        )

    assert not (real_directory / "candidate.json").exists()


def test_stage_config_refuses_nested_symlinked_output_ancestor(tmp_path):
    base = _base_config(tmp_path)
    real_directory = tmp_path / "real-output"
    nested_directory = real_directory / "nested"
    nested_directory.mkdir(parents=True)
    linked_directory = tmp_path / "linked-output"
    linked_directory.symlink_to(real_directory, target_is_directory=True)

    with pytest.raises(ActivationPreparationError, match="non-symlink directory"):
        stage_activation_config(
            base,
            linked_directory / "nested" / "candidate.json",
            candidate_executable=_candidate_executable(tmp_path),
            model_digest=MODEL_DIGEST,
        )

    assert not (nested_directory / "candidate.json").exists()


def test_stage_config_preserves_concurrently_created_output(tmp_path, monkeypatch):
    base = _base_config(tmp_path)
    output = tmp_path / "candidate.json"

    def collide(_source, destination, *, follow_symlinks):
        assert follow_symlinks is False
        Path(destination).write_text("concurrent-owner", encoding="utf-8")
        raise FileExistsError(destination)

    monkeypatch.setattr("enfold.activation.os.link", collide)

    with pytest.raises(ActivationPreparationError, match="already exists"):
        stage_activation_config(
            base,
            output,
            candidate_executable=_candidate_executable(tmp_path),
            model_digest=MODEL_DIGEST,
        )

    assert output.read_text(encoding="utf-8") == "concurrent-owner"


def test_stage_config_removes_its_publication_if_directory_fsync_fails(
    tmp_path, monkeypatch
):
    base = _base_config(tmp_path)
    output = tmp_path / "candidate.json"
    real_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(descriptor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("synthetic directory fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr("enfold.activation.os.fsync", fail_directory_fsync)

    with pytest.raises(ActivationPreparationError, match="strict validation"):
        stage_activation_config(
            base,
            output,
            candidate_executable=_candidate_executable(tmp_path),
            model_digest=MODEL_DIGEST,
        )

    assert not output.exists()


def test_stage_config_requires_canonical_digest_and_distinct_destination(tmp_path):
    base = _base_config(tmp_path)
    with pytest.raises(ActivationPreparationError, match="must not replace"):
        stage_activation_config(
            base,
            base,
            candidate_executable=_candidate_executable(tmp_path),
            model_digest=MODEL_DIGEST,
        )
    with pytest.raises(ActivationPreparationError, match="could not be measured"):
        stage_activation_config(
            base,
            tmp_path / "candidate.json",
            candidate_executable=_candidate_executable(tmp_path),
            model_digest="not-a-digest",
        )


def test_stage_config_rejects_mismatched_command_identity(tmp_path):
    base = _base_config(tmp_path)
    value = json.loads(base.read_text())
    argv = value["extraction"]["host"]["argv"]
    argv[argv.index("--model-identity") + 1] = "ollama:different-model"
    base.write_text(json.dumps(value))
    base.chmod(0o600)

    with pytest.raises(ActivationPreparationError, match="do not match"):
        stage_activation_config(
            base,
            tmp_path / "candidate.json",
            candidate_executable=_candidate_executable(tmp_path),
            model_digest=MODEL_DIGEST,
        )
