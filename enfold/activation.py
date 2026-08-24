"""Generate a private, fail-closed Enfold activation configuration.

This module never installs packages, changes the live configuration, opens the
configured database, controls services, or invokes a model.  It converts one
existing daemon configuration into a side-by-side candidate configuration
whose bundled Ollama extractor is pinned to an immutable model artifact and an
exact measured inference recipe.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

from .extraction_contract import PROMPT_IDENTITY
from .extractor_artifact import (
    InferenceRecipe,
    bundled_ollama_components,
    require_sha256_digest,
)
from .host_extractor import HostExtractorConfig
from .ollama_extractor_child import DEFAULT_ENDPOINT
from .server import load_config


ACTIVATION_CONFIG_VERSION = "enfold-activation-config-v1"


class ActivationPreparationError(RuntimeError):
    """A candidate configuration cannot be prepared safely."""


@dataclass(frozen=True, slots=True)
class StagedActivationConfig:
    schema_version: str
    status: str
    config_path: str
    config_sha256: str
    model: str
    model_identity: str
    prompt_identity: str
    recipe_version: int


def _private_regular_file(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ActivationPreparationError(f"{label} does not exist") from exc
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ActivationPreparationError(
            f"{label} must be a regular, non-symlink file"
        )
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ActivationPreparationError(
            f"{label} must be user-owned and not group/world writable"
        )


def _candidate_executable(path: Path) -> None:
    _private_regular_file(path, "candidate extractor executable")
    if not path.is_absolute() or not os.access(path, os.X_OK):
        raise ActivationPreparationError(
            "candidate extractor executable must be absolute and executable"
        )


def _private_directory(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ActivationPreparationError(f"{label} does not exist") from exc
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise ActivationPreparationError(
            f"{label} must be a non-symlink directory"
        )
    if info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise ActivationPreparationError(
            f"{label} must be user-owned and not group/world writable"
        )


def _absolute(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        raise ActivationPreparationError(f"{label} must be absolute")
    if candidate.is_symlink():
        raise ActivationPreparationError(f"{label} must be a non-symlink path")
    if any(parent.is_symlink() for parent in candidate.parents):
        raise ActivationPreparationError(
            f"{label} parent must be a non-symlink directory"
        )
    try:
        return candidate.resolve()
    except (OSError, RuntimeError) as exc:
        raise ActivationPreparationError(f"{label} cannot be safely resolved") from exc


def _option(argv: Sequence[str], name: str, *, default: str | None = None) -> str:
    values: list[str] = []
    prefix = f"{name}="
    for index, value in enumerate(argv):
        if value.startswith(prefix):
            values.append(value[len(prefix) :])
        elif value == name:
            if index + 1 >= len(argv):
                raise ActivationPreparationError(f"{name} has no value")
            values.append(argv[index + 1])
    if len(values) > 1 or any(not value for value in values):
        raise ActivationPreparationError(f"{name} must have exactly one value")
    if values:
        return values[0]
    if default is None:
        raise ActivationPreparationError(f"{name} is required")
    return default


def _reject_json_constant(value: str) -> None:
    raise ActivationPreparationError(
        f"non-finite JSON number is not allowed: {value}"
    )


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ActivationPreparationError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _load_base(path: Path) -> dict[str, Any]:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ActivationPreparationError(
            "base configuration must be a regular, non-symlink file"
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ActivationPreparationError(
                "base configuration must be a regular, non-symlink file"
            )
        if info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ActivationPreparationError(
                "base configuration must be user-owned and not group/world writable"
            )
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            value = json.load(
                handle,
                parse_constant=_reject_json_constant,
                object_pairs_hook=_unique_json_object,
            )
    except ActivationPreparationError:
        raise
    except (OSError, UnicodeError, ValueError) as exc:
        raise ActivationPreparationError("base configuration is not valid JSON") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise ActivationPreparationError("base configuration must be an object")
    return value


def _host_config(value: Mapping[str, Any], executable: Path) -> tuple[HostExtractorConfig, str]:
    extraction = value.get("extraction")
    if not isinstance(extraction, dict) or extraction.get("mode") != "daemon-supervised":
        raise ActivationPreparationError(
            "base configuration must use daemon-supervised extraction"
        )
    host = extraction.get("host")
    if not isinstance(host, dict):
        raise ActivationPreparationError("base extraction host is missing")
    argv = host.get("argv")
    if not isinstance(argv, list) or not all(
        isinstance(item, str) and item for item in argv
    ):
        raise ActivationPreparationError("base extraction argv is invalid")
    model = _option(argv, "--model")
    endpoint = _option(argv, "--endpoint", default=DEFAULT_ENDPOINT)
    model_identity = host.get("model_identity")
    if not isinstance(model_identity, str) or not model_identity.strip():
        raise ActivationPreparationError("base model identity is invalid")
    command_identity = _option(
        argv, "--model-identity", default=model_identity.strip()
    )
    if command_identity != model_identity.strip():
        raise ActivationPreparationError(
            "base command and host model identities do not match"
        )
    candidate_argv = (
        str(executable),
        "--endpoint",
        endpoint,
        "--model",
        model,
        "--model-identity",
        model_identity.strip(),
        "--prompt-identity",
        PROMPT_IDENTITY,
    )
    try:
        config = HostExtractorConfig(
            argv=candidate_argv,
            model_identity=model_identity.strip(),
            prompt_identity=PROMPT_IDENTITY,
            timeout_seconds=host.get("timeout_seconds", 180.0),
            terminate_grace_seconds=host.get("terminate_grace_seconds", 2.0),
            max_input_bytes=host.get("max_input_bytes", 16 * 1024),
            max_output_bytes=host.get("max_output_bytes", 64 * 1024),
            max_error_bytes=host.get("max_error_bytes", 16 * 1024),
            environment={},
        )
    except (TypeError, ValueError) as exc:
        raise ActivationPreparationError("base extraction host is invalid") from exc
    return config, model


def _atomic_private_json(path: Path, value: Mapping[str, Any]) -> bytes:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    _private_directory(path.parent, "candidate configuration directory")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    published = False
    complete = False
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("candidate configuration write made no progress")
            offset += written
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        load_config(temporary_path, allow_live=True)
        try:
            # Hard-link publication is atomic and, unlike os.replace(), cannot
            # overwrite a file created after the caller's initial check.
            os.link(temporary_path, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ActivationPreparationError(
                "candidate configuration already exists"
            ) from exc
        published = True
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        complete = True
        return encoded
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if published and not complete:
            try:
                if path.exists() and os.path.samefile(temporary_path, path):
                    path.unlink()
            except OSError:
                pass
        temporary_path.unlink(missing_ok=True)
        if complete and not path.exists():
            raise ActivationPreparationError(
                "candidate configuration publication was lost"
            )


def stage_activation_config(
    base_config: str | Path,
    output_config: str | Path,
    *,
    candidate_executable: str | Path,
    model_digest: str,
    database_path: str | Path | None = None,
    socket_path: str | Path | None = None,
) -> StagedActivationConfig:
    """Build and validate a side-by-side candidate without opening its database."""

    if Path(output_config).expanduser().is_symlink():
        raise ActivationPreparationError("candidate configuration already exists")
    base = _absolute(base_config, "base configuration")
    output = _absolute(output_config, "candidate configuration")
    executable = _absolute(candidate_executable, "candidate extractor executable")
    if base == output:
        raise ActivationPreparationError(
            "candidate configuration must not replace the base configuration"
        )
    if output.exists():
        raise ActivationPreparationError("candidate configuration already exists")
    _candidate_executable(executable)
    value = _load_base(base)
    host, model = _host_config(value, executable)
    try:
        canonical_model_digest = require_sha256_digest(
            model_digest, name="candidate model digest"
        )
        components = bundled_ollama_components(executable)
        recipe = host.inference_recipe(
            model_artifact_digest=canonical_model_digest,
            **components,
        )
    except Exception as exc:
        raise ActivationPreparationError(
            "candidate extraction recipe could not be measured"
        ) from exc

    candidate = json.loads(json.dumps(value, allow_nan=False))
    extraction = candidate["extraction"]
    extraction["host"] = {
        "type": "subprocess",
        "argv": list(host.argv),
        "model_identity": host.model_identity,
        "prompt_identity": host.prompt_identity,
        "timeout_seconds": host.timeout_seconds,
        "terminate_grace_seconds": host.terminate_grace_seconds,
        "max_input_bytes": host.max_input_bytes,
        "max_output_bytes": host.max_output_bytes,
        "max_error_bytes": host.max_error_bytes,
        "environment": {},
    }
    extraction["artifact"] = {
        "provider": "ollama",
        "model": model,
        "model_digest": canonical_model_digest,
        "recipe_digest": recipe.digest,
    }
    extraction["artifact_recheck_seconds"] = extraction.get(
        "artifact_recheck_seconds", 60
    )
    if database_path is not None:
        candidate["database_path"] = str(Path(database_path).expanduser().resolve())
    if socket_path is not None:
        candidate["socket_path"] = str(Path(socket_path).expanduser().resolve())

    try:
        encoded = _atomic_private_json(output, candidate)
    except Exception as exc:
        if isinstance(exc, ActivationPreparationError):
            raise
        raise ActivationPreparationError(
            "candidate configuration failed strict validation"
        ) from exc
    return StagedActivationConfig(
        schema_version=ACTIVATION_CONFIG_VERSION,
        status="staged-not-activated",
        config_path=str(output),
        config_sha256=f"sha256:{hashlib.sha256(encoded).hexdigest()}",
        model=model,
        model_identity=host.model_identity,
        prompt_identity=host.prompt_identity,
        recipe_version=InferenceRecipe.VERSION,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a private pinned Enfold candidate configuration."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    stage = subcommands.add_parser("stage-config")
    stage.add_argument("base_config", type=Path)
    stage.add_argument("output_config", type=Path)
    stage.add_argument("--candidate-executable", type=Path, required=True)
    stage.add_argument("--model-digest", required=True)
    stage.add_argument("--database-path", type=Path)
    stage.add_argument("--socket-path", type=Path)
    args = parser.parse_args(argv)
    try:
        report = stage_activation_config(
            args.base_config,
            args.output_config,
            candidate_executable=args.candidate_executable,
            model_digest=args.model_digest,
            database_path=args.database_path,
            socket_path=args.socket_path,
        )
    except ActivationPreparationError as exc:
        parser.error(str(exc))
    print(json.dumps(asdict(report), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
