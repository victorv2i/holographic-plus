"""Deterministic, credential-free attestation of an extraction recipe.

Model extraction is defined by more than a mutable model name.  The adapter
command and its bounded runtime configuration, the model identity, and the
exact prompt, schema, and adapter source artifacts all affect the result.  This
module represents those inputs as a canonical manifest and hashes that
manifest without opening files, consulting live configuration, or accepting
credential values.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import hmac
import importlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any, ClassVar, Sequence


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTITY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_ARGUMENT_PARTS = (
    "secret",
    "credential",
    "key",
    "token",
    "password",
    "auth",
    "bearer",
)


class ExtractorArtifactError(RuntimeError):
    """The configured inference recipe could not be attested."""


def require_sha256_digest(value: object, *, name: str = "artifact digest") -> str:
    """Return a canonical SHA-256 identity or fail closed."""

    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ExtractorArtifactError(
            f"{name} must be sha256:<64 lowercase hexadecimal characters>"
        )
    return value


def digest_bytes(value: bytes) -> str:
    """Return the canonical digest of caller-supplied artifact bytes."""

    if not isinstance(value, bytes):
        raise TypeError("artifact bytes must be bytes")
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def digest_file(path: str | Path) -> str:
    """Stream a regular local artifact into a canonical digest."""

    artifact = Path(path)
    if not artifact.is_file():
        raise ExtractorArtifactError("artifact file cannot be measured")
    digest = hashlib.sha256()
    try:
        with artifact.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise ExtractorArtifactError("artifact file cannot be measured") from exc
    return f"sha256:{digest.hexdigest()}"


def digest_text(value: str) -> str:
    """Return the canonical UTF-8 digest of caller-supplied artifact text."""

    if not isinstance(value, str):
        raise TypeError("artifact text must be a string")
    return digest_bytes(value.encode("utf-8"))


def digest_canonical_json(value: Any) -> str:
    """Hash a JSON artifact after deterministic structural serialization."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise ExtractorArtifactError("artifact is not canonical JSON") from exc
    return digest_bytes(encoded)


@dataclass(frozen=True, slots=True)
class ArtifactComponent:
    """A named prompt, schema, or source artifact represented only by digest."""

    identity: str
    digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not _IDENTITY.fullmatch(
            self.identity
        ):
            raise ExtractorArtifactError("component identity is invalid")
        require_sha256_digest(self.digest, name="component digest")

    def manifest(self) -> dict[str, str]:
        return {"identity": self.identity, "digest": self.digest}


@dataclass(frozen=True, slots=True)
class InferenceRecipe:
    """Credential-free inputs that determine one extractor inference recipe."""

    VERSION: ClassVar[int] = 1

    adapter_command: tuple[str, ...]
    timeout_seconds: float
    terminate_grace_seconds: float
    max_input_bytes: int
    max_output_bytes: int
    max_error_bytes: int
    environment_names: tuple[str, ...]
    model_identity: str
    model_artifact_digest: str
    prompt: ArtifactComponent
    schema: ArtifactComponent
    source: ArtifactComponent

    def __post_init__(self) -> None:
        command = tuple(self.adapter_command)
        if not command or not all(
            isinstance(item, str) and item and "\0" not in item for item in command
        ):
            raise ExtractorArtifactError("adapter command is invalid")
        object.__setattr__(self, "adapter_command", command)

        for name in ("timeout_seconds", "terminate_grace_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ExtractorArtifactError(f"{name} must be positive and finite")
            object.__setattr__(self, name, float(value))
        for name in ("max_input_bytes", "max_output_bytes", "max_error_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ExtractorArtifactError(f"{name} must be a positive integer")

        environment_names = tuple(sorted(set(self.environment_names)))
        if not all(
            isinstance(name, str) and _ENV_NAME.fullmatch(name)
            for name in environment_names
        ):
            raise ExtractorArtifactError("environment name is invalid")
        object.__setattr__(self, "environment_names", environment_names)

        if not isinstance(self.model_identity, str) or not _IDENTITY.fullmatch(
            self.model_identity
        ):
            raise ExtractorArtifactError("model identity is invalid")
        require_sha256_digest(
            self.model_artifact_digest, name="model artifact digest"
        )
        for name in ("prompt", "schema", "source"):
            if not isinstance(getattr(self, name), ArtifactComponent):
                raise TypeError(f"{name} must be an ArtifactComponent")

    def _manifest(self, *, redact_command: bool) -> dict[str, Any]:
        command = list(self.adapter_command)
        if redact_command:
            redact_next = False
            for index, argument in enumerate(command):
                normalized = argument.replace("-", "").replace("_", "").casefold()
                flagged = "sk-" in argument.casefold() or any(
                    part in normalized for part in _SECRET_ARGUMENT_PARTS
                )
                if redact_next or flagged:
                    command[index] = "<redacted>"
                redact_next = (
                    flagged and argument.startswith("-") and "=" not in argument
                )
        return {
            "adapter": {
                "command": command,
                "config": {
                    "environment_names": list(self.environment_names),
                    "max_error_bytes": self.max_error_bytes,
                    "max_input_bytes": self.max_input_bytes,
                    "max_output_bytes": self.max_output_bytes,
                    "terminate_grace_seconds": self.terminate_grace_seconds,
                    "timeout_seconds": self.timeout_seconds,
                },
            },
            "model": {
                "identity": self.model_identity,
                "artifact_digest": self.model_artifact_digest,
            },
            "prompt": self.prompt.manifest(),
            "schema": self.schema.manifest(),
            "source": self.source.manifest(),
            "version": self.VERSION,
        }

    def manifest(self) -> dict[str, Any]:
        """Return the public canonical recipe; it can never contain credentials."""

        return self._manifest(redact_command=True)

    @property
    def digest(self) -> str:
        return digest_canonical_json(self._manifest(redact_command=False))


@dataclass(frozen=True, slots=True)
class ExtractorArtifactAttestation:
    """A verified inference-recipe identity safe for health and provenance."""

    digest: str
    recipe_version: int = InferenceRecipe.VERSION

    def __post_init__(self) -> None:
        require_sha256_digest(self.digest, name="extractor artifact digest")
        if self.recipe_version != InferenceRecipe.VERSION:
            raise ExtractorArtifactError("extractor recipe version is unsupported")

    def safe_state(self) -> dict[str, object]:
        return {
            "status": "verified",
            "recipe_version": self.recipe_version,
        }


def attest_inference_recipe(
    recipe: InferenceRecipe, *, expected_digest: str
) -> ExtractorArtifactAttestation:
    """Verify a pinned recipe digest without resolving any external state."""

    if not isinstance(recipe, InferenceRecipe):
        raise TypeError("recipe must be an InferenceRecipe")
    expected = require_sha256_digest(
        expected_digest, name="expected extractor artifact digest"
    )
    observed = recipe.digest
    if not hmac.compare_digest(observed, expected):
        raise ExtractorArtifactError("extractor inference recipe digest does not match")
    return ExtractorArtifactAttestation(observed)


_BUNDLED_OLLAMA_MODULES = (
    "enfold.ollama_extractor_child",
    "enfold.host_extractor",
    "enfold.extraction_contract",
    "enfold.extraction_spans",
)


def is_bundled_ollama_command(argv: Sequence[str]) -> bool:
    """Return whether argv selects Enfold's bundled local Ollama child."""

    command = tuple(argv)
    if not command:
        return False
    if Path(command[0]).name.removesuffix(".exe") == "enfold-ollama-extractor":
        return True
    return any(
        option == "-m" and module == "enfold.ollama_extractor_child"
        for option, module in zip(command, command[1:])
    )


def bundled_ollama_components(
    adapter_executable: str | Path,
) -> dict[str, ArtifactComponent]:
    """Measure the installed bundled Ollama prompt, schema, and source bundle.

    The executable and every Enfold module that defines the child-side prompt,
    schema, span construction, or transport are read as bytes.  This is an
    offline local measurement; it neither starts the adapter nor contacts
    Ollama.
    """

    executable = Path(adapter_executable)
    if not executable.is_absolute():
        raise ExtractorArtifactError("adapter executable must be absolute")
    try:
        executable_digest = digest_file(executable)
        executable_size = executable.stat().st_size
    except (OSError, ExtractorArtifactError) as exc:
        raise ExtractorArtifactError("adapter executable cannot be measured") from exc
    if executable_size <= 0:
        raise ExtractorArtifactError("adapter executable cannot be measured")
    source_paths = _foreign_bundled_ollama_source_paths(executable)

    modules: dict[str, Any] = {}
    source_manifest = {"adapter_executable": executable_digest}
    for module_name in _BUNDLED_OLLAMA_MODULES:
        if source_paths is None:
            module = importlib.import_module(module_name)
            module_file = getattr(module, "__file__", None)
            if not isinstance(module_file, str):
                raise ExtractorArtifactError(
                    "bundled adapter source cannot be measured"
                )
            path = Path(module_file)
            if path.suffix in {".pyc", ".pyo"}:
                source_path = path.with_suffix(".py")
                if source_path.exists():
                    path = source_path
            modules[module_name] = module
        else:
            path = source_paths[module_name]
        try:
            source_manifest[module_name] = digest_file(path)
        except ExtractorArtifactError as exc:
            raise ExtractorArtifactError(
                "bundled adapter source cannot be measured"
            ) from exc

    contract = modules.get("enfold.extraction_contract")
    if contract is None:
        prompt_identity, system_prompt, proposal_schema = (
            _foreign_extraction_contract(
                source_paths["enfold.extraction_contract"]
            )
        )
    else:
        prompt_identity = contract.PROMPT_IDENTITY
        system_prompt = contract.SYSTEM_PROMPT
        proposal_schema = contract.PROPOSAL_SCHEMA
    return {
        "prompt": ArtifactComponent(
            str(prompt_identity), digest_text(str(system_prompt))
        ),
        "schema": ArtifactComponent(
            f"{prompt_identity}-proposal-schema",
            digest_canonical_json(proposal_schema),
        ),
        "source": ArtifactComponent(
            "enfold-ollama-extractor-source-v1",
            digest_canonical_json(source_manifest),
        ),
    }


def _foreign_bundled_ollama_source_paths(
    executable: Path,
) -> dict[str, Path] | None:
    """Resolve foreign bundled sources, or select current-process imports."""

    interpreter = executable
    try:
        with executable.open("rb") as stream:
            first_line = stream.readline(4096)
    except OSError as exc:
        raise ExtractorArtifactError("adapter executable cannot be measured") from exc
    if first_line.startswith(b"#!"):
        try:
            shebang = shlex.split(first_line[2:].decode("utf-8").strip())
        except (UnicodeDecodeError, ValueError) as exc:
            raise ExtractorArtifactError(
                "adapter executable installation cannot be verified"
            ) from exc
        if not shebang or not Path(shebang[0]).is_absolute():
            raise ExtractorArtifactError(
                "adapter executable installation cannot be verified"
            )
        interpreter = Path(shebang[0])
    current = Path(sys.executable)
    current_environment = (
        current.is_absolute()
        and os.path.normcase(os.path.abspath(interpreter.parent))
        == os.path.normcase(os.path.abspath(current.parent))
        and os.path.normcase(os.path.abspath(executable.parent))
        == os.path.normcase(os.path.abspath(interpreter.parent))
    )
    if current_environment and executable == interpreter:
        return None

    prefixes = [interpreter.parent.parent]
    if executable != interpreter:
        prefixes.extend((executable.parent.parent, executable.parent))
    package_roots: list[Path] = []
    for prefix in prefixes:
        candidates = [
            *prefix.glob("lib/python*/site-packages/enfold"),
            *prefix.glob("lib/python*/dist-packages/enfold"),
            prefix / "Lib" / "site-packages" / "enfold",
            prefix / "site-packages" / "enfold",
            prefix / "enfold",
        ]
        for candidate in candidates:
            if candidate.is_dir() and candidate not in package_roots:
                package_roots.append(candidate)

    installations: list[dict[str, Path]] = []
    for package_root in package_roots:
        source_paths = {
            module_name: package_root / f"{module_name.rsplit('.', 1)[1]}.py"
            for module_name in _BUNDLED_OLLAMA_MODULES
        }
        if all(path.is_file() for path in source_paths.values()):
            installations.append(source_paths)
    if current_environment:
        if not installations:
            return None
        current_package_root = Path(__file__).resolve().parent
        if any(
            all(path.resolve().parent == current_package_root for path in paths.values())
            for paths in installations
        ):
            return None
    if len(installations) != 1:
        raise ExtractorArtifactError(
            "foreign Enfold installation modules cannot be located unambiguously"
        )
    return installations[0]


def _foreign_extraction_contract(path: Path) -> tuple[str, str, Any]:
    """Read literal prompt and schema constants without importing foreign code."""

    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise ExtractorArtifactError(
            "foreign extraction contract constants cannot be measured"
        ) from exc

    constants: dict[str, Any] = {}
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom):
            constants.update(_foreign_imported_constants(path, statement))
            continue
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
            value = statement.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        try:
            constants[target.id] = ast.literal_eval(
                _ForeignConstantResolver(constants).visit(value)
            )
        except (ValueError, TypeError):
            constants.pop(target.id, None)
            continue

    prompt_identity = constants.get("PROMPT_IDENTITY")
    system_prompt = constants.get("SYSTEM_PROMPT")
    proposal_schema = constants.get("PROPOSAL_SCHEMA")
    if (
        not isinstance(prompt_identity, str)
        or not isinstance(system_prompt, str)
        or not isinstance(proposal_schema, dict)
    ):
        raise ExtractorArtifactError(
            "foreign extraction contract constants cannot be measured"
        )
    return prompt_identity, system_prompt, proposal_schema


def _foreign_imported_constants(
    contract_path: Path, statement: ast.ImportFrom
) -> dict[str, Any]:
    """Read direct scalar imports from one foreign Enfold sibling module."""

    module = statement.module
    if statement.level == 1 and module is not None and module.isidentifier():
        module_name = module
    elif (
        statement.level == 0
        and module is not None
        and module.startswith("enfold.")
        and module.count(".") == 1
        and module.removeprefix("enfold.").isidentifier()
    ):
        module_name = module.removeprefix("enfold.")
    else:
        return {}

    module_path = contract_path.parent / f"{module_name}.py"
    try:
        source = module_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_path))
    except (OSError, UnicodeError, SyntaxError):
        return {}

    literals: dict[str, Any] = {}
    for imported in statement.names:
        if not imported.name.isidentifier():
            continue
        values: list[ast.expr] = []
        for candidate in tree.body:
            if (
                isinstance(candidate, ast.Assign)
                and len(candidate.targets) == 1
                and isinstance(candidate.targets[0], ast.Name)
                and candidate.targets[0].id == imported.name
            ):
                values.append(candidate.value)
            elif (
                isinstance(candidate, ast.AnnAssign)
                and isinstance(candidate.target, ast.Name)
                and candidate.target.id == imported.name
                and candidate.value is not None
            ):
                values.append(candidate.value)
        if len(values) != 1:
            continue
        try:
            value = ast.literal_eval(values[0])
        except (ValueError, TypeError):
            continue
        if isinstance(value, (str, int, float, bool)):
            literals[imported.asname or imported.name] = value
    return literals


class _ForeignConstantResolver(ast.NodeTransformer):
    """Substitute previously parsed scalar constants for literal evaluation."""

    def __init__(self, constants: dict[str, Any]) -> None:
        self._constants = constants

    def visit_Name(self, node: ast.Name) -> ast.expr:
        value = self._constants.get(node.id)
        if value is None or not isinstance(value, (str, int, float, bool)):
            return node
        return ast.copy_location(ast.Constant(value=value), node)
