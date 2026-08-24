from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

from enfold import extraction_contract, extraction_processor
from enfold.extractor_artifact import (
    ArtifactComponent,
    ExtractorArtifactError,
    InferenceRecipe,
    attest_inference_recipe,
    bundled_ollama_components,
    digest_bytes,
    digest_canonical_json,
    digest_file,
    digest_text,
)
from enfold.host_extractor import HostExtractorConfig


_MODEL_DIGEST = "sha256:" + "b" * 64


def _components():
    return {
        "prompt": ArtifactComponent(
            "durable-memory-v1", digest_text("system prompt")
        ),
        "schema": ArtifactComponent(
            "proposal-schema-v1",
            digest_canonical_json(
                {
                    "required": ["proposals"],
                    "type": "object",
                }
            ),
        ),
        "source": ArtifactComponent(
            "enfold.ollama_extractor_child:v1", digest_bytes(b"source bytes")
        ),
    }


def _config(**changes):
    values = {
        "argv": (
            "/opt/enfold/python",
            "-m",
            "enfold.ollama_extractor_child",
            "--model",
            "qwen3:30b",
        ),
        "model_identity": "ollama:qwen3-30b",
        "prompt_identity": "durable-memory-v1",
        "timeout_seconds": 180,
        "terminate_grace_seconds": 2,
        "max_input_bytes": 16_384,
        "max_output_bytes": 65_536,
        "max_error_bytes": 16_384,
        "environment": {
            "MODEL_ENDPOINT": "http://127.0.0.1:11434/api/chat",
            "OPENAI_API_KEY": "must-not-enter-the-recipe",
        },
    }
    values.update(changes)
    return HostExtractorConfig(**values)


def test_host_recipe_is_deterministic_and_never_contains_environment_values():
    recipe = _config().inference_recipe(
        model_artifact_digest=_MODEL_DIGEST, **_components()
    )
    reordered = _config(
        timeout_seconds=180.0,
        environment={
            "OPENAI_API_KEY": "a-different-secret",
            "MODEL_ENDPOINT": "a-different-endpoint",
        },
    ).inference_recipe(model_artifact_digest=_MODEL_DIGEST, **_components())

    assert recipe.digest == reordered.digest
    assert recipe.environment_names == ("MODEL_ENDPOINT", "OPENAI_API_KEY")
    serialized = repr(recipe.manifest())
    assert "must-not-enter-the-recipe" not in serialized
    assert "a-different-secret" not in serialized
    assert "127.0.0.1" not in serialized


def test_recipe_manifest_redacts_argv_credentials_but_digest_uses_real_command():
    first_secret = "sk-" + "first-secret-value"
    second_secret = "sk-" + "second-secret-value"
    first = _config(
        argv=("/opt/enfold/python", "--api-key", first_secret)
    ).inference_recipe(model_artifact_digest=_MODEL_DIGEST, **_components())
    second = _config(
        argv=("/opt/enfold/python", "--api-key", second_secret)
    ).inference_recipe(model_artifact_digest=_MODEL_DIGEST, **_components())

    assert first.manifest()["adapter"]["command"] == [
        "/opt/enfold/python",
        "<redacted>",
        "<redacted>",
    ]
    assert first_secret not in repr(first.manifest())
    assert first.digest != second.digest


@pytest.mark.parametrize(
    "option",
    [
        "--secret",
        "--credential",
        "--access-key",
        "--refresh_token",
        "--password-file",
        "--auth-header",
    ],
)
def test_recipe_manifest_redacts_broad_secret_shaped_argv_options(option):
    recipe = _config(
        argv=("/opt/enfold/python", option, "sensitive-value")
    ).inference_recipe(model_artifact_digest=_MODEL_DIGEST, **_components())

    assert recipe.manifest()["adapter"]["command"] == [
        "/opt/enfold/python",
        "<redacted>",
        "<redacted>",
    ]


@pytest.mark.parametrize(
    "change",
    [
        {"argv": ("/opt/enfold/different-extractor",)},
        {"timeout_seconds": 181},
        {"max_input_bytes": 16_385},
        {"model_identity": "ollama:qwen3-32b"},
        {"prompt_identity": "durable-memory-v3"},
        {"environment": {"ADAPTER_MODE": "strict"}},
    ],
)
def test_adapter_command_config_and_model_identity_change_recipe_digest(change):
    baseline = _config().inference_recipe(
        model_artifact_digest=_MODEL_DIGEST, **_components()
    ).digest

    components = _components()
    if "prompt_identity" in change:
        components["prompt"] = ArtifactComponent(
            change["prompt_identity"], components["prompt"].digest
        )
    assert _config(**change).inference_recipe(
        model_artifact_digest=_MODEL_DIGEST, **components
    ).digest != baseline


def test_prompt_artifact_identity_must_match_host_configuration():
    components = _components()
    components["prompt"] = ArtifactComponent(
        "different-prompt-v1", components["prompt"].digest
    )

    with pytest.raises(ExtractorArtifactError, match="does not match"):
        _config().inference_recipe(
            model_artifact_digest=_MODEL_DIGEST, **components
        )


@pytest.mark.parametrize("component", ["prompt", "schema", "source"])
def test_prompt_schema_and_source_artifacts_change_recipe_digest(component):
    components = _components()
    baseline = _config().inference_recipe(
        model_artifact_digest=_MODEL_DIGEST, **components
    ).digest
    current = components[component]
    components[component] = ArtifactComponent(
        current.identity, digest_text(f"different {component}")
    )

    assert _config().inference_recipe(
        model_artifact_digest=_MODEL_DIGEST, **components
    ).digest != baseline


@pytest.mark.parametrize("component", ["prompt", "schema", "source"])
def test_prompt_schema_and_source_identities_change_recipe_digest(component):
    components = _components()
    baseline = _config().inference_recipe(
        model_artifact_digest=_MODEL_DIGEST, **components
    ).digest
    current = components[component]
    components[component] = ArtifactComponent(
        f"{current.identity}-v2", current.digest
    )
    if component == "prompt":
        config = _config(prompt_identity=components[component].identity)
    else:
        config = _config()

    assert config.inference_recipe(
        model_artifact_digest=_MODEL_DIGEST, **components
    ).digest != baseline


def test_schema_digest_is_independent_of_mapping_key_order():
    assert digest_canonical_json({"a": 1, "b": 2}) == digest_canonical_json(
        {"b": 2, "a": 1}
    )


def test_file_digest_matches_exact_bytes(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"exact adapter bytes")

    assert digest_file(artifact) == digest_bytes(b"exact adapter bytes")


def test_recipe_attestation_verifies_exact_digest_and_exposes_only_safe_state():
    recipe = _config().inference_recipe(
        model_artifact_digest=_MODEL_DIGEST, **_components()
    )

    attestation = attest_inference_recipe(recipe, expected_digest=recipe.digest)

    assert attestation.safe_state() == {
        "status": "verified",
        "recipe_version": InferenceRecipe.VERSION,
    }

    with pytest.raises(ExtractorArtifactError, match="does not match"):
        attest_inference_recipe(
            recipe, expected_digest="sha256:" + "0" * 64
        )


@pytest.mark.parametrize(
    "digest",
    ["0" * 64, "sha256:" + "A" * 64, "sha256:" + "0" * 63],
)
def test_component_digests_must_be_canonical_sha256(digest):
    with pytest.raises(ExtractorArtifactError, match="component digest"):
        ArtifactComponent("prompt-v1", digest)


def test_model_artifact_digest_is_part_of_the_recipe():
    baseline = _config().inference_recipe(
        model_artifact_digest=_MODEL_DIGEST, **_components()
    )
    different = _config().inference_recipe(
        model_artifact_digest="sha256:" + "c" * 64, **_components()
    )

    assert baseline.digest != different.digest
    assert baseline.manifest()["model"] == {
        "identity": "ollama:qwen3-30b",
        "artifact_digest": _MODEL_DIGEST,
    }


def test_bundled_ollama_components_measure_installed_contract_and_executable():
    components = bundled_ollama_components(sys.executable)

    assert components["prompt"].identity == "durable-memory-v3"
    assert components["schema"].identity == "durable-memory-v3-proposal-schema"
    assert components["source"].identity == "enfold-ollama-extractor-source-v1"


def test_bundled_components_use_current_installation_fast_path(monkeypatch):
    imported = []
    real_import_module = __import__("importlib").import_module

    def tracking_import(module_name):
        imported.append(module_name)
        return real_import_module(module_name)

    monkeypatch.setattr(
        "enfold.extractor_artifact.importlib.import_module", tracking_import
    )

    bundled_ollama_components(sys.executable)

    assert imported == [
        "enfold.ollama_extractor_child",
        "enfold.host_extractor",
        "enfold.extraction_contract",
        "enfold.extraction_spans",
    ]


def _foreign_installation(tmp_path, *, interpreter_path=None):
    prefix = tmp_path / "other-enfold"
    bin_dir = prefix / "bin"
    bin_dir.mkdir(parents=True)
    interpreter = interpreter_path or bin_dir / "python"
    if interpreter_path is None:
        interpreter.write_bytes(b"different python installation")
    executable = bin_dir / "enfold-ollama-extractor"
    executable.write_text(f"#!{interpreter}\n", encoding="utf-8")
    package = prefix / "lib" / "python3.13" / "site-packages" / "enfold"
    package.mkdir(parents=True)
    paths = {}
    for module_name in (
        "enfold.ollama_extractor_child",
        "enfold.host_extractor",
        "enfold.extraction_contract",
        "enfold.extraction_spans",
    ):
        path = package / f"{module_name.rsplit('.', 1)[1]}.py"
        source = f"foreign source for {module_name}\n"
        if module_name == "enfold.extraction_contract":
            source = """PROMPT_IDENTITY = "foreign-memory-v9"
SYSTEM_PROMPT = "Foreign prompt text."
MAX_ITEMS = 7
PROPOSAL_SCHEMA = {"type": "object", "maxItems": MAX_ITEMS}
"""
        path.write_text(source, encoding="utf-8")
        paths[module_name] = path
    return executable, paths


def test_bundled_components_measure_foreign_installation_sources(tmp_path):
    executable, paths = _foreign_installation(tmp_path)
    expected_manifest = {"adapter_executable": digest_file(executable)}
    expected_manifest.update(
        (module_name, digest_file(path)) for module_name, path in paths.items()
    )

    components = bundled_ollama_components(executable)

    assert components["source"].digest == digest_canonical_json(expected_manifest)
    assert components["prompt"] == ArtifactComponent(
        "foreign-memory-v9", digest_text("Foreign prompt text.")
    )
    assert components["schema"].digest == digest_canonical_json(
        {"type": "object", "maxItems": 7}
    )
    assert components["schema"].identity == "foreign-memory-v9-proposal-schema"


@pytest.mark.parametrize("absolute_import", [False, True])
def test_bundled_components_measure_real_foreign_contract_imports(
    tmp_path, absolute_import
):
    executable, paths = _foreign_installation(tmp_path)
    package = paths["enfold.extraction_contract"].parent
    shutil.copyfile(
        extraction_contract.__file__, paths["enfold.extraction_contract"]
    )
    if absolute_import:
        contract_source = paths["enfold.extraction_contract"].read_text(
            encoding="utf-8"
        )
        paths["enfold.extraction_contract"].write_text(
            contract_source.replace(
                "from .extraction_processor import MAX_EXTRACTED_MEMORIES",
                "from enfold.extraction_processor import MAX_EXTRACTED_MEMORIES",
            ),
            encoding="utf-8",
        )
    shutil.copyfile(
        extraction_processor.__file__, package / "extraction_processor.py"
    )

    foreign = bundled_ollama_components(executable)
    current = bundled_ollama_components(sys.executable)

    assert foreign["prompt"] == current["prompt"]
    assert foreign["schema"] == current["schema"]


@pytest.mark.parametrize(
    "provider_source",
    [
        "MAX_EXTRACTED_MEMORIES = determine_limit()\n",
        "from .limits import MAX_EXTRACTED_MEMORIES\n",
    ],
)
def test_foreign_contract_imports_fail_closed_beyond_one_literal_level(
    tmp_path, provider_source
):
    executable, paths = _foreign_installation(tmp_path)
    paths["enfold.extraction_contract"].write_text(
        "from .extraction_processor import MAX_EXTRACTED_MEMORIES\n"
        "PROMPT_IDENTITY = 'foreign-memory-v9'\n"
        "SYSTEM_PROMPT = 'Foreign prompt text.'\n"
        "PROPOSAL_SCHEMA = {'maxItems': MAX_EXTRACTED_MEMORIES}\n",
        encoding="utf-8",
    )
    processor_path = (
        paths["enfold.extraction_contract"].parent / "extraction_processor.py"
    )
    processor_path.write_text(provider_source, encoding="utf-8")

    with pytest.raises(ExtractorArtifactError, match="constants cannot be measured"):
        bundled_ollama_components(executable)


def test_direct_script_with_current_interpreter_measures_its_foreign_tree(tmp_path):
    executable, _paths = _foreign_installation(
        tmp_path, interpreter_path=Path(sys.executable)
    )

    components = bundled_ollama_components(executable)

    assert components["prompt"].identity == "foreign-memory-v9"


def test_current_directory_match_does_not_override_foreign_module_tree(
    tmp_path, monkeypatch
):
    executable, _paths = _foreign_installation(tmp_path)
    monkeypatch.setattr(
        "enfold.extractor_artifact.sys.executable",
        str(executable.parent / "python"),
    )

    components = bundled_ollama_components(executable)

    assert components["prompt"].identity == "foreign-memory-v9"


@pytest.mark.parametrize(
    "source",
    [
        "PROMPT_IDENTITY = 'incomplete'\n",
        "PROMPT_IDENTITY =\n",
        "PROMPT_IDENTITY = build_identity()\nSYSTEM_PROMPT = 'x'\nPROPOSAL_SCHEMA = {}\n",
    ],
)
def test_foreign_contract_constants_fail_closed_when_not_literal(tmp_path, source):
    executable, paths = _foreign_installation(tmp_path)
    paths["enfold.extraction_contract"].write_text(source, encoding="utf-8")

    with pytest.raises(ExtractorArtifactError, match="constants cannot be measured"):
        bundled_ollama_components(executable)


def test_bundled_components_fail_closed_for_unresolvable_foreign_installation(
    tmp_path,
):
    bin_dir = tmp_path / "other-enfold" / "bin"
    bin_dir.mkdir(parents=True)
    interpreter = bin_dir / "python"
    interpreter.write_bytes(b"different python installation")
    executable = bin_dir / "enfold-ollama-extractor"
    executable.write_text(f"#!{interpreter}\n", encoding="utf-8")

    with pytest.raises(ExtractorArtifactError, match="modules cannot be located"):
        bundled_ollama_components(executable)


def test_bundled_source_digest_includes_host_supervisor(monkeypatch):
    host_digest = ["a"]

    def measured_digest(path):
        if Path(path).name == "host_extractor.py":
            return "sha256:" + host_digest[0] * 64
        return "sha256:" + "b" * 64

    monkeypatch.setattr("enfold.extractor_artifact.digest_file", measured_digest)
    baseline = bundled_ollama_components(sys.executable)["source"].digest

    host_digest[0] = "c"
    changed = bundled_ollama_components(sys.executable)["source"].digest

    assert changed != baseline
