from __future__ import annotations

import importlib.metadata
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
from zipfile import ZipFile

from enfold import server


ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/victorv2i/enfold"
PINNED_GIT = f"git+{REPO}@v"
# Remote tag from 2026-07-12. Its tree has no enfold / enfold-mcp scripts.
STALE_PUBLISHED_VERSION = "0.8.0"

# Bare PyPI-name installs. A git+ URL or a local path is fine.
_PYPI_NAME_INSTALL = re.compile(
    r"""(?:python3?\s+-m\s+)?(?:pip3?|uvx\s+--from)\s+(?:install\s+)?['\"]?enfold(?:\[[^\]]+\])?['\"]?(?:\s|$)""",
    re.MULTILINE,
)


def test_release_versions_are_aligned(monkeypatch):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = (ROOT / "integrations/hermes_enfold_v1/plugin.yaml").read_text(
        encoding="utf-8"
    )

    def missing_distribution(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(server.importlib.metadata, "version", missing_distribution)

    version = project["project"]["version"]
    assert version != STALE_PUBLISHED_VERSION
    assert "memory_eval.fixtures" in project["tool"]["setuptools"]["packages"]
    assert f"version: {version}\n" in manifest
    assert server._version() == version


def test_project_version_is_not_the_already_published_remote_tag():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    assert version != STALE_PUBLISHED_VERSION


def test_test_extra_includes_sqlite_vec_so_optional_index_tests_run_in_ci():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]["optional-dependencies"]
    assert "sqlite-vec==0.1.9" in extras["test"]
    assert extras["sqlite-vec"] == ["sqlite-vec==0.1.9"]


def test_built_wheel_contains_packaged_plugin_manifest(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for filename in ("pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(ROOT / filename, project / filename)
    for package in ("enfold", "memory_eval"):
        shutil.copytree(ROOT / package, project / package)
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()

    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_dir),
        ],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )

    wheel = next(wheel_dir.glob("enfold-*.whl"))
    with ZipFile(wheel) as archive:
        assert "enfold/plugin.yaml" in archive.namelist()
        assert "memory_eval/fixtures/context_arena.json" in archive.namelist()


def test_wheel_metadata_ships_mcp_by_default_and_keeps_sqlite_vec_optional():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    extras = project["project"]["optional-dependencies"]
    scripts = project["project"]["scripts"]

    assert any(item == "mcp" or item.startswith("mcp") for item in dependencies)
    assert not any("sqlite-vec" in item for item in dependencies)
    assert extras["sqlite-vec"] == ["sqlite-vec==0.1.9"]
    assert "mcp" not in extras
    assert scripts["enfold-mcp"] == "enfold.mcp_launcher:product_main"
    assert scripts["enfold"] == "enfold.bootstrap:main"


def _mcp_requirement(item: str):
    from packaging.requirements import Requirement

    req = Requirement(item)
    assert req.name == "mcp"
    return req


def test_mcp_runtime_dependency_rejects_v2_and_accepts_release_pin():
    # Fresh pip/uvx must not resolve mcp 2.x (FastMCP import path was removed).
    # The range must still accept the release-workflow pin mcp==1.28.1.
    from packaging.version import Version

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    extras = project["project"]["optional-dependencies"]
    mcp_items = [
        item for item in dependencies if item == "mcp" or item.startswith("mcp")
    ]
    assert mcp_items, "mcp must be a required dependency"
    req = _mcp_requirement(mcp_items[0])
    assert req.specifier, "mcp must be version-constrained, not a bare pinless name"
    assert Version("1.28.1") in req.specifier
    assert Version("2.0.0") not in req.specifier
    assert Version("2.1.1") not in req.specifier
    test_mcp = [
        item for item in extras["test"] if item == "mcp" or item.startswith("mcp")
    ]
    if test_mcp:
        test_req = _mcp_requirement(test_mcp[0])
        assert Version("2.1.1") not in test_req.specifier


def test_tree_does_not_name_removed_mcp_extra():
    forbidden = "enfold[" + "mcp]"
    roots = (
        ROOT / "enfold",
        ROOT / "docs",
        ROOT / "tests",
        ROOT / ".github",
    )
    named_files = (
        ROOT / "pyproject.toml",
        ROOT / "README.md",
        ROOT / "CHANGELOG.md",
    )
    hits = []
    for path in named_files:
        text = path.read_text(encoding="utf-8")
        if forbidden in text:
            hits.append(str(path.relative_to(ROOT)))
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".py", ".md", ".toml", ".yml", ".yaml"}:
                continue
            text = path.read_text(encoding="utf-8")
            if forbidden in text:
                hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def _read(*relative: str) -> str:
    return (ROOT.joinpath(*relative)).read_text(encoding="utf-8")


def test_install_docs_do_not_name_a_pypi_package():
    for relative in (
        ("README.md",),
        ("docs", "BOOTSTRAP.md"),
        ("docs", "SERVER_DEPLOYMENT.md"),
    ):
        text = _read(*relative)
        for match in _PYPI_NAME_INSTALL.finditer(text):
            line_end = text.find("\n", match.start())
            line = text[match.start() : None if line_end == -1 else line_end]
            if f"git+{REPO}" in line:
                continue
            raise AssertionError(
                f"{'/'.join(relative)} still names a PyPI package: {match.group(0)!r}"
            )
        assert "uvx --from enfold " not in text
        assert "uvx --from enfold\n" not in text


def test_install_docs_pin_github_git_and_a_version_tag():
    version = tomllib.loads(_read("pyproject.toml"))["project"]["version"]
    tag = f"v{version}"
    readme = _read("README.md")
    bootstrap = _read("docs", "BOOTSTRAP.md")
    deployment = _read("docs", "SERVER_DEPLOYMENT.md")
    releasing = _read("docs", "RELEASING.md")
    changelog = _read("CHANGELOG.md")
    for text in (readme, bootstrap, deployment):
        assert f"{PINNED_GIT}{version}" in text
        assert "@main" not in text
        assert "@HEAD" not in text
    assert f"The current release is `{tag}`" in readme
    assert f"enfold-{version}-py3-none-any.whl" in readme
    assert f"enfold-{version}-py3-none-any.whl" in bootstrap
    assert f"The current project version is `{version}`" in releasing
    assert f"## {version}" in changelog


def test_readme_quickstart_includes_host_mcp_snippets():
    readme = _read("README.md")
    assert "claude mcp add --transport stdio --scope user enfold -- uvx --from " in readme
    assert "[mcp_servers.enfold]" in readme
    assert '"mcpServers"' in readme
    assert "enfold setup --client cursor" in readme
    assert "enfold setup --client claude-code" in readme or "claude-code" in readme
    assert "enfold setup --client codex" in readme or "codex" in readme
    assert "Remember that my test preference is dark mode" in readme


def test_project_metadata_points_at_github_and_declares_license_files():
    project = tomllib.loads(_read("pyproject.toml"))["project"]
    urls = project["urls"]
    assert urls["Homepage"] == REPO
    assert urls["Repository"] == REPO
    assert urls["Issues"] == f"{REPO}/issues"
    assert urls["Changelog"] == f"{REPO}/blob/main/CHANGELOG.md"
    assert project["license"] == "MIT"
    assert "LICENSE" in project["license-files"]


def test_release_workflow_publishes_github_artifacts_not_pypi():
    workflow = _read(".github", "workflows", "release.yml")
    lowered = workflow.lower()
    assert "pypi" not in lowered
    assert "twine" not in lowered
    assert "pypa/gh-action-pypi-publish" not in lowered
    assert "id-token" not in lowered
    assert "PYPI_" not in workflow
    assert "TWINE_" not in workflow
    assert "python -m build" in workflow
    assert "--sdist" in workflow or "python -m build" in workflow
    assert "sha256" in lowered
    assert "gh release create" in workflow
    assert "pytest" in lowered
    assert "override-ini" in workflow or "pythonpath" in lowered
    assert "tags:" in workflow


def test_ci_runs_first_run_path_from_a_built_wheel():
    workflow = _read(".github", "workflows", "tests.yml")
    assert "python -m build" in workflow
    assert "enfold setup" in workflow
    assert "enfold-mcp --self-test" in workflow
    assert "enfold doctor" in workflow
    assert re.search(r"pip install .*enfold-.*\.whl", workflow)
    assert "-e .[test]" in workflow


def test_releasing_doc_pins_version_tag_schema_and_mcp_compat():
    text = _read("docs", "RELEASING.md")
    assert "pyproject.toml" in text
    assert "CHANGELOG.md" in text
    assert "v0." in text or "v${" in text or "`v" in text
    assert "schema" in text.lower()
    assert "SUPPORTED_SCHEMA_VERSION" in text or "schema version 1" in text.lower()
    assert "PROTOCOL_MAJOR" in text or "protocol 1.0" in text.lower()
    assert "migrate" in text.lower()
    assert "PyPI" in text or "pypi" in text
    assert "not published" in text.lower() or "never published" in text.lower() or "not publish" in text.lower()
