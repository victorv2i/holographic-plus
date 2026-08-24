from __future__ import annotations

import importlib.metadata
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from zipfile import ZipFile

from enfold import server


ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_are_aligned(monkeypatch):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    manifest = (ROOT / "integrations/hermes_enfold_v1/plugin.yaml").read_text(
        encoding="utf-8"
    )

    def missing_distribution(_name):
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(server.importlib.metadata, "version", missing_distribution)

    assert project["project"]["version"] == "0.8.0"
    assert "memory_eval.fixtures" in project["tool"]["setuptools"]["packages"]
    assert "version: 0.8.0\n" in manifest
    assert server._version() == "0.8.0"


def test_built_wheel_contains_packaged_plugin_manifest(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    for filename in ("pyproject.toml", "README.md"):
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
