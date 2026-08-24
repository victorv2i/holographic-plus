from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import stat
import tomllib

import pytest

from enfold.bootstrap import BootstrapError, bootstrap, main
from enfold.schema import SUPPORTED_SCHEMA_VERSION, schema_version
from enfold.server import load_config, main as server_main


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_bootstrap_creates_private_valid_minimal_instance(tmp_path: Path) -> None:
    """A new user can create a ready-to-check local instance without source paths."""

    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"

    report = bootstrap(
        config_dir=config_dir,
        data_dir=data_dir,
        owner_client_id="first-install",
    )

    config_path = config_dir / "server.json"
    database_path = data_dir / "memory.db"
    socket_path = data_dir / "enfold.sock"
    assert report.config_path == config_path
    assert report.database_path == database_path
    assert report.socket_path == socket_path
    assert report.owner_client_id == "first-install"
    assert _mode(config_dir) == 0o700
    assert _mode(data_dir) == 0o700
    assert _mode(config_path) == 0o600
    assert _mode(database_path) == 0o600
    assert not socket_path.exists()

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    assert raw == {
        "database_path": str(database_path),
        "extraction": {"mode": "disabled"},
        "grants": {"first-install": ["private"]},
        "retrieval": {
            "allow_nonproduction": True,
            "dimensions": 256,
            "mode": "ci",
            "vector_backend": "brute",
        },
        "socket_path": str(socket_path),
    }

    loaded = load_config(config_path)
    assert loaded.database_path == database_path
    assert loaded.socket_path == socket_path
    assert loaded.grants == {"first-install": ("private",)}
    with sqlite3.connect(database_path) as conn:
        assert schema_version(conn) == SUPPORTED_SCHEMA_VERSION
    assert server_main(["--config", str(config_path), "check"]) == 0


def test_cli_init_emits_machine_readable_instance_report(
    tmp_path: Path, capsys
) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"

    result = main([
        "init",
        "--config-dir", str(config_dir),
        "--data-dir", str(data_dir),
        "--client-id", "cli-install",
    ])

    assert result == 0
    assert json.loads(capsys.readouterr().out) == {
        "config_path": str(config_dir / "server.json"),
        "database_path": str(data_dir / "memory.db"),
        "owner_client_id": "cli-install",
        "socket_path": str(data_dir / "enfold.sock"),
    }


def test_bootstrap_refuses_to_overwrite_an_existing_instance(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    bootstrap(config_dir=config_dir, data_dir=data_dir)
    original_config = (config_dir / "server.json").read_bytes()
    original_database = (data_dir / "memory.db").read_bytes()

    with pytest.raises(BootstrapError, match="already exists"):
        bootstrap(config_dir=config_dir, data_dir=data_dir)

    assert (config_dir / "server.json").read_bytes() == original_config
    assert (data_dir / "memory.db").read_bytes() == original_database


def test_bootstrap_rejects_a_symlinked_destination(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked_config = tmp_path / "linked-config"
    linked_config.symlink_to(target, target_is_directory=True)

    with pytest.raises(BootstrapError, match="symlink"):
        bootstrap(config_dir=linked_config, data_dir=tmp_path / "data")

    assert not (target / "server.json").exists()


def test_packaging_exports_the_discoverable_init_command() -> None:
    project = Path(__file__).parents[1] / "pyproject.toml"
    with project.open("rb") as handle:
        metadata = tomllib.load(handle)

    assert metadata["project"]["scripts"]["enfold"] == "enfold.bootstrap:main"


def test_bootstrap_removes_exact_artifacts_after_write_or_fsync_failure(
    tmp_path: Path, monkeypatch
) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    real_write = os.write

    def fail_nonempty_write(descriptor: int, data: bytes) -> int:
        if data:
            raise OSError("injected write failure")
        return real_write(descriptor, data)

    monkeypatch.setattr("enfold.bootstrap.os.write", fail_nonempty_write)
    with pytest.raises(BootstrapError, match="cannot write configuration"):
        bootstrap(config_dir=config_dir, data_dir=data_dir)
    assert not (config_dir / "server.json").exists()
    assert not (data_dir / "memory.db").exists()

    monkeypatch.undo()
    monkeypatch.setattr(
        "enfold.bootstrap.os.fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("injected fsync failure")),
    )
    with pytest.raises(BootstrapError, match="cannot write database"):
        bootstrap(config_dir=config_dir, data_dir=data_dir)
    assert not (config_dir / "server.json").exists()
    assert not (data_dir / "memory.db").exists()


def test_bootstrap_rejects_client_ids_that_protocol_clients_cannot_use(tmp_path: Path):
    with pytest.raises(BootstrapError, match="protocol client id"):
        bootstrap(
            config_dir=tmp_path / "config",
            data_dir=tmp_path / "data",
            owner_client_id="owner client",
        )

    assert not (tmp_path / "config").exists()
    assert not (tmp_path / "data").exists()


def test_bootstrap_rejects_nonsticky_writable_destination_ancestors(tmp_path: Path):
    unsafe = tmp_path / "shared"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)

    with pytest.raises(BootstrapError, match="writable ancestor"):
        bootstrap(
            config_dir=unsafe / "config",
            data_dir=unsafe / "data",
        )
