from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import tomllib

import pytest

from enfold.bootstrap import (
    BootstrapError,
    bootstrap,
    credential_digest,
    ensure_instance,
    main,
    new_client_token,
    replace_config,
    resolve_socket_path,
    write_client_credential,
)
from enfold.schema import SUPPORTED_SCHEMA_VERSION, schema_version
from enfold.server import load_config, main as server_main


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _overlong_owner_tree(root: Path) -> Path:
    candidate = root
    while len(os.fsencode(os.fspath(candidate / "data" / "enfold.sock"))) <= 107:
        candidate = candidate / ("pad-" + "x" * 40)
    candidate.mkdir(parents=True)
    os.chmod(candidate, 0o700)
    return candidate


def test_bootstrap_keeps_store_and_uses_short_runtime_socket_when_data_dir_is_long(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    long_root = _overlong_owner_tree(tmp_path / "long")
    config_dir = long_root / "config"
    data_dir = long_root / "data"

    report = bootstrap(config_dir=config_dir, data_dir=data_dir)

    assert report.database_path == data_dir / "memory.db"
    assert report.database_path.is_file()
    assert report.socket_path.parent == runtime / "enfold"
    assert report.socket_path.name.endswith(".sock")
    assert len(os.fsencode(os.fspath(report.socket_path))) <= 107
    raw = json.loads(report.config_path.read_text(encoding="utf-8"))
    assert raw["database_path"] == str(data_dir / "memory.db")
    assert raw["socket_path"] == str(report.socket_path)
    assert not (data_dir / "enfold.sock").exists()


def test_long_data_dirs_do_not_share_a_runtime_socket(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    first_root = _overlong_owner_tree(tmp_path / "one")
    second_root = _overlong_owner_tree(tmp_path / "two")
    first = bootstrap(
        config_dir=first_root / "config", data_dir=first_root / "data"
    )
    second = bootstrap(
        config_dir=second_root / "config", data_dir=second_root / "data"
    )
    assert first.socket_path != second.socket_path
    assert first.database_path != second.database_path
    assert first.socket_path.parent == runtime / "enfold"
    assert second.socket_path.parent == runtime / "enfold"
    assert len(os.fsencode(os.fspath(first.socket_path))) <= 107
    assert len(os.fsencode(os.fspath(second.socket_path))) <= 107


def test_runtime_fallback_sockets_distinguish_8_hex_colliding_data_dirs(
    tmp_path: Path, monkeypatch
) -> None:
    runtime = tmp_path / "run"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    preferred = _overlong_owner_tree(tmp_path / "preferred") / "data" / "enfold.sock"
    first_data = Path("/tmp/enfold-collision-store-138396/data")
    second_data = Path("/tmp/enfold-collision-store-167855/data")
    first_digest = hashlib.sha256(os.fsencode(os.fspath(first_data))).hexdigest()
    second_digest = hashlib.sha256(os.fsencode(os.fspath(second_data))).hexdigest()
    assert first_digest[:8] == second_digest[:8] == "f4c00c46"
    assert first_digest != second_digest

    first = resolve_socket_path(
        preferred, data_directory=first_data, allow_runtime_fallback=True
    )
    second = resolve_socket_path(
        preferred, data_directory=second_data, allow_runtime_fallback=True
    )
    assert first != second
    assert first.name != "f4c00c46.sock"
    assert second.name != "f4c00c46.sock"
    assert len(os.fsencode(os.fspath(first))) <= 107
    assert len(os.fsencode(os.fspath(second))) <= 107


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
    token = report.owner_credential
    digest = "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert token
    assert token not in config_path.read_text(encoding="utf-8")
    assert raw == {
        "client_credentials": {"first-install": digest},
        "database_path": str(database_path),
        "extraction": {"mode": "disabled"},
        "grants": {"first-install": ["private"]},
        "retrieval": {
            "mode": "local-lexical",
        },
        "socket_path": str(socket_path),
    }

    loaded = load_config(config_path)
    assert loaded.database_path == database_path
    assert loaded.socket_path == socket_path
    assert loaded.grants == {"first-install": ("private",)}
    assert loaded.client_credentials == {"first-install": digest}
    assert report.credential_path.is_file()
    assert _mode(report.credential_path) == 0o600
    assert report.credential_path.read_text(encoding="utf-8") == token
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
    payload = json.loads(capsys.readouterr().out)
    token = payload["owner_credential"]
    digest = "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()
    raw = json.loads((config_dir / "server.json").read_text(encoding="utf-8"))
    assert payload["config_path"] == str(config_dir / "server.json")
    assert payload["database_path"] == str(data_dir / "memory.db")
    assert payload["owner_client_id"] == "cli-install"
    assert payload["socket_path"] == str(data_dir / "enfold.sock")
    assert raw["client_credentials"]["cli-install"] == digest
    assert token not in (config_dir / "server.json").read_text(encoding="utf-8")


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


def test_ensure_instance_reuse_does_not_disclose_a_requested_client_secret(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    bootstrap(config_dir=config_dir, data_dir=data_dir, owner_client_id="low")
    privileged_token = new_client_token()
    raw = json.loads((config_dir / "server.json").read_text(encoding="utf-8"))
    raw["grants"]["privileged"] = ["private"]
    raw["client_credentials"]["privileged"] = credential_digest(privileged_token)
    replace_config(config_dir / "server.json", raw)
    write_client_credential(config_dir, "privileged", privileged_token)

    report, created = ensure_instance(
        config_dir=config_dir,
        data_dir=data_dir,
        owner_client_id="privileged",
    )

    assert created is False
    assert report.owner_client_id == "privileged"
    assert report.owner_credential == ""
    assert privileged_token not in report.owner_credential
    assert (config_dir / "credentials" / "privileged").read_text(
        encoding="utf-8"
    ) == privileged_token
