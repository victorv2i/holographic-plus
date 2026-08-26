"""Create a new private, local-only Enfold instance.

The daemon intentionally never creates a database.  This explicit bootstrap
command creates only a brand-new store and configuration; it does not touch an
existing database, start a daemon, install an adapter, or configure a service.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import stat
from typing import Any, Sequence

from .daemon import (
    AF_UNIX_PATH_MAX,
    SocketPathError,
    check_unix_socket_path,
    unix_socket_path_bytes,
)
from .schema import migrate
from .server import ServerConfigError, load_config
from .protocol import ClientContext, ProtocolValidationError

_PROTOCOL_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


class BootstrapError(RuntimeError):
    """A new Enfold instance could not be created safely."""


@dataclass(frozen=True, slots=True)
class BootstrapReport:
    """Paths and principal configured for one newly-created instance."""

    config_path: Path
    database_path: Path
    socket_path: Path
    owner_client_id: str
    owner_credential: str
    credential_path: Path


def new_client_token() -> str:
    """Return a protocol-safe bearer token that is never persisted as a digest source."""

    for _ in range(8):
        token = "enf_" + secrets.token_urlsafe(32)
        if _PROTOCOL_TOKEN.fullmatch(token):
            return token
    raise BootstrapError("could not generate a protocol-safe client credential")


def credential_digest(token: str) -> str:
    """Return the sha256 digest stored in daemon configuration."""

    if not isinstance(token, str) or not _PROTOCOL_TOKEN.fullmatch(token):
        raise BootstrapError("client credential is not a protocol-safe token")
    return "sha256:" + hashlib.sha256(token.encode("utf-8")).hexdigest()


def _absolute_path(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:]:
        cursor /= part
        if cursor.is_symlink():
            raise BootstrapError(f"{label} must not contain a symlink")
        try:
            info = cursor.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise BootstrapError(f"cannot inspect {label} ancestor") from exc
        if (
            stat.S_ISDIR(info.st_mode)
            and info.st_mode & 0o022
            and not info.st_mode & stat.S_ISVTX
        ):
            raise BootstrapError(
                f"{label} must not have a non-sticky writable ancestor"
            )
    return Path(os.path.abspath(candidate))


def _private_directory(path: Path, label: str) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        info = path.lstat()
    except OSError as exc:
        raise BootstrapError(f"cannot inspect {label}") from exc
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        raise BootstrapError(f"{label} must be a non-symlink directory")
    if info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise BootstrapError(f"{label} must be owned by this user and owner-only")


def _ensure_absent(path: Path, label: str) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise BootstrapError(f"cannot inspect {label}") from exc
    raise BootstrapError(f"{label} already exists; refusing to overwrite it")


def _write_new_file(path: Path, data: bytes, label: str) -> os.stat_result:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise BootstrapError(f"{label} already exists; refusing to overwrite it") from exc
    except OSError as exc:
        raise BootstrapError(f"cannot create {label}") from exc
    created: os.stat_result | None = None
    try:
        created = os.fstat(descriptor)
        offset = 0
        while offset < len(data):
            written = os.write(descriptor, data[offset:])
            if written <= 0:
                raise OSError(f"{label} write made no progress")
            offset += written
        os.fsync(descriptor)
        info = os.fstat(descriptor)
    except OSError as exc:
        _remove_created(path, created)
        raise BootstrapError(f"cannot write {label}") from exc
    finally:
        os.close(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise BootstrapError(f"{label} must be an owner-only regular file")
    return info


def _remove_created(path: Path, created: os.stat_result | None) -> None:
    if created is None:
        return
    try:
        current = path.lstat()
        if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
            path.unlink()
    except OSError:
        pass


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BootstrapError(f"cannot sync {path}") from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise BootstrapError(f"cannot sync {path}") from exc
    finally:
        os.close(descriptor)


def _new_database(path: Path) -> os.stat_result:
    created = _write_new_file(path, b"", "database")
    try:
        with sqlite3.connect(path) as connection:
            migrate(connection)
    except Exception as exc:
        _remove_created(path, created)
        if isinstance(exc, BootstrapError):
            raise
        raise BootstrapError("cannot initialize database schema") from exc
    return created


def _runtime_socket_candidates(data_directory: Path) -> list[Path]:
    digest = hashlib.sha256(os.fsencode(os.fspath(data_directory))).hexdigest()[:16]
    uid = os.getuid()
    candidates: list[Path] = []
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime:
        configured = Path(runtime).expanduser()
        if configured.is_absolute():
            candidates.append(configured / "enfold" / f"{digest}.sock")
    run_user = Path(f"/run/user/{uid}")
    if run_user.is_dir():
        candidates.append(run_user / "enfold" / f"{digest}.sock")
    candidates.append(Path(f"/tmp/enfold-{uid}-{digest}") / "enfold.sock")
    return candidates


def resolve_socket_path(
    preferred: Path,
    *,
    explicit: str | Path | None = None,
    data_directory: Path | None = None,
    allow_runtime_fallback: bool = True,
) -> Path:
    """Return a bindable Unix socket path without moving the store."""

    if explicit is not None:
        path = _absolute_path(explicit, "socket path")
        try:
            check_unix_socket_path(path)
        except SocketPathError as exc:
            raise BootstrapError(str(exc)) from exc
        return path
    if unix_socket_path_bytes(preferred) <= AF_UNIX_PATH_MAX:
        return preferred
    if allow_runtime_fallback:
        root = data_directory if data_directory is not None else preferred.parent
        for candidate in _runtime_socket_candidates(root):
            try:
                path = _absolute_path(candidate, "socket path")
                check_unix_socket_path(path)
            except (BootstrapError, SocketPathError):
                continue
            return path
    try:
        check_unix_socket_path(preferred)
    except SocketPathError as exc:
        raise BootstrapError(str(exc)) from exc
    return preferred


def _new_config(path: Path, value: dict[str, Any]) -> os.stat_result:
    encoded = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    created = _write_new_file(path, encoded, "configuration")
    try:
        load_config(path)
    except ServerConfigError as exc:
        _remove_created(path, created)
        raise BootstrapError("generated configuration failed strict validation") from exc
    return created


def bootstrap(
    *,
    config_dir: str | Path,
    data_dir: str | Path,
    owner_client_id: str = "enfold-owner",
    socket_path: str | Path | None = None,
) -> BootstrapReport:
    """Create a schema-current store and a production-supported local configuration.

    The generated retrieval configuration uses offline local-lexical scoring,
    the same first-run mode ``enfold doctor`` exercises.
    """

    if not isinstance(owner_client_id, str) or not owner_client_id.strip():
        raise BootstrapError("owner client id must be a non-empty string")
    if owner_client_id != owner_client_id.strip():
        raise BootstrapError("owner client id must not have surrounding whitespace")
    try:
        ClientContext(
            client_id=owner_client_id,
            surface="bootstrap",
            agent_id="bootstrap",
            session_id="bootstrap",
        )
    except ProtocolValidationError as exc:
        raise BootstrapError("owner client id must be a valid protocol client id") from exc
    config_directory = _absolute_path(config_dir, "configuration directory")
    data_directory = _absolute_path(data_dir, "data directory")
    _private_directory(config_directory, "configuration directory")
    _private_directory(data_directory, "data directory")

    config_path = config_directory / "server.json"
    database_path = data_directory / "memory.db"
    socket_path = resolve_socket_path(
        data_directory / "enfold.sock",
        explicit=socket_path,
        data_directory=data_directory,
        allow_runtime_fallback=True,
    )
    if socket_path.parent != data_directory:
        _private_directory(socket_path.parent, "socket directory")
    credentials_directory = config_directory / "credentials"
    credential_path = credentials_directory / owner_client_id
    token = new_client_token()
    digest = credential_digest(token)
    _ensure_absent(config_path, "configuration")
    _ensure_absent(database_path, "database")
    _ensure_absent(socket_path, "socket path")
    _ensure_absent(credential_path, "client credential")

    database = _new_database(database_path)
    config: os.stat_result | None = None
    credential: os.stat_result | None = None
    try:
        config = _new_config(
            config_path,
            {
                "client_credentials": {owner_client_id: digest},
                "database_path": str(database_path),
                "extraction": {"mode": "disabled"},
                "grants": {owner_client_id: ["private"]},
                "retrieval": {
                    "mode": "local-lexical",
                },
                "socket_path": str(socket_path),
            },
        )
        _private_directory(credentials_directory, "credentials directory")
        credential = _write_new_file(
            credential_path, token.encode("utf-8"), "client credential"
        )
        _fsync_directory(config_directory)
        _fsync_directory(data_directory)
        _fsync_directory(credentials_directory)
    except Exception:
        _remove_created(credential_path, credential)
        _remove_created(config_path, config)
        _remove_created(database_path, database)
        raise
    return BootstrapReport(
        config_path=config_path,
        database_path=database_path,
        socket_path=socket_path,
        owner_client_id=owner_client_id,
        owner_credential=token,
        credential_path=credential_path,
    )


def _xdg_directory(variable: str, fallback: str) -> Path:
    configured = os.environ.get(variable)
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute():
            return candidate
    return Path.home() / fallback


def default_config_dir() -> Path:
    return _xdg_directory("XDG_CONFIG_HOME", ".config") / "enfold"


def default_data_dir() -> Path:
    return _xdg_directory("XDG_DATA_HOME", ".local/share") / "enfold"


def read_client_credential(credential_path: Path) -> str:
    """Return the owner-only bearer token written during init or setup."""

    try:
        token = credential_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BootstrapError("cannot read client credential") from exc
    if not _PROTOCOL_TOKEN.fullmatch(token):
        raise BootstrapError("stored client credential is not a protocol-safe token")
    return token


def write_client_credential(config_dir: str | Path, client_id: str, token: str) -> Path:
    """Write one owner-only credential file and return its path."""

    config_directory = _absolute_path(config_dir, "configuration directory")
    _private_directory(config_directory, "configuration directory")
    credentials_directory = config_directory / "credentials"
    _private_directory(credentials_directory, "credentials directory")
    path = credentials_directory / client_id
    digest = credential_digest(token)
    if path.exists():
        current = read_client_credential(path)
        if credential_digest(current) == digest:
            return path
        raise BootstrapError("client credential already exists; refusing to overwrite it")
    _write_new_file(path, token.encode("utf-8"), "client credential")
    _fsync_directory(credentials_directory)
    return path


def replace_config(path: Path, value: dict[str, Any]) -> None:
    """Replace an existing owner-only configuration after strict validation."""

    encoded = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    staging = path.with_name(path.name + ".tmp")
    _ensure_absent(staging, "configuration staging")
    created = _write_new_file(staging, encoded, "configuration staging")
    try:
        load_config(staging)
        os.replace(staging, path)
        info = path.lstat()
        if info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise BootstrapError("configuration must be an owner-only regular file")
    except Exception:
        _remove_created(staging, created)
        raise


def ensure_instance(
    *,
    config_dir: str | Path,
    data_dir: str | Path,
    owner_client_id: str = "enfold-mcp-install",
) -> tuple[BootstrapReport, bool]:
    """Create a new instance or reuse a complete existing one."""

    config_directory = _absolute_path(config_dir, "configuration directory")
    data_directory = _absolute_path(data_dir, "data directory")
    config_path = config_directory / "server.json"
    database_path = data_directory / "memory.db"
    config_exists = config_path.exists()
    database_exists = database_path.exists()
    if config_exists and database_exists:
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
            loaded = load_config(config_path)
        except (OSError, json.JSONDecodeError, ServerConfigError) as exc:
            raise BootstrapError("existing configuration is not a valid Enfold instance") from exc
        grants = loaded.grants
        client_id = (
            owner_client_id if owner_client_id in grants else next(iter(grants))
        )
        credential_path = config_directory / "credentials" / client_id
        return (
            BootstrapReport(
                config_path=config_path,
                database_path=Path(raw["database_path"]),
                socket_path=Path(raw["socket_path"]),
                owner_client_id=client_id,
                owner_credential="",
                credential_path=credential_path,
            ),
            False,
        )
    if config_exists or database_exists:
        raise BootstrapError(
            "incomplete Enfold instance; refusing to repair it in place"
        )
    return bootstrap(
        config_dir=config_directory,
        data_dir=data_directory,
        owner_client_id=owner_client_id,
    ), True


def _add_instance_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--config-dir",
        type=Path,
        default=default_config_dir(),
        help="private configuration directory (default: XDG config/enfold)",
    )
    command.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="private data and socket directory (default: XDG data/enfold)",
    )


def _add_socket_path_argument(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--socket-path",
        type=Path,
        help="absolute Unix socket path; must be at most 107 bytes",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and administer a private local Enfold instance."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser(
        "init", help="create a new private store and minimal daemon configuration"
    )
    _add_instance_arguments(initialize)
    _add_socket_path_argument(initialize)
    initialize.add_argument(
        "--client-id",
        default="enfold-owner",
        help="client id granted the initial private scope",
    )
    setup = commands.add_parser(
        "setup",
        help="create or reuse state, grant one client, and run a write/search smoke test",
    )
    _add_instance_arguments(setup)
    _add_socket_path_argument(setup)
    setup.add_argument(
        "--client",
        required=True,
        choices=("codex", "claude-code", "cursor", "hermes", "generic"),
        help="MCP host that should receive a generated grant and snippet",
    )
    uninstall = commands.add_parser(
        "uninstall",
        help="list or remove files written by enfold init/setup",
    )
    _add_instance_arguments(uninstall)
    uninstall.add_argument(
        "--dry-run",
        action="store_true",
        help="print the files that would be removed without deleting them",
    )
    uninstall.add_argument(
        "--purge-data",
        action="store_true",
        help="also delete the SQLite store and its sidecars",
    )
    commands.add_parser(
        "doctor",
        help="write, recall, and return evidence through an isolated local-lexical daemon",
    )
    commands.add_parser(
        "demo",
        help="show a conflict receipt, human resolve, history, and erasure on a disposable store",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``enfold init``, ``setup``, ``uninstall``, ``doctor``, or ``demo``."""

    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "doctor":
        from .doctor import main as run_doctor

        return run_doctor()
    if args.command == "demo":
        from .demo import main as run_demo

        return run_demo()
    if args.command == "setup":
        from .setup_cli import run_setup

        return run_setup(
            client=args.client,
            config_dir=args.config_dir,
            data_dir=args.data_dir,
            socket_path=args.socket_path,
        )
    if args.command == "uninstall":
        from .setup_cli import run_uninstall

        return run_uninstall(
            config_dir=args.config_dir,
            data_dir=args.data_dir,
            dry_run=args.dry_run,
            purge_data=args.purge_data,
        )
    try:
        report = bootstrap(
            config_dir=args.config_dir,
            data_dir=args.data_dir,
            owner_client_id=args.client_id,
            socket_path=args.socket_path,
        )
    except BootstrapError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "config_path": str(report.config_path),
        "credential_path": str(report.credential_path),
        "database_path": str(report.database_path),
        "owner_client_id": report.owner_client_id,
        "owner_credential": report.owner_credential,
        "socket_path": str(report.socket_path),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
