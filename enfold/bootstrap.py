"""Create a new private, local-only Enfold instance.

The daemon intentionally never creates a database.  This explicit bootstrap
command creates only a brand-new store and configuration; it does not touch an
existing database, start a daemon, install an adapter, or configure a service.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import stat
from typing import Any, Sequence

from .schema import migrate
from .server import ServerConfigError, load_config
from .protocol import ClientContext, ProtocolValidationError


class BootstrapError(RuntimeError):
    """A new Enfold instance could not be created safely."""


@dataclass(frozen=True, slots=True)
class BootstrapReport:
    """Paths and principal configured for one newly-created instance."""

    config_path: Path
    database_path: Path
    socket_path: Path
    owner_client_id: str


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
) -> BootstrapReport:
    """Create a schema-current store and a minimal development configuration.

    The generated retrieval configuration deliberately uses deterministic
    non-production retrieval.  Configure and validate a stored retriever before
    placing real durable memory in the instance.
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
    socket_path = data_directory / "enfold.sock"
    _ensure_absent(config_path, "configuration")
    _ensure_absent(database_path, "database")
    _ensure_absent(socket_path, "socket path")

    database = _new_database(database_path)
    config: os.stat_result | None = None
    try:
        config = _new_config(
            config_path,
            {
                "database_path": str(database_path),
                "extraction": {"mode": "disabled"},
                "grants": {owner_client_id: ["private"]},
                "retrieval": {
                    "allow_nonproduction": True,
                    "dimensions": 256,
                    "mode": "ci",
                    "vector_backend": "brute",
                },
                "socket_path": str(socket_path),
            },
        )
        _fsync_directory(config_directory)
        _fsync_directory(data_directory)
    except Exception:
        _remove_created(config_path, config)
        _remove_created(database_path, database)
        raise
    return BootstrapReport(
        config_path=config_path,
        database_path=database_path,
        socket_path=socket_path,
        owner_client_id=owner_client_id,
    )


def _xdg_directory(variable: str, fallback: str) -> Path:
    configured = os.environ.get(variable)
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_absolute():
            return candidate
    return Path.home() / fallback


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a new private local Enfold instance."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    initialize = commands.add_parser(
        "init", help="create a new private store and minimal daemon configuration"
    )
    initialize.add_argument(
        "--config-dir",
        type=Path,
        default=_xdg_directory("XDG_CONFIG_HOME", ".config") / "enfold",
        help="private configuration directory (default: XDG config/enfold)",
    )
    initialize.add_argument(
        "--data-dir",
        type=Path,
        default=_xdg_directory("XDG_DATA_HOME", ".local/share") / "enfold",
        help="private data and socket directory (default: XDG data/enfold)",
    )
    initialize.add_argument(
        "--client-id",
        default="enfold-owner",
        help="client id granted the initial private scope",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the explicit ``enfold init`` bootstrap command."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = bootstrap(
            config_dir=args.config_dir,
            data_dir=args.data_dir,
            owner_client_id=args.client_id,
        )
    except BootstrapError as exc:
        parser.error(str(exc))
    print(json.dumps({
        "config_path": str(report.config_path),
        "database_path": str(report.database_path),
        "owner_client_id": report.owner_client_id,
        "socket_path": str(report.socket_path),
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
