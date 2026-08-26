"""Reversible first-run setup and uninstall for a local Enfold instance."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets
import signal
import sqlite3
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

from .bootstrap import (
    BootstrapError,
    BootstrapReport,
    bootstrap,
    credential_digest,
    ensure_instance,
    new_client_token,
    read_client_credential,
    replace_config,
    resolve_socket_path,
    write_client_credential,
    _absolute_path,
    _private_directory,
    _remove_created,
    _write_new_file,
)
from .client import ClientConfig, EnfoldClient
from .daemon import (
    DaemonError,
    SocketPathError,
    check_unix_socket_path,
    remove_stale_socket,
    socket_liveness,
    wait_for_live_socket,
)
from .protocol import ClientContext
from .server import ServerConfigError, load_config


class SetupError(RuntimeError):
    """A setup or uninstall step failed and should be reported to the user."""


CLIENT_PROFILES = {
    "codex": {"surface": "codex", "agent_id": "codex", "kind": "toml"},
    "claude-code": {"surface": "claude-code", "agent_id": "claude-code", "kind": "cli"},
    "cursor": {"surface": "cursor", "agent_id": "cursor", "kind": "json"},
    "hermes": {"surface": "hermes", "agent_id": "hermes", "kind": "json"},
    "generic": {"surface": "generic", "agent_id": "generic", "kind": "json"},
}

AGENT_SENTENCE = (
    "Enfold is connected; tell your agent: "
    '"Remember that my test preference is dark mode, then show me the evidence '
    'for that memory."'
)

_MANIFEST_NAME = "install-manifest.json"
_PID_NAME = "enfold.pid"
_LOG_NAME = "daemon.log"


@dataclass(frozen=True, slots=True)
class _CreatedFile:
    path: Path
    created: os.stat_result


def _pid_path(data_dir: Path) -> Path:
    return data_dir / _PID_NAME


def _manifest_path(config_dir: Path) -> Path:
    return config_dir / _MANIFEST_NAME


def _stop_pid(pid_path: Path) -> None:
    if not pid_path.is_file():
        return
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.02)
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        return


def _write_pid(pid_path: Path, pid: int) -> _CreatedFile | None:
    if pid_path.exists():
        pid_path.write_text(f"{pid}\n", encoding="utf-8")
        return None
    created = _write_new_file(pid_path, f"{pid}\n".encode("utf-8"), "daemon pid")
    return _CreatedFile(pid_path, created)


def ensure_user_daemon(
    config_path: Path,
    data_dir: Path,
    *,
    python: str | None = None,
    restart_if_live: bool = False,
) -> str:
    """Start the per-user daemon or attach to a live one."""

    loaded = load_config(config_path)
    try:
        check_unix_socket_path(loaded.socket_path)
    except SocketPathError as exc:
        raise SetupError(str(exc)) from exc
    status = socket_liveness(loaded.socket_path)
    if status == "live":
        if not restart_if_live:
            return "reused"
        _stop_pid(_pid_path(data_dir))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = socket_liveness(loaded.socket_path)
            if status != "live":
                break
            time.sleep(0.02)
        if status == "live":
            raise SetupError(
                "could not restart live daemon to load new client grant"
            )
    if status == "stale":
        remove_stale_socket(loaded.socket_path)
    interpreter = python or sys.executable
    _private_directory(data_dir, "data directory")
    log_path = data_dir / _LOG_NAME
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        log_fd = os.open(log_path, flags, 0o600)
    except OSError as exc:
        raise SetupError("cannot open daemon log") from exc
    try:
        process = subprocess.Popen(
            [interpreter, "-m", "enfold.server", "--config", str(config_path), "run"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=log_fd,
            start_new_session=True,
            cwd=os.fspath(data_dir),
        )
    except OSError as exc:
        os.close(log_fd)
        raise SetupError("cannot start Enfold daemon") from exc
    os.close(log_fd)
    _write_pid(_pid_path(data_dir), process.pid)
    try:
        wait_for_live_socket(loaded.socket_path, timeout=8.0)
    except DaemonError as exc:
        _stop_pid(_pid_path(data_dir))
        detail = ""
        try:
            detail = log_path.read_text(encoding="utf-8")[-400:]
        except OSError:
            detail = ""
        message = str(exc)
        if detail:
            message = f"{message}: {detail.strip()}"
        raise SetupError(message) from exc
    return "started"


def run_protocol_smoke(
    *,
    socket_path: Path,
    client_id: str,
    surface: str,
    agent_id: str,
    credential: str,
) -> dict[str, Any]:
    """Write and recall one disposable fact through the daemon protocol."""

    nonce = secrets.token_hex(4)
    marker = f"Enfold setup smoke remembered the local install {nonce}"
    client = EnfoldClient(
        ClientConfig(
            socket_path,
            ClientContext(
                client_id=client_id,
                surface=surface,
                agent_id=agent_id,
                session_id=f"enfold-setup-{secrets.token_hex(8)}",
                access_scopes=("private",),
            ),
            credential=credential,
            connect_timeout=2.0,
            request_timeout=10.0,
        )
    )
    written = client.request(
        "memory.write",
        {
            "idempotency_key": f"enfold-setup-smoke-{secrets.token_hex(8)}",
            "content": marker,
            "source_type": "operator",
            "scope": "private",
        },
    )
    found = client.request("memory.search", {"query": marker})
    facts = found.get("facts") if isinstance(found, dict) else None
    fact_ids = [
        row.get("fact_id")
        for row in facts or ()
        if isinstance(row, dict)
    ]
    if written.get("fact_id") not in fact_ids:
        raise SetupError("smoke test wrote a fact that search did not recall")
    return {"fact_id": written["fact_id"], "query": marker}


def registered_client_surface(
    client_id: str,
    *,
    config_dir: str | Path | None = None,
    database_path: str | Path | None = None,
) -> str | None:
    """Return the surface already bound to ``client_id``, if any."""

    if config_dir is not None:
        manifest = Path(config_dir) / _MANIFEST_NAME
        if manifest.is_file():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                payload = {}
            for item in payload.get("clients") or ():
                if not isinstance(item, dict) or item.get("id") != client_id:
                    continue
                profile = CLIENT_PROFILES.get(item.get("profile"))
                if profile:
                    return profile["surface"]
    if database_path is None:
        return None
    try:
        uri = Path(database_path).resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
    except (OSError, sqlite3.Error):
        return None
    try:
        row = connection.execute(
            "SELECT surface FROM memory_clients WHERE client_id = ?",
            (client_id,),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    if row and isinstance(row[0], str) and row[0].strip():
        return row[0]
    return None


def _allocate_client_id(profile: str, grants: Iterable[str]) -> str:
    existing = set(grants)
    candidate = f"{profile}-install-1"
    index = 1
    while candidate in existing:
        index += 1
        candidate = f"{profile}-install-{index}"
    return candidate


def render_snippet(
    *,
    profile: str,
    client_id: str,
    credential: str,
    command: str = "enfold-mcp",
) -> str:
    meta = CLIENT_PROFILES[profile]
    args = [
        "--client-id",
        client_id,
        "--surface",
        meta["surface"],
        "--agent-id",
        meta["agent_id"],
        "--access-scope",
        "private",
    ]
    if meta["kind"] == "toml":
        quoted = ", ".join(json.dumps(item) for item in args)
        return (
            "[mcp_servers.enfold]\n"
            f"command = {json.dumps(command)}\n"
            f"args = [{quoted}]\n"
            "\n"
            "[mcp_servers.enfold.env]\n"
            f"ENFOLD_CLIENT_CREDENTIAL = {json.dumps(credential)}\n"
        )
    if meta["kind"] == "cli":
        joined = " ".join(args)
        return (
            f"claude mcp add --transport stdio --scope user enfold -- {command} {joined}\n"
            "Set ENFOLD_CLIENT_CREDENTIAL in the Claude Code supervisor environment; "
            "do not paste the token into an agent prompt.\n"
            f"ENFOLD_CLIENT_CREDENTIAL={credential}\n"
        )
    payload = {
        "mcpServers": {
            "enfold": {
                "type": "stdio",
                "command": command,
                "args": args,
                "env": {"ENFOLD_CLIENT_CREDENTIAL": credential},
            }
        }
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _add_grant(
    config_path: Path,
    *,
    client_id: str,
    token: str,
) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    grants = dict(raw["grants"])
    credentials = dict(raw.get("client_credentials") or {})
    grants[client_id] = ["private"]
    credentials[client_id] = credential_digest(token)
    if set(credentials) != set(grants):
        missing = sorted(set(grants) - set(credentials))
        raise SetupError(
            "existing grants have no credentials; re-run setup for those clients: "
            + ", ".join(missing)
        )
    raw["grants"] = grants
    raw["client_credentials"] = credentials
    replace_config(config_path, raw)
    return raw


def _manifest_payload(
    *,
    profile: str,
    client_id: str,
    paths: Sequence[Path],
) -> dict[str, Any]:
    kinds = {
        "server.json": "config",
        "memory.db": "database",
        _MANIFEST_NAME: "manifest",
        _PID_NAME: "pid",
        _LOG_NAME: "log",
    }
    listed = []
    for path in paths:
        kind = kinds.get(path.name, "file")
        if path.parent.name == "credentials":
            kind = "credential"
        elif path.suffix == ".snippet":
            kind = "snippet"
        listed.append({"kind": kind, "path": str(path)})
    return {
        "clients": [{"id": client_id, "profile": profile}],
        "paths": listed,
        "version": 1,
    }


def _collect_known_paths(config_dir: Path, data_dir: Path) -> list[Path]:
    paths = [
        config_dir / "server.json",
        _manifest_path(config_dir),
        data_dir / "memory.db",
        data_dir / "memory.db-wal",
        data_dir / "memory.db-shm",
        data_dir / "memory.db.enfold.lock",
        data_dir / "enfold.sock",
        _pid_path(data_dir),
        data_dir / _LOG_NAME,
    ]
    config_path = config_dir / "server.json"
    if config_path.is_file():
        try:
            configured = Path(
                json.loads(config_path.read_text(encoding="utf-8"))["socket_path"]
            )
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            configured = None
        if configured is not None and configured not in paths:
            paths.append(configured)
    credentials = config_dir / "credentials"
    if credentials.is_dir():
        paths.extend(sorted(credentials.iterdir()))
    clients = config_dir / "clients"
    if clients.is_dir():
        paths.extend(sorted(clients.iterdir()))
    return [path for path in paths if path.exists()]


def _remove_path(path: Path) -> None:
    try:
        if path.is_dir() and not path.is_symlink():
            path.rmdir()
        elif path.exists() or path.is_symlink():
            path.unlink()
    except OSError:
        pass


def run_setup(
    *,
    client: str,
    config_dir: str | Path,
    data_dir: str | Path,
    socket_path: str | Path | None = None,
) -> int:
    """Create state, grant one client, smoke-test, and print the snippet."""

    if client not in CLIENT_PROFILES:
        print(f"enfold setup: unknown client {client!r}", file=sys.stderr)
        return 2
    try:
        config_directory = _absolute_path(config_dir, "configuration directory")
        data_directory = _absolute_path(data_dir, "data directory")
    except BootstrapError as exc:
        print(f"enfold setup: {exc}", file=sys.stderr)
        return 2

    created_instance = False
    started_daemon = False
    backups: list[tuple[Path, bytes]] = []
    created_files: list[_CreatedFile] = []
    config_path = config_directory / "server.json"
    database_path = data_directory / "memory.db"
    profile = CLIENT_PROFILES[client]
    client_id = f"{client}-install-1"
    token = ""
    snippet = ""

    def rollback() -> None:
        if started_daemon:
            _stop_pid(_pid_path(data_directory))
        for path, data in reversed(backups):
            path.write_bytes(data)
        for item in reversed(created_files):
            _remove_created(item.path, item.created)
        if created_instance:
            for path in (
                config_path,
                database_path,
                data_directory / "memory.db-wal",
                data_directory / "memory.db-shm",
                data_directory / "memory.db.enfold.lock",
                data_directory / "enfold.sock",
                _pid_path(data_directory),
                data_directory / _LOG_NAME,
                _manifest_path(config_directory),
            ):
                _remove_path(path)
            if isinstance(socket_path, Path):
                _remove_path(socket_path)
                if socket_path.parent != data_directory:
                    try:
                        socket_path.parent.rmdir()
                    except OSError:
                        pass

    try:
        if config_path.exists() and database_path.exists():
            loaded = load_config(config_path)
            client_id = _allocate_client_id(client, loaded.grants)
            token = new_client_token()
            backups.append((config_path, config_path.read_bytes()))
            raw = _add_grant(config_path, client_id=client_id, token=token)
            credential_path = write_client_credential(
                config_directory, client_id, token
            )
            created_files.append(
                _CreatedFile(credential_path, credential_path.lstat())
            )
            if socket_path is not None:
                chosen = resolve_socket_path(
                    Path(raw["socket_path"]),
                    explicit=socket_path,
                    allow_runtime_fallback=False,
                )
                if chosen.parent != data_directory:
                    _private_directory(chosen.parent, "socket directory")
                if str(chosen) != raw["socket_path"]:
                    raw["socket_path"] = str(chosen)
                    replace_config(config_path, raw)
            socket_path = Path(raw["socket_path"])
        elif config_path.exists() or database_path.exists():
            raise SetupError("incomplete Enfold instance; refusing to repair it in place")
        else:
            report = bootstrap(
                config_dir=config_directory,
                data_dir=data_directory,
                owner_client_id=client_id,
                socket_path=socket_path,
            )
            created_instance = True
            token = report.owner_credential
            socket_path = report.socket_path
            created_files.extend(
                [
                    _CreatedFile(report.config_path, report.config_path.lstat()),
                    _CreatedFile(report.database_path, report.database_path.lstat()),
                    _CreatedFile(report.credential_path, report.credential_path.lstat()),
                ]
            )
        snippet = render_snippet(profile=client, client_id=client_id, credential=token)
        clients_dir = config_directory / "clients"
        _private_directory(clients_dir, "client snippet directory")
        snippet_path = clients_dir / f"{client}.snippet"
        if not snippet_path.exists():
            created = _write_new_file(
                snippet_path, snippet.encode("utf-8"), "client snippet"
            )
            created_files.append(_CreatedFile(snippet_path, created))
        else:
            backups.append((snippet_path, snippet_path.read_bytes()))
            snippet_path.write_text(snippet, encoding="utf-8")
        manifest_paths = _collect_known_paths(config_directory, data_directory)
        manifest_paths.append(_manifest_path(config_directory))
        encoded = json.dumps(
            _manifest_payload(profile=client, client_id=client_id, paths=manifest_paths),
            indent=2,
            sort_keys=True,
        ).encode("utf-8") + b"\n"
        manifest = _manifest_path(config_directory)
        if manifest.exists():
            backups.append((manifest, manifest.read_bytes()))
            manifest.write_bytes(encoded)
        else:
            created = _write_new_file(manifest, encoded, "install manifest")
            created_files.append(_CreatedFile(manifest, created))
        daemon_state = ensure_user_daemon(
            config_path,
            data_directory,
            restart_if_live=not created_instance,
        )
        started_daemon = daemon_state == "started"
        smoke = run_protocol_smoke(
            socket_path=socket_path,
            client_id=client_id,
            surface=profile["surface"],
            agent_id=profile["agent_id"],
            credential=token,
        )
    except (BootstrapError, SetupError, ServerConfigError, OSError, ValueError) as exc:
        rollback()
        print(f"enfold setup: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        rollback()
        print(f"enfold setup: {exc}", file=sys.stderr)
        return 2

    payload = {
        "agent_sentence": AGENT_SENTENCE,
        "client_credential": token,
        "client_id": client_id,
        "config_path": str(config_path),
        "database_path": str(database_path),
        "snippet": snippet,
        "smoke": smoke,
        "socket_path": str(socket_path),
    }
    print(AGENT_SENTENCE, file=sys.stderr)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0


def run_uninstall(
    *,
    config_dir: str | Path,
    data_dir: str | Path,
    dry_run: bool,
    purge_data: bool,
) -> int:
    """List or remove files written by init/setup."""

    try:
        config_directory = _absolute_path(config_dir, "configuration directory")
        data_directory = _absolute_path(data_dir, "data directory")
    except BootstrapError as exc:
        print(f"enfold uninstall: {exc}", file=sys.stderr)
        return 2
    manifest = _manifest_path(config_directory)
    listed: list[str] = []
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            listed = [item["path"] for item in payload.get("paths", [])]
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            listed = []
    if not listed:
        listed = [str(path) for path in _collect_known_paths(config_directory, data_directory)]
    preserved = []
    removable = []
    for item in listed:
        path = Path(item)
        if path.name == "memory.db" and not purge_data:
            preserved.append(str(path))
            continue
        if path.name.startswith("memory.db") and not purge_data:
            preserved.append(str(path))
            continue
        removable.append(str(path))
    report = {
        "dry_run": dry_run,
        "paths": listed,
        "preserved": preserved,
        "purge_data": purge_data,
        "removed": [] if dry_run else removable,
    }
    if dry_run:
        print(json.dumps(report, sort_keys=True))
        return 0
    _stop_pid(_pid_path(data_directory))
    try:
        loaded = load_config(config_directory / "server.json")
        if socket_liveness(loaded.socket_path) == "stale":
            remove_stale_socket(loaded.socket_path)
    except (OSError, ServerConfigError, DaemonError):
        pass
    for item in removable:
        _remove_path(Path(item))
    for directory in (
        config_directory / "credentials",
        config_directory / "clients",
        config_directory,
    ):
        if directory.is_dir() and not any(directory.iterdir()):
            _remove_path(directory)
    print(json.dumps(report, sort_keys=True))
    return 0


def prepare_stdio_session(
    *,
    config_dir: str | Path,
    data_dir: str | Path,
    client_id: str | None = None,
) -> tuple[BootstrapReport, str, str, bool]:
    """Create or reuse the local store and ensure the daemon is accepting."""

    report, created = ensure_instance(
        config_dir=config_dir,
        data_dir=data_dir,
        owner_client_id=client_id or "enfold-mcp-install",
    )
    env_token = os.environ.get("ENFOLD_CLIENT_CREDENTIAL", "")
    if client_id and not created:
        token = env_token
    elif created:
        token = env_token or report.owner_credential
    else:
        token = env_token or report.owner_credential
        if not token and report.credential_path.is_file():
            token = read_client_credential(report.credential_path)
    if not token:
        raise SetupError(
            "no client credential is available; run enfold setup or set "
            "ENFOLD_CLIENT_CREDENTIAL in the supervisor environment"
        )
    state = ensure_user_daemon(report.config_path, Path(data_dir))
    return report, token, state, created
