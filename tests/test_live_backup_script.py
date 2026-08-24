from __future__ import annotations

import os
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).parents[1] / "scripts" / "enfold-live-backup"
HEALTH_SCRIPT = Path(__file__).parents[1] / "scripts" / "enfold-health-check"
READY_SCRIPT = Path(__file__).parents[1] / "scripts" / "enfold-wait-ready"


def _fake_python(tmp_path: Path, *, fail: bool = False) -> Path:
    executable = tmp_path / ("backup-failure" if fail else "backup-success")
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'test "$1" = "-m"\n'
        'test "$2" = "enfold.ops"\n'
        'test "$3" = "backup"\n'
        'cp -- "$4" "$5"\n'
        + (
            "printf '{\"ok\":false}\\n'\nexit 7\n"
            if fail
            else "printf '{\"ok\":true}\\n'\n"
        ),
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable


def _run(
    tmp_path: Path, *, python: Path
) -> tuple[subprocess.CompletedProcess[str], Path]:
    source = tmp_path / "memory.db"
    source.write_bytes(b"verified-backup-fixture")
    destination = tmp_path / "backups"
    env = {
        **os.environ,
        "ENFOLD_BACKUP_SOURCE": str(source),
        "ENFOLD_BACKUP_DEST_DIR": str(destination),
        "ENFOLD_BACKUP_PYTHON": str(python),
        "ENFOLD_BACKUP_RETENTION_DAYS": "14",
    }
    result = subprocess.run(
        [str(SCRIPT)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, destination


def _expired_generation(tmp_path: Path, *, unknown_entry: bool = False) -> Path:
    generation = tmp_path / "backups" / "generations" / "expired-generation"
    generation.mkdir(parents=True)
    for name in ("memory_store.sqlite", "memory_store.json", "SHA256SUMS"):
        (generation / name).write_text("old backup", encoding="utf-8")
    if unknown_entry:
        (generation / "operator-note.txt").write_text(
            "retain for review", encoding="utf-8"
        )
    os.utime(generation, (1, 1))
    return generation


def test_live_backup_publishes_one_complete_verified_generation(tmp_path):
    result, destination = _run(
        tmp_path, python=_fake_python(tmp_path)
    )

    assert result.returncode == 0, result.stderr
    generations = [
        path
        for path in (destination / "generations").iterdir()
        if not path.name.startswith(".")
    ]
    assert len(generations) == 1
    generation = generations[0]
    assert sorted(path.name for path in generation.iterdir()) == [
        "SHA256SUMS",
        "memory_store.json",
        "memory_store.sqlite",
    ]
    assert (generation / "memory_store.sqlite").read_bytes() == (
        b"verified-backup-fixture"
    )
    subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"],
        cwd=generation,
        text=True,
        capture_output=True,
        check=True,
    )
    assert (destination / "latest").resolve() == generation.resolve()
    assert str(generation / "memory_store.sqlite") in result.stdout


def test_live_backup_failure_leaves_no_partial_or_published_generation(tmp_path):
    result, destination = _run(
        tmp_path, python=_fake_python(tmp_path, fail=True)
    )

    assert result.returncode == 7
    generations = destination / "generations"
    assert list(generations.iterdir()) == []
    assert not (destination / "latest").exists()


def test_live_backup_retention_prunes_only_exact_known_generation(tmp_path):
    expired = _expired_generation(tmp_path)

    result, _destination = _run(tmp_path, python=_fake_python(tmp_path))

    assert result.returncode == 0, result.stderr
    assert not expired.exists()


def test_live_backup_retention_preserves_unknown_generation_entries(tmp_path):
    expired = _expired_generation(tmp_path, unknown_entry=True)

    result, _destination = _run(tmp_path, python=_fake_python(tmp_path))

    assert result.returncode == 0, result.stderr
    assert sorted(path.name for path in expired.iterdir()) == [
        "SHA256SUMS",
        "memory_store.json",
        "memory_store.sqlite",
        "operator-note.txt",
    ]


def test_health_wrapper_uses_dedicated_public_health_identity(tmp_path):
    server = tmp_path / "enfold-server"
    server.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    server.chmod(0o700)
    config = tmp_path / "server.json"
    config.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [str(HEALTH_SCRIPT)],
        env={
            **os.environ,
            "ENFOLD_HEALTH_SERVER": str(server),
            "ENFOLD_HEALTH_CONFIG": str(config),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.splitlines() == [
        "--config",
        str(config),
        "--allow-live",
        "health",
        "--client-id",
        "healthcheck-install",
        "--surface",
        "operator-health",
        "--agent-id",
        "operator-health",
        "--access-scope",
        "public",
    ]


def test_wait_ready_retries_official_authenticated_protocol_health(tmp_path):
    attempts = tmp_path / "attempts"
    server = tmp_path / "enfold-server"
    server.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'attempts_file="${ENFOLD_TEST_ATTEMPTS:?}"\n'
        'attempts="$(cat "$attempts_file" 2>/dev/null || printf 0)"\n'
        'attempts="$((attempts + 1))"\n'
        'printf "%s" "$attempts" >"$attempts_file"\n'
        'test "$*" = "--config '
        + str(tmp_path / "server.json")
        + " --allow-live health --client-id healthcheck-install "
        "--surface operator-health --agent-id operator-health "
        '--access-scope public --readiness"\n'
        'test "$attempts" -ge 3\n',
        encoding="utf-8",
    )
    server.chmod(0o700)
    config = tmp_path / "server.json"
    config.write_text("{}", encoding="utf-8")

    result = subprocess.run(
        [str(READY_SCRIPT)],
        env={
            **os.environ,
            "ENFOLD_HEALTH_SERVER": str(server),
            "ENFOLD_HEALTH_CONFIG": str(config),
            "ENFOLD_READY_TIMEOUT_SECONDS": "2",
            "ENFOLD_READY_INTERVAL_SECONDS": "0.01",
            "ENFOLD_TEST_ATTEMPTS": str(attempts),
        },
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert attempts.read_text(encoding="utf-8") == "3"
