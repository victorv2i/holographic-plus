from __future__ import annotations

import json
from pathlib import Path

import pytest

from enfold.bootstrap import main as enfold_main
from enfold.demo import main, run_demo


ROOT = Path(__file__).resolve().parents[1]
LIVE_HERMES = Path.home() / ".hermes" / "memory_store.db"
HOSTS = ("claude-code", "codex", "cursor", "hermes")


def _hermes_fingerprint() -> tuple[object, ...]:
    if not LIVE_HERMES.exists():
        return ()
    info = LIVE_HERMES.stat()
    return (info.st_ino, info.st_size, info.st_mtime_ns)


def test_demo_walks_conflict_receipt_resolution_history_evidence_and_erasure():
    before = _hermes_fingerprint()
    report = run_demo()

    assert report["ok"] is True
    assert report["network"] == "disabled"
    assert report["retrieval"] == "local-lexical"
    assert report["store_discarded"] is True
    assert report["live_hermes_path"] == str(LIVE_HERMES)
    assert report["touched_live_hermes"] is False
    assert _hermes_fingerprint() == before

    assert report["client_a"]["outcome"] == "add"
    assert report["client_b"]["outcome"] == "conflict"
    assert isinstance(report["client_a"]["fact_id"], int)
    assert isinstance(report["client_b"]["fact_id"], int)
    assert report["client_a"]["object_value"] == "3100"
    assert report["client_b"]["object_value"] == "3200"
    assert report["client_a"]["source_authority"] == report["client_b"]["source_authority"]

    conflict = report["conflict_recall"]
    assert conflict["facts"] == []
    contents = " ".join(str(row.get("content") or "") for row in conflict["facts"])
    assert "3100" not in contents
    assert "3200" not in contents
    receipts = conflict["open_conflicts"]
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["conflict_id"] == report["conflict_id"]
    assert receipt["conflict_id"]
    assert set(receipt["member_fact_ids"]) == {
        report["client_a"]["fact_id"],
        report["client_b"]["fact_id"],
    }
    assert "do not treat either as current" in receipt["summary"]

    resolved = report["resolution"]
    assert resolved["resolution_fact_id"] == report["client_b"]["fact_id"]
    assert report["client_a"]["fact_id"] in resolved["superseded_fact_ids"]

    settled = report["settled_recall"]
    settled_contents = [str(row.get("content") or "") for row in settled["facts"]]
    assert any("3200" in item for item in settled_contents)
    assert all("3100" not in item for item in settled_contents)
    assert settled["open_conflicts"] == []

    history_contents = " ".join(
        str(row.get("content") or "") for row in report["history"]["facts"]
    )
    assert "3100" in history_contents
    assert "3200" in history_contents
    older = next(
        row
        for row in report["history"]["facts"]
        if row.get("fact_id") == report["client_a"]["fact_id"]
    )
    assert older.get("superseded_by") == report["client_b"]["fact_id"]

    named = set(report["evidence_client_ids"])
    assert report["client_a"]["client_id"] in named
    assert report["client_b"]["client_id"] in named

    assert report["erasure"]["shown"] is True
    export_text = report["erasure"]["export_text"]
    assert "3200" not in export_text
    assert "The staging port is 3200" not in export_text


def test_enfold_demo_is_on_the_product_cli_and_prints_a_live_walkthrough(capsys):
    before = _hermes_fingerprint()
    result = enfold_main(["demo"])
    captured = capsys.readouterr()
    assert result == 0
    text = captured.out
    assert "conflict" in text.lower()
    assert "3100" in text
    assert "3200" in text
    assert "do not treat either as current" in text
    assert "demo-client-a" in text
    assert "demo-client-b" in text
    assert "human" in text.lower()
    assert str(LIVE_HERMES) not in text or "not used" in text.lower() or "never" in text.lower()
    assert _hermes_fingerprint() == before
    assert not any(
        line.strip().startswith("{") and line.strip().endswith("}")
        for line in text.splitlines()
        if line.strip()
    )


def test_demo_main_exits_zero_and_matches_the_live_report(capsys):
    assert main([]) == 0
    text = capsys.readouterr().out
    assert "staging port" in text.lower()
    assert "conflict" in text.lower()


def test_enfold_help_lists_demo(capsys):
    with pytest.raises(SystemExit) as exited:
        enfold_main(["--help"])
    assert exited.value.code == 0
    assert "demo" in capsys.readouterr().out


def test_host_skills_tell_agents_to_recall_write_typed_state_and_not_obey_memory():
    for host in HOSTS:
        path = ROOT / "integrations" / host / "SKILL.md"
        assert path.is_file(), f"missing host skill: {path}"
        text = path.read_text(encoding="utf-8")
        lowered = text.lower()
        assert "recall" in lowered
        assert "state" in lowered
        assert "instruction" in lowered
        assert len(text) < 1200
        assert "\u2014" not in text
