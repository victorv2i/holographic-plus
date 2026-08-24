from __future__ import annotations

import json
import sqlite3

import pytest

from enfold.extraction_processor import ExtractionProcessor
from enfold.policy import MemoryPolicy
from enfold.schema import MigrationError, _needs_extraction_queue_patch, migrate
from enfold.service import EnfoldService


def test_migration_preserves_but_quarantines_legacy_bare_list_snapshot(tmp_path):
    conn = sqlite3.connect(tmp_path / "legacy-snapshot.db")
    migrate(conn)
    transcript = "USER: Avery prefers local tools."
    proposal = {
        "content": "Avery prefers local tools.",
        "category": "user_pref",
        "tags": "local,tools",
    }
    conn.execute(
        "INSERT INTO extract_queue(payload, status, proposal_json) "
        "VALUES (?, 'pending', ?)",
        (transcript, json.dumps([proposal], indent=2)),
    )
    conn.commit()

    assert migrate(conn) == 1
    proposal_json, proposal_hash = conn.execute(
        "SELECT proposal_json, proposal_hash FROM extract_queue"
    ).fetchone()
    assert proposal_json == json.dumps(
        [proposal],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert proposal_hash is None
    assert not _needs_extraction_queue_patch(conn)

    class NoRecallExtractor:
        identity = "unused:v1"
        calls = 0

        def extract(self, _envelope):
            self.calls += 1
            return ()

    extractor = NoRecallExtractor()
    service = EnfoldService(conn, MemoryPolicy({}))
    first = ExtractionProcessor(
        conn, service, extractor, retry_delay_seconds=0
    ).process_one()

    assert (first.outcome, first.error) == ("dead", "legacy_extraction_quarantined")
    assert extractor.calls == 0
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize(
    ("proposal_json", "proposal_hash"),
    ((None, "orphaned-hash"), ("not json", None), ('{"content":"fact"}', None)),
)
def test_migration_rejects_nonlegacy_partial_snapshots(
    tmp_path, proposal_json, proposal_hash
):
    conn = sqlite3.connect(tmp_path / "invalid-partial-snapshot.db")
    migrate(conn)
    conn.execute(
        "INSERT INTO extract_queue(payload, status, proposal_json, proposal_hash) "
        "VALUES ('legacy transcript', 'pending', ?, ?)",
        (proposal_json, proposal_hash),
    )
    conn.commit()

    assert _needs_extraction_queue_patch(conn)

    with pytest.raises(
        MigrationError, match="proposal snapshot columns are inconsistent"
    ):
        migrate(conn)
    assert conn.execute(
        "SELECT proposal_json, proposal_hash FROM extract_queue"
    ).fetchone() == (proposal_json, proposal_hash)
    conn.close()
