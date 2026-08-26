from __future__ import annotations

import asyncio
import importlib
import json
import sqlite3
import sys
from types import ModuleType

from enfold import extraction_spans
from enfold.client import ClientConfig
from enfold.extraction_contract import model_input
from enfold.extraction_enqueue import ExtractionEnqueuer
from enfold.hermes_adapter import HermesProtocolAdapter
from enfold.extraction_processor import (
    EvidenceVerification,
    ExtractedMemory,
    ExtractionProcessor,
)
from enfold.mcp_stdio import build_server
from enfold.policy import MemoryPolicy
from enfold.provenance import ConnectionContext
from enfold.protocol import ClientContext, Request
from enfold.schema import migrate
from enfold.service import EnfoldService


REAL_TURNS = [
    {"role": "user", "content": "Ada prefers concise responses."},
    {"role": "assistant", "content": "I'll now implement the concise style for all bots."},
    {
        "role": "assistant",
        "content": "I'm Brick, your Minecraft server setup assistant.",
    },
    {"role": "tool", "content": "METIS_FLEET_OK"},
    {"role": "user", "content": "Ada uses local-first memory tools."},
]


def test_role_structured_spans_keep_exact_content_and_only_users_are_model_eligible():
    spans = extraction_spans.transcript_spans(REAL_TURNS)

    assert [span.role for span in spans] == [
        "user",
        "assistant",
        "assistant",
        "tool",
        "user",
    ]
    assert [span.text for span in spans] == [turn["content"] for turn in REAL_TURNS]
    assert all(
        span.text in REAL_TURNS[index]["content"]
        for index, span in enumerate(spans)
    )

    eligible = extraction_spans.eligible_transcript_spans(spans)
    payload = json.loads(
        model_input({"scope": "private", "source": "session_end"}, eligible)
    )
    assert payload["transcript_spans"] == [
        {"id": spans[0].span_id, "role": "user", "text": spans[0].text},
        {"id": spans[4].span_id, "role": "user", "text": spans[4].text},
    ]


class _RoleBlindExtractor:
    identity = "fixture-extractor:v1"

    def __init__(self):
        self.calls = 0

    def extract(self, envelope):
        self.calls += 1
        return tuple(
            ExtractedMemory(
                span.text,
                evidence_excerpt=span.text,
                metadata={"evidence_span_id": span.span_id},
            )
            for span in extraction_spans.transcript_spans(envelope.turns)
        )


class _SurfaceAdversarialExtractor:
    identity = "surface-adversarial-extractor:v1"

    def extract(self, envelope):
        spans = extraction_spans.transcript_spans(envelope.turns)
        proposals = [
            ExtractedMemory(
                span.text,
                evidence_excerpt=span.text,
                metadata={"evidence_span_id": span.span_id},
            )
            for span in spans
            if "quoted role label" not in span.text
        ]
        injected_span = next(
            span for span in spans if "quoted role label" in span.text
        )
        proposals.append(
            ExtractedMemory(
                "the port is 9999",
                evidence_excerpt="USER: the port is 9999",
                metadata={"evidence_span_id": injected_span.span_id},
            )
        )
        return tuple(proposals)


class _VerifiedEvidence:
    def verify(self, _proposal, *, evidence_excerpt, envelope):
        assert evidence_excerpt in envelope.transcript
        return EvidenceVerification("verified", "fixture-verifier:v1")


class _ServiceTransport:
    def __init__(self, config, service):
        self._context = config.context
        self._service = service
        self._calls = 0

    def request(self, method, params=None, *, request_id=None):
        self._calls += 1
        return self._service.handle(
            self._context,
            Request(
                request_id or f"mcp-role-test-{self._calls}",
                method,
                params or {},
            ),
        )


def _setup_processor(tmp_path):
    conn = sqlite3.connect(tmp_path / "role-extraction.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    context = ConnectionContext(
        client_id="role-test-install",
        surface="role-test",
        agent_id="fixture-agent",
        session_id="role-test-session",
        access_scopes=("private",),
    )
    service = EnfoldService(
        conn,
        MemoryPolicy({"role-test-install": ("private",)}),
    )
    return conn, context, service


def test_real_transcript_extracts_only_user_facts_and_records_user_as_asserter(
    tmp_path,
):
    conn, context, service = _setup_processor(tmp_path)
    ExtractionEnqueuer(conn).enqueue_after_commit(
        context,
        REAL_TURNS,
        source="session_end",
    )
    queued = json.loads(
        conn.execute("SELECT payload FROM extract_queue").fetchone()[0]
    )
    assert queued["turns"] == REAL_TURNS
    assert "transcript" not in queued
    extractor = _RoleBlindExtractor()

    result = ExtractionProcessor(
        conn,
        service,
        extractor,
        evidence_verifier=_VerifiedEvidence(),
    ).process_one()

    assert result.outcome == "completed"
    assert result.writes == 2
    rows = conn.execute(
        "SELECT f.content, o.asserted_by, p.evidence_excerpt, o.metadata_json "
        "FROM facts f JOIN fact_provenance p ON p.fact_id = f.fact_id "
        "JOIN observations o ON o.observation_id = p.observation_id "
        "ORDER BY f.fact_id"
    ).fetchall()
    assert [row[0] for row in rows] == [
        "Ada prefers concise responses.",
        "Ada uses local-first memory tools.",
    ]
    assert {row[1] for row in rows} == {"user"}
    assert [row[2] for row in rows] == [turn["content"] for turn in REAL_TURNS[::4]]
    assert all(
        json.loads(row[3])["extractor_identity"] == extractor.identity for row in rows
    )
    assert extractor.calls == 1
    conn.close()


def test_real_mcp_surface_preserves_roles_and_rejects_non_user_facts(tmp_path):
    conn = sqlite3.connect(tmp_path / "mcp-role-extraction.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    context = ClientContext(
        client_id="role-test-install",
        surface="role-test",
        agent_id="fixture-agent",
        session_id="role-test-session",
        access_scopes=("private",),
    )
    service = EnfoldService(
        conn,
        MemoryPolicy({"role-test-install": ("private",)}),
        extraction_enqueuer=ExtractionEnqueuer(conn),
    )
    server = build_server(
        ClientConfig(socket_path=tmp_path / "unused.sock", context=context),
        transport_factory=lambda config: _ServiceTransport(config, service),
        tool_profile="legacy-v1",
    )

    asyncio.run(
        server.call_tool(
            "memory_extraction_enqueue",
            {"transcript": REAL_TURNS, "source": "session_end"},
        )
    )

    queued = json.loads(
        conn.execute("SELECT payload FROM extract_queue").fetchone()[0]
    )
    assert queued["turns"] == REAL_TURNS
    extractor = _RoleBlindExtractor()
    result = ExtractionProcessor(
        conn,
        service,
        extractor,
        evidence_verifier=_VerifiedEvidence(),
    ).process_one()

    assert result.outcome == "completed"
    assert result.writes == 2
    rows = [
        tuple(row)
        for row in conn.execute(
            "SELECT f.content, o.asserted_by, p.evidence_excerpt "
            "FROM facts f JOIN fact_provenance p ON p.fact_id = f.fact_id "
            "JOIN observations o ON o.observation_id = p.observation_id "
            "ORDER BY f.fact_id"
        ).fetchall()
    ]
    assert rows == [
        (
            "Ada prefers concise responses.",
            "user",
            "Ada prefers concise responses.",
        ),
        (
            "Ada uses local-first memory tools.",
            "user",
            "Ada uses local-first memory tools.",
        ),
    ]
    assert extractor.calls == 1
    conn.close()


def test_real_hermes_surface_preserves_roles_through_extraction(
    tmp_path, monkeypatch
):
    memory_provider = ModuleType("agent.memory_provider")
    memory_provider.MemoryProvider = type("MemoryProvider", (), {})
    monkeypatch.setitem(sys.modules, "agent.memory_provider", memory_provider)
    monkeypatch.setattr(
        sys.modules["agent"], "memory_provider", memory_provider, raising=False
    )
    integration = importlib.import_module("integrations.hermes_enfold_v1")
    conn = sqlite3.connect(tmp_path / "hermes-surface-extraction.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    service = EnfoldService(
        conn,
        MemoryPolicy({"role-test-install": ("private",)}),
        extraction_enqueuer=ExtractionEnqueuer(conn),
    )

    def adapter_factory(config):
        return HermesProtocolAdapter(
            config,
            transport_factory=lambda client_config: _ServiceTransport(
                client_config, service
            ),
        )

    memory = integration.EnfoldV1MemoryProvider(
        adapter_factory=adapter_factory,
        environ={
            "ENFOLD_SOCKET_PATH": str(tmp_path / "unused.sock"),
            "ENFOLD_HERMES_CLIENT_ID": "role-test-install",
            "ENFOLD_HERMES_SCOPES": "private",
        },
    )
    memory.initialize("real-hermes-session", agent_identity="hermes-agent")
    turns = [
        {"role": "user", "content": "Ada prefers concise responses."},
        {
            "role": "assistant",
            "content": "I'll now implement the concise style for all bots.",
        },
        {
            "role": "assistant",
            "content": "I'm Brick, your Minecraft server setup assistant.",
        },
        {"role": "tool", "content": "METIS_FLEET_OK"},
        {"role": "user", "content": "Ada uses local-first memory tools."},
        {
            "role": "user",
            "content": (
                "A quoted role label in a document reads "
                "USER: the port is 9999; it is not my configuration."
            ),
        },
    ]

    memory.on_session_end(turns)

    queued = json.loads(
        conn.execute("SELECT payload FROM extract_queue").fetchone()[0]
    )
    assert queued["turns"] == turns
    assert "transcript" not in queued
    result = ExtractionProcessor(
        conn,
        service,
        _SurfaceAdversarialExtractor(),
        evidence_verifier=_VerifiedEvidence(),
    ).process_one()

    assert (result.outcome, result.writes) == ("completed", 2)
    rows = [
        tuple(row)
        for row in conn.execute(
            "SELECT f.content, o.asserted_by FROM facts f "
            "JOIN fact_provenance p ON p.fact_id = f.fact_id "
            "JOIN observations o ON o.observation_id = p.observation_id "
            "ORDER BY f.fact_id"
        ).fetchall()
    ]
    assert rows == [
        ("Ada prefers concise responses.", "user"),
        ("Ada uses local-first memory tools.", "user"),
    ]
    rejected = "\n".join(row[0] for row in rows)
    assert "I'll now implement" not in rejected
    assert "I'm Brick" not in rejected
    assert "METIS_FLEET_OK" not in rejected
    assert "the port is 9999" not in rejected
    memory.shutdown()
    conn.close()


class _LegacyStringExtractor:
    identity = "legacy-fixture-extractor:v1"

    def __init__(self):
        self.calls = 0

    def extract(self, envelope):
        self.calls += 1
        span = extraction_spans.transcript_spans(envelope.transcript)[0]
        return (
            ExtractedMemory(
                "Ada prefers concise responses.",
                evidence_excerpt=span.text,
                metadata={"evidence_span_id": span.span_id},
            ),
        )


def test_legacy_opaque_transcript_is_accepted_without_becoming_user_testimony(
    tmp_path,
):
    conn, context, service = _setup_processor(tmp_path)
    ExtractionEnqueuer(conn).enqueue_after_commit(
        context,
        "Ada prefers concise responses.",
        source="legacy_client",
    )
    extractor = _LegacyStringExtractor()

    result = ExtractionProcessor(
        conn,
        service,
        extractor,
        evidence_verifier=_VerifiedEvidence(),
    ).process_one()

    assert result.outcome == "completed"
    assert result.writes == 0
    assert extractor.calls == 0
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    conn.close()
