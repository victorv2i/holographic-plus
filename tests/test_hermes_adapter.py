from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

import pytest

from enfold.client import EnfoldTransportError
from enfold.hermes_adapter import (
    HERMES_CLIENT_ID,
    DegradedReadResult,
    HermesAdapterConfig,
    HermesMemorySession,
    HermesProtocolAdapter,
    HermesSessionContext,
)
from enfold.extraction_spans import TranscriptInput


class RecordingTransport:
    instances = []

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.instances.append(self)

    def request(self, method, params=None, *, request_id=None):
        self.calls.append((method, params, request_id))
        return {"method": method, "params": params}


class OfflineTransport(RecordingTransport):
    def request(self, method, params=None, *, request_id=None):
        self.calls.append((method, params, request_id))
        raise EnfoldTransportError("offline")


class ReadOnlyFallback:
    def __init__(self):
        self.calls = []

    def read(self, method, params, *, context):
        self.calls.append((method, params, context))
        return {"facts": [{"content": "cached"}]}


@dataclass
class FakeHermesHost:
    agent_id: str = "avery"
    session_id: str = "session-42"
    parent_agent_id: str | None = "avery-main"
    project_root: str = "/work/enfold"
    repository: str = "avery/enfold"
    branch: str = "adapter"
    commit_sha: str = "abc123"


@pytest.fixture(autouse=True)
def clear_transports():
    RecordingTransport.instances.clear()


def adapter(tmp_path: Path, transport=RecordingTransport, **kwargs):
    return HermesProtocolAdapter(
        HermesAdapterConfig(tmp_path / "enfold.sock"),
        transport_factory=transport,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("connect_timeout", float("nan")),
        ("connect_timeout", float("inf")),
        ("request_timeout", 0),
    ],
)
def test_adapter_config_rejects_invalid_timeouts(tmp_path, field, value):
    with pytest.raises(ValueError, match="timeouts must be positive"):
        HermesAdapterConfig(tmp_path / "enfold.sock", **{field: value})


def test_fake_host_context_becomes_immutable_handshake_provenance(tmp_path):
    memory = adapter(tmp_path).open_host_session(
        FakeHermesHost(), access_scopes=("private", "project:enfold")
    )
    context = RecordingTransport.instances[0].config.context

    assert context.client_id == HERMES_CLIENT_ID
    assert context.surface == "hermes"
    assert context.agent_id == "avery"
    assert context.session_id == "session-42"
    assert context.parent_agent_id == "avery-main"
    assert context.project_root == "/work/enfold"
    assert context.repository == "avery/enfold"
    assert context.branch == "adapter"
    assert context.commit_sha == "abc123"
    assert context.access_scopes == ("private", "project:enfold")
    assert memory.context == context


def test_adapter_passes_supervisor_credential_to_every_session(tmp_path):
    root = HermesProtocolAdapter(
        HermesAdapterConfig(
            tmp_path / "enfold.sock", credential="credential-from-supervisor"
        ),
        transport_factory=RecordingTransport,
    )

    root.open_session(HermesSessionContext("avery", "s-1", ("private",)))

    assert RecordingTransport.instances[0].config.credential == (
        "credential-from-supervisor"
    )


def test_adapter_config_repr_omits_plaintext_credential(tmp_path):
    credential = "operator-" + "credential-value"

    rendered = repr(
        HermesAdapterConfig(tmp_path / "enfold.sock", credential=credential)
    )

    assert credential not in rendered
    assert "credential=" not in rendered


def test_mapping_host_is_supported_but_scopes_are_explicit(tmp_path):
    host = {"agent_id": "delegate-1", "session_id": "job-9"}
    memory = adapter(tmp_path).open_host_session(host, access_scopes=("work",))

    assert memory.context.agent_id == "delegate-1"
    assert memory.context.access_scopes == ("work",)
    with pytest.raises(TypeError):
        adapter(tmp_path).open_host_session(host)  # type: ignore[call-arg]


def test_write_maps_fields_and_uses_stable_event_idempotency(tmp_path):
    memory = adapter(tmp_path).open_session(
        HermesSessionContext("avery", "s-1", ("private",))
    )

    first = memory.write(
        "Avery prefers concise replies",
        event_id="message-123",
        source_type="conversation",
        scope="private",
        category="user_pref",
        asserted_by="avery",
    )
    second = memory.write(
        "Avery prefers concise replies",
        event_id="message-123",
        source_type="conversation",
        scope="private",
        category="user_pref",
        asserted_by="avery",
    )

    calls = RecordingTransport.instances[0].calls
    assert first["method"] == "memory.write"
    assert calls[0][1]["idempotency_key"] == calls[1][1]["idempotency_key"]
    assert calls[0][1]["idempotency_key"].startswith("hermes-v1:")
    assert calls[0][1]["source_type"] == "conversation"
    assert calls[0][1]["scope"] == "private"
    assert second == first


def test_idempotency_is_session_and_event_specific(tmp_path):
    root = adapter(tmp_path)
    a = root.open_session(HermesSessionContext("avery", "s-1", ("private",)))
    b = root.open_session(HermesSessionContext("avery", "s-2", ("private",)))

    assert a.idempotency_key("turn-1") == a.idempotency_key("turn-1")
    assert a.idempotency_key("turn-1") != a.idempotency_key("turn-2")
    assert a.idempotency_key("turn-1") != b.idempotency_key("turn-1")
    with pytest.raises(ValueError, match="event_id"):
        a.idempotency_key(" ")


def test_all_read_operations_map_to_proxy(tmp_path):
    memory = adapter(tmp_path).open_session(
        HermesSessionContext("avery", "s-1", ("private",))
    )

    memory.search("Avery", category="person", limit=5)
    memory.memory_context("current project", token_budget=144, scope="private")
    memory.evidence(7, limit=3)
    memory.history(subject_key="avery", predicate_key="preference")
    memory.conflicts(scope="private", unresolved_only=False)

    assert RecordingTransport.instances[0].calls == [
        ("memory.search", {"query": "Avery", "category": "person", "limit": 5}, None),
        (
            "memory.context",
            {"query": "current project", "token_budget": 144, "scope": "private"},
            None,
        ),
        ("memory.evidence", {"fact_id": 7, "limit": 3}, None),
        (
            "memory.history",
            {"subject_key": "avery", "predicate_key": "preference"},
            None,
        ),
        (
            "memory.conflicts",
            {"unresolved_only": False, "scope": "private"},
            None,
        ),
    ]


def test_promote_maps_to_retry_stable_protocol_call(tmp_path):
    memory = adapter(tmp_path).open_session(
        HermesSessionContext("avery", "s-1", ("private",))
    )
    memory.promote(12, event_id="promote-1", target_scope="work")
    assert RecordingTransport.instances[0].calls == [
        (
            "memory.promote",
            {
                "fact_id": 12,
                "idempotency_key": memory.idempotency_key("promote-1"),
                "target_scope": "work",
            },
            None,
        )
    ]


def test_conflict_resolution_maps_to_authoritative_write(tmp_path):
    memory = adapter(tmp_path).open_session(
        HermesSessionContext("avery", "s-1", ("private",))
    )
    memory.resolve_conflict("conflict-1", 7, reason="Avery selected it")
    assert RecordingTransport.instances[0].calls == [
        (
            "memory.resolve_conflict",
            {
                "conflict_id": "conflict-1",
                "resolution_fact_id": 7,
                "reason": "Avery selected it",
            },
            None,
        )
    ]


def test_extraction_enqueue_maps_to_attributed_daemon_surface(tmp_path):
    memory = adapter(tmp_path).open_session(
        HermesSessionContext("avery", "s-1", ("private",))
    )
    memory.enqueue_extraction(
        "USER: durable preference", source="session_end", metadata={"hook": "session_end"}
    )
    assert RecordingTransport.instances[0].calls == [
        (
            "memory.extraction.enqueue",
            {
                "transcript": "USER: durable preference",
                "source": "session_end",
                "scope": "private",
                "metadata": {"hook": "session_end"},
            },
            None,
        )
    ]


def test_extraction_enqueue_contract_accepts_role_structured_turns():
    annotation = get_type_hints(HermesMemorySession.enqueue_extraction)["transcript"]

    assert annotation == TranscriptInput


def test_offline_write_fails_explicitly_and_never_uses_read_fallback(tmp_path):
    fallback = ReadOnlyFallback()
    memory = adapter(
        tmp_path, OfflineTransport, degraded_provider=fallback
    ).open_session(HermesSessionContext("avery", "s-1", ("private",)))

    with pytest.raises(EnfoldTransportError, match="offline"):
        memory.write(
            "do not queue this",
            event_id="event-1",
            source_type="conversation",
            scope="private",
        )
    assert fallback.calls == []


def test_read_fallback_is_opt_in_and_visibly_degraded(tmp_path):
    fallback = ReadOnlyFallback()
    memory = adapter(
        tmp_path, OfflineTransport, degraded_provider=fallback
    ).open_session(HermesSessionContext("avery", "s-1", ("private",)))

    result = memory.search("Avery", limit=2)

    assert isinstance(result, DegradedReadResult)
    assert result.degraded is True
    assert result.result == {"facts": [{"content": "cached"}]}
    assert fallback.calls[0][0:2] == ("memory.search", {"query": "Avery", "limit": 2})
    assert fallback.calls[0][2] == memory.context


def test_read_without_fallback_fails_explicitly(tmp_path):
    memory = adapter(tmp_path, OfflineTransport).open_session(
        HermesSessionContext("avery", "s-1", ("private",))
    )
    with pytest.raises(EnfoldTransportError, match="offline"):
        memory.search("Avery")


def test_reserved_fields_cannot_spoof_write_contract(tmp_path):
    memory = adapter(tmp_path).open_session(
        HermesSessionContext("avery", "s-1", ("private",))
    )
    with pytest.raises(ValueError, match="reserved"):
        memory.write(
            "fact",
            event_id="event",
            source_type="conversation",
            scope="private",
            idempotency_key="spoof",
        )
