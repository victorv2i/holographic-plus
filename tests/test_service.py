from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from enfold.core_store import insert_fact
from enfold.policy import MemoryPolicy
from enfold.extraction_enqueue import ExtractionEnqueuer
from enfold.embeddings import embedding_to_bytes
from enfold.hybrid_retrieval import HybridRetriever, deterministic_retriever_factory
from enfold.protocol import ClientContext, Request
from enfold.schema import migrate
from enfold.service import EnfoldService, MAX_WRITE_TEXT_BYTES, ServiceRequestError
from enfold.state_slots import open_state_conflict


def _store(tmp_path) -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / "enfold-service.db")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def _context(
    client: str,
    surface: str,
    agent: str,
    *,
    scopes: tuple[str, ...] = ("private", "work"),
    session: str | None = None,
) -> ClientContext:
    return ClientContext(
        client_id=client,
        surface=surface,
        agent_id=agent,
        session_id=session or f"{agent}-session",
        repository="enfold",
        branch="service-layer",
        commit_sha="abc123",
        access_scopes=scopes,
    )


def _request(request_id: str, method: str, **params) -> Request:
    return Request(request_id, method, params)


def _write(
    service: EnfoldService,
    context: ClientContext,
    key: str,
    content: str,
    **params,
):
    return service.handle(
        context,
        _request(
            f"req-{key}",
            "memory.write",
            idempotency_key=key,
            content=content,
            source_type="agent_report",
            **params,
        ),
    )


@pytest.mark.parametrize("field", ["content", "observation_content"])
def test_write_text_over_service_byte_limit_is_typed_validation_error(
    tmp_path, field
):
    conn = _store(tmp_path)
    service = EnfoldService(conn, MemoryPolicy({"terminal-install": ("private",)}))
    context = _context("terminal-install", "terminal", "terminal", scopes=("private",))
    oversized = "é" * (MAX_WRITE_TEXT_BYTES // 2 + 1)
    params = {
        "idempotency_key": f"oversized-{field}",
        "content": "Small fact text.",
        "source_type": "agent_report",
        field: oversized,
    }

    with pytest.raises(ServiceRequestError, match="UTF-8 bytes") as rejected:
        service.handle(context, _request("oversized", "memory.write", **params))

    assert rejected.value.code == "invalid_params"
    assert conn.execute("SELECT count(*) FROM facts").fetchone()[0] == 0
    conn.close()


def test_atomic_write_batch_rolls_back_prior_write_on_policy_rejection(tmp_path):
    conn = _store(tmp_path)
    service = EnfoldService(conn, MemoryPolicy({"terminal-install": ("private",)}))
    context = _context("terminal-install", "terminal", "terminal", scopes=("private",))
    requests = (
        _request(
            "batch-1",
            "memory.write",
            idempotency_key="batch-1",
            content="A valid first batch fact.",
            source_type="automatic_extraction",
        ),
        _request(
            "batch-2",
            "memory.write",
            idempotency_key="batch-2",
            content="api_key = abcdefghijklmnopqrstuv",
            source_type="automatic_extraction",
        ),
    )

    batch = service.handle_write_batch(context, requests)

    assert batch.committed is False
    assert [response["outcome"] for response in batch.responses] == [
        "inserted",
        "rejected",
    ]
    for table in (
        "facts",
        "observations",
        "fact_provenance",
        "memory_write_log",
    ):
        assert conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0] == 0
    conn.close()


@pytest.mark.parametrize(
    ("near_dedup_enabled", "expected_outcome", "expected_history_size"),
    [(True, "near_dedup", 2), (False, "inserted", 1)],
)
def test_service_near_duplicate_merge_and_off_switch(
    tmp_path, near_dedup_enabled, expected_outcome, expected_history_size
):
    conn = _store(tmp_path)
    identity = "fake:service:document:none:v1"
    service = EnfoldService(
        conn,
        MemoryPolicy({"terminal-install": ("private",)}),
        embedding_identity=identity,
        query_embedder=lambda _content: np.asarray((1.0, 0.0), dtype=np.float32),
        near_dedup_enabled=near_dedup_enabled,
    )
    context = _context("terminal-install", "terminal", "terminal", scopes=("private",))
    existing = _write(
        service, context, "existing", "The build uses port 3100.", trust_score=0.8
    )
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, 2, ?)",
        (
            existing["fact_id"],
            embedding_to_bytes(np.asarray((1.0, 0.0), dtype=np.float32)),
            identity,
        ),
    )
    conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
    conn.commit()

    result = _write(
        service,
        context,
        "paraphrase",
        "Build service listens on port 3100.",
        trust_score=0.4,
    )
    history = service.handle(
        context,
        _request("history-near-dedup", "memory.history", fact_id=result["fact_id"]),
    )["facts"]

    assert result["outcome"] == expected_outcome
    assert len(history) == expected_history_size
    if near_dedup_enabled:
        assert result["fact_id"] == existing["fact_id"]
        assert history[1]["superseded_by"] == existing["fact_id"]
    conn.close()


def test_retrieval_metadata_can_use_a_caller_supplied_connection(tmp_path):
    writer = _store(tmp_path)
    builds: list[sqlite3.Connection] = []
    inner = deterministic_retriever_factory()

    def factory(used_conn, scopes):
        builds.append(used_conn)
        return inner(used_conn, scopes)

    service = EnfoldService(
        writer,
        MemoryPolicy({"terminal-install": ("private",)}),
        retriever_factory=factory,
    )
    inspector = sqlite3.connect(tmp_path / "enfold-service.db")
    inspector.row_factory = sqlite3.Row
    builds.clear()

    metadata = service.retrieval_metadata_for(inspector)

    assert builds == [inspector]
    assert writer not in builds
    assert metadata["retrieval_stack"]
    inspector.close()
    writer.close()


def test_service_defaults_near_dedup_off_for_preference_paraphrase(tmp_path):
    conn = _store(tmp_path)
    identity = "fake:service:document:none:v1"
    service = EnfoldService(
        conn,
        MemoryPolicy({"terminal-install": ("private",)}),
        embedding_identity=identity,
        query_embedder=lambda _content: np.asarray((1.0, 0.0), dtype=np.float32),
    )
    context = _context("terminal-install", "terminal", "terminal", scopes=("private",))
    existing = _write(
        service, context, "existing-pref", "Avery prefers tea.", trust_score=0.8
    )
    conn.execute(
        "INSERT INTO fact_embeddings(fact_id, embedding, dim, embedding_identity) "
        "VALUES (?, ?, 2, ?)",
        (
            existing["fact_id"],
            embedding_to_bytes(np.asarray((1.0, 0.0), dtype=np.float32)),
            identity,
        ),
    )
    conn.execute("INSERT INTO facts_fts(facts_fts) VALUES ('rebuild')")
    conn.commit()

    result = _write(
        service,
        context,
        "incoming-pref",
        "Avery prefers tea with honey.",
        trust_score=0.9,
    )

    assert result["outcome"] == "inserted"
    assert result["fact_id"] != existing["fact_id"]
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM facts WHERE invalid_at IS NULL"
        ).fetchone()[0]
        == 2
    )
    conn.close()


@pytest.fixture
def setup(tmp_path):
    conn = _store(tmp_path)
    grants = {
        "client-a-install": ("private", "work", "secret"),
        "client-b-install": ("private",),
        "hermes-install": ("private", "work"),
    }
    service = EnfoldService(
        conn,
        MemoryPolicy(
            grants,
            correction_authorities=("hermes-install",),
            conflict_resolution_authorities=("hermes-install",),
        ),
    )
    contexts = {
        "client-a": _context("client-a-install", "client-a", "client-a"),
        "client-b": _context("client-b-install", "client-b", "client-b"),
        "hermes": _context("hermes-install", "hermes", "avery"),
    }
    yield conn, service, contexts
    conn.close()


def test_cross_agent_writes_have_trusted_client_and_hermes_provenance(setup):
    conn, service, contexts = setup
    outcomes = {
        name: _write(service, context, f"{name}-1", f"{name} built the memory bridge")
        for name, context in contexts.items()
    }

    rows = conn.execute(
        """SELECT client_id, session_id, performed_by, repository, branch, commit_sha
           FROM observations ORDER BY observation_id"""
    ).fetchall()
    assert [tuple(row) for row in rows] == [
        (
            "client-a-install",
            "client-a-session",
            "client-a",
            "enfold",
            "service-layer",
            "abc123",
        ),
        (
            "client-b-install",
            "client-b-session",
            "client-b",
            "enfold",
            "service-layer",
            "abc123",
        ),
        (
            "hermes-install",
            "avery-session",
            "avery",
            "enfold",
            "service-layer",
            "abc123",
        ),
    ]
    for name, result in outcomes.items():
        evidence = service.handle(
            contexts[name],
            _request(f"evidence-{name}", "memory.evidence", fact_id=result["fact_id"]),
        )
        assert evidence["evidence"][0]["client_id"] == f"{name}-install"
        assert evidence["evidence"][0]["performed_by"] == contexts[name].agent_id


def test_server_grants_narrow_reads_and_writes_without_cross_scope_oracles(setup):
    conn, service, contexts = setup
    work = _write(
        service,
        contexts["client-a"],
        "work-1",
        "Project Zephyr deployment token rotation completed",
        scope="work",
    )
    private = _write(
        service,
        contexts["client-b"],
        "private-1",
        "Project Zephyr private notes are indexed",
    )
    client_b_requested_too_much = _context(
        "client-b-install", "client-b", "client-b", scopes=("private", "work")
    )

    results = service.handle(
        client_b_requested_too_much,
        _request("search-1", "memory.search", query="Zephyr"),
    )["facts"]
    assert [fact["fact_id"] for fact in results] == [private["fact_id"]]
    rejected = _write(
        service,
        client_b_requested_too_much,
        "work-denied",
        "Client B cannot self-grant a work memory",
        scope="work",
    )
    assert rejected["outcome"] == "rejected"
    assert rejected["fact_id"] is None
    assert (
        conn.execute(
            "SELECT COUNT(*) FROM facts WHERE fact_id = ?", (work["fact_id"],)
        ).fetchone()[0]
        == 1
    )

    with pytest.raises(ServiceRequestError, match="not found") as hidden:
        service.handle(
            client_b_requested_too_much,
            _request("evidence-hidden", "memory.evidence", fact_id=work["fact_id"]),
        )
    assert hidden.value.code == "not_found"


def test_sensitive_facts_require_matching_capability_on_every_read_surface(tmp_path):
    conn = _store(tmp_path)
    policy = MemoryPolicy(
        {
            "sensitive-install": ("private", "sensitive"),
            "ordinary-install": ("private",),
        }
    )
    service = EnfoldService(conn, policy)
    sensitive_context = _context(
        "sensitive-install",
        "terminal",
        "terminal",
        scopes=("private", "sensitive"),
    )
    ordinary_context = _context(
        "ordinary-install", "terminal", "terminal", scopes=("private",)
    )
    sensitive_not_requested = _context(
        "sensitive-install",
        "terminal",
        "terminal",
        scopes=("private",),
        session="terminal-not-sensitive",
    )
    rejected = _write(
        service,
        sensitive_not_requested,
        "sensitive-capability-not-requested",
        "Avery hidden deployment plan",
        scope="private",
        sensitivity="sensitive",
    )
    assert rejected["outcome"] == "rejected"
    assert rejected["fact_id"] is None
    stored = _write(
        service,
        sensitive_context,
        "sensitive-private",
        "Avery sensitive deployment planning",
        scope="private",
        sensitivity="sensitive",
    )

    assert service.handle(
        ordinary_context,
        _request("search-sensitive", "memory.search", query="deployment planning"),
    )["facts"] == []
    assert service.handle(
        ordinary_context,
        _request(
            "context-sensitive",
            "memory.context",
            query="deployment planning",
            token_budget=256,
        ),
    )["facts"] == []
    for method in ("memory.evidence", "memory.history"):
        with pytest.raises(ServiceRequestError, match="not found") as hidden:
            service.handle(
                ordinary_context,
                _request(f"hidden-{method}", method, fact_id=stored["fact_id"]),
            )
        assert hidden.value.code == "not_found"

    visible = service.handle(
        sensitive_context,
        _request("visible-sensitive", "memory.evidence", fact_id=stored["fact_id"]),
    )
    assert visible["fact"]["fact_id"] == stored["fact_id"]
    conn.close()


def test_sensitive_provenance_requires_matching_capability(tmp_path):
    conn = _store(tmp_path)
    policy = MemoryPolicy(
        {
            "sensitive-install": ("private", "sensitive"),
            "ordinary-install": ("private",),
        }
    )
    service = EnfoldService(conn, policy)
    sensitive_context = _context(
        "sensitive-install",
        "terminal",
        "terminal",
        scopes=("private", "sensitive"),
    )
    ordinary_context = _context(
        "ordinary-install", "terminal", "terminal", scopes=("private",)
    )
    normal = _write(
        service,
        ordinary_context,
        "normal-observation",
        "Avery uses Enfold for shared memory",
    )
    _write(
        service,
        sensitive_context,
        "sensitive-observation",
        "Avery uses Enfold for shared memory",
        sensitivity="sensitive",
    )

    ordinary = service.handle(
        ordinary_context,
        _request("ordinary-evidence", "memory.evidence", fact_id=normal["fact_id"]),
    )
    assert [item["client_id"] for item in ordinary["evidence"]] == [
        "ordinary-install"
    ]
    ordinary_search = service.handle(
        ordinary_context,
        _request("ordinary-search", "memory.search", query="shared memory"),
    )["facts"]
    assert ordinary_search[0]["attribution"]["performed_by"] == "terminal"
    assert ordinary_search[0]["attribution"]["evidence_count"] == 1

    privileged = service.handle(
        sensitive_context,
        _request(
            "privileged-evidence", "memory.evidence", fact_id=normal["fact_id"]
        ),
    )
    assert {item["client_id"] for item in privileged["evidence"]} == {
        "ordinary-install",
        "sensitive-install",
    }
    conn.close()


def test_history_filters_sensitive_successors_from_visible_anchor(tmp_path):
    conn = _store(tmp_path)
    policy = MemoryPolicy(
        {
            "sensitive-install": ("private", "sensitive"),
            "ordinary-install": ("private",),
        }
    )
    service = EnfoldService(conn, policy)
    ordinary_context = _context(
        "ordinary-install", "terminal", "terminal", scopes=("private",)
    )
    sensitive_context = _context(
        "sensitive-install",
        "terminal",
        "terminal",
        scopes=("private", "sensitive"),
    )
    state = {"subject_key": "project:enfold", "predicate_key": "release_plan"}
    normal = _write(
        service,
        ordinary_context,
        "normal-history",
        "Enfold release plan is an alpha",
        state={**state, "object_value": "alpha", "valid_from": "2026-08-01T00:00:00Z"},
    )
    _write(
        service,
        sensitive_context,
        "sensitive-history",
        "Enfold release plan has a sensitive codename",
        sensitivity="sensitive",
        state={
            **state,
            "object_value": "sensitive-codename",
            "valid_from": "2026-08-02T00:00:00Z",
        },
    )

    ordinary = service.handle(
        ordinary_context,
        _request("ordinary-history", "memory.history", fact_id=normal["fact_id"]),
    )
    assert [fact["fact_id"] for fact in ordinary["facts"]] == [normal["fact_id"]]
    privileged = service.handle(
        sensitive_context,
        _request("privileged-history", "memory.history", fact_id=normal["fact_id"]),
    )
    assert len(privileged["facts"]) == 2
    conn.close()


def test_conflict_reads_require_visible_member_sensitivity(tmp_path):
    conn = _store(tmp_path)
    fact_id = insert_fact(
        conn,
        "Enfold release codename is sensitive",
        scope="private",
        sensitivity="sensitive",
        memory_kind="state",
        subject_key="project:enfold",
        predicate_key="release_codename",
    )
    conflict = open_state_conflict(
        conn,
        "project:enfold",
        "release_codename",
        (fact_id,),
        scope="private",
    )
    conn.commit()
    service = EnfoldService(
        conn,
        MemoryPolicy(
            {
                "sensitive-install": ("private", "sensitive"),
                "ordinary-install": ("private",),
            },
            conflict_resolution_authorities=(
                "sensitive-install",
                "ordinary-install",
            ),
        ),
    )
    ordinary_context = _context(
        "ordinary-install", "terminal", "terminal", scopes=("private",)
    )
    sensitive_context = _context(
        "sensitive-install",
        "terminal",
        "terminal",
        scopes=("private", "sensitive"),
    )

    assert service.handle(
        ordinary_context,
        _request("ordinary-conflicts", "memory.conflicts"),
    )["conflicts"] == []
    with pytest.raises(ServiceRequestError, match="not found") as hidden_resolution:
        service.handle(
            ordinary_context,
            _request(
                "ordinary-resolution",
                "memory.resolve_conflict",
                conflict_id=conflict.conflict_id,
                resolution_fact_id=fact_id,
                reason="must not resolve a hidden conflict",
            ),
        )
    assert hidden_resolution.value.code == "not_found"
    privileged = service.handle(
        sensitive_context,
        _request("sensitive-conflicts", "memory.conflicts"),
    )["conflicts"]
    assert privileged[0]["conflict_id"] == conflict.conflict_id
    assert privileged[0]["members"][0]["fact_id"] == fact_id
    conn.close()


def test_search_answers_who_did_or_learned_with_visible_only_attribution(setup):
    conn, service, contexts = setup
    private = _write(
        service,
        contexts["client-b"],
        "attribution-private",
        "Client B learned that Zephyr deploys on Tuesday",
    )
    work = _write(
        service,
        contexts["client-a"],
        "attribution-work",
        "Client A independently verified the Zephyr Tuesday deployment",
        scope="work",
    )
    work_observation = conn.execute(
        "SELECT observation_id FROM memory_write_log WHERE fact_id = ?",
        (work["fact_id"],),
    ).fetchone()[0]
    conn.execute(
        "INSERT INTO fact_provenance(fact_id, observation_id, relation, created_at) "
        "VALUES (?, ?, 'supports', '2099-01-01T00:00:00Z')",
        (private["fact_id"], work_observation),
    )
    conn.commit()

    client_b = service.handle(
        contexts["client-b"],
        _request("who-client-b", "memory.search", query="Zephyr deploys Tuesday"),
    )["facts"][0]["attribution"]
    assert client_b == {
        "performed_by": "client-b",
        "agent_id": "client-b",
        "session_id": "client-b-session",
        "source_type": "agent_report",
        "repository": "enfold",
        "branch": "service-layer",
        "commit_sha": "abc123",
        "evidence_count": 1,
    }

    hermes = service.handle(
        contexts["hermes"],
        _request("who-hermes", "memory.search", query="Zephyr deploys Tuesday"),
    )["facts"][0]["attribution"]
    assert hermes["performed_by"] == "client-a"
    assert hermes["agent_id"] == "client-a"
    assert hermes["evidence_count"] == 2


def test_secret_and_credential_writes_are_factless_and_replayable(setup):
    conn, service, contexts = setup
    client_a_secret = _context(
        "client-a-install",
        "client-a",
        "client-a",
        scopes=("private", "secret"),
        session="client-a-secret-session",
    )
    before = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    secret = _write(
        service,
        client_a_secret,
        "secret-1",
        "Do not persist this material",
        scope="secret",
        sensitivity="secret",
    )
    credential = _write(
        service,
        contexts["hermes"],
        "secret-2",
        "api_key=supersecretcredentialvalue",
    )
    replay = _write(
        service,
        client_a_secret,
        "secret-1",
        "Do not persist this material",
        scope="secret",
        sensitivity="secret",
    )

    assert secret["outcome"] == credential["outcome"] == "rejected"
    assert secret["fact_id"] is credential["fact_id"] is None
    assert replay["replayed"] is True
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == before
    assert conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0] == 0


def test_state_and_attribution_fields_are_credential_screened(setup):
    conn, service, contexts = setup
    before = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    result = _write(
        service,
        contexts["client-a"],
        "screen-state",
        "An otherwise harmless claim",
        asserted_by="Avery",
        state={
            "subject_key": "service:database",
            "predicate_key": "connection",
            "object_value": (
                "postgresql://user:" + "password123" + "@example.test/db"
            ),
        },
    )
    assert result["outcome"] == "rejected"
    assert result["fact_id"] is None
    assert conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0] == before
    assert (
        "password123"
        not in conn.execute(
            "SELECT detail_json FROM memory_write_log WHERE idempotency_key = 'screen-state'"
        ).fetchone()[0]
    )


def test_state_supersession_conflict_history_evidence_and_settled_search(setup):
    _conn, service, contexts = setup
    state = {"subject_key": "agent:avery", "predicate_key": "preferred_model"}
    first = _write(
        service,
        contexts["hermes"],
        "state-1",
        "Avery prefers model Terra 5.5",
        source_authority=0.8,
        state={
            **state,
            "object_value": "terra-5.5",
            "valid_from": "2026-07-11T10:00:00Z",
        },
        evidence_excerpt="Observed in the Hermes configuration.",
    )
    replacement = _write(
        service,
        contexts["client-a"],
        "state-2",
        "Avery prefers model Terra 5.6",
        source_authority=0.8,
        state={
            **state,
            "object_value": "terra-5.6",
            "valid_from": "2026-07-12T10:00:00Z",
        },
    )
    conflicting = _write(
        service,
        contexts["client-b"],
        "state-3",
        "Avery prefers model Unknown 1",
        source_authority=0.2,
        state={
            **state,
            "object_value": "unknown-1",
            "valid_from": "2026-07-13T10:00:00Z",
        },
    )

    assert first["outcome"] == "add"
    assert replacement["outcome"] == "supersede"
    assert conflicting["outcome"] == "conflict"
    history = service.handle(
        contexts["hermes"],
        _request("history-1", "memory.history", **state, scope="private"),
    )["facts"]
    assert [fact["fact_id"] for fact in history] == [
        first["fact_id"],
        replacement["fact_id"],
        conflicting["fact_id"],
    ]
    assert history[0]["superseded_by"] == replacement["fact_id"]

    evidence = service.handle(
        contexts["hermes"],
        _request("evidence-1", "memory.evidence", fact_id=first["fact_id"]),
    )
    assert evidence["evidence"][0]["evidence_excerpt"] == (
        "Observed in the Hermes configuration."
    )
    assert evidence["evidence"][0]["client_id"] == "hermes-install"

    conflicts = service.handle(
        contexts["hermes"], _request("conflicts-1", "memory.conflicts")
    )["conflicts"]
    assert len(conflicts) == 1
    assert set(conflicts[0]["member_fact_ids"]) == {
        replacement["fact_id"],
        conflicting["fact_id"],
    }
    assert {member["fact_id"] for member in conflicts[0]["members"]} == set(
        conflicts[0]["member_fact_ids"]
    )
    assert (
        service.handle(
            contexts["hermes"],
            _request("search-conflict", "memory.search", query="prefers model"),
        )["facts"]
        == []
    )


def test_authorized_conflict_resolution_restores_settled_truth_and_audits(setup):
    conn, service, contexts = setup
    slot = {"subject_key": "agent:avery", "predicate_key": "model"}
    first = _write(
        service,
        contexts["hermes"],
        "resolve-1",
        "Avery uses Terra",
        source_authority=0.8,
        state={**slot, "object_value": "terra", "valid_from": "2026-07-11T10:00:00Z"},
    )
    other = _write(
        service,
        contexts["client-b"],
        "resolve-2",
        "Avery uses Model Z",
        source_authority=0.2,
        state={**slot, "object_value": "model-z", "valid_from": "2026-07-12T10:00:00Z"},
    )
    conflict_id = other["detail"]["conflict_id"]

    with pytest.raises(ServiceRequestError) as denied:
        service.handle(
            contexts["client-b"],
            _request(
                "resolve-denied",
                "memory.resolve_conflict",
                conflict_id=conflict_id,
                resolution_fact_id=first["fact_id"],
                reason="self-claimed authority",
            ),
        )
    assert denied.value.code == "access_denied"

    result = service.handle(
        contexts["hermes"],
        _request(
            "resolve-ok",
            "memory.resolve_conflict",
            conflict_id=conflict_id,
            resolution_fact_id=first["fact_id"],
            reason="Avery confirmed Terra",
        ),
    )["resolution"]
    assert result["resolution_fact_id"] == first["fact_id"]
    assert result["superseded_fact_ids"] == [other["fact_id"]]
    assert (
        service.handle(
            contexts["hermes"], _request("settled", "memory.search", query="Terra")
        )["facts"][0]["fact_id"]
        == first["fact_id"]
    )
    audit = conn.execute(
        """SELECT resolver_client_id, resolver_session_id, resolver_agent_id, reason
           FROM fact_conflict_resolutions WHERE conflict_id = ?""",
        (conflict_id,),
    ).fetchone()
    assert tuple(audit) == (
        "hermes-install",
        "avery-session",
        "avery",
        "Avery confirmed Terra",
    )


def test_undated_state_uses_observed_time_and_untyped_dedup_is_scope_local(setup):
    conn, service, contexts = setup
    slot = {"subject_key": "project:enfold", "predicate_key": "phase"}
    first = _write(
        service,
        contexts["client-a"],
        "time-1",
        "Enfold is in alpha",
        observed_at="2026-07-11T10:00:00Z",
        source_authority=0.5,
        state={**slot, "object_value": "alpha"},
    )
    second = _write(
        service,
        contexts["client-a"],
        "time-2",
        "Enfold is in beta",
        observed_at="2026-07-12T10:00:00Z",
        source_authority=0.5,
        state={**slot, "object_value": "beta"},
    )
    assert first["outcome"] == "add"
    assert second["outcome"] == "supersede"
    assert (
        conn.execute(
            "SELECT valid_from FROM facts WHERE fact_id = ?", (second["fact_id"],)
        ).fetchone()[0]
        == "2026-07-12T10:00:00Z"
    )

    private = _write(service, contexts["client-a"], "dedup-1", "Exact shared text")
    duplicate = _write(service, contexts["hermes"], "dedup-2", "Exact shared text")
    work = _write(
        service, contexts["client-a"], "dedup-3", "Exact shared text", scope="work"
    )
    assert duplicate["outcome"] == "dedup"
    assert duplicate["fact_id"] == private["fact_id"]
    assert work["fact_id"] != private["fact_id"]


def test_context_is_scoped_cited_current_and_conflict_safe(setup):
    _conn, service, contexts = setup
    private = _write(
        service,
        contexts["client-b"],
        "context-private",
        "CedarContext private registry has the private endpoint.",
    )
    work = _write(
        service,
        contexts["client-a"],
        "context-work",
        "CedarContext work registry has the work endpoint.",
        scope="work",
    )
    slot = {"subject_key": "atlas", "predicate_key": "backup_schedule"}
    stale = _write(
        service,
        contexts["hermes"],
        "context-stale",
        "Atlas backup schedule runs Monday.",
        source_authority=0.8,
        state={**slot, "object_value": "monday", "valid_from": "2026-07-10T10:00:00Z"},
    )
    current = _write(
        service,
        contexts["client-a"],
        "context-current",
        "Atlas backup schedule runs Tuesday.",
        source_authority=0.8,
        state={**slot, "object_value": "tuesday", "valid_from": "2026-07-11T10:00:00Z"},
    )
    conflict_slot = {"subject_key": "atlas", "predicate_key": "unstable_model"}
    _write(
        service,
        contexts["hermes"],
        "context-conflict-first",
        "UnstableAtlas model is Terra.",
        source_authority=0.8,
        state={
            **conflict_slot,
            "object_value": "terra",
            "valid_from": "2026-07-10T10:00:00Z",
        },
    )
    conflicted = _write(
        service,
        contexts["client-b"],
        "context-conflict-second",
        "UnstableAtlas model is Model Z.",
        source_authority=0.2,
        state={
            **conflict_slot,
            "object_value": "model-z",
            "valid_from": "2026-07-11T10:00:00Z",
        },
    )
    conflict_id = conflicted["detail"]["conflict_id"]
    _conn.execute(
        "UPDATE facts SET correction_status = 'human_confirmed' "
        "WHERE fact_id IN (?, ?)",
        (private["fact_id"], current["fact_id"]),
    )
    _conn.commit()

    private_pack = service.handle(
        contexts["client-a"],
        _request(
            "context-private-request",
            "memory.context",
            query="CedarContext registry",
            scope="private",
            token_budget=256,
        ),
    )
    assert private["fact_id"] in [fact["fact_id"] for fact in private_pack["facts"]]
    assert work["fact_id"] not in [fact["fact_id"] for fact in private_pack["facts"]]
    assert private_pack["facts"][0]["attribution"]["agent_id"] == "client-b"
    assert "fact:" + str(private["fact_id"]) in private_pack["markdown"]
    assert private_pack["token_estimate"]["used"] <= 256

    current_pack = service.handle(
        contexts["client-a"],
        _request(
            "context-current-request",
            "memory.context",
            query="Atlas backup schedule",
            token_budget=256,
        ),
    )
    current_ids = [fact["fact_id"] for fact in current_pack["facts"]]
    assert current["fact_id"] in current_ids
    assert stale["fact_id"] not in current_ids

    conflict_pack = service.handle(
        contexts["client-a"],
        _request(
            "context-conflict-request",
            "memory.context",
            query="UnstableAtlas model",
            token_budget=256,
        ),
    )
    assert conflict_pack["abstained"] is True
    assert f"[conflict:{conflict_id}" in conflict_pack["markdown"]
    assert "do not treat either as current" in conflict_pack["markdown"]
    assert "UnstableAtlas model is Terra" not in conflict_pack["markdown"]
    assert "UnstableAtlas model is Model Z" not in conflict_pack["markdown"]
    assert conflict_pack["open_conflicts"][0]["conflict_id"] == conflict_id
    assert all(fact.get("fact_id") for fact in conflict_pack["facts"])


def test_context_validates_scope_and_token_budget(setup):
    _conn, service, contexts = setup
    _write(service, contexts["client-a"], "context-validation", "Orchid context note")

    with pytest.raises(ServiceRequestError, match="token_budget"):
        service.handle(
            contexts["client-a"],
            _request(
                "context-budget", "memory.context", query="Orchid", token_budget=15
            ),
        )
    with pytest.raises(ServiceRequestError) as denied:
        service.handle(
            contexts["client-b"],
            _request(
                "context-scope",
                "memory.context",
                query="Orchid",
                token_budget=64,
                scope="work",
            ),
        )
    assert denied.value.code == "access_denied"


def test_unknown_clients_and_nested_identity_spoofing_fail_closed(setup):
    _conn, service, contexts = setup
    unknown = _context("unknown-install", "client-a", "client-a")
    with pytest.raises(ServiceRequestError) as denied:
        service.handle(
            unknown, _request("search-denied", "memory.search", query="anything")
        )
    assert denied.value.code == "access_denied"

    with pytest.raises(ServiceRequestError) as spoofed:
        service.handle(
            contexts["client-a"],
            _request(
                "write-spoof",
                "memory.write",
                idempotency_key="spoof",
                content="Nested identity must not be trusted",
                source_type="agent_report",
                metadata={"audit": {"client_id": "forged"}},
            ),
        )
    assert spoofed.value.code == "invalid_params"


def test_search_accepts_natural_language_and_reports_hybrid_capabilities(setup):
    _conn, service, contexts = setup
    written = _write(
        service,
        contexts["client-a"],
        "natural-search",
        "Orchid backups run every Tuesday",
    )

    response = service.handle(
        contexts["client-a"],
        _request(
            "natural-search-request",
            "memory.search",
            query='When does Orchid backup run? (current) + "schedule"',
        ),
    )

    assert response["facts"][0]["fact_id"] == written["fact_id"]
    assert "score" in response["facts"][0]
    assert response["retrieval"]["filter_before_dense_ranking"] is True
    assert response["retrieval"]["embedder_production_ready"] is False
    assert (
        response["retrieval"]["natural_language_query_parser"] == "quoted_token_or_v1"
    )


def test_service_search_serializes_dense_scores_as_builtin_json_numbers(tmp_path):
    class NumpyEmbedder:
        identity = "numpy-service-regression"
        production_ready = False

        def embed_query(self, _text):
            return np.asarray((1.0, 0.0), dtype=np.float32)

        def embed_documents(self, texts):
            return tuple(np.asarray((1.0, 0.0), dtype=np.float32) for _text in texts)

    conn = _store(tmp_path)
    service = EnfoldService(
        conn,
        MemoryPolicy({"client-a-install": ("private",)}),
        retriever_factory=lambda connection, scopes: HybridRetriever(
            connection, NumpyEmbedder(), allowed_scopes=scopes
        ),
    )
    context = _context("client-a-install", "client-a", "client-a", scopes=("private",))
    _write(
        service,
        context,
        "numpy-score",
        "The service must serialize dense scores through JSON.",
    )

    response = service.handle(
        context,
        _request("numpy-search", "memory.search", query="dense scores"),
    )

    assert type(response["facts"][0]["dense_score"]) is float
    assert type(response["facts"][0]["score"]) is float
    json.dumps(response, allow_nan=False)
    conn.close()


def test_daemon_owned_extraction_surface_enqueues_without_model_call(tmp_path):
    conn = _store(tmp_path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS extract_queue (id INTEGER PRIMARY KEY, payload TEXT NOT NULL, "
        "status TEXT NOT NULL DEFAULT 'pending', payload_hash TEXT)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS extraction_active_hash ON extract_queue(payload_hash) "
        "WHERE status IN ('pending', 'processing')"
    )
    conn.commit()
    service = EnfoldService(
        conn,
        MemoryPolicy({"client-a-install": ("private",)}),
        extraction_enqueuer=ExtractionEnqueuer(conn),
    )
    context = _context("client-a-install", "client-a", "client-a", scopes=("private",))
    request = _request(
        "extract-1",
        "memory.extraction.enqueue",
        transcript="Avery wants a shared local second brain.",
        source="session_end",
    )

    first = service.handle(context, request)
    second = service.handle(context, request)

    assert first["outcome"] == "queued"
    assert first["automatic_llm_extraction"] == "deferred"
    assert second["replayed"] is True
    assert conn.execute("SELECT count(*) FROM extract_queue").fetchone()[0] == 1
    conn.close()


def test_extraction_rejects_oversized_canonical_payload_before_enqueue(tmp_path):
    conn = _store(tmp_path)
    service = EnfoldService(
        conn,
        MemoryPolicy({"client-a-install": ("private",)}),
        extraction_enqueuer=ExtractionEnqueuer(conn),
    )
    context = _context(
        "client-a-install", "client-a", "client-a", scopes=("private",)
    )

    with pytest.raises(
        ServiceRequestError, match="canonical extraction payload"
    ) as error:
        service.handle(
            context,
            _request(
                "oversized-extraction",
                "memory.extraction.enqueue",
                transcript="x" * (12 * 1024),
                source="session_end",
            ),
        )

    assert error.value.code == "invalid_params"
    assert conn.execute("SELECT count(*) FROM extract_queue").fetchone()[0] == 0
    conn.close()


def test_huge_numeric_parameter_is_typed_validation_error(setup):
    _conn, service, contexts = setup

    with pytest.raises(ServiceRequestError) as error:
        service.handle(
            contexts["hermes"],
            _request(
                "huge-min-trust",
                "memory.search",
                query="anything",
                min_trust=10**10_000,
            ),
        )

    assert error.value.code == "invalid_params"


def test_conflicts_limit_batches_member_facts_and_reports_truncation(setup):
    conn, service, contexts = setup
    for index in range(3):
        subject = f"agent:conflict-{index}"
        predicate = "setting"
        fact_id = insert_fact(
            conn,
            f"conflict value {index}",
            memory_kind="state",
            subject_key=subject,
            predicate_key=predicate,
            object_value=str(index),
            scope="private",
        )
        open_state_conflict(
            conn,
            subject,
            predicate,
            (fact_id,),
            detected_at=f"2026-07-0{index + 1}T00:00:00Z",
        )
    conn.commit()
    statements = []
    conn.set_trace_callback(statements.append)

    result = service.handle(
        contexts["hermes"],
        _request("bounded-conflicts", "memory.conflicts", scope="private", limit=2),
    )

    conn.set_trace_callback(None)
    assert len(result["conflicts"]) == 2
    assert result["output_truncated"] is True
    member_reads = [
        statement for statement in statements
        if "FROM facts WHERE fact_id IN" in statement
    ]
    assert len(member_reads) == 1
    assert not any(
        "FROM facts WHERE fact_id =" in statement for statement in statements
    )


def test_run_scoped_writes_are_hidden_from_default_search_until_promoted(tmp_path):
    conn = _store(tmp_path)
    service = EnfoldService(conn, MemoryPolicy({"terminal-install": ("private",)}))
    context = _context("terminal-install", "terminal", "terminal", scopes=("private",))
    run = _write(
        service,
        context,
        "run-note",
        "The child found that lock ownership is safe.",
        scope="run:child-session",
    )

    default = service.handle(
        context, _request("default-search", "memory.search", query="lock ownership")
    )
    assert default["facts"] == []

    scoped = service.handle(
        context,
        _request(
            "run-search",
            "memory.search",
            query="lock ownership",
            scope="run:child-session",
        ),
    )
    assert [fact["fact_id"] for fact in scoped["facts"]] == [run["fact_id"]]

    promoted = service.handle(
        context,
        _request(
            "promote-1",
            "memory.promote",
            fact_id=run["fact_id"],
            idempotency_key="promote-lock",
            target_scope="private",
        ),
    )
    assert promoted["outcome"] in {"inserted", "deduped"}
    assert promoted["fact_id"] != run["fact_id"]

    durable = service.handle(
        context, _request("after-promote", "memory.search", query="lock ownership")
    )
    assert [fact["fact_id"] for fact in durable["facts"]] == [promoted["fact_id"]]
    fact = conn.execute(
        "SELECT scope, content FROM facts WHERE fact_id = ?",
        (promoted["fact_id"],),
    ).fetchone()
    assert tuple(fact) == ("private", "The child found that lock ownership is safe.")
    conn.close()


def test_promote_rejects_non_run_facts_and_run_targets(tmp_path):
    conn = _store(tmp_path)
    service = EnfoldService(conn, MemoryPolicy({"terminal-install": ("private",)}))
    context = _context("terminal-install", "terminal", "terminal", scopes=("private",))
    durable = _write(
        service, context, "already-private", "This fact is already durable."
    )
    run = _write(
        service,
        context,
        "run-only",
        "A worker-only observation.",
        scope="run:child-session",
    )

    with pytest.raises(ServiceRequestError, match="run-scoped") as not_run:
        service.handle(
            context,
            _request(
                "promote-private",
                "memory.promote",
                fact_id=durable["fact_id"],
                idempotency_key="promote-private",
            ),
        )
    assert not_run.value.code == "invalid_params"

    with pytest.raises(ServiceRequestError, match="durable") as run_target:
        service.handle(
            context,
            _request(
                "promote-to-run",
                "memory.promote",
                fact_id=run["fact_id"],
                idempotency_key="promote-to-run",
                target_scope="run:other",
            ),
        )
    assert run_target.value.code == "invalid_params"

    with pytest.raises(ServiceRequestError, match="not visible") as missing:
        service.handle(
            context,
            _request(
                "promote-missing",
                "memory.promote",
                fact_id=99999,
                idempotency_key="promote-missing",
            ),
        )
    assert missing.value.code == "access_denied"
    conn.close()


def test_run_fact_evidence_and_history_are_visible_to_private_clients(tmp_path):
    conn = _store(tmp_path)
    service = EnfoldService(conn, MemoryPolicy({"terminal-install": ("private",)}))
    context = _context("terminal-install", "terminal", "terminal", scopes=("private",))
    run = _write(
        service,
        context,
        "run-evidence",
        "The child found that lock ownership is safe.",
        scope="run:child-session",
    )

    evidence = service.handle(
        context, _request("run-evidence", "memory.evidence", fact_id=run["fact_id"])
    )
    assert evidence["fact"]["fact_id"] == run["fact_id"]
    assert evidence["evidence"][0]["scope"] == "run:child-session"

    history = service.handle(
        context, _request("run-history", "memory.history", fact_id=run["fact_id"])
    )
    assert any(fact["fact_id"] == run["fact_id"] for fact in history["facts"])
    conn.close()


def test_promote_hides_sensitive_run_facts_from_private_only_clients(tmp_path):
    conn = _store(tmp_path)
    service = EnfoldService(
        conn,
        MemoryPolicy(
            {
                "private-only": ("private",),
                "sensitive-writer": ("private", "sensitive"),
            }
        ),
    )
    writer = _context(
        "sensitive-writer",
        "terminal",
        "writer",
        scopes=("private", "sensitive"),
    )
    reader = _context("private-only", "terminal", "reader", scopes=("private",))
    run = _write(
        service,
        writer,
        "sensitive-run",
        "A sensitive worker observation.",
        scope="run:secret-run",
        sensitivity="sensitive",
    )

    with pytest.raises(ServiceRequestError, match="not visible") as hidden:
        service.handle(
            reader,
            _request(
                "promote-sensitive",
                "memory.promote",
                fact_id=run["fact_id"],
                idempotency_key="promote-sensitive",
            ),
        )
    assert hidden.value.code == "access_denied"
    conn.close()


def test_run_scope_search_requires_private_grant(tmp_path):
    conn = _store(tmp_path)
    service = EnfoldService(conn, MemoryPolicy({"work-only": ("work",)}))
    context = _context("work-only", "terminal", "terminal", scopes=("work",))

    with pytest.raises(ServiceRequestError, match="not authorized") as denied:
        service.handle(
            context,
            _request(
                "run-search-denied",
                "memory.search",
                query="worker only",
                scope="run:child-session",
            ),
        )
    assert denied.value.code == "access_denied"
    conn.close()
