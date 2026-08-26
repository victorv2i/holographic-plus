"""Adoption contract for the public Enfold MCP surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enfold import mcp_stdio
from enfold.client import ClientConfig, EnfoldRemoteError
from enfold.mcp_stdio import build_server, parse_config
from enfold.protocol import ClientContext, ProtocolError


class FakeToolError(Exception):
    pass


class FakeMCP:
    def __init__(self, name, instructions=None, **_kwargs):
        self.name = name
        self.instructions = instructions
        self.tools = {}
        self.tool_meta = {}
        self.runs = []

    def tool(self, *args, **kwargs):
        def register(function):
            self.tools[function.__name__] = function
            self.tool_meta[function.__name__] = kwargs
            return function

        return register

    def run(self, *, transport):
        self.runs.append(transport)


class RecordingTransport:
    instances = []

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.result = {"ok": True}
        self.error = None
        self.__class__.instances.append(self)

    def request(self, method, params=None, *, request_id=None):
        self.calls.append((method, params, request_id))
        if self.error:
            raise self.error
        return self.result


def _config(tmp_path: Path) -> ClientConfig:
    return ClientConfig(
        socket_path=tmp_path / "enfold.sock",
        context=ClientContext(
            client_id="client-a-install",
            surface="client-a",
            agent_id="client-a",
            session_id="thread-7",
            project_root="/workspace/project",
            access_scopes=("private", "work"),
        ),
    )


def _build(tmp_path: Path, *, tool_profile=None):
    RecordingTransport.instances.clear()
    kwargs = {
        "server_factory": FakeMCP,
        "transport_factory": RecordingTransport,
        "tool_error_type": FakeToolError,
    }
    if tool_profile is not None:
        kwargs["tool_profile"] = tool_profile
    server = build_server(_config(tmp_path), **kwargs)
    return server, RecordingTransport.instances[0]


def _tokens(value) -> int:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (len(encoded) + 3) // 4


CORE_TOOLS = ("memory_recall", "memory_remember", "memory_inspect")
REVIEW_TOOLS = CORE_TOOLS + ("memory_review", "memory_resolve")
LEGACY_V1_TOOLS = {
    "memory_write",
    "memory_search",
    "memory_context",
    "memory_evidence",
    "memory_history",
    "memory_changes",
    "memory_timeline",
    "memory_entities",
    "memory_entity",
    "memory_conflicts",
    "memory_resolve_conflict",
    "memory_extraction_enqueue",
    "memory_promote",
}

EMPTY_RECALL_MESSAGE = (
    "No supporting memory found. This does not mean the claim is false. "
    "If the user wants persistent shared memory, offer to remember one "
    "durable fact and wait for agreement."
)

RANKER_KEYS = {
    "score",
    "fts_score",
    "jaccard_score",
    "dense_score",
    "trust_score_component",
    "memory_kind_score",
    "recency_score",
    "retrieval_count",
    "helpful_count",
    "retrieval",
    "vector_backend",
    "schema_version",
}


def test_default_profile_is_core_three_tools(tmp_path):
    server, transport = _build(tmp_path)

    assert server.name == "enfold-memory"
    assert tuple(server.tools) == CORE_TOOLS
    assert transport.calls == []
    assert "memory_search" not in server.tools
    assert "memory_write" not in server.tools
    assert "Call `memory_recall`" in (server.instructions or "")


def test_review_profile_adds_review_and_resolve(tmp_path):
    server, _transport = _build(tmp_path, tool_profile="review")
    assert tuple(server.tools) == REVIEW_TOOLS


def test_legacy_v1_profile_keeps_the_thirteen_v1_names(tmp_path):
    server, _transport = _build(tmp_path, tool_profile="legacy-v1")
    assert set(server.tools) == LEGACY_V1_TOOLS


def test_unknown_profile_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="tool profile"):
        _build(tmp_path, tool_profile="everything")


def test_parse_tool_profile_defaults_to_core_and_accepts_cli(tmp_path):
    assert mcp_stdio.parse_tool_profile([], environ={}) == "core"
    assert (
        mcp_stdio.parse_tool_profile(
            ["--tool-profile", "review"],
            environ={"ENFOLD_TOOL_PROFILE": "legacy-v1"},
        )
        == "review"
    )
    config = parse_config(
        [
            "--socket-path",
            str(tmp_path / "enfold.sock"),
            "--client-id",
            "cli-install",
            "--surface",
            "client-a",
            "--agent-id",
            "client-a",
            "--session-id",
            "cli-session",
            "--tool-profile",
            "review",
        ],
        environ={},
    )
    assert config.context.client_id == "cli-install"


def test_recall_empty_state_tells_the_agent_what_to_do_next(tmp_path):
    server, transport = _build(tmp_path)
    transport.result = {
        "markdown": "",
        "facts": [],
        "abstained": True,
        "token_estimate": {"method": "unicode_chars_divided_by_four", "budget": 512, "used": 0},
        "omitted_fact_count": 0,
        "unsafe_fact_count": 0,
        "prompt_unsafe_fact_count": 0,
        "retrieval": {"vector_backend": "brute", "score_formula": "hidden"},
        "output_truncated": False,
    }

    result = server.tools["memory_recall"]("Ada employer")

    assert result == {
        "facts": [],
        "truncated": False,
        "message": EMPTY_RECALL_MESSAGE,
    }
    method, params, _request_id = transport.calls[-1]
    assert method == "memory.context"
    assert params["query"] == "Ada employer"
    assert params["token_budget"] == 512
    assert "min_trust" not in params


def test_recall_projects_compact_facts_and_strips_ranker_telemetry(tmp_path):
    server, transport = _build(tmp_path)
    transport.result = {
        "markdown": "## Enfold Memory\n> note\n",
        "facts": [
            {
                "fact_id": 42,
                "content": "Ada prefers local-first tools.",
                "prompt_eligible": True,
                "review_status": "confirmed",
                "correction_status": "human_confirmed",
                "attribution": {
                    "source_type": "human",
                    "evidence_count": 2,
                    "commit_sha": "abc123",
                    "session_id": "secret-session",
                },
                "score": 0.91,
                "fts_score": 0.5,
                "jaccard_score": 0.2,
                "dense_score": 0.4,
                "trust_score_component": 0.5,
                "memory_kind_score": 0.5,
                "recency_score": 0.9,
                "retrieval_count": 12,
                "helpful_count": 3,
                "schema_version": 1,
                "created_at": "2026-08-01T00:00:00Z",
                "updated_at": "2026-08-01T00:00:00Z",
                "scope": "private",
                "conflict_group": None,
            }
        ],
        "abstained": False,
        "retrieval": {
            "vector_backend": "sqlite-vec",
            "score_formula": "0.5*fts",
            "weights": {"fts": 0.5},
        },
        "output_truncated": False,
        "omitted_fact_count": 0,
        "unsafe_fact_count": 0,
        "prompt_unsafe_fact_count": 0,
        "token_estimate": {"used": 40, "budget": 512},
    }

    result = server.tools["memory_recall"]("local-first tools")

    assert result["facts"] == [
        {
            "id": 42,
            "text": "Ada prefers local-first tools.",
            "review": "confirmed",
            "source": "human",
            "evidence": 2,
        }
    ]
    assert result["truncated"] is False
    assert result.get("next_cursor") is None
    blob = json.dumps(result)
    for key in RANKER_KEYS:
        assert key not in blob
    assert "markdown" not in result
    assert "commit_sha" not in blob
    assert _tokens(result) <= 512


def test_recall_surfaces_open_conflicts_instead_of_a_clean_miss(tmp_path):
    server, transport = _build(tmp_path)
    transport.result = {
        "facts": [],
        "open_conflicts": [
            {
                "conflict_id": "c_7",
                "scope": "private",
                "subject_key": "Ada",
                "predicate_key": "employer",
                "member_fact_ids": [11, 12],
                "summary": (
                    "[conflict:c_7 slot:Ada.employer members:2 - "
                    "do not treat either as current]"
                ),
            }
        ],
        "output_truncated": True,
        "retrieval": {"vector_backend": "brute", "score_formula": "hidden"},
    }

    result = server.tools["memory_recall"]("Ada employer")

    assert result["facts"] == []
    assert "message" not in result
    assert result["truncated"] is True
    assert result["open_conflicts"] == [
        {
            "id": "c_7",
            "subject": "Ada",
            "predicate": "employer",
            "member_fact_ids": [11, 12],
            "summary": (
                "[conflict:c_7 slot:Ada.employer members:2 - "
                "do not treat either as current]"
            ),
        }
    ]
    blob = json.dumps(result)
    assert "score_formula" not in blob
    assert "retrieval" not in blob
    assert EMPTY_RECALL_MESSAGE not in blob


def test_recall_keeps_open_conflicts_beside_current_facts(tmp_path):
    server, transport = _build(tmp_path)
    transport.result = {
        "facts": [
            {
                "fact_id": 42,
                "content": "Ada prefers local-first tools.",
                "prompt_eligible": True,
                "review_status": "confirmed",
                "attribution": {"source_type": "human", "evidence_count": 1},
                "conflict_group": None,
            }
        ],
        "open_conflicts": [
            {
                "conflict_id": "c_7",
                "subject_key": "Ada",
                "predicate_key": "employer",
                "member_fact_ids": [11, 12],
                "summary": "do not treat either as current",
            }
        ],
        "output_truncated": False,
        "retrieval": {},
    }

    result = server.tools["memory_recall"]("Ada")

    assert [row["id"] for row in result["facts"]] == [42]
    assert result["open_conflicts"][0]["id"] == "c_7"
    assert result["open_conflicts"][0]["member_fact_ids"] == [11, 12]


def test_recall_budget_keeps_a_conflict_receipt_before_facts(tmp_path):
    server, transport = _build(tmp_path)
    transport.result = {
        "facts": [
            {
                "fact_id": fact_id,
                "content": "A current fact with long content. " + "x" * 240,
                "prompt_eligible": True,
                "review_status": "confirmed",
                "attribution": {"source_type": "human", "evidence_count": 1},
                "conflict_group": None,
            }
            for fact_id in range(1, 9)
        ],
        "open_conflicts": [
            {
                "conflict_id": "c_7",
                "subject_key": "Ada",
                "predicate_key": "employer",
                "member_fact_ids": list(range(1, 101)),
                "summary": (
                    "[conflict:c_7 slot:Ada.employer members:100 - "
                    "do not treat any as current]"
                ),
            }
        ],
        "output_truncated": False,
        "retrieval": {},
    }

    result = server.tools["memory_recall"]("Ada", token_budget=128)

    assert result["facts"] == []
    assert result["truncated"] is True
    assert "message" not in result
    assert result["open_conflicts"][0]["id"] == "c_7"
    assert result["open_conflicts"][0]["subject"] == "Ada"
    assert result["open_conflicts"][0]["predicate"] == "employer"
    assert _tokens(result) <= 128


def test_recall_forwards_as_of_axes_through_search(tmp_path):
    server, transport = _build(tmp_path)
    transport.result = {
        "facts": [
            {
                "fact_id": 9,
                "content": "Ada worked at OldCo.",
                "prompt_eligible": True,
                "review_status": "confirmed",
                "attribution": {"source_type": "human", "evidence_count": 1},
                "conflict_group": None,
            }
        ],
        "open_conflicts": [],
        "output_truncated": False,
        "retrieval": {"vector_backend": "brute", "score_formula": "hidden"},
    }

    result = server.tools["memory_recall"](
        "Ada employer",
        as_of_valid="2021-01-01T00:00:00Z",
        as_of_tx="2021-06-01T00:00:00Z",
    )

    method, params, _request_id = transport.calls[-1]
    assert method == "memory.search"
    assert params["query"] == "Ada employer"
    assert params["as_of_valid"] == "2021-01-01T00:00:00Z"
    assert params["as_of_tx"] == "2021-06-01T00:00:00Z"
    assert result["facts"][0]["id"] == 9
    assert "retrieval" not in result
    assert "score_formula" not in json.dumps(result)


def test_recall_omits_conflicted_and_unreviewed_and_prompt_unsafe_rows(tmp_path):
    server, transport = _build(tmp_path)
    transport.result = {
        "facts": [
            {
                "fact_id": 1,
                "content": "safe current fact",
                "prompt_eligible": True,
                "review_status": "unreviewed",
                "attribution": {"source_type": "agent_report", "evidence_count": 1},
                "conflict_group": None,
            },
            {
                "fact_id": 2,
                "prompt_eligible": False,
                "content_omitted": True,
                "exclusion_reason": "unreviewed_content",
            },
            {
                "fact_id": 3,
                "content": "conflicted should never reach an agent",
                "prompt_eligible": True,
                "conflict_group": "c_1",
                "attribution": {"source_type": "agent_report", "evidence_count": 1},
            },
            {
                "fact_id": 4,
                "prompt_eligible": False,
                "content_omitted": True,
                "exclusion_reason": "instruction_shaped_content",
            },
        ],
        "output_truncated": False,
        "retrieval": {},
    }

    result = server.tools["memory_recall"]("safe current")

    assert [row["id"] for row in result["facts"]] == [1]
    assert "conflicted should never reach an agent" not in json.dumps(result)


def test_recall_fits_default_budget_and_hard_cap(tmp_path):
    server, transport = _build(tmp_path)
    long_text = "Ada prefers local-first tools and repeats this fact. " * 20
    transport.result = {
        "facts": [
            {
                "fact_id": index,
                "content": f"{long_text} {index}",
                "prompt_eligible": True,
                "review_status": "confirmed",
                "attribution": {"source_type": "human", "evidence_count": 1},
                "conflict_group": None,
                "score": 0.9,
            }
            for index in range(1, 9)
        ],
        "output_truncated": False,
        "retrieval": {"vector_backend": "brute"},
    }

    defaulted = server.tools["memory_recall"]("preference")
    assert _tokens(defaulted) <= 512
    assert defaulted["truncated"] is True
    assert defaulted["facts"]
    assert len(defaulted["facts"]) < 8

    capped = server.tools["memory_recall"]("preference", token_budget=2048)
    assert _tokens(capped) <= 2048

    with pytest.raises(FakeToolError) as raised:
        server.tools["memory_recall"]("preference", token_budget=64)
    payload = json.loads(str(raised.value))
    assert payload["code"] == "invalid_params"
    assert payload["field"] == "token_budget"
    assert payload["next_action"]


def test_remember_derives_policy_fields_and_idempotency(tmp_path):
    server, transport = _build(tmp_path)
    transport.result = {"outcome": "inserted", "fact_id": 42, "detail": {}}

    result = server.tools["memory_remember"](
        "Ada prefers local-first tools.",
        "user",
        evidence_excerpt="prefers local-first",
        state={"subject": "Ada", "predicate": "tooling_preference", "value": "local-first"},
    )

    assert result == {"status": "stored", "fact_id": 42}
    method, params, _request_id = transport.calls[-1]
    assert method == "memory.write"
    assert params["content"] == "Ada prefers local-first tools."
    assert params["source_type"] == "user"
    assert params["idempotency_key"].startswith("client-a-install:thread-7:")
    assert params["correction_status"] is None
    assert "trust_score" in params
    assert params["state"] == {
        "subject_key": "Ada",
        "predicate_key": "tooling_preference",
        "object_value": "local-first",
    }
    assert "human_confirmed" not in json.dumps(params)

    server.tools["memory_remember"](
        "Ada prefers local-first tools.",
        "user",
        evidence_excerpt="prefers local-first",
        state={"subject": "Ada", "predicate": "tooling_preference", "value": "local-first"},
    )
    assert transport.calls[0][1]["idempotency_key"] == transport.calls[1][1]["idempotency_key"]


def test_remember_forwards_valid_to_on_typed_state(tmp_path):
    server, transport = _build(tmp_path)
    transport.result = {"outcome": "inserted", "fact_id": 8, "detail": {}}

    result = server.tools["memory_remember"](
        "Ada worked at OldCo.",
        "user",
        state={
            "subject": "Ada",
            "predicate": "employer",
            "value": "OldCo",
            "valid_from": "2018-01-01T00:00:00Z",
            "valid_to": "2020-01-01T00:00:00Z",
        },
    )

    assert result == {"status": "stored", "fact_id": 8}
    assert transport.calls[-1][1]["state"] == {
        "subject_key": "Ada",
        "predicate_key": "employer",
        "object_value": "OldCo",
        "valid_from": "2018-01-01T00:00:00Z",
        "valid_to": "2020-01-01T00:00:00Z",
    }


def test_remember_still_rejects_unknown_state_fields(tmp_path):
    server, _transport = _build(tmp_path)
    with pytest.raises(FakeToolError) as raised:
        server.tools["memory_remember"](
            "Ada worked at OldCo.",
            "user",
            state={
                "subject": "Ada",
                "predicate": "employer",
                "expired_at": "2020-01-01T00:00:00Z",
            },
        )
    payload = json.loads(str(raised.value))
    assert payload["code"] == "invalid_params"
    assert payload["field"] == "state"


def test_remember_status_variants_and_next_action(tmp_path):
    server, transport = _build(tmp_path)

    transport.result = {
        "outcome": "existing",
        "fact_id": 42,
        "existing_fact_id": 42,
        "detail": {},
    }
    assert server.tools["memory_remember"]("same fact", "conversation") == {
        "status": "deduped",
        "fact_id": 42,
    }

    transport.result = {
        "outcome": "conflict",
        "fact_id": 43,
        "existing_fact_id": 42,
        "detail": {"conflict_id": "c_123"},
    }
    conflicted = server.tools["memory_remember"](
        "Ada works at OtherCo",
        "user",
        corrects_fact_id=42,
        state={"subject": "Ada", "predicate": "employer", "value": "OtherCo"},
    )
    assert conflicted["status"] == "conflicted"
    assert conflicted["conflict_id"] == "c_123"
    assert conflicted["member_fact_ids"] == [42, 43]
    assert "Ask the user" in conflicted["next_action"]
    assert transport.calls[-1][1]["supersede_fact_id"] == 42
    assert transport.calls[-1][1].get("correction_status") is None

    transport.result = {
        "outcome": "needs_review",
        "fact_id": 43,
        "detail": {"policy_reason": "client is not authorized to assert human correction"},
    }
    held = server.tools["memory_remember"]("correction", "user", corrects_fact_id=42)
    assert held["status"] == "needs_review"
    assert "review" in held["reason"].lower() or "author" in held["reason"].lower()

    transport.result = {
        "outcome": "rejected",
        "fact_id": None,
        "detail": {"policy_reason": "secret durable writes are disabled"},
    }
    rejected = server.tools["memory_remember"]("ssn 123", "user")
    assert rejected == {
        "status": "rejected",
        "reason": "secret durable writes are disabled",
    }


def test_inspect_evidence_is_compact_and_paginated(tmp_path):
    server, transport = _build(tmp_path)
    long_excerpt = "supporting excerpt " * 40
    transport.result = {
        "fact": {
            "fact_id": 42,
            "content": "Ada prefers local-first tools.",
            "score": 0.9,
            "retrieval_count": 8,
        },
        "evidence": [
            {
                "source_type": "human",
                "asserted_by": "Ada",
                "performed_by": "hermes-install",
                "observed_at": "2026-08-01T00:00:00Z",
                "evidence_excerpt": f"{long_excerpt} {index}",
                "relation": "supports",
                "commit_sha": "deadbeef",
                "session_id": "hidden",
                "observation_id": index,
            }
            for index in range(1, 7)
        ],
        "output_truncated": False,
    }

    first = server.tools["memory_inspect"](42, "evidence")
    assert _tokens(first) <= 512
    assert first["items"]
    row = first["items"][0]
    assert set(row) <= {"source", "by", "observed_at", "excerpt", "relation"}
    assert "score" not in json.dumps(first)
    assert "commit_sha" not in json.dumps(first)
    assert first["truncated"] is True
    assert first["next_cursor"]

    second = server.tools["memory_inspect"](
        42, "evidence", cursor=first["next_cursor"]
    )
    assert second["items"]
    assert second["items"][0]["excerpt"] != first["items"][0]["excerpt"]


def test_inspect_history_is_compact(tmp_path):
    server, transport = _build(tmp_path)
    transport.result = {
        "facts": [
            {
                "fact_id": 41,
                "content": "Ada worked at OldCo.",
                "valid_from": "2024-01-01T00:00:00Z",
                "invalid_at": "2026-01-01T00:00:00Z",
                "superseded_by": 42,
                "correction_status": "human_corrected",
                "score": 0.1,
            },
            {
                "fact_id": 42,
                "content": "Ada works at NewCo.",
                "valid_from": "2026-01-01T00:00:00Z",
                "invalid_at": None,
                "superseded_by": None,
                "correction_status": "human_confirmed",
            },
        ],
        "output_truncated": False,
    }

    result = server.tools["memory_inspect"](42, "history")
    assert result["items"][0] == {
        "id": 41,
        "text": "Ada worked at OldCo.",
        "valid_from": "2024-01-01T00:00:00Z",
        "invalid_at": "2026-01-01T00:00:00Z",
        "replaced_by": 42,
        "review": "corrected",
    }
    assert "score" not in json.dumps(result)
    method, params, _request_id = transport.calls[-1]
    assert method == "memory.history"
    assert params["fact_id"] == 42


def test_inspect_history_projects_valid_to_and_expired_at(tmp_path):
    server, transport = _build(tmp_path)
    transport.result = {
        "facts": [
            {
                "fact_id": 41,
                "content": "Ada worked at OldCo.",
                "valid_from": "2018-01-01T00:00:00Z",
                "valid_to": "2020-01-01T00:00:00Z",
                "invalid_at": "2022-01-01T00:00:00Z",
                "expired_at": "2022-01-01T00:00:00Z",
                "superseded_by": 42,
                "correction_status": "human_corrected",
            }
        ],
        "output_truncated": False,
    }

    result = server.tools["memory_inspect"](42, "history")
    assert result["items"][0]["valid_to"] == "2020-01-01T00:00:00Z"
    assert result["items"][0]["expired_at"] == "2022-01-01T00:00:00Z"
    assert result["items"][0]["valid_from"] == "2018-01-01T00:00:00Z"


def test_inspect_page_stays_within_declared_token_budget(tmp_path):
    server, transport = _build(tmp_path)
    excerpt = "x" * 351
    transport.result = {
        "evidence": [
            {
                "source_type": "human",
                "asserted_by": "Ada",
                "observed_at": "2026-08-01T00:00:00Z",
                "evidence_excerpt": excerpt,
                "relation": "supports",
            }
            for _ in range(2)
        ],
        "output_truncated": False,
    }

    result = server.tools["memory_inspect"](42, "evidence", token_budget=128)
    assert _tokens(result) <= 128
    assert result["truncated"] is True
    assert result["next_cursor"]


def test_review_profile_returns_summaries_not_embedded_facts(tmp_path):
    server, transport = _build(tmp_path, tool_profile="review")
    transport.result = {
        "conflicts": [
            {
                "conflict_id": "c_123",
                "subject_key": "Ada",
                "predicate_key": "employer",
                "member_fact_ids": [42, 43],
                "members": [
                    {
                        "fact_id": 42,
                        "content": "Ada works at NewCo.",
                        "score": 0.8,
                        "retrieval_count": 4,
                    },
                    {
                        "fact_id": 43,
                        "content": "Ada works at OtherCo.",
                        "score": 0.7,
                    },
                ],
            }
        ],
        "output_truncated": False,
    }

    result = server.tools["memory_review"]()
    assert result["items"] == [
        {
            "kind": "conflict",
            "id": "c_123",
            "subject": "Ada",
            "predicate": "employer",
            "choices": [
                {"fact_id": 42, "text": "Ada works at NewCo."},
                {"fact_id": 43, "text": "Ada works at OtherCo."},
            ],
        }
    ]
    assert "score" not in json.dumps(result)
    assert "retrieval_count" not in json.dumps(result)


def test_resolve_requires_review_profile_and_forwards_winning_fact(tmp_path):
    core, _transport = _build(tmp_path)
    assert "memory_resolve" not in core.tools

    server, transport = _build(tmp_path, tool_profile="review")
    transport.result = {
        "resolution": {
            "conflict_id": "c_123",
            "resolution_fact_id": 42,
            "superseded_fact_ids": [43],
        }
    }
    result = server.tools["memory_resolve"]("c_123", 42, "Ada chose NewCo")
    assert result["status"] == "resolved"
    assert result["winning_fact_id"] == 42
    assert transport.calls[-1][:2] == (
        "memory.resolve_conflict",
        {
            "conflict_id": "c_123",
            "resolution_fact_id": 42,
            "reason": "Ada chose NewCo",
        },
    )


def test_errors_carry_next_action(tmp_path):
    server, transport = _build(tmp_path)
    transport.error = EnfoldRemoteError(
        ProtocolError("not_found", "fact was not found"),
        request_id="req-9",
    )
    with pytest.raises(FakeToolError) as raised:
        server.tools["memory_inspect"](99, "evidence")
    payload = json.loads(str(raised.value))
    assert payload["code"] == "not_found"
    assert payload["next_action"]
    assert "memory_recall" in payload["next_action"]


def test_core_tool_descriptions_and_annotations_are_deliberate(tmp_path):
    server, _transport = _build(tmp_path)
    recall_meta = server.tool_meta["memory_recall"]
    remember_meta = server.tool_meta["memory_remember"]
    inspect_meta = server.tool_meta["memory_inspect"]

    recall_text = recall_meta.get("description") or server.tools["memory_recall"].__doc__
    remember_text = remember_meta.get("description") or server.tools["memory_remember"].__doc__
    inspect_text = inspect_meta.get("description") or server.tools["memory_inspect"].__doc__
    assert "Use this first" in recall_text
    assert "Do not use this to audit" in recall_text
    assert "Do not store secrets" in remember_text
    assert "Do not use this for ordinary recall" in inspect_text

    recall_hints = recall_meta["annotations"]
    remember_hints = remember_meta["annotations"]
    inspect_hints = inspect_meta["annotations"]
    assert _hint(recall_hints, "readOnlyHint") is True
    assert _hint(remember_hints, "readOnlyHint") is False
    assert _hint(inspect_hints, "idempotentHint") is True
    assert _hint(remember_hints, "destructiveHint") is False


def test_legacy_surface_is_quarantined_from_the_public_contract():
    root = Path(__file__).resolve().parents[1]
    server_text = (root / "enfold" / "mcp_server.py").read_text(encoding="utf-8")
    provider_text = (root / "enfold" / "mcp_provider.py").read_text(encoding="utf-8")
    manifest = (root / "enfold" / "plugin.yaml").read_text(encoding="utf-8")
    docs = (root / "docs" / "MCP_PROXY.md").read_text(encoding="utf-8")

    assert "enfold-memory-legacy" in server_text
    assert "Hermes compatibility extra" in server_text
    assert "Hermes compatibility extra" in provider_text
    assert "not the public enfold contract" in manifest.lower()
    assert "memory_recall" in docs
    assert "core" in docs
    assert "legacy-v1" in docs


def _hint(annotations, name: str):
    if annotations is None:
        return None
    if isinstance(annotations, dict):
        return annotations.get(name)
    return getattr(annotations, name, None)
