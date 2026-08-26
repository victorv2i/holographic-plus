"""Guards for the two extraction write paths and shared payload bounds.

F3: the Hermes llm_extract drain must not be a rememberer that can retire
existing facts. F7: a long session must not be silently reduced to a tail.
"""
from __future__ import annotations

import json
import threading
import types

from enfold.llm_extract import _format_conversation, insert_facts


def _llm_response(content):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
    )


def _stop_worker(provider):
    provider._queue_stop.set()
    provider._queue_wake.set()
    if provider._queue_worker:
        provider._queue_worker.join(timeout=2.0)


def test_hermes_drain_does_not_supersede_existing_facts(make_provider, aux_module):
    """Live Hermes extraction cannot retire a fact already in the store."""
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel"
    )
    _stop_worker(provider)
    old_id = provider._store.add_fact(
        "The Skylark dashboard port is 3100.", category="project"
    )
    aux_module.call_llm = lambda **_kwargs: _llm_response(
        json.dumps(
            [
                {
                    "content": "The Skylark dashboard port is 3200.",
                    "category": "project",
                    "tags": "skylark,port",
                }
            ]
        )
    )

    provider._extract_queue.enqueue(
        [{"role": "user", "content": "the dashboard port is now 3200"}]
    )
    provider._drain_extract_queue(threading.Event(), provider._extract_queue)

    old = provider._store._conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?",
        (old_id,),
    ).fetchone()
    assert old["superseded_by"] is None
    assert old["invalid_at"] is None


def test_insert_facts_refuses_to_supersede_an_existing_value(make_provider):
    """Direct insert_facts is the same write path and must not retire facts."""
    provider = make_provider()
    old_id = provider._store.add_fact(
        "The Skylark dashboard port is 3100.", category="project"
    )
    calls = []

    def record_supersede(old_fact_id, new_fact_id):
        calls.append((old_fact_id, new_fact_id))
        return provider._supersede_fact(old_fact_id, new_fact_id)

    result = insert_facts(
        provider._store,
        [
            {
                "content": "The Skylark dashboard port is 3200.",
                "category": "project",
                "tags": "skylark,port",
            }
        ],
        dedup_check=provider._find_near_duplicate,
        update_check=provider._find_update_target,
        supersede=record_supersede,
    )

    assert calls == []
    assert result.failed == 0
    old = provider._store._conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?",
        (old_id,),
    ).fetchone()
    assert old["superseded_by"] is None
    assert old["invalid_at"] is None


def test_format_conversation_fails_closed_instead_of_dropping_the_head():
    head = "USER: remember the first decision about Postgres."
    tail = "ASSISTANT: later chatter only."
    messages = [
        {"role": "user", "content": head},
        {"role": "assistant", "content": "x" * 200},
        {"role": "assistant", "content": tail},
    ]

    try:
        rendered = _format_conversation(messages, max_chars=40)
    except ValueError as exc:
        assert "exceeds" in str(exc).lower() or "limit" in str(exc).lower()
        return

    assert head in rendered
    assert "earlier content omitted" not in rendered


def test_enqueue_fails_closed_on_oversized_session(make_provider):
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel"
    )
    _stop_worker(provider)
    messages = [
        {"role": "user", "content": "remember the first decision about Postgres."},
        {"role": "assistant", "content": "x" * 20_000},
    ]

    queued = provider._enqueue_extraction(messages, "session_end")

    assert queued is False
    assert provider._extract_queue.pending_count() == 0


def test_session_end_persists_role_structured_turns_without_flattening(make_provider):
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel"
    )
    _stop_worker(provider)
    messages = [
        {"role": "user", "content": "Avery prefers concise responses."},
        {
            "role": "assistant",
            "content": "I will apply that preference to every future response.",
        },
        {"role": "tool", "content": "METIS_FLEET_OK"},
    ]

    provider.on_session_end(messages)

    payload = provider._store._conn.execute(
        "SELECT payload FROM extract_queue"
    ).fetchone()[0]
    assert payload == json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def test_session_end_legacy_drain_exposes_only_user_turns_to_model(
    make_provider, aux_module
):
    provider = make_provider(
        extraction_provider="testprov", extraction_model="testmodel"
    )
    _stop_worker(provider)
    messages = [
        {"role": "user", "content": "Avery prefers concise responses."},
        {
            "role": "assistant",
            "content": "The assistant applies this preference to every response.",
        },
        {"role": "tool", "content": "METIS_FLEET_OK"},
    ]
    seen_prompts = []

    def facts_from_visible_turns(**kwargs):
        prompt = kwargs["messages"][1]["content"]
        seen_prompts.append(prompt)
        facts = []
        for evidence, content in (
            (
                "Avery prefers concise responses.",
                "Avery prefers concise responses.",
            ),
            (
                "The assistant applies this preference to every response.",
                "The assistant applies this preference to every response.",
            ),
            ("METIS_FLEET_OK", "The tool reports METIS_FLEET_OK."),
        ):
            if evidence in prompt:
                facts.append({"content": content, "category": "general", "tags": ""})
        return _llm_response(json.dumps(facts))

    aux_module.call_llm = facts_from_visible_turns
    provider.on_session_end(messages)
    provider._drain_extract_queue(threading.Event(), provider._extract_queue)

    contents = [
        fact["content"]
        for fact in provider._store.list_facts(min_trust=0.0, limit=20)
    ]
    assert seen_prompts
    assert "Avery prefers concise responses." in seen_prompts[0]
    assert "The assistant applies this preference" not in seen_prompts[0]
    assert "METIS_FLEET_OK" not in seen_prompts[0]
    assert contents == ["Avery prefers concise responses."]
