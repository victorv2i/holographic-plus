from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from types import ModuleType

from enfold.policy import MemoryPolicy
from enfold.protocol import ClientContext, Request
from enfold.service import DEFAULT_OUTPUT_BOUNDS, EnfoldService, _FACT_FIELDS


def test_history_selector_fact_id_mode_passes_service_validation_and_limit(monkeypatch):
    agent = ModuleType("agent")
    memory_provider = ModuleType("agent.memory_provider")
    memory_provider.MemoryProvider = type("MemoryProvider", (), {})
    agent.memory_provider = memory_provider
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.memory_provider", memory_provider)
    module = importlib.import_module("integrations.hermes_enfold_v1")

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    columns = ", ".join(
        "fact_id INTEGER PRIMARY KEY" if name == "fact_id" else f"{name} TEXT"
        for name in _FACT_FIELDS
    )
    conn.execute(f"CREATE TABLE facts ({columns})")
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, created_at) "
        "VALUES (8, 'current', 'private', '2026-01-02T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO facts(fact_id, content, scope, created_at, invalid_at, superseded_by) "
        "VALUES (7, 'stale', 'private', '2026-01-01T00:00:00Z', "
        "'2026-01-02T00:00:00Z', 8)"
    )
    conn.commit()
    service = object.__new__(EnfoldService)
    service._conn = conn
    service._policy = MemoryPolicy({"hermes-install": ("private",)})
    service._output_bounds = DEFAULT_OUTPUT_BOUNDS
    context = ClientContext(
        "hermes-install",
        "hermes",
        "avery",
        "session-1",
        access_scopes=("private",),
    )
    selector = module._history_selector({
        "fact_id": 7,
        "subject_key": "must-be-dropped",
        "predicate_key": "must-be-dropped",
        "scope": "private",
        "limit": 1,
    })

    result = service.handle(
        context, Request("history-1", "memory.history", selector)
    )

    assert selector == {"fact_id": 7, "limit": 1}
    assert [fact["fact_id"] for fact in result["facts"]] == [7]
    assert result["output_truncated"] is True
    conn.close()


def test_search_tool_passes_advertised_category_filter(monkeypatch, tmp_path):
    agent = ModuleType("agent")
    memory_provider = ModuleType("agent.memory_provider")
    memory_provider.MemoryProvider = type("MemoryProvider", (), {})
    agent.memory_provider = memory_provider
    monkeypatch.setitem(sys.modules, "agent", agent)
    monkeypatch.setitem(sys.modules, "agent.memory_provider", memory_provider)
    module = importlib.import_module("integrations.hermes_enfold_v1")

    class Session:
        def __init__(self):
            self.calls = []

        def search(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"facts": []}

    session = Session()

    class Adapter:
        def __init__(self, config):
            del config

        def open_session(self, context):
            del context
            return session

    provider = module.EnfoldV1MemoryProvider(
        adapter_factory=Adapter,
        environ={"ENFOLD_SOCKET_PATH": str(tmp_path / "enfold.sock")},
    )
    provider.initialize("session-1")

    payload = json.loads(provider.handle_tool_call(
        "enfold_memory",
        {
            "action": "search",
            "query": "Atlas launch",
            "category": "project",
            "limit": 2,
        },
    ))

    assert payload["ok"] is True
    assert session.calls == [
        (("Atlas launch",), {"category": "project", "limit": 2})
    ]
