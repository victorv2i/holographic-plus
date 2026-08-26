from __future__ import annotations

from enfold.core_store import (
    connect_database,
    link_fact_entities,
    resolve_entity,
)
from enfold.hybrid_retrieval import lexical_retriever_factory
from enfold.schema import migrate
from memory_eval.retrieval_scorecard import retrieval_scorecard
from memory_eval.runner import EvalCase, run_retrieval_cases


def _store(tmp_path):
    conn = connect_database(tmp_path / "multihop.db")
    migrate(conn)
    return conn


def _fact(conn, fact_id: int, content: str, **fields):
    values = {
        "category": "general",
        "tags": "",
        "trust_score": 1.0,
        "memory_kind": "state",
        "scope": "private",
        "sensitivity": "normal",
        "correction_status": "human_confirmed",
        "schema_version": 1,
        **fields,
    }
    columns = ("fact_id", "content", *values)
    conn.execute(
        f"INSERT INTO facts({', '.join(columns)}) "
        f"VALUES ({', '.join('?' for _ in columns)})",
        (fact_id, content, *values.values()),
    )


def _link(conn, fact_id: int, *names: str):
    link_fact_entities(conn, fact_id, (resolve_entity(conn, name) for name in names))


def test_multihop_expansion_is_off_by_default(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "Nora owns Project Quartz")
    _link(conn, 1, "Nora", "Project Quartz")
    _fact(conn, 2, "Project Quartz deploys from Reykjavik")
    _link(conn, 2, "Project Quartz")
    conn.commit()
    statements = []
    conn.set_trace_callback(statements.append)
    retriever = lexical_retriever_factory()(conn, ("private",))

    rows = retriever.search("Nora ownership", limit=5)

    assert 2 not in {row["fact_id"] for row in rows}
    assert not any(
        "E.NAME AS VIA_ENTITY" in statement.upper() for statement in statements
    )
    conn.close()


def test_associative_slice_improves_recall_at_five(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "Alice sponsors Project Lantern")
    _link(conn, 1, "Alice", "Project Lantern")
    _fact(conn, 2, "Project Lantern stores archives in Oslo")
    _link(conn, 2, "Project Lantern")

    _fact(conn, 3, "Bruno designed Initiative Cedar")
    _link(conn, 3, "Bruno", "Initiative Cedar")
    _fact(conn, 4, "Initiative Cedar uses a cobalt interface")
    _link(conn, 4, "Initiative Cedar")

    _fact(conn, 5, "Dana advises Program Harbor")
    _link(conn, 5, "Dana", "Program Harbor")
    _fact(conn, 6, "Program Harbor depends on Service Kestrel")
    _link(conn, 6, "Program Harbor", "Service Kestrel")
    _fact(conn, 7, "Service Kestrel rotates credentials on Fridays")
    _link(conn, 7, "Service Kestrel")
    conn.commit()

    cases = [
        EvalCase(
            id="alice-lantern",
            query="Alice sponsorship",
            gold_fact_id=2,
            case_type="associative_multihop",
        ),
        EvalCase(
            id="bruno-cedar",
            query="Bruno design",
            gold_fact_id=4,
            case_type="associative_multihop",
        ),
        EvalCase(
            id="dana-kestrel",
            query="Dana advisory",
            gold_fact_id=7,
            case_type="associative_multihop",
        ),
    ]
    retriever = lexical_retriever_factory(entity_expansion=True)(conn, ("private",))

    results = run_retrieval_cases(retriever, cases, limit=5)
    card = retrieval_scorecard(results, k_values=(5,))

    assert card["recall@5"] == 1.0
    by_gold = {
        result.case.gold_fact_id: next(
            row for row in result.results if row["fact_id"] == result.case.gold_fact_id
        )
        for result in results
    }
    assert by_gold[2]["hop_distance"] == 1
    assert by_gold[4]["hop_distance"] == 1
    assert by_gold[7]["hop_distance"] == 2
    conn.close()


def test_entity_hops_never_cross_ineligible_facts(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "Nora owns Project Quartz")
    _link(conn, 1, "Nora", "Project Quartz")
    _fact(conn, 2, "Project Quartz deploys from Reykjavik")
    _link(conn, 2, "Project Quartz")
    _fact(conn, 3, "Project Quartz used a retired relay", invalid_at="2026-08-01")
    _link(conn, 3, "Project Quartz", "Retired Relay")
    _fact(conn, 4, "Retired Relay exposes the restricted route")
    _link(conn, 4, "Retired Relay")
    _fact(conn, 5, "Project Quartz formerly deployed from Lima", superseded_by=2)
    _link(conn, 5, "Project Quartz")
    _fact(conn, 6, "Project Quartz may deploy from Bern", conflict_group="quartz-1")
    _link(conn, 6, "Project Quartz")
    _fact(
        conn,
        7,
        "Project Quartz has an unreviewed destination",
        correction_status="unreviewed",
    )
    _link(conn, 7, "Project Quartz", "Unreviewed Relay")
    _fact(conn, 8, "Project Quartz has a shared-only note", scope="shared")
    _link(conn, 8, "Project Quartz")
    _fact(conn, 9, "Project Quartz has a low trust rumor", trust_score=0.2)
    _link(conn, 9, "Project Quartz")
    _fact(conn, 10, "Unreviewed Relay exposes another restricted route")
    _link(conn, 10, "Unreviewed Relay")
    conn.commit()
    retriever = lexical_retriever_factory(entity_expansion=True)(conn, ("private",))

    rows = retriever.search("Nora ownership", min_trust=0.3, limit=20)
    ids = {row["fact_id"] for row in rows}

    assert 2 in ids
    assert ids.isdisjoint({3, 4, 5, 6, 7, 8, 9, 10})

    review_rows = retriever.search(
        "Nora ownership", min_trust=0.3, limit=20, include_unreviewed=True
    )
    assert {row["fact_id"] for row in review_rows}.isdisjoint({7, 10})
    conn.close()


def test_hub_entity_does_not_expand_unrelated_facts(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "Project Beacon coordination schedule is weekly")
    _link(conn, 1, "Avery", "Project Beacon")
    _fact(conn, 2, "Project Beacon publishes the release map")
    _link(conn, 2, "Project Beacon")
    filler_ids = set()
    for fact_id in range(3, 30):
        _fact(conn, fact_id, f"Avery recorded unrelated item {fact_id}")
        _link(conn, fact_id, "Avery")
        filler_ids.add(fact_id)
    conn.commit()
    retriever = lexical_retriever_factory(entity_expansion=True)(conn, ("private",))

    rows = retriever.search("coordination schedule", limit=10)
    ids = {row["fact_id"] for row in rows}

    assert 2 in ids
    assert ids.isdisjoint(filler_ids)
    conn.close()


def test_non_hub_entity_does_not_crowd_out_its_subject_neighbor(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "Nora owns Project Quartz")
    _link(conn, 1, "Nora", "Project Quartz")
    _fact(conn, 2, "Project Quartz deploys from Reykjavik")
    _link(conn, 2, "Project Quartz")
    filler_ids = set()
    for fact_id in range(3, 22):
        _fact(conn, fact_id, f"Morgan recorded unrelated Project Quartz item {fact_id}")
        _link(conn, fact_id, "Project Quartz")
        filler_ids.add(fact_id)
    conn.commit()
    retriever = lexical_retriever_factory(entity_expansion=True)(conn, ("private",))

    rows = retriever.search("Nora ownership", limit=20)
    ids = {row["fact_id"] for row in rows}

    assert 2 in ids
    assert ids.isdisjoint(filler_ids)
    conn.close()


def test_imported_note_titles_do_not_crowd_out_a_subject_neighbor(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "Nora owns Project Quartz")
    _link(conn, 1, "Nora", "Project Quartz")
    _fact(conn, 2, "Project Quartz deploys from Reykjavik")
    _link(conn, 2, "Project Quartz")
    for fact_id in range(3, 15):
        _fact(conn, fact_id, f"Morgan recorded unrelated Project Quartz item {fact_id}")
        _link(conn, fact_id, "Project Quartz")
    imported_ids = set()
    for fact_id in range(15, 23):
        _fact(
            conn,
            fact_id,
            "Claude Code memory topic `project-quartz-notes` (project): "
            f"Morgan recorded unrelated Project Quartz item {fact_id}",
        )
        _link(conn, fact_id, "Project Quartz")
        imported_ids.add(fact_id)
    conn.commit()
    retriever = lexical_retriever_factory(entity_expansion=True)(conn, ("private",))

    rows = retriever.search("Nora ownership", limit=5)
    ids = {row["fact_id"] for row in rows}

    assert 2 in ids
    assert ids.isdisjoint(imported_ids)
    conn.close()


def test_each_multihop_destination_load_is_bounded_in_sql(tmp_path):
    conn = _store(tmp_path)
    _fact(conn, 1, "Dana advises Program Harbor")
    _link(conn, 1, "Dana", "Program Harbor")
    _fact(conn, 2, "Program Harbor depends on Service Kestrel")
    _link(conn, 2, "Program Harbor", "Service Kestrel")
    _fact(conn, 3, "Service Kestrel rotates credentials on Fridays")
    _link(conn, 3, "Service Kestrel")
    conn.commit()
    statements = []
    conn.set_trace_callback(statements.append)
    retriever = lexical_retriever_factory(entity_expansion=True)(conn, ("private",))

    retriever.search("Dana advisory", limit=5)

    destination_loads = [
        statement
        for statement in statements
        if "E.NAME AS VIA_ENTITY" in statement.upper()
    ]
    assert len(destination_loads) == 2
    assert all(" LIMIT " in statement.upper() for statement in destination_loads)
    conn.close()
