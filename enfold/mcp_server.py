"""Hermes compatibility extra: legacy v0 holographic MCP server.

This is not the public Enfold contract. The public server is
``enfold-memory`` via ``enfold.mcp_stdio`` / ``enfold-mcp-launch``
(default profile ``core``: memory_recall, memory_remember, memory_inspect).

This module opens SQLite itself, has no grants or ``include_unreviewed``
flag, and refuses schema v1 (``mcp_provider.py``). Existing file-path
launch still works for owners who already wired Hermes to this extra.

Lets other coding agents read and write the same
fact store the Hermes gateway uses in-process, over the Model Context
Protocol, instead of each agent keeping its own disconnected memory.

Tools:
    memory_search(query, limit)                          -- hybrid search
    memory_add(content, category, tags, source)           -- write, dedup-gated
    memory_supersede(old_fact_id, new_content, source)     -- explicit update
    memory_explain(query, limit)                           -- scoring breakdown
    memory_history(fact_id)                                -- supersession chain

memory_search/memory_explain/memory_history are read-only and always
registered. memory_add/memory_supersede are writes and are omitted entirely
in --read-only mode (registered but return errors is NOT the read-only
contract here: the tools simply do not exist, so a read-only client can
never even attempt one).

In --read-only mode the provider is also opened read-only. Startup skips all
mutating initialization work: schema migrations, WAL checkpoints, embedding
backfill, extraction queue workers, and reflection passes. The server can
search existing stores but will not create or repair database objects.

Run directly (run the file by path, NOT `python -m enfold.mcp_server`;
see the warning below for why):

    python enfold/mcp_server.py \\
        --db-path ~/.hermes/memory_store.db \\
        --ollama-url http://localhost:11434 \\
        --ollama-model embeddinggemma:latest

See mcp_provider.py for how the parent hermes modules and db connection are
resolved and configured.

IMPORTANT -- run this file by path, never via `-m`: importing
enfold as a package (``python -m enfold.mcp_server``, or
any ``import enfold`` before this module has resolved its parent)
runs enfold/__init__.py first, Python's own package-import
semantics, and that file does its own unconditional
``from plugins.memory.holographic import HolographicMemoryProvider`` at
module level. On a host with a *separate* Hermes install already on
sys.path (e.g. a pip-installed hermes-agent), that import silently wins the
race and this module's own ENFOLD_HERMES_SRC resolution never gets a
chance to run. Executing this file directly (``python
enfold/mcp_server.py ...``) sidesteps the package __init__.py
entirely, which is why every example here uses the file path.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar

_THIS_DIR = Path(__file__).resolve().parent

# mcp_provider decides which parent hermes modules to load (real checkout vs
# the bundled fake_hermes stubs) and must run BEFORE enfold itself
# is imported as a package, since `import enfold` runs its
# __init__.py, which does its own unconditional parent import at module
# level. Load it by file path so this module never triggers that.


def _load_mcp_provider():
    name = "_enfold_mcp_provider"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _THIS_DIR / "mcp_provider.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mcp_provider = _load_mcp_provider()


def _load_write_lock():
    name = "_enfold_write_lock"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _THIS_DIR / "write_lock.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_cross_process_write_lock = _load_write_lock().cross_process_write_lock

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only without the dep
    raise ImportError(
        "The 'mcp' package is required to run the enfold MCP server "
        "(pip install 'mcp>=1.28.1,<2'). It is a required dependency of this "
        "repo, needed for mcp_server.py / mcp_provider.py, not for the Hermes "
        "plugin itself."
    ) from exc


PUBLIC_SERVER_NAME = "enfold-memory-legacy"
DEPRECATION_WARNING = (
    "enfold.mcp_server is a Hermes compatibility extra and is not the public "
    "Enfold contract. Use enfold-mcp-launch / enfold.mcp_stdio (default "
    "profile: core). Existing file-path launch still works."
)

SOURCE_ID_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
MAX_CONTENT_CHARS = 16_000
MAX_QUERY_CHARS = 16_000
MAX_TAGS_CHARS = 2_000
MAX_CATEGORY_CHARS = 128
MIN_LIMIT = 1
MAX_LIMIT = 50

_T = TypeVar("_T")
logger = logging.getLogger(__name__)


def _retry_on_locked(fn: Callable[[], _T], attempts: int = 30, base_delay: float = 0.02) -> _T:
    """Retry *fn* through transient SQLITE_BUSY from another writer.

    Belt-and-suspenders on top of _cross_process_write_lock and
    PRAGMA busy_timeout (see mcp_provider._apply_busy_timeout): the
    background embed/backfill/extraction threads inside the provider itself
    (not the cross-process write lock's concern) can still occasionally
    collide with a foreground write. Exponential backoff, capped, never
    masking a non-lock failure.
    """
    delay = base_delay
    for attempt in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == attempts - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 1.5, 1.0)
    raise AssertionError("unreachable")  # pragma: no cover


def _require_source(args: Dict[str, Any]) -> Optional[str]:
    """Validate the required `source` tag; returns an error string or None."""
    source = args.get("source")
    if not source:
        return "missing required argument: source"
    if not isinstance(source, str):
        return "invalid source: must be a string"
    if SOURCE_ID_RE.fullmatch(source) is None:
        return "invalid source: must be a lowercase source identifier"
    return None


def _require_string(name: str, value: Any, max_chars: int, *, non_blank: bool = False) -> Optional[str]:
    if not isinstance(value, str):
        return f"invalid {name}: must be a string"
    if len(value) > max_chars:
        return f"invalid {name}: exceeds {max_chars} characters"
    if non_blank and not value.strip():
        return f"invalid {name}: must not be blank"
    return None


def _validated_limit(limit: Any) -> tuple[Optional[int], Optional[str]]:
    if isinstance(limit, bool):
        return None, "invalid limit: must be an integer"
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return None, "invalid limit: must be an integer"
    return max(MIN_LIMIT, min(MAX_LIMIT, value)), None


def _validate_fact_args(content: Any, category: Any, tags: Any) -> Optional[str]:
    for name, value, max_chars, non_blank in (
        ("content", content, MAX_CONTENT_CHARS, True),
        ("category", category, MAX_CATEGORY_CHARS, False),
        ("tags", tags, MAX_TAGS_CHARS, False),
    ):
        error = _require_string(name, value, max_chars, non_blank=non_blank)
        if error:
            return error
    return None


def _json_safe_fact(fact: Dict[str, Any]) -> Dict[str, Any]:
    """Drop non-JSON-serializable columns (the raw hrr_vector BLOB) from a fact row.

    temporal.fact_history() does ``SELECT *``, which includes hrr_vector;
    every other read path in this package (search, explain_search) already
    excludes it before returning.
    """
    return {k: v for k, v in fact.items() if k != "hrr_vector"}


def _tag_source(tags: str, source: str) -> str:
    """Replace inbound source markers with exactly one canonical marker."""
    marker = f"source:{source}"
    existing = [
        t.strip()
        for t in (tags or "").split(",")
        if t.strip() and not t.strip().lower().startswith("source:")
    ]
    existing.append(marker)
    return ",".join(existing)


class _DeferredTransactionConnection:
    """Delegate SQLite work while leaving transaction ownership to the caller."""

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass


class _SupersedeFailed(Exception):
    pass


class _InvalidReplacementFact(Exception):
    pass


def _atomic_store_write(provider, fn: Callable[[], _T]) -> _T:
    """Run legacy store operations inside one caller-owned transaction."""
    store = getattr(provider, "_store", None)
    conn = getattr(store, "_conn", None)
    if conn is None:
        raise RuntimeError("provider store has no writable connection")
    lock = getattr(store, "_lock", None)

    def _write() -> _T:
        if conn.in_transaction:
            raise RuntimeError("provider store connection already has a transaction")
        conn.execute("BEGIN IMMEDIATE")
        store._conn = _DeferredTransactionConnection(conn)
        try:
            result = fn()
            conn.commit()
            return result
        except BaseException:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            store._conn = conn

    if lock is not None:
        with lock:
            return _write()
    return _write()


def _active_fact_exists(provider, fact_id: int) -> bool:
    store = getattr(provider, "_store", None)
    conn = getattr(store, "_conn", None)
    if conn is None:
        return False
    row = conn.execute(
        "SELECT fact_id FROM facts WHERE fact_id = ? AND invalid_at IS NULL",
        (fact_id,),
    ).fetchone()
    return row is not None


def _replacement_fact_is_distinct_and_active(
    provider, old_fact_id: int, new_fact_id: int
) -> bool:
    if old_fact_id == new_fact_id:
        return False
    store = getattr(provider, "_store", None)
    conn = getattr(store, "_conn", None)
    if conn is None:
        return False
    row = conn.execute(
        "SELECT invalid_at, superseded_by FROM facts WHERE fact_id = ?",
        (new_fact_id,),
    ).fetchone()
    return (
        row is not None
        and row["invalid_at"] is None
        and row["superseded_by"] is None
    )


def _heuristic_supersede_forbidden(provider, fact_id: int) -> bool:
    store = getattr(provider, "_store", None)
    conn = getattr(store, "_conn", None)
    if conn is None:
        return False
    columns = {row[1] for row in conn.execute("PRAGMA table_info(facts)")}
    memory_expr = "memory_kind" if "memory_kind" in columns else "NULL"
    conflict_expr = "conflict_group" if "conflict_group" in columns else "NULL"
    row = conn.execute(
        f"SELECT {memory_expr}, {conflict_expr} FROM facts WHERE fact_id = ?",
        (fact_id,),
    ).fetchone()
    if row is None:
        return False
    return row[0] == "state" or row[1] is not None


def _supersede_with_rowcount(provider, old_fact_id: int, new_fact_id: int) -> bool:
    if not _replacement_fact_is_distinct_and_active(
        provider, old_fact_id, new_fact_id
    ):
        raise _InvalidReplacementFact
    store = getattr(provider, "_store", None)
    conn = getattr(store, "_conn", None)
    if conn is None:
        return False
    cur = conn.execute(
        """
        UPDATE facts
           SET invalid_at = CURRENT_TIMESTAMP,
               superseded_by = ?
         WHERE fact_id = ? AND invalid_at IS NULL
        """,
        (new_fact_id, old_fact_id),
    )
    if int(cur.rowcount) != 1:
        return False
    try:
        reflection = importlib.import_module("enfold.reflection")
        reflection.invalidate_insights_citing(conn, old_fact_id)
    except Exception as exc:
        logger.debug("enfold MCP: insight invalidation failed: %s", exc)
    return True


def build_server(provider, read_only: bool = False) -> "FastMCP":
    """Register enfold tools against *provider* and return the FastMCP app.

    *provider* must already be initialized (see mcp_provider.build_provider).
    When *read_only* is true, memory_add and memory_supersede are never
    registered at all.
    """
    server = FastMCP(PUBLIC_SERVER_NAME)
    db_path = str(Path(provider._store.db_path).expanduser().resolve())

    @server.tool()
    def memory_search(query: Any, limit: Any = 10) -> Dict[str, Any]:
        """Hybrid search (FTS + Jaccard + HRR + dense embedding) over the shared fact store."""
        error = _require_string("query", query, MAX_QUERY_CHARS)
        if error:
            return {"error": error}
        safe_limit, error = _validated_limit(limit)
        if error:
            return {"error": error}
        results = provider.search(query, limit=safe_limit, bump=False)
        return {"results": results, "count": len(results)}

    @server.tool()
    def memory_explain(query: Any, limit: Any = 10) -> Dict[str, Any]:
        """Per-candidate scoring breakdown for *query* (same pass memory_search uses)."""
        error = _require_string("query", query, MAX_QUERY_CHARS)
        if error:
            return {"error": error}
        safe_limit, error = _validated_limit(limit)
        if error:
            return {"error": error}
        breakdown = provider.explain_search(query, limit=safe_limit)
        return {"breakdown": breakdown}

    @server.tool()
    def memory_history(fact_id: Any) -> Dict[str, Any]:
        """Full supersession chain containing *fact_id*, oldest first."""
        try:
            safe_fact_id = int(fact_id)
        except (TypeError, ValueError):
            return {"error": "invalid fact_id: must be an integer"}
        return {"history": [_json_safe_fact(f) for f in provider.fact_history(safe_fact_id)]}

    if read_only:
        return server

    @server.tool()
    def memory_add(
        content: Any,
        source: Any,
        category: Any = "general",
        tags: Any = "",
    ) -> Dict[str, Any]:
        """Add a fact, tagged with its originating agent.

        Routes through the same near-duplicate dedup gate and value-update
        supersession as the live Hermes write path: a near-verbatim
        restatement is rejected (status "deduped", the existing fact_id is
        returned instead), and a genuine value update (same wording, a
        changed number/id/state word) supersedes the prior fact rather than
        leaving both live.

        source must be a lowercase source identifier.
        """
        args = {"content": content, "category": category, "source": source}
        error = _require_source(args)
        if error:
            return {"error": error}
        error = _validate_fact_args(content, category, tags)
        if error:
            return {"error": error}

        tagged = _tag_source(tags, source)

        def _do_add() -> Dict[str, Any]:
            dup = provider._find_near_duplicate(content, category=category)
            if dup is not None:
                return {
                    "fact_id": dup.get("fact_id"),
                    "status": "deduped",
                    "note": (
                        f"near-duplicate of existing fact {dup.get('fact_id')}; "
                        "not stored again"
                    ),
                }

            update_target = provider._find_update_target(content, category=category)
            old_fact_id = (
                int(update_target["fact_id"]) if update_target is not None else None
            )
            if old_fact_id is not None and _heuristic_supersede_forbidden(
                provider, old_fact_id
            ):
                old_fact_id = None

            def _insert() -> tuple[int, bool]:
                row = provider._store._conn.execute(
                    "SELECT COALESCE(MAX(fact_id), 0) AS max_fact_id FROM facts"
                ).fetchone()
                previous_max_fact_id = int(row["max_fact_id"])
                fact_id = provider._store.add_fact(
                    content, category=category, tags=tagged
                )
                if old_fact_id is not None:
                    if not _replacement_fact_is_distinct_and_active(
                        provider, old_fact_id, int(fact_id)
                    ):
                        raise _InvalidReplacementFact
                    if not provider._supersede_fact(old_fact_id, int(fact_id)):
                        raise _SupersedeFailed
                return int(fact_id), int(fact_id) > previous_max_fact_id

            try:
                fact_id, inserted = _atomic_store_write(provider, _insert)
            except _InvalidReplacementFact:
                return {
                    "status": "failed",
                    "superseded": old_fact_id,
                    "error": (
                        "supersede failed: replacement fact is not distinct and active"
                    ),
                }
            except _SupersedeFailed:
                return {
                    "status": "failed",
                    "superseded": old_fact_id,
                    "error": "supersede failed: existing fact was not updated",
                }
            if not inserted:
                if old_fact_id is not None:
                    return {
                        "fact_id": fact_id,
                        "status": "superseded_with_existing",
                        "superseded": old_fact_id,
                    }
                return {"fact_id": fact_id, "status": "deduplicated"}
            provider._embed_cb(fact_id, content)

            result = {"fact_id": fact_id, "status": "added"}
            if old_fact_id is not None:
                result["superseded"] = old_fact_id
                result["supersede_via"] = "value_update"
            return result

        with _cross_process_write_lock(db_path):
            return _retry_on_locked(_do_add)

    @server.tool()
    def memory_supersede(
        old_fact_id: Any,
        new_content: Any,
        source: Any,
        category: Any = "general",
        tags: Any = "",
    ) -> Dict[str, Any]:
        """Explicitly supersede old_fact_id with a new fact (invalidate-not-delete).

        Unlike memory_add's implicit value-update detection, this always
        marks old_fact_id invalid, regardless of how similar the wording is.
        An exact existing fact is reused instead of inserted again. The source
        must be one of the server's supported source identifiers.
        """
        args = {"content": new_content, "source": source}
        error = _require_source(args)
        if error:
            return {"error": error}
        error = _validate_fact_args(new_content, category, tags)
        if error:
            return {"error": error}
        try:
            safe_old_fact_id = int(old_fact_id)
        except (TypeError, ValueError):
            return {"error": "invalid old_fact_id: must be a positive integer"}
        if safe_old_fact_id <= 0:
            return {"error": "invalid old_fact_id: must be a positive integer"}

        tagged = _tag_source(tags, source)

        def _do_supersede() -> Dict[str, Any]:
            if not _active_fact_exists(provider, safe_old_fact_id):
                return {"error": "invalid old_fact_id: active fact not found"}

            def _insert() -> tuple[int, bool]:
                row = provider._store._conn.execute(
                    "SELECT COALESCE(MAX(fact_id), 0) AS max_fact_id FROM facts"
                ).fetchone()
                previous_max_fact_id = int(row["max_fact_id"])
                new_fact_id = provider._store.add_fact(
                    new_content, category=category, tags=tagged
                )
                if not _supersede_with_rowcount(
                    provider, safe_old_fact_id, int(new_fact_id)
                ):
                    raise _SupersedeFailed
                return int(new_fact_id), int(new_fact_id) > previous_max_fact_id

            try:
                new_fact_id, inserted = _atomic_store_write(provider, _insert)
            except _InvalidReplacementFact:
                return {
                    "status": "failed",
                    "error": (
                        "supersede failed: replacement fact is not distinct and active"
                    ),
                }
            except _SupersedeFailed:
                return {
                    "status": "failed",
                    "error": "supersede failed: old fact was not updated",
                }
            if inserted:
                provider._embed_cb(new_fact_id, new_content)
            return {
                "fact_id": new_fact_id,
                "status": (
                    "superseded" if inserted else "superseded_with_existing"
                ),
                "superseded": safe_old_fact_id,
            }

        with _cross_process_write_lock(db_path):
            return _retry_on_locked(_do_supersede)

    return server


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db-path",
        default=None,
        help="path to memory_store.db (default: live ~/.hermes/memory_store.db)",
    )
    parser.add_argument(
        "--hermes-src",
        default=None,
        help="path to a hermes source checkout providing plugins.memory.holographic "
        "(default: $ENFOLD_HERMES_SRC or ~/hermes-migration-stage/src)",
    )
    parser.add_argument("--embedding-backend", default="ollama",
                         help="ollama or fastembed (default ollama)")
    parser.add_argument("--ollama-url", default=mcp_provider.DEFAULT_OLLAMA_URL)
    parser.add_argument("--ollama-model", default=mcp_provider.DEFAULT_OLLAMA_MODEL)
    parser.add_argument("--embedding-prefix-policy", default=mcp_provider.DEFAULT_PREFIX_POLICY)
    parser.add_argument("--hrr-dim", type=int, default=1024)
    parser.add_argument("--dedup-jaccard", type=float, default=0.9)
    parser.add_argument("--dedup-cosine", type=float, default=0.92)
    parser.add_argument(
        "--busy-timeout-ms", type=int, default=mcp_provider.DEFAULT_BUSY_TIMEOUT_MS,
        help="sqlite busy_timeout in ms for concurrent writers (default 5000)",
    )
    parser.add_argument(
        "--read-only", action="store_true",
        help="register only memory_search/memory_explain/memory_history",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    print(DEPRECATION_WARNING, file=sys.stderr)
    args = _parse_args(argv)

    db_path = args.db_path
    if db_path is None:
        try:
            from hermes_constants import get_hermes_home
            db_path = str(get_hermes_home() / "memory_store.db")
        except Exception:
            db_path = str(Path.home() / ".hermes" / "memory_store.db")

    try:
        provider = mcp_provider.build_provider(
            db_path=db_path,
            hermes_src=args.hermes_src,
            embedding_backend=args.embedding_backend,
            ollama_url=args.ollama_url,
            ollama_model=args.ollama_model,
            embedding_prefix_policy=args.embedding_prefix_policy,
            hrr_dim=args.hrr_dim,
            dedup_jaccard=args.dedup_jaccard,
            dedup_cosine=args.dedup_cosine,
            busy_timeout_ms=args.busy_timeout_ms,
            session_id="mcp-server",
            read_only=args.read_only,
        )
    except RuntimeError as exc:
        print(f"enfold MCP startup failed: {exc}", file=sys.stderr)
        return 1
    try:
        server = build_server(provider, read_only=args.read_only)
        server.run(transport="stdio")
    finally:
        provider.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
