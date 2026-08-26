"""LLM-based fact extraction from conversation transcripts.

Uses a configurable LLM, the host agent's own model by default, to extract
3-7 atomic, durable facts from a conversation. Transcripts are formatted by
the provider hooks (_format_conversation), enqueued on the persistent
extract_queue, and processed by the provider's background worker, which calls
extract_facts_from_transcript() and insert_facts().

Design principles:
- Only extract facts that will still be true next week
- One fact = one atomic, self-contained statement
- Deduplicate against existing facts in the store before inserting
- The LLM call raises on failure so the queue worker can retry with backoff
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .policy import default_credential_screen
from .prompt_safety import ephemeral_extraction_reason
from .provenance import WriteRequest

if TYPE_CHECKING:
    from plugins.memory.holographic.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InsertFactsResult:
    inserted: int = 0
    skipped: int = 0
    failed: int = 0


class ExtractionParseError(ValueError):
    """The model response could not be parsed as an extraction array."""

# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a memory extraction assistant for the Hermes client.
Your job: read a conversation and extract atomic facts worth remembering long-term.

RULES:
1. Extract 3-7 facts. Quality beats quantity: extract 0 if nothing is truly durable.
2. Each fact must be ONE atomic statement (1-3 sentences max, under 400 characters).
3. ONLY extract facts that will still be true next week, no ephemeral task details.
4. No meta-commentary ("the user asked…"). State the fact directly.
5. Assign a category from: user_pref | project | tool | general
6. Assign 2-5 comma-separated tags (lowercase, no spaces).
7. Do NOT extract facts already obvious from names/dates (e.g. "today is Monday").
8. Skip anything that's clearly a one-off instruction to the agent for this session.
9. Emit each distinct fact EXACTLY ONCE. Never output multiple rephrasings or near-duplicates of the same fact (e.g. the same SHA, URL, port, or status stated several ways); choose the single clearest phrasing.
10. Resolve pronouns and vague references to concrete entities so each fact stands alone out of context.
    11. If a fact corrects or updates something in the known facts, state the new fact with the current value; do not restate unchanged known facts. This path cannot retire existing memory.
12. Convert relative time references ("yesterday", "last week", "next month") to absolute dates using TODAY'S DATE (given below), so every fact is unambiguous out of context.

GOOD facts:
- "The user prefers Postgres over MySQL for new projects."
- "The production app deploys to Vercel from the main branch."
- "The user's working hours are roughly 9am-6pm Eastern."

BAD facts (don't extract):
- "The user asked me to summarize a document." (ephemeral task)
- "The conversation was about memory systems." (meta)
- "The user said hello." (trivial)
- The same fact restated several ways, e.g. "X is pinned at SHA abc" + "X uses commit abc" + "X's clone is at abc" (duplication, output it once)

OUTPUT: Respond with a JSON array only. No prose, no markdown fences.
Example:
[
  {"content": "The user prefers pnpm as their package manager for Node projects.", "category": "tool", "tags": "pnpm,node,package-manager"},
  {"content": "The legacy v2 service is archived; active development is on v3.", "category": "project", "tags": "v3,migration,project-status"}
]

If nothing is worth saving, output: []
"""

_USER_TEMPLATE = """\
Today's date is {today}. Here is the conversation to extract facts from.
Existing known facts (do not re-extract these):
{existing_summary}

---CONVERSATION---
{conversation}
---END---

Extract new durable facts not already in the known facts list.
"""


def _format_conversation(messages: List[Dict[str, Any]], max_chars: int = 12000) -> str:
    """Render messages as a readable transcript.

    Oversized transcripts fail closed. Silently keeping a tail would drop
    earlier decisions and can cut a session mid-thought.
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if not isinstance(content, str):
            # Handle list content (tool results etc.), flatten to string
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            else:
                content = str(content)
        if not content.strip():
            continue
        if len(content) > 2000:
            raise ValueError("legacy extraction transcript exceeds size limit")
        lines.append(f"{role.upper()}: {content}")

    transcript = "\n\n".join(lines)
    if len(transcript) > max_chars:
        raise ValueError("legacy extraction transcript exceeds size limit")
    return transcript


def _existing_summary(
    store: "MemoryStore",
    limit: int = 40,
    *,
    topic: str = "",
    search_fn=None,
    similar_limit: int = 10,
) -> str:
    """Return a compact summary of existing facts for the dedup prompt.

    Blends the most trusted facts with the most recently created ones. Top
    trust alone misses freshly inserted trust-0.5 facts, so repeated
    pre-compress events in one session would re-extract paraphrases of facts
    the store already holds. The recent slice keeps just-added facts in the
    dedup context.

    When *search_fn* is given (a callable ``search_fn(topic, limit)`` such as
    the provider's hybrid ``search()``), its top hits for *topic* are also
    included: on a large store, a fact worth deduping against may be neither
    top-trust nor recent, so the transcript's own topic must be searched for
    directly.
    """
    recent_slice = 15
    top_slice = max(limit - recent_slice, 1)
    contents: List[str] = []
    try:
        rows = store._conn.execute(
            """
            SELECT content FROM (
                SELECT fact_id, content FROM (
                    SELECT fact_id, content FROM facts
                    ORDER BY trust_score DESC, updated_at DESC LIMIT ?
                )
                UNION
                SELECT fact_id, content FROM (
                    SELECT fact_id, content FROM facts
                    ORDER BY created_at DESC, fact_id DESC LIMIT ?
                )
                ORDER BY fact_id
                LIMIT ?
            )
            """,
            (top_slice, recent_slice, limit),
        ).fetchall()
        contents.extend(row["content"] for row in rows)
    except Exception:
        return "(unavailable)"

    if search_fn is not None and topic.strip():
        try:
            hits = search_fn(topic, similar_limit)
            for hit in hits:
                content = hit.get("content", "")
                if content and content not in contents:
                    contents.append(content)
        except Exception as exc:
            logger.debug("llm_extract: topic search for dedup context failed: %s", exc)

    if not contents:
        return "(none)"
    return "\n".join(f"- {content[:120]}" for content in contents)


def _extract_json_array(text: str) -> Optional[str]:
    """Return the first balanced top-level ``[...]`` block in *text*, or None.

    Bracket-balanced and string-literal aware, so it survives ```json fences and
    any leading/trailing prose (a very common LLM output shape) without being
    confused by brackets that appear inside a fact's content.
    """
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _parse_response(raw: str) -> List[Dict[str, str]]:
    """Parse an LLM response into a list of fact dicts.

    Tolerant of ```json fences and of prose before or after the JSON array: it
    tries a direct parse, then falls back to the first balanced top-level
    ``[...]`` block so trailing/leading text never drops a valid extraction.
    """
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as direct_error:
        block = _extract_json_array(raw)
        if block is None:
            raise ExtractionParseError(
                "extraction response contains no JSON array "
                f"(length={len(raw)}, parse_error_pos={direct_error.pos})"
            ) from direct_error
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError as exc:
            raise ExtractionParseError(
                "extraction response contains invalid JSON "
                f"(length={len(raw)}, parse_error_pos={exc.pos})"
            ) from exc

    if not isinstance(parsed, list):
        raise ExtractionParseError("extraction response must be a JSON array")
    validated = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        category = str(item.get("category", "general")).strip()
        tags = str(item.get("tags", "")).strip()
        if not content or len(content) < 10:
            continue
        if category not in ("user_pref", "project", "tool", "general"):
            category = "general"
        validated.append({"content": content[:400], "category": category, "tags": tags})
    return validated


def extract_facts_from_transcript(
    transcript: str,
    store: "MemoryStore",
    *,
    provider: str,
    model: str,
    effort: str | None = None,
    search_fn=None,
) -> List[Dict[str, str]]:
    """Call the extraction LLM on an already formatted transcript.

    Returns the validated fact dicts ([] when the model finds nothing worth
    saving). Raises on transport or LLM failure so the persistent queue
    worker can retry with backoff instead of losing the transcript.

    *search_fn*, when given, widens the dedup context (see _existing_summary)
    with facts similar to this transcript, not just top-trust/recent ones.
    """
    if not provider or not model:
        raise RuntimeError("extraction provider/model not configured")
    if not transcript or not transcript.strip():
        return []

    from agent.auxiliary_client import call_llm  # ImportError propagates: retryable

    existing = _existing_summary(store, topic=transcript, search_fn=search_fn)
    from datetime import date
    user_msg = _USER_TEMPLATE.format(
        today=date.today().isoformat(),
        existing_summary=existing,
        conversation=transcript,
    )

    resp = call_llm(
        provider=provider,
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=1024,
        timeout=180,
        extra_body=({"reasoning": {"effort": effort}} if effort else None),
    )
    raw = resp.choices[0].message.content or ""
    return _parse_response(raw)


def _existing_fact_id(store: "MemoryStore", content: str) -> Optional[int]:
    conn = getattr(store, "_conn", None)
    if conn is None:
        return None
    try:
        row = conn.execute(
            "SELECT fact_id FROM facts WHERE content = ? LIMIT 1",
            (content,),
        ).fetchone()
    except Exception:
        return None
    return int(row[0]) if row is not None else None


def _credential_shaped(content: str, category: str, tags: str) -> bool:
    request = WriteRequest(
        idempotency_key="legacy-extraction-screen",
        content=content,
        source_type="automatic_extraction",
        category=category,
        tags=tags,
    )
    return default_credential_screen(request) is not None


def _refuse_extraction_update(content: str, category: str, *, update_check) -> bool:
    """Return True when extraction would have retired an existing fact."""
    if update_check is None:
        return False
    update_target = update_check(content, category=category)
    return update_target is not None


def _total_changes(store: "MemoryStore") -> Optional[int]:
    conn = getattr(store, "_conn", None)
    value = getattr(conn, "total_changes", None)
    return int(value) if value is not None else None


def insert_facts(
    store: "MemoryStore",
    facts: List[Dict[str, str]],
    embed_callback=None,  # optional: callable(fact_id, content) to trigger embedding
    dedup_check=None,  # optional: callable(content, category=...) -> existing fact dict or None
    update_check=None,  # optional: callable(content, category=...) -> existing fact dict or None
    supersede=None,  # optional: callable(old_fact_id, new_fact_id) -> None
) -> InsertFactsResult:
    """Insert extracted facts into the store and return outcome counts.

    Per-fact storage failures are logged and counted so the queue can retry
    its persisted proposal snapshot. Duplicates, unsafe credential-shaped
    proposals, empty facts, and refused updates count as skipped.

    When *dedup_check* is given (the provider's near-duplicate gate, the same
    one the interactive fact_store "add" action uses), each fact is checked
    against the store before insert so extraction never floods the store with
    duplicates the interactive path would have caught.

    When *update_check* finds a VALUE UPDATE of an existing fact, the
    proposal is skipped. This path cannot retire existing memory; only the
    daemon processor may change current state under its supersession policy.
    The *supersede* argument is accepted for caller compatibility and ignored.
    """
    inserted = 0
    skipped = 0
    failed = 0
    _ = supersede
    for fact in facts:
        try:
            content = str(fact.get("content", "")).strip()
            category = str(fact.get("category", "general")).strip() or "general"
            tags = str(fact.get("tags", "")).strip()
            if not content:
                skipped += 1
                continue
            if _credential_shaped(content, category, tags):
                logger.warning(
                    "llm_extract: dropped credential-shaped extraction proposal"
                )
                skipped += 1
                continue
            skip_reason = ephemeral_extraction_reason(content)
            if skip_reason is not None:
                logger.info(
                    "llm_extract: dropped proposal (%s): %r",
                    skip_reason,
                    content[:80],
                )
                skipped += 1
                continue
            if dedup_check is not None:
                dup = dedup_check(content, category=category)
                if dup is not None:
                    logger.debug(
                        "llm_extract: skipped near-duplicate of fact %s: %r",
                        dup.get("fact_id"), content[:80],
                    )
                    skipped += 1
                    continue
            existing_id = _existing_fact_id(store, content)
            if existing_id is not None:
                skipped += 1
                continue
            if _refuse_extraction_update(
                content, category, update_check=update_check
            ):
                logger.info(
                    "llm_extract: refused extraction update that would retire "
                    "an existing fact: %r",
                    content[:80],
                )
                skipped += 1
                continue
            before_changes = _total_changes(store)
            fact_id = store.add_fact(
                content,
                category=category,
                tags=tags,
            )
            after_changes = _total_changes(store)
            if before_changes is not None and after_changes == before_changes:
                skipped += 1
                continue
            inserted += 1
            if embed_callback and fact_id:
                try:
                    embed_callback(fact_id, content)
                except Exception as emb_exc:
                    logger.debug("llm_extract: embed callback failed for %d: %s", fact_id, emb_exc)
        except Exception as exc:
            failed += 1
            logger.debug("llm_extract: add_fact failed: %s", exc)

    if inserted:
        logger.info("llm_extract: inserted %d/%d extracted facts", inserted, len(facts))
    return InsertFactsResult(inserted=inserted, skipped=skipped, failed=failed)
