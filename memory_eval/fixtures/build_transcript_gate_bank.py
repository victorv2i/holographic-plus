#!/usr/bin/env python3
"""Build the committed capture-gate fixtures.

This script writes compacted role-structured cases already listed
below. People, courses, projects, and preferences are synthetic.
It does not open a live memory database. Do not add private
transcripts, live session ids, live fact ids, or private-arena cases.

Run from the worktree root:

    PYTHONPATH=$PWD python memory_eval/fixtures/build_transcript_gate_bank.py
"""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
CASE_SCHEMA = "enfold-transcript-gate-case-v1"
OUTPUT_SCHEMA = "enfold-transcript-gate-output-v1"


def _case(
    case_id: str,
    turns: list[dict[str, str]],
    *,
    decision: str,
    expected: list[dict],
    forbidden: list[dict],
    tags: list[str],
    source: dict,
) -> dict:
    return {
        "schema_version": CASE_SCHEMA,
        "case_id": case_id,
        "turns": turns,
        "expected_decision": decision,
        "expected_facts": expected,
        "forbidden_spans": forbidden,
        "tags": tags,
        "source": source,
    }


def _typed(key: str, patterns: list[str], excerpt: str, state: dict) -> dict:
    return {
        "key": key,
        "kind": "typed_slot",
        "speaker": "user",
        "content_patterns": patterns,
        "evidence_excerpt": excerpt,
        "state": state,
    }


def _untyped(key: str, patterns: list[str], excerpt: str) -> dict:
    return {
        "key": key,
        "kind": "untyped_durable",
        "speaker": "user",
        "content_patterns": patterns,
        "evidence_excerpt": excerpt,
    }


def _forbid(key: str, speaker: str, class_name: str, patterns: list[str]) -> dict:
    return {
        "key": key,
        "speaker": speaker,
        "class": class_name,
        "content_patterns": patterns,
    }


BRICK = "I'm Brick, your Minecraft server setup assistant."
IMPLEMENT = "I'll now implement the concise style for all bots."
FLEET = "METIS_FLEET_OK"
SKILL_PRUNE = "[Skills pruned during compression -- reload before acting on these tasks]"
OUTLOOK = (
    "The user is asking about the status of a specific command related to "
    "Outlook and CRU monitoring. Let me analyze the context."
)
MOBILE_NOVEL = (
    "The user is using the Harbor Pocket app on their desktop and mobile, "
    "and they're waiting for Apple to approve it for the App Store."
)
SYLLABUS = (
    "I need to create a concise summary of the current situation with the "
    "CourseBoard course and the syllabus issues."
)
THINKING = "Final independent review confirms the deployed Thinking fix passes:"
PRODUCT = (
    "That's the strongest product signal yet: you built it for mobile, but "
    "you're voluntarily using it on desktop because the experience is better."
)
CODE_MONOLOGUE = (
    "I need to update the Harbor bot to be more concise and conversational. "
    "Let me analyze the current codebase."
)
EIGHT_BOTS = (
    "Done. All eight bots now default to 1-3 conversational sentences, "
    "result and next move first."
)
FLEET_IMPL = (
    "I'll now implement the final configuration for the Harbor Fleet "
    "Intelligence Unit to ensure all bots are using the concise style."
)
BOT_CONFIRM = (
    "I'll help you set up a concise, action-first communication style for "
    "your bot. Notch confirmed. Brick confirmed."
)
ACK_STATUS = (
    "I need to confirm the status of the Bot Chat session. The user has "
    "sent multiple ACK messages from different bots."
)
RELOAD_SKILLS = (
    "I need to reload the skill instructions that were pruned during "
    "compression before executing any tasks."
)


def cases() -> list[dict]:
    src_junk = {
        "kind": "production_unreviewed",
        "correction_status": "unreviewed",
        "source_type": "automatic_extraction",
        "asserted_by": "subprocess:ollama:qwen3-30b:durable-memory-v3",
        "mined": "synthetic compacted production-junk shapes",
    }
    src_user = {
        "kind": "store_user_claim_shape",
        "mined": "synthetic compacted user-claim shapes",
    }
    rows = []

    rows.append(
        _case(
            "prod-autoextract-junk-replay",
            [
                {"role": "user", "content": "Ada prefers concise responses."},
                {"role": "assistant", "content": IMPLEMENT},
                {"role": "assistant", "content": BRICK},
                {"role": "tool", "content": FLEET},
                {"role": "user", "content": "Ada uses local-first memory tools."},
                {"role": "assistant", "content": OUTLOOK},
                {"role": "assistant", "content": SKILL_PRUNE},
                {"role": "assistant", "content": EIGHT_BOTS},
            ],
            decision="add",
            expected=[
                _typed(
                    "ada-prefers-concise",
                    [r"(?i)\bada\b.*\bprefers?\b.*\bconcise\b"],
                    "Ada prefers concise responses.",
                    {
                        "kind": "preference",
                        "subject": "person:ada",
                        "predicate": "response_style",
                        "value": "concise",
                    },
                ),
                _untyped(
                    "ada-local-first-tools",
                    [r"(?i)\bada\b.*\blocal-first memory tools\b"],
                    "Ada uses local-first memory tools.",
                ),
            ],
            forbidden=[
                _forbid(
                    "brick-self-desc",
                    "assistant",
                    "assistant_authored",
                    [r"(?i)\bi'?m brick\b"],
                ),
                _forbid(
                    "implement-concise",
                    "assistant",
                    "assistant_authored",
                    [r"(?i)\bi'll now implement the concise style\b"],
                ),
                _forbid(
                    "metis-banner",
                    "tool",
                    "tool_banner",
                    [r"METIS_FLEET_OK"],
                ),
                _forbid(
                    "outlook-monologue",
                    "assistant",
                    "assistant_monologue",
                    [r"(?i)\bthe user is asking about the status\b"],
                ),
                _forbid(
                    "skill-prune-banner",
                    "assistant",
                    "assistant_monologue",
                    [r"(?i)skills pruned during compression"],
                ),
                _forbid(
                    "eight-bots-done",
                    "assistant",
                    "assistant_authored",
                    [r"(?i)\ball eight bots now default\b"],
                ),
            ],
            tags=["production-failure", "replay", "real-transcript"],
            source={
                **src_junk,
                "session_ids": [
                    "fixture-sess-fleet-01",
                    "fixture-sess-brick-01",
                    "fixture-sess-metis-01",
                    "fixture-cron-campus-01",
                ],
                "fact_ids": [
                    81001,
                    81008,
                    81009,
                    81010,
                    81012,
                    81013,
                ],
            },
        )
    )

    junk_specs = [
        (
            "junk-campus-outlook-monologue",
            "cron",
            OUTLOOK,
            "assistant_monologue",
            r"(?i)\boutlook and cru monitoring\b",
            81001,
        ),
        (
            "junk-mobile-assistant-novel",
            "fixture-sess-mobile-01",
            MOBILE_NOVEL,
            "assistant_monologue",
            r"(?i)\bharbor pocket app on their desktop\b",
            81002,
        ),
        (
            "junk-course-syllabus-summary",
            "fixture-sess-course-01",
            SYLLABUS,
            "assistant_monologue",
            r"(?i)\bconcise summary of the current situation\b",
            81003,
        ),
        (
            "junk-thinking-placeholder-review",
            "fixture-sess-think-01",
            THINKING,
            "assistant_authored",
            r"(?i)\bthinking fix passes\b",
            81004,
        ),
        (
            "junk-mobile-product-signal",
            "fixture-sess-mobile-01",
            PRODUCT,
            "assistant_authored",
            r"(?i)\bstrongest product signal yet\b",
            81005,
        ),
        (
            "junk-code-change-monologue",
            "fixture-sess-mobile-01",
            CODE_MONOLOGUE,
            "assistant_monologue",
            r"(?i)\banalyze the current codebase\b",
            81006,
        ),
        (
            "junk-skill-prune-banner",
            "fixture-sess-fleet-01",
            SKILL_PRUNE,
            "assistant_monologue",
            r"(?i)skills pruned during compression",
            81008,
        ),
        (
            "junk-eight-bots-done",
            "fixture-sess-fleet-01",
            EIGHT_BOTS,
            "assistant_authored",
            r"(?i)\ball eight bots now default\b",
            81009,
        ),
        (
            "junk-brick-self-description",
            "fixture-sess-brick-01",
            BRICK,
            "assistant_authored",
            r"(?i)\bi'?m brick\b",
            81010,
        ),
        (
            "junk-implement-harbor-fleet",
            "fixture-sess-fleet-01",
            FLEET_IMPL,
            "assistant_authored",
            r"(?i)\bi'll now implement the final configuration\b",
            81011,
        ),
        (
            "junk-implement-concise-style",
            "fixture-sess-fleet-01",
            IMPLEMENT,
            "assistant_authored",
            r"(?i)\bi'll now implement the concise style\b",
            81012,
        ),
        (
            "junk-metis-fleet-ok",
            "fixture-sess-metis-01",
            FLEET,
            "tool_banner",
            r"METIS_FLEET_OK",
            81013,
        ),
        (
            "junk-bot-mode-confirmation",
            "fixture-sess-fleet-01",
            BOT_CONFIRM,
            "assistant_authored",
            r"(?i)\bnotch confirmed\. brick confirmed\b",
            81014,
        ),
        (
            "junk-ack-session-status",
            "fixture-sess-ack-01",
            ACK_STATUS,
            "assistant_monologue",
            r"(?i)\bmultiple ack messages from different bots\b",
            81015,
        ),
        (
            "junk-reload-pruned-skills",
            "fixture-sess-fleet-01",
            RELOAD_SKILLS,
            "assistant_monologue",
            r"(?i)\breload the skill instructions\b",
            81009,
        ),
    ]
    for case_id, session_id, text, class_name, pattern, fact_id in junk_specs:
        role = "tool" if class_name == "tool_banner" else "assistant"
        rows.append(
            _case(
                case_id,
                [
                    {"role": "user", "content": "ACK"},
                    {"role": role, "content": text},
                ],
                decision="abstain",
                expected=[],
                forbidden=[_forbid(case_id + "-span", role, class_name, [pattern])],
                tags=["real-transcript", "junk-negative", "unreviewed"],
                source={**src_junk, "session_id": session_id, "fact_id": fact_id},
            )
        )

    rows.append(
        _case(
            "junk-desktop-just-works-wrapped",
            [
                {
                    "role": "user",
                    "content": (
                        "not because experience is better its because it just "
                        "works on my desktop but it works on phone well too"
                    ),
                },
                {"role": "assistant", "content": MOBILE_NOVEL},
            ],
            decision="add",
            expected=[
                _untyped(
                    "bot-mobile-works-desktop-and-phone",
                    [r"(?i)\bjust works on my desktop\b", r"(?i)\bworks on phone well too\b"],
                    "it just works on my desktop but it works on phone well too",
                )
            ],
            forbidden=[
                _forbid(
                    "wrapped-as-assistant-novel",
                    "assistant",
                    "assistant_monologue",
                    [r"(?i)\bwaiting for apple to approve\b"],
                )
            ],
            tags=["real-transcript", "junk-negative", "user-claim-buried"],
            source={**src_junk, "fact_id": 81016, "session_id": "fixture-sess-mobile-01"},
        )
    )

    rows.extend(
        [
            _case(
                "mixed-concise-plus-implement",
                [
                    {"role": "user", "content": "Ada prefers concise responses."},
                    {"role": "assistant", "content": IMPLEMENT},
                ],
                decision="add",
                expected=[
                    _typed(
                        "concise-pref",
                        [r"(?i)\bada\b.*\bprefers?\b.*\bconcise\b"],
                        "Ada prefers concise responses.",
                        {
                            "kind": "preference",
                            "subject": "person:ada",
                            "predicate": "response_style",
                            "value": "concise",
                        },
                    )
                ],
                forbidden=[
                    _forbid(
                        "implement-span",
                        "assistant",
                        "assistant_authored",
                        [r"(?i)\bi'll now implement\b"],
                    )
                ],
                tags=["real-transcript", "mixed"],
                source=src_junk,
            ),
            _case(
                "mixed-local-first-plus-brick",
                [
                    {"role": "user", "content": "Ada uses local-first memory tools."},
                    {"role": "assistant", "content": BRICK},
                ],
                decision="add",
                expected=[
                    _untyped(
                        "local-first",
                        [r"(?i)\blocal-first memory tools\b"],
                        "Ada uses local-first memory tools.",
                    )
                ],
                forbidden=[
                    _forbid(
                        "brick-span",
                        "assistant",
                        "assistant_authored",
                        [r"(?i)\bi'?m brick\b"],
                    )
                ],
                tags=["real-transcript", "mixed"],
                source=src_junk,
            ),
            _case(
                "mixed-ack-plus-metis",
                [
                    {"role": "user", "content": "ACK"},
                    {"role": "tool", "content": FLEET},
                ],
                decision="abstain",
                expected=[],
                forbidden=[
                    _forbid("metis-span", "tool", "tool_banner", [r"METIS_FLEET_OK"])
                ],
                tags=["real-transcript", "mixed"],
                source=src_junk,
            ),
            _case(
                "mixed-just-works-plus-novel",
                [
                    {
                        "role": "user",
                        "content": (
                            "i just need to wait till apple emails me back so "
                            "i can finish this process"
                        ),
                    },
                    {"role": "assistant", "content": PRODUCT},
                ],
                decision="add",
                expected=[
                    _untyped(
                        "waiting-on-apple-email",
                        [r"(?i)\bwait till apple emails me back\b"],
                        "i just need to wait till apple emails me back so i can finish this process",
                    )
                ],
                forbidden=[
                    _forbid(
                        "product-signal",
                        "assistant",
                        "assistant_authored",
                        [r"(?i)\bstrongest product signal yet\b"],
                    )
                ],
                tags=["real-transcript", "mixed"],
                source={**src_junk, "fact_id": 81016},
            ),
            _case(
                "mixed-pref-plus-status-report",
                [
                    {
                        "role": "user",
                        "content": (
                            "Ada wants every Harbor bot to default to concise, "
                            "conversational replies: usually 1-3 sentences, result "
                            "and next move first."
                        ),
                    },
                    {"role": "assistant", "content": EIGHT_BOTS},
                ],
                decision="add",
                expected=[
                    _typed(
                        "every-bot-concise",
                        [r"(?i)\bevery harbor bot\b.*\bconcise\b"],
                        "Ada wants every Harbor bot to default to concise, conversational replies: usually 1-3 sentences, result and next move first.",
                        {
                            "kind": "preference",
                            "subject": "person:ada",
                            "predicate": "bot_response_style",
                            "value": "concise-1-3",
                        },
                    )
                ],
                forbidden=[
                    _forbid(
                        "done-eight-bots",
                        "assistant",
                        "assistant_authored",
                        [r"(?i)\ball eight bots now default\b"],
                    )
                ],
                tags=["real-transcript", "mixed"],
                source={**src_user, "fact_id": 81017, "correction_status": "human_confirmed"},
            ),
            _case(
                "mixed-version-plus-tool-ok",
                [
                    {
                        "role": "user",
                        "content": (
                            "As of 2026-08-24, live Harbor Agent on this machine "
                            "is v0.20.5 at the harbor-agent checkout on main."
                        ),
                    },
                    {"role": "tool", "content": FLEET},
                ],
                decision="add",
                expected=[
                    _typed(
                        "harbor-live-version",
                        [r"(?i)\blive harbor agent\b.*\bv0\.20\.5\b"],
                        "live Harbor Agent on this machine is v0.20.5 at the harbor-agent checkout on main.",
                        {
                            "kind": "state",
                            "subject": "harbor_runtime",
                            "predicate": "live_version",
                            "value": "0.20.5",
                        },
                    )
                ],
                forbidden=[
                    _forbid("tool-ok", "tool", "tool_banner", [r"METIS_FLEET_OK"])
                ],
                tags=["real-transcript", "mixed", "typed-slot"],
                source={**src_user, "fact_id": 81018},
            ),
            _case(
                "mixed-accommodation-plus-thinking",
                [
                    {
                        "role": "user",
                        "content": "STA 4410 RVC has 2.0x quiz-time accommodation.",
                    },
                    {"role": "assistant", "content": THINKING},
                ],
                decision="add",
                expected=[
                    _typed(
                        "sta4410-quiz-time",
                        [r"(?i)\bsta 4410\b.*\b2\.0x quiz-time\b"],
                        "STA 4410 RVC has 2.0x quiz-time accommodation.",
                        {
                            "kind": "state",
                            "subject": "course:sta4410",
                            "predicate": "quiz_time_accommodation",
                            "value": "2.0x",
                        },
                    )
                ],
                forbidden=[
                    _forbid(
                        "thinking-review",
                        "assistant",
                        "assistant_authored",
                        [r"(?i)\bthinking fix passes\b"],
                    )
                ],
                tags=["real-transcript", "mixed", "typed-slot"],
                source={**src_user, "fact_id": 81019},
            ),
            _case(
                "mixed-meshlink-plus-appstore-novel",
                [
                    {
                        "role": "user",
                        "content": (
                            "Ada created a Harbor Pocket client and uses "
                            "its MeshLink link to access the live Harbor system "
                            "from mobile."
                        ),
                    },
                    {"role": "assistant", "content": MOBILE_NOVEL},
                ],
                decision="add",
                expected=[
                    _untyped(
                        "bot-mobile-meshlink",
                        [r"(?i)\bharbor pocket\b.*\bmeshlink\b"],
                        "Ada created a Harbor Pocket client and uses its MeshLink link to access the live Harbor system from mobile.",
                    )
                ],
                forbidden=[
                    _forbid(
                        "appstore-novel",
                        "assistant",
                        "assistant_monologue",
                        [r"(?i)\bwaiting for apple to approve\b"],
                    )
                ],
                tags=["real-transcript", "mixed"],
                source={**src_user, "fact_id": 81020, "correction_status": "human_confirmed"},
            ),
        ]
    )

    typed_rows = [
        (
            "typed-prefers-concise-style",
            "Ada prefers concise, conversational, priority-ranked answers: default to 1-3 sentences, give the result and next move.",
            r"(?i)\bada\b.*\bprefers?\b.*\bconcise, conversational\b",
            {
                "kind": "preference",
                "subject": "person:ada",
                "predicate": "response_style",
                "value": "concise-1-3",
            },
            81021,
        ),
        (
            "typed-prefers-online-cru-schedule",
            "As of 2026-08-23, Ada prefers an online-only Fall 2026 CRU schedule so coursework does not compete with the external job search.",
            r"(?i)\bada\b.*\bprefers?\b.*\bonline-only fall 2026 cru schedule\b",
            {
                "kind": "preference",
                "subject": "person:ada",
                "predicate": "cru_schedule_mode",
                "value": "online-only",
                "valid_from": "2026-08-23",
            },
            81022,
        ),
        (
            "typed-prefers-native-cli-routing",
            "Ada prefers multi-model work routed by native CLI: Codex for GPT, Claude Code for Opus, and Cursor Grok for Grok.",
            r"(?i)\bada\b.*\bprefers?\b.*\bnative cli\b",
            {
                "kind": "preference",
                "subject": "person:ada",
                "predicate": "model_routing",
                "value": "native-cli",
            },
            81023,
        ),
        (
            "typed-prefers-steering-circle",
            "Ada prefers a reusable Steering Circle for consequential decisions: native Codex for GPT, Claude Code for Claude, Cursor only for other models.",
            r"(?i)\bada\b.*\bprefers?\b.*\bsteering circle\b",
            {
                "kind": "preference",
                "subject": "person:ada",
                "predicate": "decision_council",
                "value": "steering-circle",
            },
            81024,
        ),
        (
            "typed-quiz-accommodation-none",
            "STA 6204 RXU has no quiz-time multiplier detected.",
            r"(?i)\bsta 6204\b.*\bno quiz-time multiplier\b",
            {
                "kind": "state",
                "subject": "course:sta6204",
                "predicate": "quiz_time_accommodation",
                "value": "none",
            },
            81025,
        ),
        (
            "typed-quiz-accommodation-mgt2101",
            "MGT 2101 RVC has 2.0x quiz-time accommodation.",
            r"(?i)\bmgt 2101\b.*\b2\.0x quiz-time\b",
            {
                "kind": "state",
                "subject": "course:mgt2101",
                "predicate": "quiz_time_accommodation",
                "value": "2.0x",
            },
            81026,
        ),
        (
            "typed-harborkit-typography",
            "As of 2026-08-11, Ada explicitly prefers HarborKit typography to feel modern and LMS-like, with no Times or serif appearance.",
            r"(?i)\bada\b.*\bprefers?\b.*\bharborkit typography\b",
            {
                "kind": "preference",
                "subject": "project:harborkit",
                "predicate": "typography",
                "value": "modern-lms-like",
            },
            81027,
        ),
        (
            "typed-bio-online-over-csc",
            "The user prefers to take BIO 3308 Biostatistics I online to satisfy the Data Analysis requirement, rather than CSC 4401.",
            r"(?i)\bprefers?\b.*\bbio 3308\b.*\bonline\b",
            {
                "kind": "preference",
                "subject": "person:ada",
                "predicate": "data_analysis_course",
                "value": "BIO 3308 online",
            },
            81028,
        ),
        (
            "typed-fairmart-clothing-fit",
            "The user prefers to buy clothes at FairMart for fit and value over other stores.",
            r"(?i)\bprefers?\b.*\bbuy clothes at fairmart\b",
            {
                "kind": "preference",
                "subject": "person:ada",
                "predicate": "clothing_store",
                "value": "FairMart",
            },
            81029,
        ),
        (
            "typed-threadwell-specific-pieces",
            "The user prefers Threadwell only for exact pieces already proven to work.",
            r"(?i)\bprefers?\b.*\bthreadwell only for exact pieces\b",
            {
                "kind": "preference",
                "subject": "person:ada",
                "predicate": "threadwell_use",
                "value": "proven-pieces-only",
            },
            81030,
        ),
        (
            "typed-simple-bullet-results",
            "The user prefers the final result in simple bullets.",
            r"(?i)\bprefers?\b.*\bfinal result in simple bullets\b",
            {
                "kind": "preference",
                "subject": "person:ada",
                "predicate": "result_format",
                "value": "simple-bullets",
            },
            81031,
        ),
    ]
    for case_id, content, pattern, state, fact_id in typed_rows:
        rows.append(
            _case(
                case_id,
                [{"role": "user", "content": content}],
                decision="add",
                expected=[_typed(case_id + "-fact", [pattern], content, state)],
                forbidden=[],
                tags=["real-transcript", "typed-slot"],
                source={**src_user, "fact_id": fact_id},
            )
        )

    untyped_rows = [
        (
            "untyped-harbor-safe-update",
            "Harbor runtime updates must go through harbor-safe-update.sh, which rebases carried patches. The bundled harbor update does git reset --hard origin/main and would wipe them.",
            r"(?i)\bharbor-safe-update\.sh\b",
            81032,
        ),
        (
            "untyped-retired-checkout-path",
            "The harbor-agent path on main is a retired reference checkout and is not the live gateway source.",
            r"(?i)\bretired reference checkout\b",
            81033,
        ),
        (
            "untyped-courseboard-migration-zero",
            "CourseBoard migration 44001227 completed with zero reported import issues.",
            r"(?i)\bcourseboard migration\b.*\bzero reported import issues\b",
            81034,
        ),
        (
            "untyped-browser-use-3",
            "Browser Use 3.0 support was added and tested for the Access Office quiz-extension skill.",
            r"(?i)\bbrowser use 3\.0 support\b",
            81035,
        ),
        (
            "untyped-created-bot-mobile",
            "Ada created a Harbor Pocket client and uses its MeshLink link to access the live Harbor system from mobile.",
            r"(?i)\bcreated a harbor pocket client\b",
            81020,
        ),
        (
            "untyped-access-skill-validated",
            "The Access Office quiz-extension skill was created and validated.",
            r"(?i)\baccess office quiz-extension skill\b.*\bvalidated\b",
            81036,
        ),
        (
            "untyped-daily-loop-carry-forward",
            "Carry-forward: ECE 2204 supported-section timed-assessment check, then STA 4410 accommodation review.",
            r"(?i)\bece 2204\b.*\bsta 4410 accommodation review\b",
            81037,
        ),
        (
            "untyped-email-review-phrase",
            "The Access Office email template now includes the phrase review the accommodations, and let me know if you have any questions.",
            r"(?i)\breview the accommodations\b",
            81038,
        ),
    ]
    for case_id, content, pattern, fact_id in untyped_rows:
        rows.append(
            _case(
                case_id,
                [{"role": "user", "content": content}],
                decision="add",
                expected=[_untyped(case_id + "-fact", [pattern], content)],
                forbidden=[],
                tags=["real-transcript", "untyped-durable"],
                source={**src_user, "fact_id": fact_id},
            )
        )

    rows.extend(
        [
            _case(
                "abstain-ephemeral-ack",
                [
                    {"role": "user", "content": "ACK"},
                    {"role": "assistant", "content": "ACK"},
                ],
                decision="abstain",
                expected=[],
                forbidden=[],
                tags=["real-transcript", "ephemeral"],
                source={"kind": "session_shape", "note": "multi-bot ACK flood"},
            ),
            _case(
                "abstain-skill-prune-only",
                [{"role": "assistant", "content": SKILL_PRUNE}],
                decision="abstain",
                expected=[],
                forbidden=[
                    _forbid(
                        "prune-only",
                        "assistant",
                        "assistant_monologue",
                        [r"(?i)skills pruned during compression"],
                    )
                ],
                tags=["real-transcript", "ephemeral"],
                source=src_junk,
            ),
            _case(
                "reject-secret-token",
                [
                    {
                        "role": "user",
                        "content": "My deployment API token is DEMO_TOKEN_REDACTED. Save it for later.",
                    },
                    {
                        "role": "assistant",
                        "content": "Stored DEMO_TOKEN_REDACTED for later use.",
                    },
                ],
                decision="reject",
                expected=[],
                forbidden=[
                    _forbid(
                        "api-token",
                        "assistant",
                        "assistant_authored",
                        [r"(?i)\bdemo_token_redacted\b"],
                    )
                ],
                tags=["real-transcript", "secret"],
                source={"kind": "safety-shape", "note": "same class as seed reject-secret"},
            ),
            _case(
                "abstain-tool-banner-only",
                [{"role": "tool", "content": FLEET}],
                decision="abstain",
                expected=[],
                forbidden=[
                    _forbid("banner-only", "tool", "tool_banner", [r"METIS_FLEET_OK"])
                ],
                tags=["real-transcript", "tool-only"],
                source=src_junk,
            ),
            _case(
                "abstain-assistant-self-id-only",
                [{"role": "assistant", "content": BRICK}],
                decision="abstain",
                expected=[],
                forbidden=[
                    _forbid(
                        "self-id-only",
                        "assistant",
                        "assistant_authored",
                        [r"(?i)\bi'?m brick\b"],
                    )
                ],
                tags=["real-transcript", "self-description"],
                source=src_junk,
            ),
            _case(
                "abstain-cron-monitor-dump",
                [{"role": "assistant", "content": OUTLOOK}],
                decision="abstain",
                expected=[],
                forbidden=[
                    _forbid(
                        "cron-dump",
                        "assistant",
                        "assistant_monologue",
                        [r"(?i)\bthe user is asking about the status\b"],
                    )
                ],
                tags=["real-transcript", "cron"],
                source={**src_junk, "session_id": "fixture-cron-campus-01"},
            ),
        ]
    )
    return rows


def gold_output(case: dict) -> dict:
    facts = []
    for expected in case["expected_facts"]:
        facts.append(
            {
                "content": expected["evidence_excerpt"],
                "asserted_by": "user",
                "evidence_role": "user",
                "evidence_excerpt": expected["evidence_excerpt"],
                "state": expected.get("state"),
            }
        )
    return {
        "schema_version": OUTPUT_SCHEMA,
        "case_id": case["case_id"],
        "decision": case["expected_decision"],
        "facts": facts,
        "metadata": {"provider": "role-gold", "seed_is_not_the_gate": True},
    }


def production_junk_output() -> dict:
    facts = [
        {
            "content": OUTLOOK,
            "asserted_by": "subprocess:ollama:qwen3-30b:durable-memory-v3",
            "evidence_role": "assistant",
            "evidence_excerpt": OUTLOOK,
            "state": None,
        },
        {
            "content": BRICK,
            "asserted_by": "subprocess:ollama:qwen3-30b:durable-memory-v3",
            "evidence_role": "assistant",
            "evidence_excerpt": BRICK,
            "state": None,
        },
        {
            "content": IMPLEMENT,
            "asserted_by": "subprocess:ollama:qwen3-30b:durable-memory-v3",
            "evidence_role": "assistant",
            "evidence_excerpt": IMPLEMENT,
            "state": None,
        },
        {
            "content": FLEET,
            "asserted_by": "subprocess:ollama:qwen3-30b:durable-memory-v3",
            "evidence_role": "tool",
            "evidence_excerpt": FLEET,
            "state": None,
        },
        {
            "content": EIGHT_BOTS,
            "asserted_by": "subprocess:ollama:qwen3-30b:durable-memory-v3",
            "evidence_role": "assistant",
            "evidence_excerpt": EIGHT_BOTS,
            "state": None,
        },
        {
            "content": SKILL_PRUNE,
            "asserted_by": "subprocess:ollama:qwen3-30b:durable-memory-v3",
            "evidence_role": "assistant",
            "evidence_excerpt": SKILL_PRUNE,
            "state": None,
        },
    ]
    return {
        "schema_version": OUTPUT_SCHEMA,
        "case_id": "prod-autoextract-junk-replay",
        "decision": "add",
        "facts": facts,
        "metadata": {
            "provider": "production-unreviewed-replay",
            "note": "compacted distinctive spans from the synthetic junk wave",
        },
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    bank = cases()
    header = {
        "schema_version": "enfold-transcript-gate-bank-v1",
        "tags": ["seed-7-is-not-the-gate", "real-transcript"],
        "case_count": len(bank),
    }
    _write_jsonl(HERE / "transcript_gate_cases.jsonl", [header, *bank])
    _write_jsonl(HERE / "transcript_gate_gold.jsonl", [gold_output(case) for case in bank])
    _write_jsonl(HERE / "transcript_gate_production_junk.jsonl", [production_junk_output()])
    print(f"wrote {len(bank)} cases")


if __name__ == "__main__":
    main()
