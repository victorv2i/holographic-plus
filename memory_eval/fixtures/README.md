# Public Arena fixtures

`public_arena.json` is a deliberately synthetic, privacy-safe case bank. Names,
projects, facts, and events are fictional. It is safe to publish raw queries and
retrieved fixture text in benchmark reports.

Its `memory_state` block explicitly classifies current and superseded facts and
declares durable conflict groups. The core-FTS Arena provider materializes that
state into a temporary migrated Enfold database so stale exclusion is a storage
and retrieval invariant, not a score-time fixture filter.

The fixture targets retrieval shapes that exact-fact smoke tests miss:
paraphrase, current-state updates, stale-fact exclusion, contradictions, changed
preferences, abstention, and multi-fact retrieval. It is a foundation for public
regression testing, not a claim of broad real-world benchmark coverage.

The optional `offline-hybrid-ci` provider exercises the standalone FTS,
Jaccard, dense-blending, authorization, and current-truth path without network
access. Its dense component is deterministic feature hashing, **not a semantic
model and not the production embedding stack**. It is meant to catch plumbing
and filtering regressions; model-quality claims require a separately identified
local production embedder and a disclosed benchmark run.

`locomo_smoke.json` and `longmemeval_smoke.json` are tiny fictional fixtures
for adapter tests. They are not LOCOMO or LongMemEval-S and must not be
reported as those benchmarks. Acquire the real files outside git using the
steps in `docs/BENCHMARK_PROTOCOL.md`.

`transcript_gate_cases.jsonl` is the capture ship-gate fixture. Cases
are synthetic and production-shaped: invented people, courses, and
projects with the same turn mix and adversarial traps as a live
capture failure. The seven-case `extraction_arena_seed.jsonl` is not
that gate. Rebuild the committed bank with
`build_transcript_gate_bank.py`. Do not paste private transcripts,
live session ids, live fact ids, or private-arena cases into git.
Do not re-mine a live store into this repository.
