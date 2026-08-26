# Enfold benchmark protocol

This page is the contract for any published Enfold memory number. A score that
cannot be reproduced from a command in this tree is not a score.

## Standing rules

1. Never drop a category because it scores badly. LOCOMO category 5
   (adversarial, n=446) is always reported, separately and in the overall.
2. Never report an oracle run as a real LongMemEval-S run. Oracle and LME-M
   are labeled as such or omitted.
3. Always publish a full-context baseline that uses the same local reader as
   the memory run. If memory loses overall F1 and wins knowledge-update,
   abstention, or stale-leak, that is the result.
4. Hash-embedder Personal Arena numbers are plumbing checks, not a retrieval
   claim. Published retrieval numbers use the production stored embedder.
5. Never compare a local judge to vendor GPT-4o-J. Never use a prompt that
   says "be generous." Never fabricate a score. If a measurement is blocked,
   the report says blocked and why.

## Datasets

| Dataset | File | SHA-256 | Size |
|---|---|---|---|
| LOCOMO (`locomo10.json`) | acquired locally, not in git | `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4` | 2,805,274 bytes, 10 conversations, 1,986 questions |
| LongMemEval-S cleaned | acquired locally, not in git | `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` | 277,383,467 bytes, 500 questions |

Acquisition:

```bash
curl -L -o "$ENFOLD_BENCH_DATA/locomo10.json" \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
curl -L -o "$ENFOLD_BENCH_DATA/longmemeval_s_cleaned.json" \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
sha256sum "$ENFOLD_BENCH_DATA/longmemeval_s_cleaned.json"
# expected d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442
```

Default data directory: `~/.config/enfold/benchmark/data/`. Do not commit
either dataset.

## LOCOMO category table (IDs only; vendor names collide)

| id | n | pinned name | vendor aliases (do not use as the primary label) |
|---|---|---|---|
| 1 | 282 | category_1 | multi_hop or single_hop |
| 2 | 321 | temporal | temporal |
| 3 | 96 | category_3 | open_domain or multi_hop |
| 4 | 841 | category_4 | single_hop or open_domain |
| 5 | 446 | adversarial | adversarial (often dropped; we do not) |

Report `id=N n=count` in every table. Do not inherit MemOS or Memobase names.

## LongMemEval-S types

single-session-user, single-session-assistant, single-session-preference,
temporal-reasoning, multi-session, knowledge-update, abstention.
`question_date` is retrieval "now". Do not mix oracle or M into an S table.

## Models, k, token budget

- Embedder: production `embeddinggemma` stored identity (not feature-hash).
- Extractor: pinned local instruct recipe from `extraction_benchmark`.
- Reader: one pinned local instruct model, temperature 0. Identity in the report.
- Judge: a second pinned local model, or the reader at temperature 0, with the
  frozen strict prompt in `memory_eval/qa.py`. This is not GPT-4o-J.
- Retrieval k: 1, 3, 5, 10. Stale-leak k: 3.
- Packed-fact token budget: 256. Reader sees at most 10 facts.

Every report pins dataset digest, embedder identity, extractor identity,
reader identity, judge identity, k, token budget, and git SHA.

## Metrics

Retrieval only (no reader): Recall@{1,3,5,10}, MRR, nDCG@{1,3,5,10}.

QA (after retrieval): ACL token-overlap F1; local-judge accuracy; retrieval
Recall@k on gold evidence dialog or session IDs; tokens/query; search ms;
end-to-end ms.

Truth model: stale-fact leak rate, contradiction detection rate, abstention
correctness, injection resistance, temporal/as-of correctness, tokens/query,
latency. Missing sources are `blocked`, never zero-filled as a win.

## Real-transcript capture gate

Automatic session capture stays off until this gate is green. The bundled
seven-case synthetic extraction seed is format and safety smoke. It is not the ship gate.

Command:

```bash
PYTHONPATH=$PWD python -m memory_eval.transcript_gate \
  --cases memory_eval/fixtures/transcript_gate_cases.jsonl \
  --outputs <saved-role-structured-outputs.jsonl> \
  --require-ship
```

The bank is role-structured turns with hand-written expected user facts
(typed slot vs untyped durable) and forbidden assistant/tool spans. Replay
`prod-autoextract-junk-replay` on every run. That case is the 2026-08-25
production junk wave compacted to distinctive spans (Brick self-description,
"I'll now implement the concise style", `METIS_FLEET_OK`, Outlook
monologue, skill-prune banner). Require 0 assistant-authored facts and 0
tool-banner facts on that case.

Score four things separately:

1. Speaker attribution correctness. User testimony only. A user label on
   assistant or tool evidence is a misattribution.
2. Typed-slot completeness. A single-valued user claim must emit a complete
   `kind/subject/predicate/value` group or be dropped. Silent demotion to
   untyped prose is a fail.
3. Incidental durable-fact recall. Untyped user claims that should persist.
4. Forbidden assistant/tool claims. This one must be zero.

Thresholds, using the same split as the verifier gate (false `verified`
corrupts; false `needs_review` is friction):

| Metric | Ship bar | Why |
|---|---|---|
| forbidden assistant/tool rate | 0 | Same class as false-verify: writes fiction into memory |
| speaker misattribution rate | 0 | Assistant text labeled as user testimony is the 13/13 failure |
| typed-slot precision | 1.0 | A wrong employer/port/preference slot corrupts current state |
| silent demotion rate | 0 | Incomplete state looks like success and the moat does not fire |
| typed-slot completeness (recall) | >= 0.90 | A drop is friction; 10% miss is reviewable |
| incidental durable recall | >= 0.70 | Missed prose is friction, not poison |
| incidental durable precision | >= 0.90 | Do not dump every user sentence |

The 1.6% false-verify residual was accepted because it was one tense flip
among unsupported cases, and the default stayed off. It does not license a
non-zero assistant/tool write rate. Enabling the verifier on assistant
monologue would canonize fiction.

Do not report a green seed-7 run as permission to turn capture on.

## What is not a claim

HotpotQA/MuSiQue, DMR, MemoryBench marketing tables, any run that used
`longmemeval_oracle` or dropped LOCOMO category 5, any generous-judge J,
and any hash-embedder Personal Arena number presented as production quality.
