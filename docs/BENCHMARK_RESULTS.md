# Enfold benchmark results

Numbers a stranger can reproduce from this tree are labeled **HEAD**.
Personal-arena figures labeled **HEAD** were measured on this tree
(`38ea39ff64b669d88b50b089c8eeb63f1c9cb39d`) against a private store
that is not in git, using production stored `embeddinggemma` identities.
The earlier `8bf26d9` scorecard (Recall@1 51.7%) is kept below and
marked **historical**. Later ranking commits (`e980503`, `a27bf34`)
changed the retriever; those commits were not separately re-published.
A metric that was not computed from a command in this repository is
marked **UNRUN** or **historical**. No vendor judge number was produced.
Hash-embedder Personal Arena numbers are not a retrieval claim.

## Run identity

| Field | Value |
|---|---|
| git SHA (personal-arena HEAD) | `38ea39ff64b669d88b50b089c8eeb63f1c9cb39d` |
| git SHA (personal-arena historical) | `8bf26d90f4ce6af2acd0b866bbf45b71bf2661d7` |
| date (HEAD personal-arena) | 2026-08-25, this tree |
| date (historical personal-arena) | 2026-08-24 (America/New_York), 2026-08-25T01:46Z to 2026-08-25T01:51Z |
| embedder | production stored `ollama:embeddinggemma:latest` |
| query identity | `ollama:embeddinggemma:latest:query:sha256-9fec2002477dbe163e7c83f842da13c793e96f34fcb303590502b1927304afcf:sha256:85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1` |
| document identity | `ollama:embeddinggemma:latest:document:sha256-9fec2002477dbe163e7c83f842da13c793e96f34fcb303590502b1927304afcf:sha256:85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1` |
| embedding version | `sha256:85462619ee721b466c5927d109d4cb765861907d5417b9109caebc4e614679f1` |
| dimensions | 768 |
| `embedder_production_ready` | true |
| vector backend this run | `brute` (sqlite-vec generation ledger absent; see warning below) |
| `vector_fallback_active` | false |
| reader / judge (QA smoke only) | `ollama:qwen2.5:3b-instruct:temperature-0` |
| extractor (this run) | UNRUN as a ranked model bake-off; seed outputs are pinned fixtures |
| retrieval k | 1, 3, 5, 10 |
| stale-leak k | 3 |
| personal cases | 92 (private JSONL and private store, not in git) |
| public arena | 14 synthetic cases / 22 facts (`memory_eval/fixtures/public_arena.json`) |
| extraction seed | 7 cases (`memory_eval/fixtures/extraction_arena_seed.jsonl`) |
| LOCOMO | `locomo10.json` sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`, 2,805,274 bytes, 10 conversations, 1,986 questions, 272 sessions |
| LongMemEval-S | `longmemeval_s_cleaned.json` sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`, 277,383,467 bytes, 500 questions, 23,867 sessions |

sqlite-vec warning printed on both production personal-arena runs:

```
sqlite-vec health warning: vec0 generation ledger is absent; rebuild the index; falling back to brute
```

Stored `embeddinggemma` identities were present and production-ready. Ranking
used brute stored-vector scoring, not ANN sqlite-vec. That is a measurement
condition, not a hidden fallback to the hash embedder.

## 1. Personal Arena retrieval scorecard (production embeddings)

**HEAD.** Operator-measured on this tree
(`38ea39ff64b669d88b50b089c8eeb63f1c9cb39d`) against the owner's 92-case
private bank with production stored `embeddinggemma` identities. Reader
unused. Private corpus: do not run against a live store from this public
tree. PYTHONPATH must be the Enfold checkout.

Command:

```bash
python -m memory_eval.personal_arena \
  --cases /absolute/path/to/private-arena/cases-v0.jsonl \
  --db /absolute/path/to/private-snapshot.db --seed 0 \
  --embedder production --embedder-config /absolute/path/to/embedder.json
```

Protocol retrieval scorecard (search limit 10; Recall@k, MRR, nDCG). Gold
id for MRR/nDCG is the first expected fact id that appears in the ranked
list, else the first expected id. Abstention cases are excluded from
ranking (87 answerable).

| metric | value | commit |
|---|---|---|
| cases (answerable) | 87 | `38ea39f` |
| Recall@1 | 0.7126 | `38ea39f` |
| Recall@3 | 0.8391 | `38ea39f` |
| Recall@5 | 0.9310 | `38ea39f` |
| Recall@10 | 0.9425 | `38ea39f` |
| MRR | 0.7888 | `38ea39f` |
| nDCG@10 | 0.8266 | `38ea39f` |
| stale-fact leak | 0.0 | `38ea39f` |
| abstention | 0.8 | `38ea39f` |
| nDCG@1 / nDCG@3 / nDCG@5 | UNRUN |  |
| official CLI category breakdown | UNRUN |  |

### Historical official CLI at `8bf26d9` (search limit 3)

**historical.** Measured at `8bf26d90f4ce6af2acd0b866bbf45b71bf2661d7`.
Later ranking commits changed the retriever. Do not treat these as HEAD.

```
sqlite-vec health warning: vec0 generation ledger is absent; rebuild the index; falling back to brute
PersonalArena scorecard
- Cases: 92 (87 retrieval, 5 abstention)
- Recall@1: 51.7%
- Recall@3: 74.7%
- Stale-leak rate@3: 0.0% (6 protected cases)
- Abstention correctness: 80.0%
- Per category:
  - feedback: n=1, R@1=100.0%, R@3=100.0%, stale=0.0%, abstain=0.0%
  - project: n=62, R@1=39.7%, R@3=67.2%, stale=0.0%, abstain=100.0%
  - reference: n=25, R@1=84.0%, R@3=100.0%, stale=0.0%, abstain=0.0%
  - tool: n=3, R@1=0.0%, R@3=0.0%, stale=0.0%, abstain=0.0%
  - user_pref: n=1, R@1=0.0%, R@3=0.0%, stale=0.0%, abstain=0.0%
```

### Historical protocol scorecard at `8bf26d9` (search limit 10)

Command: out-of-repo script `measure_scorecards.py` (not in this repository). **historical**.

| metric | value |
|---|---|
| cases (answerable) | 87 |
| Recall@1 | 0.5172413793103449 |
| Recall@3 | 0.7471264367816092 |
| Recall@5 | 0.8390804597701149 |
| Recall@10 | 0.9310344827586207 |
| MRR | 0.6560071154898743 |
| nDCG@1 | 0.5172413793103449 |
| nDCG@3 | 0.6532530637931081 |
| nDCG@5 | 0.6918480633667798 |
| nDCG@10 | 0.7228198555238884 |
| reader_used | false |

Recall@1 and Recall@3 matched that historical official CLI exactly.

### Full-context baseline for this memory number

Personal Arena cases have expected fact ids, not gold answer strings. A
same-reader QA baseline on the private bank is **UNRUN** (no gold answers;
private fact text is not a public scorecard).

Retrieval ceiling: `validate_personal_cases` passed on this snapshot, so every
answerable gold is an active private fact in the asked category. Stuffing the
whole eligible set would make Recall@k = 1.0 by construction. That is a
ceiling, not a scored run.

Same-reader QA that can be scored lives in section 4 (public arena rubrics)
and section 5 (fixture-scale LOCOMO / LongMemEval smoke). Headline LOCOMO and
LongMemEval-S memory vs full-context QA is **UNRUN**.

## 2. Truth-model scorecard

### Personal Arena, production embeddings (real private snapshot)

**HEAD** truth-model rates attributed to `38ea39f` on the same 92-case
private bank: stale-fact leak 0.0, abstention 0.8. Other truth-model
axes below are **historical** from the `8bf26d9` `measure_scorecards.py`
run.

| metric | status | value | n | notes |
|---|---|---|---|---|
| stale_fact_leak_rate | ok | 0.0 |  | HEAD `38ea39f`; protected-case count UNRUN on this tree |
| contradiction_detection_rate | blocked | null |  | no `case_type=contradiction` rows in the private bank |
| abstention_correctness | ok | 0.8 |  | HEAD `38ea39f`; confusion buckets UNRUN on this tree |
| injection_resistance | blocked | null |  | extraction score not attached to this retrieval run |
| temporal_asof_correctness | historical | 0.75 | 12 | `8bf26d9` only; gold hit and no stale leak@3 |
| tokens_per_query | historical | mean 460.42, p50 462, p95 903 | 92 | `8bf26d9` only |
| latency_ms | historical | p50 91.08, p95 664.81, mean 133.20 | 92 | `8bf26d9` only |

### Public Arena, current-state FTS provider (synthetic; not production embeddings)

**HEAD.** Re-ran at `38ea39ff64b669d88b50b089c8eeb63f1c9cb39d` with the
command below. Same JSON as `8bf26d9` and `a27bf34`.

Command:

```bash
python -m memory_eval.public_arena --provider core-fts-current
```

Raw output:

```
{"answerable_recall@1": 0.8333333333333334, "arena": "enfold-public-arena", "arena_version": "1.0", "cases": 14, "false_confident": 0, "provider": "EnfoldCoreFtsCurrentProvider", "set_f1": 1.0, "set_recall": 1.0, "stale_leaks@3": 0, "true_abstain": 2}
```

Protocol scorecards from `measure_scorecards.py` on the same provider
(**historical**; official CLI above was re-run at HEAD):

| metric | status | value | n |
|---|---|---|---|
| Recall@1 / @3 / @5 / @10 | ok | 0.8333 / 1.0 / 1.0 / 1.0 | 12 answerable |
| MRR | ok | 0.9166666666666666 | 12 |
| nDCG@1 / @3 | ok | 0.8333 / 0.9385 | 12 |
| stale_fact_leak_rate | ok | 0.0 | 6 |
| contradiction_detection_rate | ok | 1.0 | 2 |
| abstention_correctness | ok | 1.0 | 2 |
| temporal_asof_correctness | ok | 1.0 | 4 |
| injection_resistance | blocked | null | no extraction score on this run |
| tokens_per_query | ok | mean 10.07, p50 9, p95 23 | 14 |
| latency_ms | ok | p50 0.14, p95 0.37 | 14 |

`offline-hybrid-ci` (hash embedder, not a retrieval claim) produced the same
retrieval and truth rates as `core-fts-current` on this 14-case fixture.

Lexical fixture provider (no current-state filter) on the same cases:

```
{"answerable_recall@1": 0.8333333333333334, "arena": "enfold-public-arena", "arena_version": "1.0", "cases": 14, "false_confident": 1, "provider": "LexicalFixtureProvider", "set_f1": 0.9305555555555555, "set_recall": 1.0, "stale_leaks@3": 6, "true_abstain": 1}
```

Truth scorecard on lexical: stale-leak 1.0 (6/6), temporal 0.0 (0/4),
abstention 0.5 (1/2), contradiction 1.0 (2/2). This is the control that the
current-state provider is doing work. It is not a production embedding claim.

### Extraction seed: injection resistance

Commands:

```bash
python -m memory_eval.extraction_arena \
  --cases memory_eval/fixtures/extraction_arena_seed.jsonl \
  --outputs memory_eval/fixtures/extraction_arena_seed_outputs.jsonl \
  --require-perfect

python -m memory_eval.extraction_runtime_arena \
  --cases memory_eval/fixtures/extraction_arena_seed.jsonl \
  --outputs memory_eval/fixtures/extraction_arena_seed_outputs.jsonl \
  --require-perfect
```

**HEAD.** Re-ran at `38ea39ff64b669d88b50b089c8eeb63f1c9cb39d`.
Offline content/evidence scoring: 7/7 passed, decision accuracy 1.0, forbidden
leak rate 0.0. Runtime replay on disposable DBs: 7/7 passed, 0 live database
writes, 0 model calls.

Truth-model injection_resistance from `truth_scorecard(extraction_score=...)`:
**1.0 on 1 tagged `prompt-injection` case** (`abstain-prompt-injection`).
Scale is the bundled seed, not the off-repo 190-case corpus. That 190-case
corpus was **UNRUN**.

## 3. LOCOMO

### Headline bench: UNRUN

The dataset was obtained and parsed. No Enfold ingest, no HybridRetriever
search over extracted facts, and no reader QA score exists. `scores` is null
on purpose.

Acquisition (operator-local; dataset is not in git):

```bash
mkdir -p ~/.config/enfold/benchmark/data
curl -L -o ~/.config/enfold/benchmark/data/locomo10.json \
  https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
sha256sum ~/.config/enfold/benchmark/data/locomo10.json
# expected 79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4
```

Parse command and raw output:

```bash
python -m memory_eval.locomo_adapter \
  --data ~/.config/enfold/benchmark/data/locomo10.json --require-published-hash
```

```
{
  "status": "parsed",
  "scores": null,
  "dataset_sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
  "conversations": 10,
  "questions": 1986,
  "sessions": 272,
  "categories": [1, 2, 3, 4, 5],
  "reason": "dataset parsed; retrieval/QA scores require a local reader run"
}
```

Category counts from the same file (category 5 not dropped): id=1 n=282,
id=2 n=321, id=3 n=96, id=4 n=841, id=5 n=446. 444 questions have no `answer`
field (almost all category 5).

Owner command to produce the missing headline scores, once ingest exists:
walk 272 sessions through the extraction queue into a disposable DB, then
search + local reader + ACL F1 / local-J / evidence Recall@k, never dropping
category 5, and run the same reader on the full conversation transcript as
the full-context baseline.

### Fixture-scale smoke (not the headline)

```bash
python -m memory_eval.locomo_adapter \
  --data memory_eval/fixtures/locomo_smoke.json
```

```
{
  "status": "parsed",
  "scores": null,
  "dataset_sha256": "782a72363c7338dc22e43d91ec05cecfca255697f1a7f1221aa7d16891f29b8c",
  "conversations": 1,
  "questions": 3,
  "sessions": 2,
  "categories": [1, 2, 5],
  "reason": "dataset parsed; retrieval/QA scores require a local reader run"
}
```

Full-context only (same reader; memory ingest UNRUN) from an out-of-repo
script, **historical**:

| id | gold | full-context answer | token F1 | local-J |
|---|---|---|---|---|
| cat 1 | Local-first databases | I don't know. | 0.0 | INCORRECT |
| cat 2 | 3 May 2023 | Riv joined the studio on 3 May 2023. | 0.6 | INCORRECT |
| cat 5 | self-care is important (adversarial_answer) | I don't know. ... | 0.0 | INCORRECT |

Mean full-context token F1 on the 3-question fixture: 0.2. Local-J correct
rate: 0.0. Reader/judge: `ollama:qwen2.5:3b-instruct:temperature-0`. This is
not a LOCOMO score.

## 4. LongMemEval-S

### Headline bench: UNRUN

The cleaned S split was downloaded and parsed. No ingest, no `question_date`
retrieval, no reader QA. `scores` is null.

Acquisition that was run:

```bash
mkdir -p ~/.config/enfold/benchmark/data
curl -L -o ~/.config/enfold/benchmark/data/longmemeval_s_cleaned.json \
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
sha256sum ~/.config/enfold/benchmark/data/longmemeval_s_cleaned.json
# got d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442
# 277383467 bytes
```

Parse command and raw output:

```bash
python -m memory_eval.longmemeval_adapter \
  --data ~/.config/enfold/benchmark/data/longmemeval_s_cleaned.json --split S
```

```
{
  "status": "parsed",
  "scores": null,
  "split": "S",
  "comparable_to_paper": true,
  "questions": 500,
  "sessions": 23867,
  "by_type": {
    "knowledge-update": 78,
    "multi-session": 133,
    "single-session-assistant": 56,
    "single-session-preference": 30,
    "single-session-user": 70,
    "temporal-reasoning": 133
  },
  "reason": "dataset parsed; retrieval/QA scores require a local reader run"
}
```

`question_type` in this cleaned file has no `abstention` label. 30
`question_id` values contain `_abs`. Oracle was not loaded as S.

Owner command for the missing headline scores: ingest each of the 500
haystacks (23,867 sessions) into disposable DBs with `haystack_dates` as
`observed_at`, retrieve at `question_date`, score ACL F1 / local-J /
evidence-session Recall@k, and run the same reader on the full haystack as
the full-context baseline. Do not report oracle or M as S.

### Fixture-scale smoke (not the headline)

```bash
python -m memory_eval.longmemeval_adapter \
  --data memory_eval/fixtures/longmemeval_smoke.json --split S
```

```
{
  "status": "parsed",
  "scores": null,
  "split": "S",
  "comparable_to_paper": true,
  "questions": 2,
  "sessions": 3,
  "by_type": {"abstention": 1, "knowledge-update": 1},
  "reason": "dataset parsed; retrieval/QA scores require a local reader run"
}
```

Full-context only (memory ingest UNRUN), same reader:

| id | type | gold | full-context answer | token F1 | local-J |
|---|---|---|---|---|---|
| lme-ku-1 | knowledge-update | Contoso Research | Contoso Research | 1.0 | CORRECT |
| lme-abs-1 | abstention | I don't know. | I don't know. There is no information... | 0.4 | INCORRECT |

Mean full-context token F1: 0.7. Local-J correct rate: 0.5. This is not a
LongMemEval-S score.

## 5. Same-reader full-context on Public Arena (synthetic)

Command: out-of-repo script `measure_full_context.py` (not in this repository). **historical**.

Reader and judge: `ollama:qwen2.5:3b-instruct:temperature-0`. Judge prompt is
`memory_eval/qa.py:LOCAL_JUDGE_PROMPT` (strict; does not say "be generous").
Memory pack: `core-fts-current` top-10 fact strings. Full context: all 16
current fixture facts. Gold is `answer_rubric.must_mention` joined, or
`I don't know.` for abstention. Two paraphrase cases have no rubric and were
excluded from the means below (empty gold would force F1 0).

| split | n | memory token F1 | full-context token F1 | memory local-J | full-context local-J |
|---|---|---|---|---|---|
| public arena, scored cases | 12 | 0.5992 | 0.4677 | 0.3333 | 0.1667 |

This gold is keyword rubrics, not human short answers. Local-J often marked
factually right longer answers INCORRECT because they were not identical to
the keyword string. Treat these as a same-reader smoke, not a market QA
claim. Memory ingest vs full-context on LOCOMO / LongMemEval-S remains
**UNRUN**.

## 6. Real-transcript capture gate

This is the ship instrument for automatic capture. The seven-case
extraction seed in section 2 is not this gate.

Bank: `memory_eval/fixtures/transcript_gate_cases.jsonl` (50
role-structured cases). Rebuild with
`memory_eval/fixtures/build_transcript_gate_bank.py`. Do not re-mine a
live store into git.

Command and role-gold control (the measuring stick, not a live extractor
claim):

```bash
python -m memory_eval.transcript_gate \
  --cases memory_eval/fixtures/transcript_gate_cases.jsonl \
  --outputs memory_eval/fixtures/transcript_gate_gold.jsonl \
  --require-ship
```

**HEAD.** Re-ran at `38ea39ff64b669d88b50b089c8eeb63f1c9cb39d`.
Role-gold: ship=true, case_pass_rate=1.0 on 50 cases, forbidden leaks=0,
speaker accuracy=1.0 (29/29), typed recall/precision=1.0 on 16 slots,
incidental recall/precision=1.0 on 13 durables. This only proves the
instrument can go green on correctly attributed user facts.

Production junk replay (the known failure, compacted from the live
unreviewed rows):

```bash
python -m memory_eval.transcript_gate \
  --cases memory_eval/fixtures/transcript_gate_cases.jsonl \
  --outputs memory_eval/fixtures/transcript_gate_production_junk.jsonl \
  --require-ship
```

Result: ship=false. Re-ran at `38ea39f`: forbidden leaks=8 (rate 0.02),
typed completeness=0.0, incidental recall=0.0. `prod-autoextract-junk-replay`
records assistant-authored facts and a `METIS_FLEET_OK` tool-banner fact.
Capture must stay off.

Live extractor against this bank (**HEAD**, this tree `38ea39f`):
operator-measured on disposable SQLite files. The operator live store was
not opened. Automatic capture stays off.

| axis | live number | ship bar | result | commit |
|---|---|---|---|---|
| forbidden assistant/tool rate | 0.0 | 0 | pass | `38ea39f` |
| speaker misattribution rate | 0.0 | 0 | pass | `38ea39f` |
| typed-slot precision | 0.3333 | 1.0 | fail | `38ea39f` |
| silent demotion rate | 0.625 | 0 | fail | `38ea39f` |
| typed-slot completeness | 0.375 | >= 0.90 | fail | `38ea39f` |
| incidental durable recall | UNRUN | >= 0.70 |  |  |
| incidental durable precision | UNRUN | >= 0.90 |  |  |

Live extractor against this bank (`6808f64`, 2026-08-25): **historical**,
ship=false. Ran all 50 cases through `qwen3:30b` / `durable-memory-v3`
on the role-structured path with `qwen3.8:27b` evidence verification.

| axis | live number | ship bar | result |
|---|---|---|---|
| forbidden assistant/tool rate | 0.0 (0 leaks / 50) | 0 | pass |
| speaker misattribution rate | 0.0 (0 / 28) | 0 | pass |
| typed-slot precision | 0.087 (2 / 23) | 1.0 | fail |
| silent demotion rate | 0.8125 (13 / 16) | 0 | fail |
| typed-slot completeness | 0.125 (2 / 16) | >= 0.90 | fail |
| incidental durable recall | 0.923 (12 / 13) | >= 0.70 | pass |
| incidental durable precision | 2.4 (12 / 5) | >= 0.90 | pass (scorer artifact; see note) |

**historical** (`6808f64`) case pass rate 0.70 (35 / 50). Production
replay `prod-autoextract-junk-replay` wrote only the two user facts
and 0 assistant/tool facts. Role-structured capture stopped the
2026-08-25 junk class. It did not produce gold-stable typed slots.
Automatic capture stays off.

Incidental precision 2.4 is the committed scorer: `incidental_matched`
can count a content hit that also carries typed state, while
`incidental_predicted` counts only facts without a complete state
group. The check is `>= 0.90` and therefore passes. It is not a
precision of 1.0.

**historical** (`6808f64`) wall time: about 221s of per-case time
(mean 4.43s; 27 model-calling cases mean 7.93s, min 2.90s, max 37.91s
cold). A from-scratch full bank is a few minutes once `qwen3:30b` is
loaded. The per-run summary file was operator-local and is not in this
repository.

## 7. Explicitly UNRUN

- Headline LOCOMO token F1, local-J, evidence Recall@k, tokens/query, end-to-end ms
- Headline LongMemEval-S accuracy by type, including knowledge-update / temporal / abstention
- LongMemEval-M, oracle-as-S (correctly refused), BEAM, ConvoMem, LongMemEval-V2
- Personal Arena same-reader QA (no gold answers)
- Extraction 190-case private corpus and any Ollama extraction bake-off
- Production-embedder Public Arena (fixture has no stored `embeddinggemma` vectors)
- sqlite-vec ANN path (ledger absent; this run used brute stored vectors)
- Any comparison of the local 3B judge to vendor GPT-4o-J
