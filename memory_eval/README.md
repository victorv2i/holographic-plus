# MemoryArena fixtures

`personal_arena.py` is the private, offline retrieval benchmark. Run it with:

```bash
# Published retrieval claim (fails closed without stored identities):
python -m memory_eval.personal_arena \
  --cases ~/.config/enfold/private-arena/cases-v0.jsonl \
  --db ~/.hermes/memory_store.db --seed 0 \
  --embedder production --embedder-config ~/.config/enfold/benchmark/embedder.json

# Offline CI plumbing (not a retrieval claim):
python -m memory_eval.personal_arena \
  --cases ~/.config/enfold/private-arena/cases-v0.jsonl \
  --db ~/.hermes/memory_store.db --seed 0 \
  --embedder hash
```

It opens the live database read-only and makes a SQLite backup in a temporary
directory before every run. The CLI default is `--embedder production`, which
uses the daemon's stored `embeddinggemma` identity and fails closed when that
identity is missing. Pass `--embedder hash` for the offline CI plumbing path
(deterministic feature-hash). Hash numbers are not a retrieval claim. Neither
path measures MCP transport.

Each private JSONL row has `id`, `query`, and `category`; it has either
`expected_fact_ids` and/or `expected_content_regexes`, or neither for an
abstention case. `forbidden_content_regexes` marks text that must not appear in
the top three and contributes to stale-leak rate. `asof` is a human-auditable
temporal annotation; current-truth filtering comes from the snapshot schema.
Expected ids/regexes are alternatives, so a case passes recall when any one
matches. The top result is considered a confident answer at score >= 0.35
(configurable with `--abstention-min-score`).

## Offline extraction Arena

Score already-saved, provider-neutral extraction outputs with:

```bash
python -m memory_eval.extraction_arena \
  --cases memory_eval/fixtures/extraction_arena_seed.jsonl \
  --outputs memory_eval/fixtures/extraction_arena_seed_outputs.jsonl \
  --require-perfect
```

This command makes no provider, model, network, service, or database calls.
The bundled seven-case synthetic seed is format and safety smoke coverage; it
is not the capture ship gate and not the full 190-case corpus used for
extraction-model selection. Capture stays off until the real-transcript
gate is green:

```bash
python -m memory_eval.transcript_gate \
  --cases memory_eval/fixtures/transcript_gate_cases.jsonl \
  --outputs memory_eval/fixtures/transcript_gate_gold.jsonl \
  --require-ship
```

Replay the known production failure with
`memory_eval/fixtures/transcript_gate_production_junk.jsonl`. That run must
not ship. Regenerate compacted fixtures with
`memory_eval/fixtures/build_transcript_gate_bank.py`. Do not commit full
private monologues.

After offline content/evidence scoring, replay the same saved proposals through
Enfold's real enqueue policy, extraction processor, state transitions, and
write log using one disposable migrated database per case:

```bash
python -m memory_eval.extraction_runtime_arena \
  --cases memory_eval/fixtures/extraction_arena_seed.jsonl \
  --outputs memory_eval/fixtures/extraction_arena_seed_outputs.jsonl \
  --require-perfect
```

Runtime replay never opens the configured or live Enfold database, makes no
provider/model/network call, and ignores each candidate's self-reported
`decision`. It derives `reject` from enqueue policy, `abstain` from a completed
zero-write job, and write decisions from authoritative `memory_write_log`
outcomes.

Run an actual candidate through the same two Arenas with the reproducible
benchmark runner. It writes a proposal-only artifact separately from the
scored report, so no model-generated lifecycle label can influence the score:

```bash
python -m memory_eval.extraction_benchmark \
  --adapter-config ~/.config/enfold/benchmark/qwen35-27b.json \
  --cases memory_eval/fixtures/extraction_arena_seed.jsonl \
  --proposals ~/.config/enfold/benchmark/results/qwen35-27b-proposals.json \
  --report ~/.config/enfold/benchmark/results/qwen35-27b-report.json \
  --repetitions 3 --timing-class warm --require-perfect
```

Validate a recipe and corpus without launching the adapter with `--dry-run`.
The adapter config is explicit and uses the same bounded subprocess boundary
as supervised extraction:

```json
{
  "schema_version": "enfold-extraction-benchmark-adapter-v1",
  "host": {
    "type": "subprocess",
    "argv": [
      "/absolute/path/to/python",
      "-m", "enfold.ollama_extractor_child",
      "--model", "qwen3.5:27b",
      "--model-identity", "qwen3.5:27b-pinned",
      "--prompt-identity", "durable-memory-v2"
    ],
    "model_identity": "qwen3.5:27b-pinned",
    "prompt_identity": "durable-memory-v2",
    "environment": {}
  },
  "recipe": {
    "model_artifact_digest": "sha256:<64 lowercase hex characters>",
    "runtime": {"name": "ollama", "version": "<pinned version>"},
    "decoder": {
      "temperature": 0,
      "thinking": false,
      "context_tokens": "<pinned value>",
      "output_tokens": "<pinned value>"
    }
  }
}
```

Environment values may be supplied to the child, but reports record only
their names. Public recipe metadata rejects credential-bearing fields. Enfold
policy preflight runs before inference, so rejected cases are never sent to a
local or cloud adapter. The bundled seven cases remain smoke coverage; use the
full frozen corpus and documented repetitions for ranked model selection.
Proposal artifacts contain model-generated content and exact source evidence;
the runner writes artifacts atomically with owner-only permissions, but they
must still be treated as private data and kept out of source control.

The repository contains only `fixtures/personal_arena_sample.jsonl`, a
synthetic format sample. It intentionally does not match any real database.
Keep real cases, results, and source facts under
`~/.config/enfold/private-arena/`; do not add them to git. Add a case only
after checking its expected fact against a read-only snapshot, choose a stable
id, and record a precise stale regex whenever the question has a known prior
answer.

## Published scorecards

The standing rules live in `docs/BENCHMARK_PROTOCOL.md`. Retrieval quality and
reader QA are never one number.

```bash
# Retrieval only (Recall@k, MRR, nDCG). No reader.
PYTHONPATH=$PWD python -c "from memory_eval.retrieval_scorecard import retrieval_scorecard"

# Truth model (stale leak, contradiction, abstain, injection, as-of, tokens, latency)
PYTHONPATH=$PWD python -c "from memory_eval.truth_scorecard import truth_scorecard"
```

A metric that cannot be computed is reported as `blocked`, never as a made-up
zero.

## LOCOMO and LongMemEval-S

Adapters parse public datasets offline and refuse to invent scores. Tiny
fictional smoke fixtures live under `fixtures/`. The real datasets stay out of
git.

```bash
# After downloading locomo10.json (see docs/BENCHMARK_PROTOCOL.md):
PYTHONPATH=$PWD python -m memory_eval.locomo_adapter \
  --data ~/.config/enfold/benchmark/data/locomo10.json --require-published-hash

PYTHONPATH=$PWD python -m memory_eval.longmemeval_adapter \
  --data ~/.config/enfold/benchmark/data/longmemeval_s_cleaned.json --split S
```

Both commands print a parse report with `scores: null` until a local reader
run is attached. Category 5 is never dropped. Oracle files cannot be loaded
as LongMemEval-S.
