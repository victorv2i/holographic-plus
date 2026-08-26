# Operator detail

This page holds ranking defaults, write rules, backup commands, and a
source checkout. First install stays in [BOOTSTRAP.md](BOOTSTRAP.md).
Production config stays in [SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md).
MCP tool profiles stay in [MCP_PROXY.md](MCP_PROXY.md).

## Retrieval defaults

First run writes `retrieval.mode=local-lexical`: FTS, Jaccard, and ranking
priors, with dense scoring off. That is the path `enfold doctor` exercises.
It needs no model and no `sqlite-vec`. `ci` mode is a non-production plumbing
test and requires `allow_nonproduction: true`. Stored embeddings are an
explicit upgrade (`retrieval.mode=stored`).

Two-hop entity expansion is off unless a caller constructs
`HybridRetriever(..., entity_expansion=True)`. Default search does not expand.

When dense scoring is configured, ranking is reciprocal-rank fusion of a
lexical list and a dense list, then priors. The `RankingConfig` defaults in
`enfold/hybrid_retrieval.py` are:

```text
score = 0.76 × RRF(lexical, dense)
      + 0.05 × trust + 0.02 × kind + 0.03 × recency
      + 0.08 × review + 0.06 × named_subject
```

Lexical list order uses `0.35 × FTS + 0.25 × Jaccard` (renormalized when
dense is disabled). FTS itself blends reciprocal BM25 rank (`0.25`) with
distinct query-token coverage (`0.75`). The kind prior is state `1.00`,
insight `0.75`, untyped `0.50`, and event `0.25`. Recency uses an exponential
365-day half-life. `score_floor: 0.12` rejects weak candidates and
`ambiguity_margin: 0.005` abstains when the top two results are too close.
A named anchor in the query also requires that anchor in a candidate.

`memory_context` produces a bounded, cited Markdown block. It estimates tokens
as Unicode characters divided by four, truncates individual facts to fit, omits
unsafe or duplicate state slots, and can use maximal marginal relevance (MMR)
to choose diverse context. Prompt-ready rendering treats memory as reference
claims, not control instructions. Rows with
`correction_status='unreviewed'` and instruction-shaped content are omitted
from Markdown and returned only as redacted receipts. Writes with a null
correction status may still render and are labeled untrusted. Reviewed facts
(`human_confirmed` / `human_corrected`) are packed first. Default
`memory_search` also excludes `correction_status='unreviewed'`; pass
`include_unreviewed=true` when a review surface must list those rows on
purpose.

## Writes and typed state

Writes are idempotent and carry provenance. Exact duplicates reuse the current
fact. For untyped near duplicates, the service finds an FTS-bounded semantic
candidate, writes the incoming observation, and merges the two by retaining one
surviving fact while superseding the other. This is not a claim that every
similar sentence is rejected.

Automatic extraction is off. `enfold setup` and `enfold init` write
`extraction.mode=disabled`. Typed extraction, when an operator later enables
it, accepts `state`, `preference`, `commitment`, or `event` labels at
confidence `0.8` or higher. Only `state` is routed into a structured
`(scope, subject_key, predicate_key)` state slot. A changed state can
supersede the prior slot value; competing current state is recorded as a
conflict. The other labels remain attributed facts with their extracted type in
metadata. The real-transcript capture gate is still red, so automatic capture
must stay disabled. See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).

## sqlite-vec

`sqlite-vec` is an optional acceleration extra, pinned to `0.1.9`, and is not
required to install or recall facts. The canonical `fact_embeddings` table
remains the source of truth. If the extension, its index, or its metadata is
unavailable, retrieval automatically uses brute-force cosine scoring instead.
Rebuild an index only in a maintenance window:

```bash
python -m enfold.ops rebuild-vector-index /absolute/path/to/memory.db \
  --embedding-identity 'provider:model:document:policy:version' \
  --dimensions 768
```

## Backup, rehearsal, and browse

Use SQLite's backup API rather than copying a live database file:

```bash
python -m enfold.ops backup SOURCE.sqlite BACKUP.sqlite \
  --secondary-directory /mounted/offsite \
  --age-recipient-path /secure/recipients.txt
```

The primary backup is verified. A secondary destination is best effort: failure
does not invalidate a completed primary backup. When `age` is available,
Enfold uses `age -R` and places only an encrypted `.age` artifact in the
secondary destination. Without `age`, it warns and makes a private plain copy.
Keep identities and recipient files outside the repository and server config.

Rehearse the newest `*.sqlite` backup without changing the live database:

```bash
python -m enfold.backup_rehearsal LIVE_DB BACKUP_DIR STATE_DIR \
  --fact-count-tolerance 100
```

The rehearsal restores to a temporary directory, runs `quick_check`, compares
fact counts, writes a dated JSON pass/fail report, and exits nonzero on failure.

For a read-only local browser, configure `browse_scopes`, then create a filtered
snapshot. It includes only current, settled, normal-sensitivity facts in those
scopes and a small `metadata.json` for Datasette. Serve the resulting immutable
SQLite file with a local Datasette installation:

```bash
python -m enfold.ops browse-snapshot /absolute/path/to/server.json
```

Regenerate the snapshot when browser-visible facts change. It does not serve or
modify the live database.

## Production daemon

For an existing or production deployment, create a user-owned JSON configuration
with explicit database, socket, retrieval, and client grants. See
[SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md).

```bash
enfold-server --config /absolute/path/to/server.json check
enfold-server --config /absolute/path/to/server.json run
```

The optional MCP bridge connects to an already-running daemon and never opens
SQLite itself:

```bash
enfold-mcp-proxy --socket-path /absolute/private/enfold.sock \
  --client-id workstation-1 --surface local --agent-id worker-1 \
  --session-id session-1
```

If the default data-dir socket would exceed the AF_UNIX 107-byte limit, setup
keeps the store where it is and binds a shorter owner-only socket under
`XDG_RUNTIME_DIR` or `/tmp`. Pass `--socket-path` to choose the socket
explicitly. The store is never moved silently.

## Contributors and source checkout

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
```

`enfold init` still creates a new owner-only instance without starting a
daemon. Prefer `enfold setup` or `enfold-mcp` for a complete first run.
See [BOOTSTRAP.md](BOOTSTRAP.md) for paths, credentials, uninstall, and
release-wheel verification. See [RELEASING.md](RELEASING.md) for tag,
changelog, schema, and MCP compatibility rules.

## Evaluation

The public Arena is a synthetic regression harness. It does not read a live
store or measure production embedding quality:

```bash
python -m memory_eval.public_arena --provider core-fts-current --limit 5
```

The personal Arena harness is public, but its corpus remains private by design.
Keep real cases and reports outside the repository. Full scorecards, UNRUN
rows, and the capture gate live in [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md).
The measurement contract is [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md).
