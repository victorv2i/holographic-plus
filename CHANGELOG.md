# Changelog

## Unreleased

## 0.8.1

First installable GitHub Release from this tree. The older `v0.8.0` tag
(2026-07-12) has no `enfold` or `enfold-mcp` scripts; do not pin it.

- Added bitemporal valid time so event time stays distinct from recorded time.
- Shipped an installable wheel with `enfold setup` and per-client credentials.
- Collapsed the default MCP core profile to three tools: `memory_recall`,
  `memory_remember`, and `memory_inspect`. Profile `legacy-v1` keeps the
  previous thirteen names for one transition release.
- Added `enfold doctor` for an isolated local-lexical write/recall/evidence check.
- Added `enfold demo` for the conflict-receipt walk on a disposable store.
- Fixed ranking so typed current state stays visible and recovered Recall@1
  cases that later retriever commits had lost.
- Made entity-graph multi-hop expansion opt-in on `HybridRetriever`.
- Added role-structured extraction so speaker and tool banners cannot be
  stored as user facts.
- Added the real-transcript capture ship gate. Automatic capture stays off
  until the gate is green.
- Conflict drain now requires `can_resolve_conflicts`.
- Install and release documentation describe GitHub-only distribution.
  Enfold is not published to PyPI. Pin a released version tag.

## 0.8.0

- Added optional `sqlite-vec==0.1.9` indexing with validated automatic
  brute-force fallback and an explicit `rebuild-vector-index` operation.
- Added standalone hybrid ranking priors, confidence abstention, bounded MMR
  context packing, and typed state-slot handling.
- Added near-duplicate merge handling for untyped writes, preserving one fact
  and superseding the other after recording the incoming observation.
- Added MCP change, timeline, entity-list, and entity-detail tools.
- Added verified backup secondary destinations, optional age encryption, restore
  rehearsal reports, and filtered browse snapshots for Datasette.
- Added public synthetic and private-corpus personal Arena evaluation surfaces.
