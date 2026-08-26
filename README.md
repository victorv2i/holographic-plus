<p align="center">
  <img src="assets/logo.png" alt="Enfold logo" width="280">
</p>

<h1 align="center">Enfold</h1>

<p align="center"><strong>When two agents disagree, you get a conflict you can settle, not a silent guess.</strong></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license"></a>
  <a href=".github/workflows/tests.yml"><img src="https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg" alt="Python 3.11, 3.12, and 3.13"></a>
  <a href="https://github.com/victorv2i/enfold/actions/workflows/tests.yml"><img src="https://github.com/victorv2i/enfold/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
</p>

Enfold stores durable facts on your machine, with evidence and history on
every claim. When two writers put different values in the same typed slot at
equal authority, it opens a conflict, returns a receipt instead of a value,
and waits for a human. There is no hosted service. Memory stays local.

## The moment

`python -m enfold.demo` (or `enfold demo`) walks that contract on a disposable
store. It never opens `~/.hermes/memory_store.db`. Conflict ids change each
run. This is a real run on this tree:

```
Enfold demo
Disposable store. Offline. Local-lexical retrieval.
This is not a bag of notes: current state, conflict, and human resolve are first-class.

1. Client A writes typed state
   client: demo-client-a
   claim:  env:staging.port = 3100
   write:  outcome=add fact_id=1 authority=0.5

2. Client B writes the same slot at equal authority
   client: demo-client-b
   claim:  env:staging.port = 3200
   write:  outcome=conflict fact_id=2 authority=0.5

3. Recall returns a conflict receipt, not 3100, not 3200
   current facts returned: none
   receipt: [conflict:f6a352c0-8976-457e-b57b-e953dddd5bcb slot:env:staging.port members:2 - do not treat either as current]
   conflict_id: f6a352c0-8976-457e-b57b-e953dddd5bcb
   members: [1, 2]
   This is the moment. A competitor would silently return one value, or both as if they were both current.

4. Human authority resolves once
   resolver: demo-human
   winner: fact_id=2 (staging port 3200)
   superseded: [1]
   reason: operator chose the current staging port

5. Recall now returns 3200. History keeps 3100.
   current: The staging port is 3200.
   history: fact_id=1 value=3100 superseded_by=2 content=The staging port is 3100.
   history: fact_id=2 value=3200 superseded_by=None content=The staging port is 3200.
   evidence names: demo-client-a, demo-client-b

6. Erase the current fact. Export cannot recover it.
   path: maintenance erase_fact then export_current
   erased fact_id=2; export omitted_erased=0
   export text does not contain 3200.

Diagnosis: wrote competing typed state, returned a conflict receipt, resolved as the human authority, then showed history, evidence, and erasure
Disposable store discarded.
Live /home/wonny/.hermes/memory_store.db was not used.
```

## Five-minute setup

Enfold keeps one private SQLite store on your machine. The default sends
nothing to a cloud service, starts no model, and grants only the local install
access to the private store. Client identity is a generated credential, not a
handshake claim.

Distribution is GitHub-only. There is no PyPI package. Pin a released
version tag, not a moving branch. The current release is `v0.8.1`.
Do not pin `v0.8.0`; that tag has no `enfold` or `enfold-mcp` scripts.

Primary path, the form MCP hosts already use:

```bash
uvx --from git+https://github.com/victorv2i/enfold@v0.8.1 enfold-mcp --self-test
```

`enfold-mcp` creates or reuses the XDG store, starts or attaches to the
per-user daemon, waits for readiness, then serves MCP on stdio. Add one
host snippet and ask the connected agent: "Remember that my test preference is dark mode, then show me the evidence for that memory."

Claude Code:

```bash
claude mcp add --transport stdio --scope user enfold -- uvx --from git+https://github.com/victorv2i/enfold@v0.8.1 enfold-mcp
```

Codex (`~/.codex/config.toml`):

```toml
[mcp_servers.enfold]
command = "uvx"
args = ["--from", "git+https://github.com/victorv2i/enfold@v0.8.1", "enfold-mcp"]
```

Cursor (merge into `~/.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "enfold": {
      "args": [
        "--from",
        "git+https://github.com/victorv2i/enfold@v0.8.1",
        "enfold-mcp"
      ],
      "command": "uvx",
      "type": "stdio"
    }
  }
}
```

Those snippets match the host shapes `enfold setup` writes. They omit
`--client-id` so first run can create the local grant. After a PATH install,
`enfold setup --client {codex|claude-code|cursor|hermes|generic}` writes the
host snippet under `~/.config/enfold/clients/` and runs a smoke test. The
token is printed once. Put it in the supervisor environment as
`ENFOLD_CLIENT_CREDENTIAL`, never in an agent prompt.

Keep the tools on PATH (alternative):

```bash
uv tool install git+https://github.com/victorv2i/enfold@v0.8.1
enfold setup --client cursor
enfold-mcp --self-test
enfold doctor
```

`enfold-mcp --self-test` reuses that grant and surface.
`enfold doctor` writes a fact, recalls it, and returns evidence on an isolated
local-lexical daemon.

pip from the same tag (alternative):

```bash
python3 -m pip install "git+https://github.com/victorv2i/enfold@v0.8.1"
```

Wheel from a GitHub Release (alternative). Download `enfold-*.whl` and
`SHA256SUMS`, check the digest, then:

```bash
python3 -m pip install ./enfold-0.8.1-py3-none-any.whl
```

See exactly what was written, then remove it:

```bash
enfold uninstall --dry-run
enfold uninstall --purge-data
```

Paths, credentials, socket length, and wheel checks:
[first local instance](docs/BOOTSTRAP.md).

## What it does not do yet

Automatic extraction ships disabled (`extraction.mode=disabled` from
`enfold setup` and `enfold init`). The real-transcript capture gate is red:
typed-slot completeness is 0.375 against a bar of 0.90, and precision and
silent-demotion fail the same instrument. Capture stays off until that gate
is green. Unrun work is marked UNRUN, not zero-filled. See
[BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md).

First run is `retrieval.mode=local-lexical` and needs no model. Stored
embeddings are an explicit upgrade:
[SERVER_DEPLOYMENT.md](docs/SERVER_DEPLOYMENT.md).

## Measured, with the scope attached

Author's 92-case private bank, against a real store, with production
embeddings. Not a public benchmark. Ranking metrics use the 87 answerable
cases. Measured on the tree released as `v0.8.1` (`3a66bc1`), documented in
[BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md):

| metric | value |
|---|---|
| Recall@1 | 0.7126 |
| Recall@10 | 0.9425 |
| MRR | 0.7888 |
| stale-fact leak | 0.0 |

LOCOMO and LongMemEval headline scores are deliberately UNRUN: datasets
parsed, no ingest, no reader QA. These private-bank numbers are not a
comparison to those benches. The 14-case public Arena is synthetic, not a
production-embedding claim. Full tables:
[BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md),
[BENCHMARK_PROTOCOL.md](docs/BENCHMARK_PROTOCOL.md).

## Tools

Default MCP profile `core`: `memory_recall`, `memory_remember`,
`memory_inspect`. Review and `legacy-v1`: [MCP proxy](docs/MCP_PROXY.md).

More: [operator detail](docs/OPERATOR.md),
[server deployment](docs/SERVER_DEPLOYMENT.md),
[releasing](docs/RELEASING.md),
`integrations/{claude-code,codex,cursor,hermes}/SKILL.md`.

## License

MIT, see [LICENSE](LICENSE).
