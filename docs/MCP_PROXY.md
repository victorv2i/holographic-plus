# Enfold MCP proxy

`enfold-mcp-proxy` / `enfold-mcp-launch` is the public MCP surface. The
server name is `enfold-memory`. It is a thin adapter for an already-running
standalone Enfold v1 daemon. It never opens SQLite and does not import
Hermes. Every tool call uses `EnfoldClient`, which opens a fresh Unix-socket
connection, negotiates the fixed client context, performs one request, and
disconnects.

## Tool profiles

`--tool-profile` (or `ENFOLD_TOOL_PROFILE`) is fixed for the process:

| Profile | Tools |
|---|---|
| `core` (default) | `memory_recall`, `memory_remember`, `memory_inspect` |
| `review` | core plus `memory_review`, `memory_resolve` |
| `legacy-v1` | the previous thirteen v1 names, for one transition release |

A new agent should use `core`. Operators reviewing conflicts opt into
`review`. Do not list the old names beside the new ones in `core`.

`enfold.mcp_launcher` does not yet grow a `--tool-profile` flag (that file
is owned by another change). Until it does, set `ENFOLD_TOOL_PROFILE` or
launch `enfold-mcp-proxy` directly with `--tool-profile`.

`mcp` is a base install dependency. Inspect startup options:

```bash
python -m pip install -e .
enfold-mcp-proxy --help
```

For a static MCP client registration, use `enfold-mcp-launch`. It creates
a fresh cryptographically random session ID for every proxy process and safely
captures the process CWD, Git root, credential-free origin, branch, and commit
when available. Client, surface, agent, socket, and requested scopes remain
explicit immutable registration arguments; environment variables and tool
parameters cannot override them. Git discovery uses bounded subprocess argv
without a shell, prompt, pager, or inherited environment.

MCP client A registration (the `client-a-install-1` grant must exist in the
daemon configuration with the same or broader scopes):

```bash
mcp-client-a add enfold -- /path/to/enfold/.venv/bin/python \
  -m enfold.mcp_launcher \
  --socket-path /path/to/enfold.sock \
  --client-id client-a-install-1 \
  --surface mcp-client-a \
  --agent-id client-a \
  --access-scope private \
  --access-scope work \
  --access-scope project:enfold
```

MCP client B registration (use a distinct server grant):

```bash
mcp-client-b add enfold -- \
  /path/to/enfold/.venv/bin/python -m enfold.mcp_launcher \
  --socket-path /path/to/enfold.sock \
  --client-id client-b-install-1 \
  --surface mcp-client-b \
  --agent-id client-b \
  --access-scope private \
  --access-scope work \
  --access-scope project:enfold
```

Do not put `--session-id` in a static registration: omission is what gives
each proxy process a fresh session. An explicit session is available only for
a trusted supervisor that already owns a stable session identifier.

Direct proxy launch remains useful for diagnostics (only after an Enfold
daemon is running):

```bash
enfold-mcp-proxy \
  --socket-path /path/to/enfold.sock \
  --client-id client-a-install-1 \
  --surface mcp-client-a \
  --agent-id client-a \
  --session-id client-a-thread-123 \
  --project-root /path/to/project \
  --repository owner/project \
  --branch main \
  --access-scope private \
  --access-scope work \
  --tool-profile core
```

The required values also accept `ENFOLD_SOCKET_PATH`, `ENFOLD_CLIENT_ID`,
`ENFOLD_SURFACE`, `ENFOLD_AGENT_ID`, and `ENFOLD_SESSION_ID`. Optional
provenance and scope variables are listed by `--help`.

## Core tools

`memory_recall` is the only everyday recall path. It uses the daemon's
prompt-safe context packer, then returns a compact projection: `id`, `text`,
`review`, `source`, and `evidence`. Ranker scores, retrieval telemetry, and
storage internals are stripped. The default whole-result budget is 512
estimated tokens, with a hard cap of 2048. An empty result tells the agent
what to do next; it is not a broken tool.

`memory_remember` stores one durable fact. The model supplies `content` and
`origin` (`user`, `conversation`, `tool`, `document`, `agent_inference`).
Trust, authority, sensitivity, relation, correction status, and retry
idempotency are assigned by policy. The model cannot assert
`human_confirmed`. Optional typed `state` uses `subject` / `predicate` /
`value` so Enfold can detect conflicts.

`memory_inspect` pages evidence or history for one recalled fact ID. Use it
only when a claim is surprising, disputed, high-impact, or historical.

`review` adds `memory_review` (compact conflict summaries) and
`memory_resolve` (human-chosen winner only).

Writer/session/project identity is not present in tool schemas: it comes only
from startup context.

## Errors

Daemon application failures become MCP tool errors whose message is compact
JSON containing `code`, `message`, `retryable`, `details`, `request_id`, and
an actionable `next_action`. Transport outages use
`code="daemon_unavailable"` and are retryable. Successful results are
checked and normalized to JSON before they cross the MCP boundary.

## Legacy surfaces

`legacy-v1` keeps `memory_write`, `memory_promote`, `memory_search`,
`memory_context`, `memory_evidence`, `memory_history`, `memory_changes`,
`memory_timeline`, `memory_entities`, `memory_entity`, `memory_conflicts`,
`memory_resolve_conflict`, and `memory_extraction_enqueue` for one
transition release. New registrations should not use it.

`enfold.mcp_server` / `enfold.mcp_provider` / `enfold/plugin.yaml` are a
Hermes compatibility extra for the v0 holographic store. They are not the
public contract. Existing owners who launch `python enfold/mcp_server.py`
by file path can keep doing that; the server now advertises
`enfold-memory-legacy` and prints a deprecation warning on stderr. New
installs should use `enfold-mcp-launch` and the v1 daemon.

This adapter establishes reliable attribution between cooperative local
clients. It is not an authentication boundary against another process running
as the same operating-system user. `enfold init` and `enfold setup` write a per-client credential digest in
`server.json` and print the raw token once. `health` then reports
`client-credential`. Without those digests, any same-UID process can claim
a configured client id. Do not omit `ENFOLD_CLIENT_CREDENTIAL` in a
supervisor registration.
