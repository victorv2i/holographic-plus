# Standalone daemon packaging

`enfold-server` is the foreground entry point for the shared Enfold service.
It never creates or migrates a database. Run
`python -m enfold.ops migrate /absolute/path/to/memory.db` during an explicit
maintenance window before starting it against an older store.

Example configuration (store as a user-owned, non-group/world-writable file):

```json
{
  "database_path": "/absolute/path/to/memory.db",
  "socket_path": "/absolute/private/directory/enfold.sock",
  "retrieval": {
    "mode": "ci",
    "allow_nonproduction": true,
    "dimensions": 256
  },
  "grants": {
    "client-a-install-1": ["private", "work", "sensitive", "project:enfold"],
    "client-b-install-1": ["private", "work", "project:enfold"],
    "hermes-native": ["private", "work", "project:enfold"]
  },
  "client_credentials": {
    "client-a-install-1": "sha256:<64-lowercase-hex>",
    "client-b-install-1": "sha256:<64-lowercase-hex>",
    "hermes-native": "sha256:<64-lowercase-hex>"
  }
}
```

When `client_credentials` is present it must contain exactly every client in
`grants`, and every client must have a distinct digest. Enfold stores only
SHA-256 digests in the server configuration. A
trusted supervisor supplies the corresponding URL-safe token to official MCP
launchers as `ENFOLD_CLIENT_CREDENTIAL`; the Hermes adapter accepts the token
as `HermesAdapterConfig.credential`. Invalid or missing tokens fail the
handshake before grant policy is evaluated. Live `health` reports
`identity_authentication` as either `client-credential` or
`trusted-local-uid`.

This credential is defense in depth for isolated or supervised clients, not a
new security principal inside one Unix account. A malicious process running as
the daemon's Unix user may be able to inspect another process or its launch
environment. Without `client_credentials`, every process under that UID is
explicitly trusted to claim any configured client ID. Do not offer mutually
untrusted tenants one daemon: use separate Unix accounts and separate Enfold
daemon/database/socket instances. The daemon intentionally rejects socket
peers whose kernel-reported UID differs from its own.

Sensitivity is an additional capability, not an alternative name for a data
scope. A fact with `scope: "private"` and `sensitivity: "sensitive"` is visible
only when the connection is authorized for and requests both `private` and
`sensitive`. Clients without the `sensitive` capability cannot create
sensitive durable writes. `secret` durable writes remain disabled. The same
filter applies to search, context, evidence, history, projections, and conflict
members.

Prompt-ready `memory.context` is stricter than search. It renders only facts
whose correction status is `human_confirmed` or `human_corrected`, then applies
normalized control-syntax rejection as defense in depth. Other matches return
only redacted receipts with an exclusion reason. Grant correction authority
only to a trusted review surface; ordinary extraction or agent writes are not
implicitly prompt-trusted. Arbitrary stored text remains inspectable through
the structured `memory.search` path.

The `retrieval` object is required. `ci` mode is an offline plumbing test, not
a semantic production retriever, and therefore requires the conspicuous
`allow_nonproduction: true` opt-in. Health output preserves
`embedder_production_ready: false`.

The staged stored retrieval stack has a concrete SQLite backend and validates all
of the following before use: exact query-to-document identity-role mapping,
embedding version, dimension, active-fact coverage, candidate-only vector
loading, finite vectors, and complete per-search coverage. Missing or malformed
vectors fail closed. A representative configuration is:

```json
{
  "retrieval": {
    "mode": "stored",
    "provider": "ollama",
    "model": "embeddinggemma",
    "dimensions": 768,
    "query_identity": "ollama:embeddinggemma:query:none:sha256:<64-lowercase-hex>",
    "document_identity": "ollama:embeddinggemma:document:none:sha256:<64-lowercase-hex>",
    "embedding_version": "sha256:<64-lowercase-hex>",
    "model_fingerprint": "sha256:<64-lowercase-hex>",
    "prefix_policy": "none",
    "processor": {"mode": "daemon-supervised"},
    "query_prefix": ""
  }
}
```

`model_fingerprint` is mandatory and must be an immutable
`sha256:<64-lowercase-hex>` Ollama artifact digest. `embedding_version` must
equal that digest and remain the final component of both identities.

Before stored-mode `check`, `status`, or startup can proceed, Enfold resolves
the configured local Ollama tag through `/api/tags` and fails closed unless its
reported digest matches exactly. A mutable tag is therefore never an
attestation. Safe check and authenticated health output exposes only
`artifact_attestation: {"provider": "ollama", "status": "verified"}`; it
does not expose provider payloads or artifact details.

The model fingerprint equals the final identity version component, and the
identity is derived from provider, model, role, prefix policy, and fingerprint.
Non-empty prefixes require a full SHA-256 prefix policy.

Daemon-supervised stored mode currently permits Ollama only because its request
timeout bounds shutdown. FastEmbed remains blocked until embedding inference is
isolated in a killable worker process; an unbounded in-process ONNX call must
not be able to strand the sole-writer daemon during shutdown.

Stored mode provisions a durable `embedding_jobs` outbox. Every new fact and
its exact identity/version/dimension job commit in one transaction; no model
call occurs on `memory.write`. `EmbeddingJobProcessor.process_one()` is an
explicit lease/retry/dead-letter worker run by the daemon on a dedicated SQLite
connection. `check` verifies the configured artifact and reports safe
attestation state; live health reports the
worker heartbeat, last success/error, and outbox state. The worker stops and
joins before its connection closes. No model work occurs on `memory.write`.

An explicit maintenance flow can call `EmbeddingOutbox.enqueue_backfill()` for
preexisting active facts; the daemon supervisor also owns this at startup.
`check` remains read-only. A missing vector is temporarily eligible only when
an exact pending/processing job exists; that
candidate remains searchable lexically with zero dense contribution. Missing
work without a viable job, malformed vectors, identity mismatch, and dead
letters make health unsafe and block activation or fail retrieval. Expired
leases are reclaimable, retries are bounded, and completion revalidates the
lease, content hash, and current/non-erased fact state after the model call.

When the configured sqlite-vec index is healthy, Enfold scores every eligible
fact that has the configured stored vector, then unions the global dense window
with FTS candidates. This keeps an old semantic-only fact discoverable after
the store exceeds the candidate-window size. Pending embedding jobs remain
lexical-only with a zero dense score. Health metadata reports
`dense_candidate_coverage: "global"` and
`candidate_generation: "global-index-plus-lexical"`; non-indexed and explicit
CI modes instead honestly report the bounded `"recent-plus-lexical"` path.

The vec0 table is derived state. Rebuild validates all canonical vectors and
records a source/index generation ledger; normal fresh opens use that ledger
instead of a full source-to-index membership scan. A canonical vector mutation
that does not update the derived index makes the generations diverge and forces
an honest brute fallback. `vector_fallback_active` therefore describes the
current state, while `vector_fallback_count` and
`vector_fallback_recovery_count` retain process-lifetime diagnostic history.

The socket directory must already exist, be owned by the current user, and
must not be group/world writable. Validate without binding a socket:

```bash
enfold-server --config /absolute/path/to/server.json check
enfold-server --config /absolute/path/to/server.json status
```

For stored mode, `status` deliberately does not call a bare socket connection
“healthy”: socket probing cannot attest the worker heartbeat. It reports live
health as unverified and exits nonzero; query the authenticated protocol
`health` method to evaluate heartbeat, errors, dead letters, and pending age.

Run the foreground process only after validation:

```bash
enfold-server --config /absolute/path/to/server.json run
```

Any configuration, database, or socket path below `~/.hermes` is refused
unless `--allow-live` is supplied explicitly. The flag acknowledges a live
deployment; it does not migrate, back up, install, or register anything.

Optional top-level configuration fields are `busy_timeout_ms`, `client_timeout`,
`shutdown_timeout`, `max_frame_bytes`, `backlog`, and
`cleanup_stale_socket`. `client_credentials` enables credential-bound client
IDs as described above. The optional `extraction` object is described below.
Unknown fields fail validation.

## Automatic host-model extraction

Automatic model extraction is opt-in and defaults to
`{"mode": "disabled"}`. A daemon-supervised deployment currently requires the
bundled local Ollama child plus immutable model and recipe pins. The implemented
subprocess boundary uses explicit argv,
does not invoke a shell or inherit the daemon environment, bounds JSON input,
output, errors, and execution time, validates structured proposals, and
terminates the child process group on failure. A representative configuration
is:

Extractor output is a proposal, not evidence. Canonical writes additionally
require an independently configured evidence verifier to confirm that the
cited transcript span supports the proposed claim. With no verifier,
automatic extraction remains fail-closed: proposals are dead-lettered as
`proposal_support_unverified`, health reports the verifier as unconfigured and
degraded, and no fact is written. Exact span identity and excerpt matching are
necessary integrity checks but are not treated as semantic entailment.

```json
{
  "extraction": {
    "mode": "daemon-supervised",
    "host": {
      "type": "subprocess",
      "argv": [
        "/absolute/path/to/enfold-ollama-extractor",
        "--model", "qwen3:30b",
        "--model-identity", "ollama:qwen3-30b",
        "--prompt-identity", "durable-memory-v2"
      ],
      "model_identity": "ollama:qwen3-30b",
      "prompt_identity": "durable-memory-v2",
      "timeout_seconds": 180,
      "terminate_grace_seconds": 2,
      "max_input_bytes": 16384,
      "max_output_bytes": 65536,
      "max_error_bytes": 16384,
      "environment": {}
    },
    "artifact": {
      "provider": "ollama",
      "model": "qwen3:30b",
      "model_digest": "sha256:<64-lowercase-hex>",
      "recipe_digest": "sha256:<64-lowercase-hex>"
    },
    "artifact_recheck_seconds": 60,
    "poll_seconds": 1,
    "drain_limit": 4,
    "lease_seconds": 300,
    "heartbeat_seconds": 30,
    "retry_delay_seconds": 1,
    "max_attempts": 3,
    "heartbeat_stale_seconds": 240,
    "pending_stale_seconds": 900
  }
}
```

The executable path must be absolute. Environment inheritance is disabled and
an attested Ollama deployment rejects environment overrides, so every
inference-affecting child option remains in the pinned argv. `model_digest`
must be the immutable digest returned for the configured local tag, not the tag
itself. `recipe_digest` covers that model digest, exact argv and supervisor
limits, prompt, proposal schema, adapter executable, and installed extraction
source modules. Compute it from the candidate configuration with
`HostExtractorConfig.inference_recipe(model_artifact_digest=...,` followed by
`**bundled_ollama_components(argv[0]))`, then pin the returned `.digest`.

Both `check` and startup resolve the mutable Ollama tag and verify the exact
recipe before opening the database. While running, the extraction worker
repeats the verification at the configured interval before claiming more queue
work; a failure pauses claims and degrades health until verification recovers.
Any tag drift, executable/source change, prompt/schema change, or configuration
change therefore fails closed. Authenticated health and check output expose
only `status=verified`, `provider=ollama`, the recipe format version, and a
stable worker failure code; configured or observed digests and Ollama payloads
are never returned.

Keep the user-owned server configuration private and do not commit
credentials. The daemon gives the worker a dedicated SQLite connection,
exposes worker and queue state through authenticated health, stops new claims
during shutdown, and stops the worker before closing its connection.
Each validated non-empty proposal batch is applied through the authoritative
write policy in one transaction that also deletes the leased queue row. A
policy rejection or late failure rolls back the entire batch, including state
supersession, provenance, write logs, and embedding-outbox jobs; the persisted
proposal snapshot remains available for a model-free retry. Query embeddings
used for near-duplicate checks are computed before transaction ownership.
Repository verification is not a real-store rehearsal or permission to
activate; follow
[`ACTIVATION_CHECKLIST.md`](ACTIVATION_CHECKLIST.md).

### Side-by-side activation configuration

`enfold-activation stage-config` prepares a new private configuration without
replacing the base configuration, opening its database, controlling a service,
or invoking a model. It preserves the existing grants and retrieval settings,
switches the bundled extractor to the current grounded contract, and measures
and pins the exact model artifact and inference recipe for the candidate
executable.

```bash
enfold-activation stage-config \
  /absolute/path/to/current-server.json \
  /absolute/path/to/candidate-server.json \
  --candidate-executable /absolute/path/to/enfold-ollama-extractor \
  --model-digest sha256:<64-lowercase-hex>
```

Inputs must be private, user-owned, non-symlink files; the output must not
exist and is atomically published with mode 0600 only after strict config
validation. The emitted report identifies the staged model and contract but
does not disclose artifact or recipe digests. A staged config is not an
activation: run the copied-store rehearsal and maintenance-window gates before
any service cutover.

### Bundled local Ollama child

`enfold-ollama-extractor` implements the subprocess contract for a local
Ollama `/api/chat` endpoint. It accepts only loopback HTTP URLs, disables
ambient proxies, supplies a strict system prompt and JSON Schema `format`,
bounds and validates the response, and never writes transcript or model text
to stderr. The `durable-memory-v2` contract presents deterministic bounded
transcript spans to the model. A proposal selects an `evidence_span_id`; the
child copies the exact source text into the host-facing `evidence_excerpt`, and
the processor independently revalidates that excerpt before persistence. The
model is configurable; `qwen3:30b` is only the example default.

```json
{
  "type": "subprocess",
  "argv": [
    "/absolute/path/to/enfold-ollama-extractor",
    "--endpoint", "http://127.0.0.1:11434/api/chat",
    "--model", "qwen3:30b",
    "--model-identity", "ollama:qwen3-30b",
    "--prompt-identity", "durable-memory-v2"
  ],
  "model_identity": "ollama:qwen3-30b",
  "prompt_identity": "durable-memory-v2",
  "environment": {}
}
```

The child verifies both request identities. Alternatively, the only non-secret
variables it reads are `ENFOLD_OLLAMA_ENDPOINT`, `ENFOLD_OLLAMA_MODEL`,
`ENFOLD_OLLAMA_TIMEOUT_SECONDS`, and `ENFOLD_OLLAMA_MAX_RESPONSE_BYTES`.
Authenticated and non-local endpoints are intentionally unsupported. Success
emits exactly one canonical `{"proposals": [...], "version": 1}` object.
Configuration, invalid-data, unavailable-service, and unexpected failures use
stable statuses 64, 65, 69, and 70 without a diagnostic body.

### Bundled OpenAI Luna child

`enfold-openai-extractor` implements the same `durable-memory-v2` subprocess
contract for benchmark and evaluation use with the OpenAI Responses API. It is
pinned to the official
`https://api.openai.com/v1/responses` endpoint, sends no tools, uses strict
Structured Outputs, sets `store` to `false`, hashes the Enfold client identity
into a privacy-preserving `safety_identifier`, and applies the same local span
resolution and processor grounding checks as the Ollama child.

```json
{
  "type": "subprocess",
  "argv": [
    "/absolute/path/to/enfold-openai-extractor",
    "--model", "gpt-5.6-luna",
    "--model-identity", "openai:gpt-5.6-luna",
    "--prompt-identity", "durable-memory-v2",
    "--reasoning-effort", "none"
  ],
  "model_identity": "openai:gpt-5.6-luna",
  "prompt_identity": "durable-memory-v2",
  "environment": {
    "OPENAI_API_KEY": "<private-platform-api-key>",
    "OPENAI_PROJECT_ID": "<optional-project-id>"
  }
}
```

The API key is accepted only through the child's explicitly allowlisted
environment, never through argv. Keep the user-owned server configuration
private and permission-restricted. Optional environment fields are
`OPENAI_ORG_ID`, `OPENAI_PROJECT_ID`, `ENFOLD_OPENAI_MODEL`,
`ENFOLD_OPENAI_TIMEOUT_SECONDS`, `ENFOLD_OPENAI_MAX_RESPONSE_BYTES`, and
`ENFOLD_OPENAI_MAX_OUTPUT_TOKENS`. Start with reasoning effort `none` and
compare `low` in the extraction Arena before activation. Authentication and
schema/request errors fail permanently; throttling and service failures remain
retryable.

This child requires a dedicated OpenAI Platform API key and usage billing.
Consumer-product sessions and cached login credentials are unsupported and
must never be copied, parsed, or forwarded. If no Platform API key is
available, leave the child disabled. Daemon-supervised activation also rejects
hosted mutable model names that cannot be resolved to a locally verifiable
immutable artifact.
