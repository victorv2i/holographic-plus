# First local Enfold instance

`enfold setup --client {codex|claude-code|cursor|hermes|generic}` is the
complete first-run path. It creates the owner-only store if needed, generates a
client id and bearer token, adds the grant and credential digest atomically,
starts or reuses the per-user daemon, writes the host snippet, runs a
write/search smoke test through the daemon protocol, and records an uninstall
manifest. If any step fails, it rolls back files it created and says what
happened.

`enfold-mcp` is the product stdio entry. With no extra flags it creates or
reuses the XDG store, starts or attaches to the daemon, waits for readiness,
and then serves MCP. `enfold-mcp --self-test` stops after a successful
remembered-and-recalled fact. After `enfold setup`, self-test reuses the
client id and surface that setup registered.

`enfold doctor` writes a fact, recalls it, and returns its evidence through
an isolated local-lexical daemon, then prints a pass/fail diagnosis. It does
not use or modify the live store.

`enfold demo` uses the same isolated path to show the moat: equal-authority
typed writes produce a conflict receipt, a human authority resolves it, and
history plus evidence remain. It does not touch `~/.hermes/memory_store.db`.
Host instruction snippets are in `integrations/{claude-code,codex,cursor,hermes}/SKILL.md`.

`enfold init` remains the explicit "create files only" command. It does not
start a daemon or register a client.

Linux AF_UNIX sockets accept at most 107 bytes. When the default
`$XDG_DATA_HOME/enfold/enfold.sock` path is longer than that, a new instance
keeps the SQLite store in the data directory and binds an owner-only socket
under `$XDG_RUNTIME_DIR/enfold` or `/tmp/enfold-<uid>-<hash>`. An existing
instance with a too-long configured socket is not relocated; pass a shorter
`--socket-path` or edit `socket_path` in `server.json`.

## Install

Enfold is distributed from GitHub only. There is no PyPI package. Pin a
released version tag. The published GitHub Release wheel includes the MCP
bridge. `sqlite-vec` is an optional upgrade, not a requirement for first
recall.

Primary path:

```bash
uvx --from git+https://github.com/victorv2i/enfold@v0.8.1 enfold-mcp --self-test
```

Keep the tools on PATH, then run the first-run path:

```bash
uv tool install git+https://github.com/victorv2i/enfold@v0.8.1
enfold setup --client cursor
enfold-mcp --self-test
enfold doctor
```

pip from the same tag:

```bash
python -m pip install "git+https://github.com/victorv2i/enfold@v0.8.1"
# Optional accelerator:
python -m pip install "enfold[sqlite-vec] @ git+https://github.com/victorv2i/enfold@v0.8.1"
```

Wheel from a GitHub Release: download `enfold-*.whl` and `SHA256SUMS`, check
the digest, then `python -m pip install ./enfold-0.8.1-py3-none-any.whl`.

From a source checkout:

```bash
python -m pip install -e .
enfold setup --client generic --config-dir /tmp/enfold-dev/config --data-dir /tmp/enfold-dev/data
```

## What is written to disk

By default those paths are:

| Purpose | Default path |
| --- | --- |
| Configuration | `$XDG_CONFIG_HOME/enfold/server.json`, or `~/.config/enfold/server.json` |
| Client credential | `$XDG_CONFIG_HOME/enfold/credentials/<client-id>` |
| Install manifest | `$XDG_CONFIG_HOME/enfold/install-manifest.json` |
| Host snippet | `$XDG_CONFIG_HOME/enfold/clients/<client>.snippet` |
| Database | `$XDG_DATA_HOME/enfold/memory.db`, or `~/.local/share/enfold/memory.db` |
| Daemon socket | `$XDG_DATA_HOME/enfold/enfold.sock`, or a shorter owner-only runtime path when that default exceeds 107 bytes |
| Daemon pid / log | `$XDG_DATA_HOME/enfold/enfold.pid`, `$XDG_DATA_HOME/enfold/daemon.log` |

Both generated directories are mode `0700`; configuration, credentials, and the
SQLite database are mode `0600`. Initialization refuses symlinked paths and
refuses non-sticky writable ancestor directories. It validates the client ID
against the wire protocol, removes exact files it created if initialization
fails, and refuses to overwrite a configuration, database, or socket path.

## Credentials

`enfold init` and `enfold setup` generate a per-client bearer token, store only
its sha256 digest in `server.json`, print the raw token once, and write the
raw token to the owner-only credentials file so `enfold-mcp` can reuse it.
Host snippets pass the token as `ENFOLD_CLIENT_CREDENTIAL` in the supervisor
environment. Do not paste the token into an agent prompt. The daemon still
checks `SO_PEERCRED` and refuses a peer whose UID is not this user.

Without that token, claiming a granted `client_id` is rejected.

## Uninstall

```bash
enfold uninstall --dry-run
enfold uninstall
enfold uninstall --purge-data
```

`--dry-run` prints every path recorded in the install manifest. The default
uninstall stops a daemon we started and removes package-created config,
credentials, snippets, pid, and log files while preserving `memory.db`.
`--purge-data` also deletes the store and SQLite sidecars.

## Generated retrieval

Bootstrap writes `retrieval.mode=local-lexical` and
`extraction.mode=disabled`. Local-lexical is the production-honest first-run
mode: FTS, Jaccard, and ranking priors, with dense scoring off. It is the
same path `enfold doctor` exercises. `ci` retrieval is a separate
non-production plumbing test and is not written by setup. Configure and
validate stored retrieval before relying on semantic ranking; see
[SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md).

## Release verification

Before releasing a wheel, verify the installed surface in an isolated temporary
environment. The wheel must expose `enfold`, `enfold-mcp`, and `enfold-server`,
include `mcp` as a base dependency, and retain `enfold/plugin.yaml` as package
data.

```bash
python -m build
python -m venv /tmp/enfold-wheel-check
/tmp/enfold-wheel-check/bin/python -m pip install dist/enfold-*.whl
/tmp/enfold-wheel-check/bin/enfold --help
/tmp/enfold-wheel-check/bin/enfold setup --client generic \
  --config-dir /tmp/enfold-wheel-check/config \
  --data-dir /tmp/enfold-wheel-check/data
/tmp/enfold-wheel-check/bin/enfold-mcp --self-test \
  --config-dir /tmp/enfold-wheel-check/config \
  --data-dir /tmp/enfold-wheel-check/data
/tmp/enfold-wheel-check/bin/enfold doctor
```

This check uses fresh temporary directories and does not exercise or modify an
existing live store.
