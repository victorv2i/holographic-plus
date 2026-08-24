# First local Enfold instance

`enfold init` is the explicit first-run path for a **new** local Enfold
instance. It creates a database and a server configuration, but never starts a
daemon, installs an MCP client, changes a service, or opens an existing store.

Install a released Enfold wheel (and the optional MCP dependency only when an
MCP bridge is needed), then initialize a new instance:

```bash
python -m pip install /path/to/enfold-0.8.0-py3-none-any.whl
# Optional, for enfold-mcp-proxy only:
python -m pip install mcp
enfold init --client-id workstation-1
```

The command prints JSON with the generated paths. By default those are:

| Purpose | Default path |
| --- | --- |
| Configuration | `$XDG_CONFIG_HOME/enfold/server.json`, or `~/.config/enfold/server.json` |
| Database | `$XDG_DATA_HOME/enfold/memory.db`, or `~/.local/share/enfold/memory.db` |
| Daemon socket | `$XDG_DATA_HOME/enfold/enfold.sock`, or `~/.local/share/enfold/enfold.sock` |

Use explicit private locations when appropriate:

```bash
enfold init \
  --config-dir /absolute/private/config/enfold \
  --data-dir /absolute/private/data/enfold \
  --client-id workstation-1
```

Both generated directories are mode `0700`; the configuration and SQLite
database are mode `0600`. Initialization refuses symlinked paths and refuses
non-sticky writable ancestor directories. It validates the initial client ID
against the wire protocol, removes exact files it created if initialization
fails, and refuses to overwrite a configuration, database, or socket path.
Choose a new directory instead of pointing it at an existing Enfold deployment.
In particular, it does not target `~/.hermes`.

Validate the new instance and then run it in the foreground:

```bash
enfold-server --config ~/.config/enfold/server.json check
enfold-server --config ~/.config/enfold/server.json run
```

## What the generated configuration means

Bootstrap grants only the selected client ID the `private` scope. It disables
automatic extraction and selects 256-dimensional deterministic `ci` retrieval
with the brute-force backend. That retrieval mode is deliberately marked
non-production: it is useful for checking the local daemon and its wiring, not
for semantic production memory. Do not place real durable memory in an instance
until you have configured and validated the stored retrieval profile described
in [SERVER_DEPLOYMENT.md](SERVER_DEPLOYMENT.md).

Add every MCP client as a distinct explicit grant before registering it. The
MCP bridge is a cooperative same-OS-user adapter, not an authentication boundary
between processes owned by that user; see [MCP_PROXY.md](MCP_PROXY.md).

## Release verification

Before releasing a wheel, verify the installed surface in an isolated temporary
environment. The wheel must expose `enfold`, `enfold-server`, and any adapters
advertised by the release, and it must retain `enfold/plugin.yaml` as package
data.

```bash
python -m build
python -m venv /tmp/enfold-wheel-check
/tmp/enfold-wheel-check/bin/python -m pip install dist/enfold-*.whl
/tmp/enfold-wheel-check/bin/enfold --help
/tmp/enfold-wheel-check/bin/enfold init \
  --config-dir /tmp/enfold-wheel-check/config \
  --data-dir /tmp/enfold-wheel-check/data \
  --client-id wheel-check
/tmp/enfold-wheel-check/bin/enfold-server \
  --config /tmp/enfold-wheel-check/config/server.json check
```

This check intentionally uses fresh temporary directories and does not exercise
or modify an existing live store.
