# Staging artifact: not installed

This directory is a Hermes `MemoryProvider` bridge that Enfold does not
install, import, register, or copy. Repository tests may import it.
Nothing in `enfold setup`, `enfold-mcp`, or the published wheel starts
this provider or connects it to a live store.

To use it, an operator must copy it to `$HERMES_HOME/plugins/enfold_v1`
during a controlled maintenance window after the standalone daemon is
already running. See `docs/STAGING_ACTIVATION.md`.

The public MCP surface is `enfold-mcp` / `enfold-mcp-proxy`, not this
folder.
