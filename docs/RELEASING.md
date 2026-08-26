# Releasing Enfold

Enfold is distributed from GitHub only. It is not published to PyPI.
A release is a git tag plus a GitHub Release that attaches the wheel,
sdist, and SHA256 checksums. Users pin that tag.

## Version agreement

Three strings must match before a tag is pushed:

| Location | Form |
| --- | --- |
| `pyproject.toml` `[project].version` | `X.Y.Z` |
| git tag | `vX.Y.Z` |
| `CHANGELOG.md` heading | `## X.Y.Z` |

`integrations/hermes_enfold_v1/plugin.yaml` `version:` must also equal
`X.Y.Z`. `tests/test_packaging_fixes.py` checks this alignment.

Do not move a published tag. If `vX.Y.Z` already points at an older
tree, bump the project version and cut `vX.Y.Z+1`.

The current project version is `0.8.1`. An older `v0.8.0` tag exists on
the remote (2026-07-12) and predates the installable `enfold` /
`enfold-mcp` scripts. Do not move `v0.8.0`. The first GitHub Release
from this tree is `v0.8.1`. README, BOOTSTRAP, and SERVER_DEPLOYMENT
pin `@v0.8.1`.

## Cutting a release

1. Move notes from `## Unreleased` into `## X.Y.Z` in `CHANGELOG.md`.
2. Set `[project].version` and the Hermes plugin version to `X.Y.Z`.
3. Confirm `docs/BOOTSTRAP.md` and `README.md` pin `@vX.Y.Z`.
4. Run the full test suite from the repo root:
   `PYTHONPATH=$PWD python -m pytest tests/ -q`
5. Tag `vX.Y.Z` on that commit and push the tag. The
   `.github/workflows/release.yml` workflow builds the wheel and sdist,
   runs the test suite against the installed wheel, writes `SHA256SUMS`,
   and attaches those files to a GitHub Release. It does not publish
   anywhere else.

Install what that release produced, not a moving branch:

```bash
uvx --from git+https://github.com/victorv2i/enfold@vX.Y.Z enfold-mcp --self-test
```

## Schema compatibility

The on-disk store is schema version 1 (`SUPPORTED_SCHEMA_VERSION = 1`
in `enfold/schema.py`).

- A newer Enfold binary opens a v1 store.
- A writer may require an explicit
  `python -m enfold.ops migrate /absolute/path/to/memory.db` before it
  will accept a v1 file that is missing later writer patches. Opening the
  daemon does not migrate.
- A store created by a newer schema than this code supports is refused
  (`SchemaTooNewError`). Upgrade Enfold; do not downgrade the file.
- Downgrades are not supported.

A version bump that needs a new schema must ship a registered migration
and increment `SUPPORTED_SCHEMA_VERSION`. Until that happens, every
`0.8.x` release reads and writes the same v1 store. Pinning the install
tag keeps the binary and the store on a known pair. An unpinned branch
install can orphan a store the older binary can no longer open.

## MCP and protocol compatibility

The daemon handshake is protocol `1.0` (`PROTOCOL_MAJOR = 1`,
`PROTOCOL_MINOR = 0` in `enfold/protocol.py`).

- Major mismatches are refused.
- A client minor newer than the server minor is refused.
- The default stdio MCP profile is `core`: `memory_recall`,
  `memory_remember`, and `memory_inspect`. Profile `review` adds
  `memory_review` and `memory_resolve`. Profile `legacy-v1` keeps the
  previous thirteen v1 names for one transition release.

Patch releases (`0.8.x`) keep this protocol major and these profiles.
A protocol major bump is a breaking release and must be called out in
`CHANGELOG.md` before the tag is cut.
