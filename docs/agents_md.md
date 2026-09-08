# AGENTS.md

An `AGENTS.md` file in a project gives Millie standing instructions for that
codebase: how to build and test, conventions to follow, things to avoid. Millie
reads it at the start of every session. `/init` creates one.

## Hierarchical agents message

When the `child_agents_md` feature flag is enabled (via `[features]` in
`config.toml`), Millie appends additional guidance about AGENTS.md scope and
precedence to the user instructions message and emits that message even when
no AGENTS.md is present.
