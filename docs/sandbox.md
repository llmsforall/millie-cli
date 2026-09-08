# Sandbox and approvals

Two settings decide how much Millie can do without asking you.

## `sandbox_mode`

What commands are allowed to touch while they run:

- `read-only` – commands can read the filesystem but not write to it.
- `workspace-write` – commands can write inside the current project (and a
  few scratch locations) but nowhere else. This is the usual setting.
- `danger-full-access` – no sandbox at all. Use only when you know why.

Network access from inside the sandbox is off unless you enable it.

## `approval_policy`

When Millie pauses to ask before running a command:

- `untrusted` – only known-safe, read-only commands run without asking.
- `on-request` – Millie decides when something is risky enough to ask. The
  default for interactive use.
- `on-failure` – commands run in the sandbox; if one fails, Millie asks
  whether to retry it outside the sandbox.
- `never` – never ask. Intended for `millie exec` in scripts.

`/permissions` inside a session switches between these presets, and both
settings can be changed for one run with `-c`, for example
`millie -c approval_policy=never -c sandbox_mode=read-only`.
