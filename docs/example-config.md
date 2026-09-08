# Example configuration

Ordinary launches need no hand-written model path or memory overrides: use
`millie --model select`, which remembers your choice. If you want a small
editable starting point, merge this into `~/.millie/config.toml`:

```toml
model = "millie-35B-A3B-9GB"
approval_policy = "on-request"
sandbox_mode = "workspace-write"

# Optional: request a shorter context to leave more memory for other apps.
# model_context_window = 16384

[llamacpp]
# Optional: disable images. Leave unset to use the serving profile default.
# vision = false
```

Put top-level keys before `[llamacpp]`; do not create duplicate tables when
editing an existing file. Leaving settings absent lets the catalog's profile
choose context, vision and placement for your hardware. Change the model slug
to your preferred catalog model, or use the chooser again.

For a separate user configuration, put overrides in
`~/.millie/work.config.toml` and launch `millie --profile work`. The last
explicit model selected under that profile is saved there.

See [memory help](memory.md) for smaller-context, CPU/GPU and image-support
examples, and [configuration](config.md) for advanced overrides. Close sessions
using the old server before applying new server settings.
