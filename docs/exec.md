# Non-interactive mode

`millie exec` runs one task without the interactive UI and exits, which suits
scripts and CI:

```shell
millie exec "fix the failing test in tests/parser.rs"
echo "summarise this repository" | millie exec -
```

If model files are missing, explicitly approve their download with `--download`.
For example, `millie exec --model millie-35B-A3B-9GB --download "explain the tests"`.
A remembered model choice alone does not approve a download. Once cached, omit
`--download`. See [memory help](memory.md) if startup cannot allocate enough memory.

Useful flags:

- `--json` – print events as JSON lines instead of human-readable text.
- `-o FILE`, `--output-last-message FILE` – write the agent's final message to
  a file.
- `--output-schema FILE` – a JSON Schema the final response must match.
- `--ephemeral` – do not save the session to disk.
- `--skip-git-repo-check` – allow running outside a Git repository.
- `--llama-model-path FILE`, `--llama-gpu 0,1`, `--llama-port PORT` – override
  the `[llamacpp]` settings for this run.

`millie exec resume --last` (or `resume <SESSION_ID>`) continues an earlier
session, and `millie exec review` runs a code review of the current changes.
Settings and approval behaviour are the same as the interactive UI; see
[config.md](./config.md) and [sandbox.md](./sandbox.md).
