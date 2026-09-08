# Configuration

Millie reads `~/.millie/config.toml` (set `MILLIE_HOME` to move the whole
directory). A project can add a `.millie/config.toml` of its own, and any
setting can be overridden for one run with `-c key=value` on the command line,
for example `millie -c sandbox_mode=read-only`.

The complete list of settings, with types and descriptions, is the JSON schema
at [`codex-rs/core/config.schema.json`](../codex-rs/core/config.schema.json).
For a first launch, see [Getting started](getting-started.md). For allocation
failures or room for other applications, see [Make Millie fit in memory](memory.md).
The settings most people touch:

| Setting | What it does |
| --- | --- |
| `model` | The remembered model choice; use `millie --model select` to choose again. |
| `model_provider` | Where the model is served from. Defaults to `llamacpp`, the bundled local server. |
| `approval_policy` | When Millie stops to ask before running a command: `untrusted`, `on-request`, `on-failure` or `never`. See [sandbox.md](./sandbox.md). |
| `sandbox_mode` | What the sandbox lets commands do: `read-only`, `workspace-write` or `danger-full-access`. See [sandbox.md](./sandbox.md). |
| `[llamacpp]` | Settings for the bundled local server; see below. |
| `[mcp_servers.<name>]` | External MCP tool servers to make available; see below. |
| `notify` | A command to run when a turn finishes, for desktop notifications. |
| `log_dir` | Directory for a plaintext log of the session. |
| `model_context_window` | Requested conversation context; automatic serving profiles may cap it to fit memory. |
| `--profile <name>` | Layers `$MILLIE_HOME/<name>.config.toml` over the base user config. Explicit model choices are saved there. |

## The system prompt

The system prompt is a plain text file, not part of the binary, so it can be
changed without rebuilding. Millie looks for it in this order and uses the
first one it finds:

1. the file named by the `MILLIE_SYSTEM_PROMPT` environment variable,
2. `~/.millie/system_prompt.md`,
3. `system_prompt.md` next to the `millie` executable (release packages ship
   one there),
4. `prompts/system_prompt.md` in a source checkout (debug builds only).

If none exists Millie refuses to start and says where it looked. The file may
contain `{{ personality }}` once; it is replaced by the text for the selected
personality (`/personality`). `model_instructions_file` still overrides the
whole prompt for one configuration if you want a different one per profile.

## The model catalog

Model names and capabilities are not compiled into Millie. They come from
`millie-models.json`, a file with one entry per served model (slug, display
name, context window, reasoning levels), looked up like the system prompt:
`MILLIE_MODELS=<file>`, then `$MILLIE_HOME/millie-models.json`, then the file
next to the `millie` binary, then the source checkout in debug builds. Adding or renaming a
model is an edit to that file. The initial recommendation uses the catalog
and hardware; subsequent launches use the remembered explicit selection.
Millie refuses to start without the catalog.

## Everyday model and memory settings

Normal catalog selection needs no manually entered GGUF path. For example:

```toml
model = "millie-35B-A3B-9GB"
# model_context_window = 16384             # optional smaller context

[llamacpp]
# vision = false                           # optional text-only launch
```

`millie --model select` opens the chooser even when files are already cached.
`millie --model millie-35B-A3B-9GB` selects directly. Both remember the last
explicit choice in the active user config, including if startup or download
approval fails. Unknown names are errors. A failed startup does not substitute
another model. With no remembered choice, interactive startup offers a
hardware-based recommendation; 9GB is preferred on 16 GB Apple Silicon Macs.

Selection does not approve downloads. Missing files need interactive approval
or `--download`; a `model` entry in config is not download permission. Scripts
can use `millie exec --model millie-35B-A3B-9GB --download "your task"`.
Installed files launch offline. Files are cached under `$MILLIE_HOME/models/`;
let Millie manage their paths and revisions. Text-only profiles omit the vision
tower until image support is requested.

## Serving profiles

The runtime catalog supplies profiles per model, in preference order. Automatic
selection uses the actual selected GPU's memory and total system RAM, with
Apple Silicon profiles accounting for unified memory. It estimates fit from
capacity, not currently free memory. Leave room for other applications.

Examples from the bundled catalog:

| Model/device | Profile | Main defaults |
| --- | --- | --- |
| 9GB, 16 GB Apple Silicon | `metal-16gb` | 65,536-token context, vision off, weights loaded into RAM |
| 11GB, 16 GB Apple Silicon | `metal-16gb` | 32k context, vision off, mapped weights, 9 expert layers on CPU; tight fit |
| 7GB, 16 GB Apple Silicon | `metal-full` | 32k context, vision on |
| 9GB/11GB, larger Mac | `metal-full` | 131k context, vision on; catalog threshold 22 GB RAM |
| 9GB/11GB, 16 GB GPU class | `gpu-full` | 131k context, vision on; catalog threshold 15 GB VRAM |
| 9GB/11GB, 8 GB GPU + 16 GB RAM class | `hybrid` | 131k context, vision off, 32 expert layers on CPU |
| 9GB/11GB, 4 GB GPU + 16 GB RAM class | `deep-hybrid` | 32k context, vision off, all experts on CPU |

These are profile defaults, not measured usage or guarantees. CPU profiles and
thresholds also vary by model. Consult `millie-models.json` for the full current
list. Forcing an unknown profile is an error. Forcing a known profile does not
establish that the machine has enough memory.

`--serving-profile NAME` pins a serving profile; `--profile NAME` selects a
named user configuration. For each setting, explicit CLI/config overrides take
precedence over profile defaults, which precede generic fallback defaults.
The context request `model_context_window` is normally capped by the profile;
`--model-context-window` overrides that user setting, while the advanced
`llamacpp.ctx_size` bypasses the profile context cap. The confirmed server
context is used by the session.

Startup reports the resolved profile/settings. Settings apply on a new server
launch; an existing server is reused only for a matching model and vision mode.
A context difference is reported, and the session uses the running server's
context. See [memory help](memory.md) for examples and restart instructions.

## Shared server lifetime

The session that starts a managed server owns it. Same-model, same-vision-mode
sessions may attach. Exiting a borrowing session leaves the server running;
exiting the owner stops it, even if another window is open. That window can
restart with `millie resume SESSION_ID` to attempt loading the server again.
Clearing a conversation does not release its server connection.

Different-model or opposite-vision-mode launches are refused without replacing
the running server. Close existing sessions before changing those settings.
`--keep-model-server` explicitly preserves a newly started server after owner
exit. Independently started servers are not automatically terminated by Millie.

## Advanced local server settings

These keys belong under `[llamacpp]`. Normally leave them unset so the selected
model's serving profile can provide defaults. See [memory help](memory.md) for
tradeoffs before copying overrides into your config.

| Key | Purpose |
| --- | --- |
| `gpu = [0]` | Device indices; `[]` selects CPU-only serving |
| `port = 8095` | Local server port; default 8095 |
| `profile = "hybrid"` | Pin a profile supported by the selected model |
| `ctx_size = 32768` | Force total server context, bypassing the profile cap |
| `parallel = 2` | Concurrent request slots; total context is divided among slots; default 1 |
| `vision = false` | Disable image input and vision-tower loading |
| `kv_cache = "q8_0"` | Profile default cache type; `f16` uses more memory; `q4_0` is not broadly validated |
| `mmap = true` | Memory-map weights; paging can slow responses |
| `n_cpu_moe = 32` | Number of expert layers placed on CPU |
| `server_bin = "/path/to/llama-server"` | Override runtime executable discovery |
| `model_path = "/path/to/model.gguf"` | Serve a manually managed local model |
| `mmproj_path = "/path/to/mmproj.gguf"` | Matching vision tower for a manual model path |
| `model_hf = "org/repo/file.gguf"` | Track a custom Hugging Face GGUF; see model updates below |
| `resamples = 1` | Enable one clean retry per pathology episode; default off |

Sampling defaults live in the external catalog and can be overridden through
configuration; they do not require a rebuild. Catalog-managed models require every sampling setting to resolve from catalog
or configuration; missing or invalid settings are startup errors, not a request
to use compiled defaults. Generic custom-source fallback behavior is unchanged.
Leave the model-specific defaults in place for normal use. Memory tuning does not require sampling or prompt edits.
Custom Hugging Face downloads use published revision/checksum metadata. Manual
GGUF files have no tracked remote update source. Millie's prompt and tool format
are tested with Millie models; arbitrary models are not quality-tested.

## MCP servers

```toml
[mcp_servers.docs]
command = "npx"
args = ["-y", "some-mcp-server"]
```

Each entry names a command to run; Millie starts it and offers its tools to
the model. `/mcp` inside a session lists what is configured.

## str_replace autocorrect

```toml
[str_replace_autocorrect]
mode = "log"                  # correct | log (default) | off
# posterior_threshold = 0.9999   # how sure the corrector must be
# max_distance_fraction = 0.25   # decline above this fraction of old_str changed
# h0_prior = 0.5                 # prior that the text is not in the file at all
# max_old_str_chars = 4000
# max_file_bytes = 2097152
```

When `str_replace` fails because `old_str` is not in the file, Millie looks
for the single span the model most plausibly meant: first exact matches after
normalizing line endings, trailing whitespace, interior whitespace runs and
indentation; then a fuzzy search that aligns candidate spans and accepts one
only when a probability model says it is overwhelmingly the intended text
(posterior at least `posterior_threshold` against every other candidate and
against "the text is not in this file") and the edit distance is under
`max_distance_fraction` of `old_str`. `correct` applies it and tells the model
exactly what differed (a short diff); `log` returns the normal failure but
records the would-be correction, with the original and corrected calls, in
the model-call log; `off` does nothing. Two or more plausible spans always
mean no correction. All of these can be set per run with `-c`, for example
`-c str_replace_autocorrect.mode=correct`.

## Edit reminder

```toml
[tools]
edit_reminder = "Reminder: validate this change where applicable before you finish -- run the relevant build, tests or a quick execution, and fix what fails."
```

Off by default. When set, the text is appended to every successful
`str_replace` and `create_file` result, so the model sees it at the moment of
the edit and conversation compaction cannot lose it. Any wording works; the
one above is a reasonable start. No rebuild is needed to change or remove it.

## Harness notes

```toml
[tools]
harness_notes = "tool"   # tool (default) | user
```

Messages Millie itself sends the model (the continuation prompt after a
cut-off or malformed tool call, the loop guard's note after a discarded
completion) are delivered as tool results by default, which keeps the turn's
reasoning visible to the model (this model family drops earlier reasoning at
every real user message). With `user` they are sent as user messages instead.

## Repeated tool calls

```toml
[tools]
repeat_guard = "warn"   # warn (default) | log | off
```

The remembered calls reset when a new user message is accepted, including input
added during an active turn. Asking Millie to reread a changed file therefore
starts a fresh allowance. Internal continuation notes do not reset the guard.

When the model issues the same tool call again -- identical to its previous
call, or three times within its last ten calls -- `warn` runs the call and
appends a warning to the result telling the model which call it is repeating
and to try something different; `log` only records the repeat in the
model-call log (if that is on); `off` does nothing. The call is never blocked.

## Loop guard

```toml
[llamacpp]
repeat_stop = true               # stop a completion that is looping (default true)
ngram_penalty_scale = 1.0        # penalty per extra matched token; 0 turns the penalty off
ngram_penalty_step = 0.5         # added to the scale for each further failing completion
ngram_penalty_start_n = 3        # shortest match penalized on the first failure; drops by 1 per further failure
ngram_penalty_max = 10.0         # cap on the penalty for one token
backtrack_on_repetition = true   # discard a looped completion instead of keeping it
```

Two things count as a failure: a completion that loops, and a tool call
repeated identically (see "Repeated tool calls" above).

With `repeat_stop` on, the server stops a completion as soon as one stretch
of text has repeated at least three times and those repeats cover at least
80% of the last 300, 600 or 900 generated tokens. A completion that reaches
the token cap is checked the same way. Legitimate structured output is not
affected: a list that alternates between two names, or many different names
each used a few times, does not meet the rule.

After a failure, the text that repeated (the looped stretch, or the last ten
tool calls) is sent with the next completion as reference text for an n-gram
penalty: any token that would extend a verbatim match of
`ngram_penalty_start_n` or more tokens against that text has
`(match length - start_n + 1) * scale` subtracted from its logit, capped at
`ngram_penalty_max`. The model's own new output is never penalized, only
matches against the reference text. Each further consecutive failing
completion raises the scale by `ngram_penalty_step` and lowers the first
penalized match length by one, down to 1: with the defaults the second
failure penalizes from 2-token matches at scale 1.5, and from the third
failure on every token that occurs in the reference text is penalized
(1 x scale with no context, rising with the match length). The first
completion that finishes cleanly (no loop, no repeated call) clears the
penalty. The state is not cleared by a new user message.

With `backtrack_on_repetition` on, a looped completion is removed from the
conversation, its tool calls are not executed, and a one-line note tells the
model its previous attempt repeated itself and was discarded; the model
then retries under the penalty. With it off, the looped text stays in the
conversation and the model receives an error message with a short excerpt of
what repeated.

Every key can also be set for one run with `-c`, for example
`millie -c llamacpp.ngram_penalty_scale=2.0`. The model-call log records the
server's repetition statistics (`repetition_stats`) and every penalty
activation (`pathology`).

## vLLM backend

```toml
model_provider = "vllm"
model = "millie-35B-A3B-9GB"            # catalog entry matching the weights served remotely

[vllm]
base_url = "http://127.0.0.1:8300/v1"   # the server's OpenAI-compatible API (default port 8300)
served_model = "millie"                 # the --served-model-name it was started with
```

Millie can talk to a vLLM server instead of llama-server. The server is
started separately and must support the selected model and Millie’s tool-call
format (see [external vLLM servers](./vllm.md)); Millie only checks that it
answers and then sends requests to it. The generation settings under
`[llamacpp]` (sampling, `max_output_tokens`, `thinking_budget`, the loop
guard keys) apply to both backends.

Tool calls are constrained the same way as in llama-server: the serving script
loads a tool-parser plugin that builds the identical lazy grammar from the tool
schemas for every request (xgrammar structural tag, trigger `<tool_call>`).

Differences from llama-server: the repetition stop runs inside Millie on the
streamed output (one unit per streamed token) and cancels the request when it
fires, with the same rule and the same logging; the n-gram penalty spans are
tokenized through the server's `/tokenize` endpoint; the model-call log has no
`raw_generation` field for this backend.

## Model-call log (local, opt-in)

```toml
[model_call_log]
enabled = true                 # or MILLIE_MODEL_CALL_LOG=1
# dir = "/path/to/dir"         # default ~/.millie/model-calls
```

Writes one JSON-lines file per run with every model call (the full request,
the model's raw generated text, the parsed reasoning/content/tool calls,
finish reason, usage) and every tool call (arguments and result), all keyed by
thread id, turn id and a per-thread call number so a run can be reconstructed
exactly. Nothing is ever sent anywhere; the files stay on your machine. They
contain your prompts and code, so treat them as sensitive.

## Lifecycle hooks

Admins can set top-level `allow_managed_hooks_only = true` in
`requirements.toml` to ignore user, project, and session hook configs while
still allowing managed hooks from requirements and managed config layers. This
setting is only supported in `requirements.toml`; putting it in `config.toml`
does not enable managed-hooks-only mode.

## Image input

For vision-enabled catalog profiles, the vision tower downloads alongside the
model. `--no-vision` and text-only profiles skip that download. To add vision
later, close existing Millie sessions and run `millie --vision --download`.
Millie downloads the tower matching the installed model revision and reuses the
model weights. If the installed model has no recorded remote revision and its tower is
missing, provide the matching local tower or explicitly run `millie models update`
to install a matched set. Existing model weights remain usable with `--no-vision`. Set `vision = true` under `[llamacpp]` to enable it on ordinary
future launches. Opposite vision modes cannot share a running server. Attach images in
the TUI or with `millie exec -i image.png`. With an explicit `model_path`,
set `mmproj_path` to the matching vision tower file.

## Completion retries

When a completion fails -- looping output, an identically repeated tool
call, or hitting the generation token cap -- Millie first discards it and
samples the identical request again (fresh seed, no penalties), up to
`resamples` times per episode. Only when that budget is exhausted does the
escalating repetition-penalty machinery engage. `resamples = 0` disables the
retry layer entirely. `backtrack_on_repetition` separately controls backtracking
on repetition and defaults to `false`.

## Model updates

`millie models check-updates` checks the tracked files on Hugging Face without
transferring model weights. `millie models update` explicitly downloads the newest
revision, verifies its published SHA256 checksums, and activates the complete
file set for the next server launch. No software reinstall or hosted catalog is
required. Sampling continues to come from your local catalog/configuration.

These commands manage the local GGUF cache; they do not update an externally
started vLLM server. `HF_ENDPOINT` can select an alternate Hugging Face endpoint.

Both commands default to your remembered model or configured `llamacpp.model_hf`.
To target another catalog model without changing your selection:

```sh
millie models check-updates millie-35B-A3B-9GB
millie models update millie-35B-A3B-9GB
millie --profile work models check-updates
```

Updates follow the existing filenames on the repository's `main` branch. Put
experimental weights on another branch or filename. Metadata checks pin the
whole download to one immutable commit, including the catalog vision tower when
present. These commands currently support public repositories with published
SHA256 metadata. A manual local GGUF path has no tracked remote update source.

Interrupted transfers resume. Failed verification or any failed file leaves the
previous active revision intact. Running servers and their files are untouched;
close the server-owning session before launching the updated model. Old files
are retained, so allow disk space for the replacement. An unchanged model file
can be reused when another file in the set changes.

Fresh installations also resolve the current revision and published checksums
from Hugging Face; the catalog's historical checksums do not pin new downloads.
Installed revisions launch offline. Startup does not automatically replace
installed weights with newer ones.

Catalog and custom Hugging Face cache locations include both repository and full file path.
Existing catalog model files are reused at their current paths without a startup
checksum scan or a copy of the weights. Millie records their locations without
claiming a remote revision or checksum. Both the original model and matching local
vision tower remain usable after a software update. An explicit model update
verifies and activates the new revision while retaining the original files.
Custom Hugging Face files still use repository-specific cache locations; an
arbitrary basename is not enough to identify their repository.

Unknown keys in `[llamacpp]` and catalog sampling blocks are errors, including
when `--strict-config` is absent. vLLM requires an explicit catalog model (or a
remembered selection); `vllm.served_model` is the server alias, a separate value.

An explicit unknown `llamacpp.profile` is an error, rather than a request for
automatic profile selection. Explicit `--model` choices are remembered in the
active configuration for both llama.cpp and vLLM, even when startup fails.
