# Local model startup

Millie remembers the last model you explicitly select. Use `millie --model
select` to open the chooser, including when model files are already downloaded,
or use `millie --model MODEL` to select a catalog entry directly. A valid choice
is saved before downloading or loading begins. A failed startup leaves that
choice intact; free memory and retry. An invalid model name is rejected.

Selecting a model and approving its download are separate actions. Use
`--download` to approve missing model files, or answer the terminal download
prompt. Existing partial downloads are preserved on HTTP errors and concurrent
downloads of the same file are coordinated.

On 16GB Apple Silicon Macs the catalog recommends the 9GB model with ternary experts. The
11GB option remains available and is labeled "fits, but tight". Recommendations
do not replace a remembered choice. These recommendations and sampling defaults
live in `millie-models.json`.

The first Millie process to launch a model server owns it. Other windows using
the same model and port can attach. Closing an attached window does not stop
the server; closing the owning window does, including terminal closure or a
crash. A disconnected session reports a `millie resume SESSION_ID` command.
Requests are not automatically replayed after a server failure.

Pass `--keep-model-server` to explicitly keep a newly launched server alive
after the owner exits. It does not transfer ownership of an existing server.
Millie never stops an independently launched server. Changing models requires
closing the existing sessions and stopping any persistent or external server.
Clearing a conversation does not release its server attachment.

Simultaneous launches coordinate startup and recognize a server that is still
loading. A failed or timed-out managed launch cleans up its own server. Advanced
setups may adjust `MILLIE_LLAMACPP_STARTUP_TIMEOUT_SECS` (default 300 seconds).

Memory profiles follow the selected server device indices, including CPU-only
selection. Cache preparation is best-effort and bounded; a new turn or
interruption cancels the current session's background warmup. Failure to prepare
a cache does not claim success or prevent ordinary inference on a healthy server.

`--oss` has been removed. Use the ordinary Millie command; `--local-provider`
remains available when selecting another provider.

## Startup configuration and request handling

- Explicit model selection updates the active user profile, including when startup
  fails. Other profiles retain their selections.
- A conflicting running model is checked before download approval; attachment
  checks again under the startup lock after files are ready.
- Managed llama-server launch and requests use `llamacpp.port` (exported as
  `MILLIE_LLAMACPP_PORT`, falling back to `MILLIE_OSS_PORT`). A conflicting remote
  API URL is rejected. External vLLM uses `vllm.base_url`.
- The chat transport requires the supplied `millie-native.jinja`. Restore a missing
  installation file or correct `MILLIE_LLAMACPP_CHAT_TEMPLATE`; explicit raw
  transport does not require a server-side template.
- Downloads time out on connection failure or inactivity, retaining `.part`
  progress for retry. `MILLIE_DOWNLOAD_CONNECT_TIMEOUT_SECS` (30) and
  `MILLIE_DOWNLOAD_IDLE_TIMEOUT_SECS` (60) accept positive seconds. There is no
  whole-file deadline, so an active large download can continue.
- Interrupted or errored streams do not complete tool calls. Cancelling a turn
  closes its request even when the server is silent. Raw length and repetition
  stops remain visible to the continuation logic.
- Optional penalty tokenization is bounded by `MILLIE_TOKENIZE_TIMEOUT_SECS`
  (10 seconds for all spans). Failure skips optional penalty spans for that call.
- Both local backends read generation settings from `[llamacpp]`; placing sampling
  keys under `[vllm]` is a configuration error. Explicit zero values are preserved.
