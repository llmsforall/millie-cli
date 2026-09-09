//! Local `llama.cpp` OSS provider support.
//!
//! Unlike Ollama/LM Studio, `llama-server` is not expected to already be
//! running as a user-managed background service: Millie launches it itself,
//! pointed at a specific `.gguf` file, unless a health check shows one is
//! already listening on the target port (for example because another Millie
//! invocation launched it first).

mod download_range;
mod lifecycle;
pub mod model_updates;
pub use lifecycle::SUPERVISOR_ARG;
pub use lifecycle::run_supervisor;
mod chat;
mod format;
mod grammar;
mod parse;
mod settings;

pub use settings::parse_vision_setting;

pub use chat::build_chat_messages;
pub use chat::chat_tools_json;
pub use format::MEDIA_MARKER;
pub use format::collect_media;
pub use format::flatten_tool_name;
pub use format::flattened_tools_json;
pub use format::render_prompt;
pub use format::tool_param_types;
pub use format::unflatten_tool_name;
pub use grammar::tool_call_grammar;
pub use parse::ExtractedToolCalls;
pub use parse::ParsedToolCall;
pub use parse::arguments_to_json_string;
pub use parse::coerce_param_value;
pub use parse::extract_tool_calls;
pub use parse::ordered_object_pairs;

use std::io;
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use tokio::process::Command;
use tokio::time::Instant;
use tokio::time::sleep;

/// Display/config name for the model served by this provider. The actual
/// weights are whatever `llamacpp.model_path` points at; this label is what
/// the UI shows and what `model` resolves to when unset.

/// Default port `llama-server` listens on, and where Millie looks for an
/// already-running instance before launching a new one.
pub const DEFAULT_PORT: u16 = 8090;

/// Default `--ctx-size` when `llamacpp.ctx_size` is not configured. The
/// default model's KV is cheap (interval-4 attention), so 128k fits
/// comfortably in 16GB of unified memory alongside the weights.
pub const DEFAULT_CTX_SIZE: u32 = 131072;

/// Default single-GPU selection when nothing else is configured.
pub const DEFAULT_GPU: &[u32] = &[0];

const ENV_MODEL_PATH: &str = "MILLIE_LLAMACPP_MODEL_PATH";
const ENV_GPU: &str = "MILLIE_LLAMACPP_GPU";
const ENV_CTX_SIZE: &str = "MILLIE_LLAMACPP_CTX_SIZE";
const ENV_SERVER_BIN: &str = "MILLIE_LLAMACPP_SERVER_BIN";
const ENV_PARALLEL: &str = "MILLIE_LLAMACPP_PARALLEL";
const ENV_MODEL_URL: &str = "MILLIE_LLAMACPP_MODEL_URL";
const ENV_MODEL_SHA256: &str = "MILLIE_LLAMACPP_MODEL_SHA256";
const ENV_MMPROJ_PATH: &str = "MILLIE_LLAMACPP_MMPROJ_PATH";
const ENV_MMPROJ_URL: &str = "MILLIE_LLAMACPP_MMPROJ_URL";
const ENV_MMPROJ_SHA256: &str = "MILLIE_LLAMACPP_MMPROJ_SHA256";
const ENV_VISION: &str = "MILLIE_LLAMACPP_VISION";
const ENV_KV_CACHE: &str = "MILLIE_LLAMACPP_KV_CACHE";
const ENV_MMAP: &str = "MILLIE_LLAMACPP_MMAP";
const ENV_N_CPU_MOE: &str = "MILLIE_LLAMACPP_N_CPU_MOE";
/// Base directory for persisted prompt caches (MILLIE_HOME/kvcache), exported
/// by the CLI layer.
const ENV_CACHE_BASE: &str = "MILLIE_LLAMACPP_CACHE_BASE";
/// Exported by `ensure_oss_ready` once a server is confirmed healthy: the
/// ACTUAL context size of the running llama-server (from `/props`), whether
/// this process spawned it or reused one already listening. The agent side
/// reads this so its context-window accounting (auto-compaction thresholds,
/// token displays) tracks the real KV budget instead of the catalog value --
/// `model_context_window` stays the user's one knob, and this feeds reality
/// back when the machine-fit profile or a reused server clamps it.
pub const ENV_RESOLVED_CTX: &str = "MILLIE_LLAMACPP_RESOLVED_CTX";
/// The default system prompt, exported by the CLI layer, used to warm and
/// persist the base prompt cache so the first message never waits on the
/// system-prompt prefill. Empty/unset disables the warm-up.
const ENV_SYSTEM_PROMPT: &str = "MILLIE_LLAMACPP_SYSTEM_PROMPT";

/// KV cache quantization used when nothing else is configured. q8_0 halves
/// KV memory against f16 at negligible quality cost and needs flash
/// attention, which is turned on alongside it.
pub const DEFAULT_KV_CACHE: &str = "q8_0";
const ENV_TEMPERATURE: &str = "MILLIE_LLAMACPP_TEMPERATURE";
const ENV_MAX_OUTPUT_TOKENS: &str = "MILLIE_LLAMACPP_MAX_OUTPUT_TOKENS";

/// Max tokens the model may spend inside a thinking block per completion;
/// the server force-closes the block when the budget is hit. 0 disables.
pub fn thinking_budget_from_env() -> u32 {
    match std::env::var("MILLIE_LLAMACPP_THINKING_BUDGET")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
    {
        Some(v) => v,
        None => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| eprintln!("millie: WARNING: thinking_budget not set by catalog/config; using compiled fallback 1024"));
            1024
        }
    }
}

/// Settings for the loop guard: the server-side repetition stop and the
/// escalating n-gram penalty Millie applies after a detected pathology.
/// Populated from `[llamacpp]` config via env vars, like the sampling params.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PathologyParams {
    /// Send `repeat_stop` on every request so the server stops looping output.
    pub repeat_stop: bool,
    /// Penalty per extra matched token after the first pathology (subtracted from the logit).
    pub ngram_penalty_scale: f32,
    /// Added to the scale for every further consecutive failing call.
    pub ngram_penalty_step: f32,
    /// First match length that is penalized.
    pub ngram_penalty_start_n: u32,
    /// Cap on the penalty for one token.
    pub ngram_penalty_max: f32,
    /// Discard a looped generation from the conversation instead of keeping it.
    pub backtrack_on_repetition: bool,
    /// Clean retries per pathology episode before any penalty machinery
    /// engages: the failed completion is discarded and the identical request
    /// is sampled again (fresh seed, no penalties, no injected notes).
    pub resamples: u32,
}

impl Default for PathologyParams {
    fn default() -> Self {
        Self {
            repeat_stop: true,
            ngram_penalty_scale: 1.0,
            ngram_penalty_step: 0.5,
            ngram_penalty_start_n: 3,
            ngram_penalty_max: 10.0,
            backtrack_on_repetition: false,
            resamples: 0,
        }
    }
}

pub fn pathology_params_from_env() -> PathologyParams {
    fn read<T: std::str::FromStr>(key: &str, default: T) -> T {
        std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse::<T>().ok())
            .unwrap_or(default)
    }
    let d = PathologyParams::default();
    PathologyParams {
        repeat_stop: read("MILLIE_LLAMACPP_REPEAT_STOP", d.repeat_stop),
        ngram_penalty_scale: read("MILLIE_LLAMACPP_NGRAM_PENALTY_SCALE", d.ngram_penalty_scale),
        ngram_penalty_step: read("MILLIE_LLAMACPP_NGRAM_PENALTY_STEP", d.ngram_penalty_step),
        ngram_penalty_start_n: read(
            "MILLIE_LLAMACPP_NGRAM_PENALTY_START_N",
            d.ngram_penalty_start_n,
        ),
        ngram_penalty_max: read("MILLIE_LLAMACPP_NGRAM_PENALTY_MAX", d.ngram_penalty_max),
        backtrack_on_repetition: read(
            "MILLIE_LLAMACPP_BACKTRACK_ON_REPETITION",
            d.backtrack_on_repetition,
        ),
        resamples: read("MILLIE_LLAMACPP_RESAMPLES", d.resamples),
    }
}

/// Served model name to put in requests to a vLLM server (`[vllm] served_model`).
pub const DEFAULT_VLLM_MODEL: &str = "millie";

pub fn vllm_model_from_env() -> String {
    std::env::var("MILLIE_VLLM_MODEL")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_VLLM_MODEL.to_string())
}

/// Readiness for an externally started OpenAI-compatible server (vLLM): only
/// checks that `{base_url}/models` answers. Nothing is launched.
pub async fn ensure_remote_ready(base_url: &str) -> io::Result<()> {
    let client = reqwest::Client::builder()
        .timeout(HEALTH_CHECK_TIMEOUT)
        .build()
        .map_err(io::Error::other)?;
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => Ok(()),
        Ok(resp) => Err(io::Error::other(format!(
            "vLLM server at {base_url} answered {} on /models; is the Millie model served there?",
            resp.status()
        ))),
        Err(e) => Err(io::Error::other(format!(
            "no vLLM server reachable at {base_url} ({e}); start it first (see docs/config.md, \"vLLM backend\")"
        ))),
    }
}

pub fn max_output_tokens_from_env() -> u32 {
    match std::env::var(ENV_MAX_OUTPUT_TOKENS)
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
    {
        Some(v) => v,
        None => {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| eprintln!("millie: WARNING: max_output_tokens not set by catalog/config; using compiled fallback 8192"));
            8192
        }
    }
}
const ENV_MIN_P: &str = "MILLIE_LLAMACPP_MIN_P";
const ENV_TOP_K: &str = "MILLIE_LLAMACPP_TOP_K";
const ENV_TOP_P: &str = "MILLIE_LLAMACPP_TOP_P";

/// Sampling parameters for `/completion` requests. Managed models receive
/// their defaults from the catalog and explicit user configuration.
#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    pub temperature: f32,
    pub min_p: f32,
    pub top_k: i32,
    pub top_p: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.5,
            min_p: 0.001,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

/// Read sampling overrides from `MILLIE_LLAMACPP_*` environment variables.
/// Managed-model startup validates the catalog/config values before launch;
/// custom models may use the generic fallback values.
pub fn sampling_params_from_env() -> SamplingParams {
    fn read<T: std::str::FromStr>(
        key: &str,
        default: T,
        missing: &mut Vec<&'static str>,
        label: &'static str,
    ) -> T {
        match std::env::var(key)
            .ok()
            .and_then(|v| v.trim().parse::<T>().ok())
        {
            Some(v) => v,
            None => {
                missing.push(label);
                default
            }
        }
    }
    let d = SamplingParams::default();
    let mut missing = Vec::new();
    let out = SamplingParams {
        temperature: read(ENV_TEMPERATURE, d.temperature, &mut missing, "temperature"),
        min_p: read(ENV_MIN_P, d.min_p, &mut missing, "min_p"),
        top_k: read(ENV_TOP_K, d.top_k, &mut missing, "top_k"),
        top_p: read(ENV_TOP_P, d.top_p, &mut missing, "top_p"),
    };
    // Config-first rule: sampling should come from the catalog/config. The
    // compiled constants are a model-agnostic last resort; say so once so a
    // run on silent defaults is never mistaken for a configured one.
    if !missing.is_empty() {
        static WARNED: std::sync::Once = std::sync::Once::new();
        WARNED.call_once(|| {
            eprintln!(
                "millie: WARNING: sampling parameter(s) [{}] not set by catalog/config; using compiled fallback (temp {}, min_p {}, top_k {}, top_p {})",
                missing.join(", "), out.temperature, out.min_p, out.top_k, out.top_p
            );
        });
    }
    out
}

const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(2);
const STARTUP_POLL_INTERVAL: Duration = Duration::from_millis(500);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(300);
fn startup_timeout() -> Duration {
    std::env::var("MILLIE_LLAMACPP_STARTUP_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .map(Duration::from_secs)
        .unwrap_or(STARTUP_TIMEOUT)
}

#[derive(Debug, Clone)]
pub struct LlamaCppLaunchConfig {
    pub model_path: PathBuf,
    pub port: u16,
    /// GPU indices for `CUDA_VISIBLE_DEVICES`. Empty means CPU-only.
    pub gpu: Vec<u32>,
    /// Whether `gpu` came from explicit configuration (flag, config.toml,
    /// or a serving profile) rather than the built-in default. Only explicit
    /// choices override device-mask env vars the user already set.
    pub gpu_explicit: bool,
    pub ctx_size: Option<u32>,
    /// Server slots (concurrent requests); `ctx_size` is divided among
    /// them. None leaves the server's automatic default.
    pub parallel: Option<u32>,
    pub server_bin: PathBuf,
    /// Where to download the model from when `model_path` does not exist.
    pub model_url: Option<String>,
    pub model_sha256: Option<String>,
    /// Vision tower (mmproj) to serve alongside the model, with its own
    /// optional download source.
    pub mmproj_path: Option<PathBuf>,
    pub mmproj_url: Option<String>,
    pub mmproj_sha256: Option<String>,
    /// Whether to serve the vision tower. None follows mmproj availability;
    /// Some(false) forces text-only even when an mmproj file exists.
    pub vision: Option<bool>,
    /// KV cache type for both K and V ("f16", "q8_0", "q4_0"). Quantized
    /// types imply flash attention on. "f16" leaves the server defaults.
    pub kv_cache: String,
    /// Some(false) passes --no-mmap (load fully into RAM); None/Some(true)
    /// leave mmap on, which under memory pressure degrades gracefully by
    /// re-reading pages from disk instead of failing to load.
    pub mmap: Option<bool>,
    /// Number of MoE layers whose experts stay on the CPU (--n-cpu-moe),
    /// keeping attention and dense weights on the GPU. Values past the
    /// layer count mean "all experts on CPU".
    pub n_cpu_moe: Option<u32>,
}

impl LlamaCppLaunchConfig {
    /// Reads launch settings from the `CODEX_LLAMACPP_*` environment
    /// variables, following the same pattern as `CODEX_OSS_PORT`/
    /// `CODEX_OSS_BASE_URL` for the built-in Ollama/LM Studio providers: the
    /// CLI entrypoint resolves config.toml + CLI-flag overrides once, then
    /// exports the effective values so this crate doesn't need its own copy
    /// of the config-resolution logic.
    pub fn from_env() -> io::Result<Self> {
        let model_path = std::env::var(ENV_MODEL_PATH)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from)
            .ok_or_else(|| {
                io::Error::other(
                    "no llama.cpp model path configured; set `llamacpp.model_path` in \
                     config.toml or pass --llama-model-path",
                )
            })?;

        // `CODEX_LLAMACPP_PORT` is provider-specific; `CODEX_OSS_PORT` is the
        // generic OSS-provider port env var also used to build this
        // provider's `base_url` (see `create_oss_provider`), so it's honored
        // here too to keep "where requests go" and "what port we launch on"
        // in sync without the CLI layer having to set two env vars.
        let port = codex_model_provider_info::llamacpp_port();

        let (gpu, gpu_explicit) = match std::env::var(ENV_GPU) {
            // "none" (what the CLI layer exports for an empty `gpu = []`
            // list or a CPU-only profile) means CPU-only; an unset or empty
            // env var falls back to the default GPU list.
            Ok(v) if v.trim().eq_ignore_ascii_case("none") => (Vec::new(), true),
            Ok(v) if !v.trim().is_empty() => (
                v.split(',')
                    .map(|s| s.trim().parse::<u32>())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| io::Error::other(format!("invalid {ENV_GPU} value: {e}")))?,
                true,
            ),
            _ => (DEFAULT_GPU.to_vec(), false),
        };

        let ctx_size = std::env::var(ENV_CTX_SIZE)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .and_then(|v| v.parse::<u32>().ok())
            .or(Some(DEFAULT_CTX_SIZE));

        // Explicit config first; then the llama-server shipped next to the
        // millie executable (release bundles); then PATH.
        let server_bin = std::env::var(ENV_SERVER_BIN)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from)
            .or_else(|| {
                let sibling = std::env::current_exe()
                    .ok()?
                    .parent()?
                    .join(format!("llama-server{}", std::env::consts::EXE_SUFFIX));
                sibling.is_file().then_some(sibling)
            })
            .unwrap_or_else(|| PathBuf::from("llama-server"));

        let opt_env = |key: &str| std::env::var(key).ok().filter(|v| !v.trim().is_empty());
        Ok(Self {
            model_path,
            port,
            gpu,
            gpu_explicit,
            ctx_size,
            parallel: opt_env(ENV_PARALLEL).and_then(|v| v.parse::<u32>().ok()),
            server_bin,
            model_url: opt_env(ENV_MODEL_URL),
            model_sha256: opt_env(ENV_MODEL_SHA256),
            mmproj_path: opt_env(ENV_MMPROJ_PATH).map(PathBuf::from),
            mmproj_url: opt_env(ENV_MMPROJ_URL),
            mmproj_sha256: opt_env(ENV_MMPROJ_SHA256),
            vision: parse_vision_setting(opt_env(ENV_VISION).as_deref()),
            kv_cache: opt_env(ENV_KV_CACHE).unwrap_or_else(|| DEFAULT_KV_CACHE.to_string()),
            mmap: opt_env(ENV_MMAP).map(|v| matches!(v.trim(), "1" | "true" | "on")),
            n_cpu_moe: opt_env(ENV_N_CPU_MOE).and_then(|v| v.parse::<u32>().ok()),
        })
    }

    fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Directory the server persists its prompt cache into, and the validation
    /// key that ties the cached state to this exact model + settings + server
    /// build. Config-isolated: a different model or setting produces a
    /// different key (and directory), so a stale cache is never loaded. None
    /// when prompt caching cannot be set up (no cache base resolvable).
    fn prompt_cache(&self) -> Option<(PathBuf, String)> {
        use sha2::Digest;
        // Base dir: exported by the CLI layer (MILLIE_HOME/kvcache), else
        // derived next to the model file.
        let base = std::env::var(ENV_CACHE_BASE)
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from)
            .or_else(|| self.model_path.parent().map(|p| p.join("kvcache")))?;

        let server_id = self
            .server_bin
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut h = sha2::Sha256::new();
        h.update(self.model_path.to_string_lossy().as_bytes());
        h.update(self.model_sha256.clone().unwrap_or_default().as_bytes());
        h.update(format!("ctx={:?}", self.ctx_size).as_bytes());
        h.update(format!("kv={}", self.kv_cache).as_bytes());
        h.update(format!("mmap={:?}", self.mmap).as_bytes());
        h.update(format!("ncmoe={:?}", self.n_cpu_moe).as_bytes());
        h.update(format!("gpu={:?}", self.gpu).as_bytes());
        h.update(format!("vision={:?}", self.vision).as_bytes());
        h.update(format!("srv={server_id}").as_bytes());
        let key = format!("{:x}", h.finalize());
        let dir = base.join(&key[..16]);
        Some((dir, key))
    }
}

/// Prepare the local llama.cpp environment when the `llamacpp` OSS provider
/// is selected: launch `llama-server` unless one is already reachable on the
/// configured port. Takes no config parameter -- unlike Ollama/LM Studio,
/// there's no shared `Config` needed here; all launch settings come from
/// the `CODEX_LLAMACPP_*` env vars the CLI layer resolves and exports
/// before this runs (see `LlamaCppLaunchConfig::from_env`). This also keeps
/// this crate free of a `codex-core` dependency, since `codex-core` itself
/// depends on this crate for the raw-completions transport.
pub async fn ensure_oss_ready() -> io::Result<()> {
    let launch_config = LlamaCppLaunchConfig::from_env()?;
    check_running_model(
        &launch_config.model_path,
        launch_config.vision != Some(false) && launch_config.mmproj_path.is_some(),
    )
    .await?;

    ensure_local_file(
        &launch_config.model_path,
        launch_config.model_url.as_deref(),
        launch_config.model_sha256.as_deref(),
        "model",
    )
    .await?;
    // Skip the vision tower entirely when the profile disables vision: a
    // text-only profile never loads the mmproj, so there is no reason to spend
    // the bandwidth and disk fetching it (these are the memory-constrained
    // profiles where both are scarce).
    if launch_config.vision != Some(false)
        && let Some(mmproj_path) = &launch_config.mmproj_path
    {
        ensure_local_file(
            mmproj_path,
            launch_config.mmproj_url.as_deref(),
            launch_config.mmproj_sha256.as_deref(),
            "vision tower",
        )
        .await?;
    }

    // Whether a persisted base prompt cache already exists for this exact
    // config -- if so the server loads it and no warm-up is needed.
    let cache_present = launch_config
        .prompt_cache()
        .map(|(dir, _)| dir.join("millie-prompt-cache.bin").exists())
        .unwrap_or(false);

    let owned = lifecycle::connect(&launch_config).await?;

    // First launch for this config: warm the system prompt into the cache and
    // persist it, so this machine never waits on the system-prompt prefill
    // again. Best-effort -- any failure just means the first message prefills
    // normally.
    if owned && !cache_present && launch_config.prompt_cache().is_some() {
        let sys = std::env::var(ENV_SYSTEM_PROMPT).unwrap_or_default();
        if !sys.trim().is_empty()
            && tokio::time::timeout(
                Duration::from_secs(30),
                warm_prompt_cache(&launch_config.base_url(), &sys),
            )
            .await
            .is_err()
        {
            eprintln!(
                "millie: startup cache preparation timed out; continuing without a prepared cache"
            );
        }
    }
    Ok(())
}

/// Prefill the shared system-prompt prefix into the server's cache and
/// persist it to disk. Best-effort: logs and returns on any error.
///
/// The warm-up prefills exactly the templated system portion --
/// `<|im_start|>system\n{prompt}<|im_end|>\n` -- via the raw completion
/// endpoint with `n_predict: 0` (prefill only). This is the prefix every
/// real chat request begins with, so the saved state is a clean prefix the
/// next request extends. It must NOT include a user turn: a divergent tail
/// would force a trim, which recurrent models cannot do, discarding the
/// whole cached state. (The chat endpoint can't help here -- its template
/// rejects a system-only message.)
async fn warm_prompt_cache(base_url: &str, system_prompt: &str) -> PromptCacheWarmup {
    eprintln!("millie: preparing the model for first use (caching the system prompt, one time)...");
    let prefix = format!("<|im_start|>system\n{system_prompt}<|im_end|>\n");
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
    {
        Ok(c) => c,
        Err(_) => return PromptCacheWarmup::NotSaved,
    };
    let body = serde_json::json!({
        "prompt": prefix,
        "n_predict": 0,
        "cache_prompt": true,
    });
    if client
        .post(format!("{base_url}/completion"))
        .json(&body)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .is_err()
    {
        eprintln!("millie: cache warmup failed; continuing without a prepared cache");
        return PromptCacheWarmup::NotSaved;
    }
    match client
        .post(format!("{base_url}/slots/0?action=cache-save"))
        .json(&serde_json::json!({}))
        .send()
        .await
    {
        Ok(response) => {
            let saved = match response.error_for_status() {
                Ok(response) => response.json::<serde_json::Value>().await.ok(),
                Err(_) => None,
            };
            if saved
                .as_ref()
                // The runtime's cache-save endpoint reports saved states as
                // n_saved. It has no success field, and n_written is always 0.
                .and_then(|v| v.get("n_saved"))
                .and_then(serde_json::Value::as_u64)
                .is_some_and(|states| states > 0)
            {
                eprintln!("millie: model prepared -- the system prompt is cached for fast startup");
                PromptCacheWarmup::Saved
            } else {
                eprintln!("millie: no usable saved cache was confirmed; continuing");
                PromptCacheWarmup::NotSaved
            }
        }
        Err(error) => {
            eprintln!("millie: cache save failed: {error}; continuing");
            PromptCacheWarmup::NotSaved
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum PromptCacheWarmup {
    Saved,
    NotSaved,
}

/// Device names from `llama-server --list-devices`, in listed order, e.g.
/// `["CUDA0", "CUDA1"]` or `["Vulkan0", "Vulkan1", "Vulkan2"]`. A GPU index
/// is a position in this list. Empty when listing fails or finds no devices.
fn list_device_names(server_bin: &std::path::Path) -> Vec<String> {
    let out = match std::process::Command::new(server_bin)
        .arg("--list-devices")
        .output()
    {
        Ok(out) => out,
        Err(_) => return Vec::new(),
    };
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    let mut names = Vec::new();
    for line in text.lines() {
        // Device lines look like `  CUDA0: <name> (...)` -- take the token
        // before the colon when it is a backend prefix followed by digits.
        let trimmed = line.trim_start();
        let Some((head, _)) = trimmed.split_once(':') else {
            continue;
        };
        if !head.is_empty()
            && head.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            && head.chars().last().is_some_and(|c| c.is_ascii_digit())
            && head.chars().all(|c| c.is_ascii_alphanumeric())
        {
            names.push(head.to_string());
        }
    }
    names
}

/// The last `n` lines of the server log, for inlining into errors.
fn server_log_tail(n: usize) -> String {
    match std::fs::read_to_string(server_log_path()) {
        Ok(text) => {
            let lines: Vec<&str> = text.lines().collect();
            let start = lines.len().saturating_sub(n);
            lines[start..].join("\n")
        }
        Err(_) => "(no server log)".to_string(),
    }
}

/// Warm the local server's prompt cache for a conversation so the next turn
/// (or the first turn after a resume) reuses the whole history instead of
/// re-encoding it. On this hybrid SSM+attention model the cached state can
/// only be extended, never trimmed, and past-turn reasoning is stripped
/// between turns -- so the warmed prefix must be exactly the think-stripped
/// history through the last completed assistant, with nothing after it.
///
/// `base_url` is the server root (no `/v1`). `messages` is the chat-format
/// history (system + prior turns). A throwaway user turn is appended so the
/// server template renders the last assistant as a *past* turn (thinking
/// stripped); the rendered prompt is then truncated back to exactly the
/// history, and prefilled with `n_predict: 0`. Best-effort: any error is
/// ignored (the next turn just prefills normally).
pub async fn warm_conversation_cache(base_url: &str, mut messages: Vec<serde_json::Value>) {
    messages.push(serde_json::json!({"role": "user", "content": "."}));
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
    {
        Ok(c) => c,
        Err(_) => return,
    };
    let rendered = match client
        .post(format!("{base_url}/apply-template"))
        .json(&serde_json::json!({"messages": messages, "add_generation_prompt": false}))
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(r) => match r.error_for_status() {
            Ok(r) => r,
            Err(_) => return,
        },
        Err(_) => return,
    };
    let body: serde_json::Value = match rendered.json().await {
        Ok(b) => b,
        Err(_) => return,
    };
    let Some(prompt) = body.get("prompt").and_then(|p| p.as_str()) else {
        return;
    };
    // Drop the throwaway trailing user turn (and anything after it): the
    // exact stripped prefix through the last assistant is everything before
    // the final `<|im_start|>user` marker.
    let Some(cut) = prompt.rfind("<|im_start|>user") else {
        return;
    };
    let prefix = &prompt[..cut];
    // Generous cap: a full history prefill can legitimately take minutes
    // (especially CPU/Metal), so this only guards against a wedged server,
    // and it bounds how long the session-held prewarm lock can stay taken.
    let result = client
        .post(format!("{base_url}/completion"))
        .json(&serde_json::json!({"prompt": prefix, "n_predict": 0, "cache_prompt": true}))
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status);
    if let Err(error) = result {
        tracing::warn!("conversation cache preparation failed: {error}");
    }
}

/// Token count of `content` per the server's tokenizer (no special tokens
/// added). Best-effort: None if `/tokenize` is unavailable or malformed.
async fn count_prompt_tokens(base_url: &str, content: &str) -> Option<usize> {
    let base = codex_model_provider_info::local_server_root(base_url);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .ok()?;
    let body: serde_json::Value = client
        .post(format!("{base}/tokenize"))
        .json(&serde_json::json!({"content": content, "add_special": false}))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    body.get("tokens")?.as_array().map(std::vec::Vec::len)
}

/// Number of tokens in the trailing generation-prompt suffix that
/// `render_prompt` appends (assistant header + primed `<think>`). On a
/// turn-start request these are exactly the tokens after the user message, so
/// this is the offset-from-prompt-end at which the server should pin the
/// turn-boundary recurrent checkpoint. Cached for the process (constant for a
/// given tokenizer). Best-effort: None if `/tokenize` is unavailable.
pub async fn generation_prompt_suffix_tokens(base_url: &str) -> Option<usize> {
    static CACHE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    if let Some(n) = CACHE.get() {
        return Some(*n);
    }
    let n = count_prompt_tokens(base_url, format::GENERATION_PROMPT_SUFFIX).await?;
    if n == 0 {
        return None;
    }
    let _ = CACHE.set(n);
    Some(n)
}

/// Record the confirmed serving context size in [`ENV_RESOLVED_CTX`] for the
/// agent side to read (see the const's docs). No-op when unknown.
fn export_resolved_ctx(ctx: Option<u32>) {
    if let Some(ctx) = ctx {
        // SAFETY: provider bootstrap, same phase as the other launch env
        // exports (before any sampling request is issued).
        unsafe { std::env::set_var(ENV_RESOLVED_CTX, ctx.to_string()) };
    }
}

/// Total context size (--ctx-size) of the running server, from /props.
/// Best-effort: None on any failure.
async fn server_ctx_size(base_url: &str) -> Option<u32> {
    let client = reqwest::Client::builder()
        .timeout(HEALTH_CHECK_TIMEOUT)
        .build()
        .ok()?;
    let props: serde_json::Value = client
        .get(format!("{base_url}/props"))
        .send()
        .await
        .ok()?
        .json()
        .await
        .ok()?;
    // `n_ctx` in default_generation_settings is per slot, and in this
    // server each slot gets the full --ctx-size, so it compares directly.
    let per_slot = props
        .get("default_generation_settings")
        .and_then(|s| s.get("n_ctx"))
        .and_then(serde_json::Value::as_u64)?;
    u32::try_from(per_slot).ok()
}

async fn is_server_healthy(base_url: &str) -> bool {
    let client = match reqwest::Client::builder()
        .timeout(HEALTH_CHECK_TIMEOUT)
        .build()
    {
        Ok(client) => client,
        Err(_) => return false,
    };
    client
        .get(format!("{base_url}/health"))
        .send()
        .await
        .is_ok_and(|resp| resp.status().is_success())
}

fn spawn_server(launch_config: &LlamaCppLaunchConfig) -> io::Result<tokio::process::Child> {
    if !launch_config.model_path.exists() {
        return Err(io::Error::other(format!(
            "llama.cpp model path does not exist: {}",
            launch_config.model_path.display()
        )));
    }

    let mut cmd = Command::new(&launch_config.server_bin);
    // Pin the server's media marker so the raw-completions transport can
    // render matching markers into its prompts (the server randomizes the
    // marker per process otherwise). Text containing this literal is
    // sanitized client-side before serialization.
    cmd.env("LLAMA_MEDIA_MARKER", crate::format::MEDIA_MARKER);
    cmd.arg("--model")
        .arg(&launch_config.model_path)
        .arg("--port")
        .arg(launch_config.port.to_string())
        .arg("--host")
        .arg("127.0.0.1");

    // Persisted prompt cache: point the server at a config-isolated cache
    // directory and hand it the validation key. The server loads any cache
    // present there at startup (the base system-prompt state, warmed below).
    if let Some((cache_dir, cache_key)) = launch_config.prompt_cache()
        && std::fs::create_dir_all(&cache_dir).is_ok()
    {
        // llama-server joins the filename onto this path verbatim, so it
        // needs a trailing separator.
        let mut path = cache_dir.into_os_string();
        path.push(std::path::MAIN_SEPARATOR_STR);
        cmd.arg("--slot-save-path").arg(&path);
        cmd.env("MILLIE_PROMPT_CACHE_KEY", cache_key);
    }

    if let Some(ctx_size) = launch_config.ctx_size {
        cmd.arg("--ctx-size").arg(ctx_size.to_string());
    }
    // Route turns to the slot with the longest matching prompt prefix. Each
    // slot owns a separate KV cache; keeping a conversation on a matching
    // slot avoids reprocessing its prefix when multiple slots are configured.
    cmd.arg("--slot-prompt-similarity").arg("0.5");
    // One interactive user is one conversation; extra slots would also split
    // --ctx-size between them. Multi-instance setups raise this explicitly.
    let parallel = launch_config.parallel.unwrap_or(1);
    cmd.arg("--parallel").arg(parallel.to_string());

    // Recurrent-state checkpoints. On the hybrid SSM+attention model the
    // server snapshots the recurrent state 4 tokens before each prompt's end
    // -- exactly the turn boundary -- so the next turn restores it and
    // recomputes only the last assistant turn instead of the whole
    // conversation. This is an append-only chat: we only ever reuse the most
    // recent boundary, so we keep a single checkpoint (eviction is
    // oldest-first, so each turn's boundary replaces the previous one). Each
    // checkpoint is the fixed-size recurrent state (~63 MiB here; NOT a copy
    // of the attention KV, which is trimmed normally), and the stock default
    // of 32 would pin ~2 GB on a long conversation -- which matters on the
    // memory-constrained profiles.
    cmd.arg("--ctx-checkpoints").arg("1");

    // Quantized KV cache needs flash attention; "f16"/"off" leave the
    // server's own defaults (f16 KV, auto flash attention).
    let kv = launch_config.kv_cache.trim();
    if !kv.is_empty() && !kv.eq_ignore_ascii_case("f16") && !kv.eq_ignore_ascii_case("off") {
        cmd.arg("--cache-type-k").arg(kv);
        cmd.arg("--cache-type-v").arg(kv);
        cmd.arg("-fa").arg("on");
    }
    if launch_config.mmap == Some(false) {
        cmd.arg("--no-mmap");
    }
    if let Some(n_cpu_moe) = launch_config.n_cpu_moe {
        cmd.arg("--n-cpu-moe").arg(n_cpu_moe.to_string());
    }

    // Vision tower: enables image input on the server's multimodal
    // endpoints. `vision = false` forces text-only even when the file is
    // available (memory-constrained serving profiles default to that).
    if launch_config.vision == Some(false) {
        tracing::info!("vision disabled by configuration -- launching text-only");
    } else if let Some(mmproj_path) = &launch_config.mmproj_path {
        if mmproj_path.exists() {
            cmd.arg("--mmproj").arg(mmproj_path);
        } else {
            tracing::warn!(
                "mmproj path does not exist: {} -- launching text-only",
                mmproj_path.display()
            );
        }
    }

    // Chat template for the server-side chat-completions path: the server
    // renders the model's own format and parses tool calls itself (PEG
    // parser + lazy grammar). Explicit override via
    // CODEX_LLAMACPP_CHAT_TEMPLATE; default is `millie-native.jinja` shipped
    // next to the server binary. Without a template file the server falls
    // back to the GGUF-embedded template, which does not match the native
    // wire format -- so only the raw-completions transport should be used
    // in that configuration.
    // The template belongs to Millie's installation, not to whichever
    // llama-server binary happens to be in use: `server_bin` may come from
    // MILLIE_LLAMACPP_SERVER_BIN or PATH (a dev build, a system install),
    // and deriving the template path from it silently yields no template in
    // those cases -- the server then falls back to the GGUF-embedded
    // base-model template, which renders the wrong tool format and selects
    // the wrong server-side parser. Search Millie's own locations first, the
    // same way system_prompt.md is resolved, and treat the server-binary
    // directory as just one candidate (dev trees where both sit together).
    let template = {
        let explicit = std::env::var("MILLIE_LLAMACPP_CHAT_TEMPLATE")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from);
        match explicit {
            // An explicitly named template that is missing is an error to
            // surface, not a reason to fall through to one the user did not
            // choose -- keep it as the selected path so startup reports the error.
            Some(path) => Some(path),
            None => {
                let mut candidates: Vec<PathBuf> = Vec::new();
                if let Some(exe_dir) = std::env::current_exe()
                    .ok()
                    .and_then(|exe| exe.parent().map(std::path::Path::to_path_buf))
                {
                    candidates.push(exe_dir.join("millie-native.jinja"));
                }
                if let Some(home) = std::env::var_os("MILLIE_HOME") {
                    candidates.push(PathBuf::from(home).join("millie-native.jinja"));
                }
                if let Some(dir) = launch_config.server_bin.parent() {
                    candidates.push(dir.join("millie-native.jinja"));
                }
                candidates.into_iter().find(|p| p.exists())
            }
        }
    };
    let raw = std::env::var("MILLIE_LLAMACPP_TRANSPORT").as_deref() == Ok("raw");
    match template {
        Some(path) if path.is_file() => {
            cmd.arg("--jinja").arg("--chat-template-file").arg(path);
        }
        Some(path) if !raw => {
            return Err(io::Error::other(format!(
                "Required Millie chat template is missing: {}. Restore millie-native.jinja or correct MILLIE_LLAMACPP_CHAT_TEMPLATE.",
                path.display()
            )));
        }
        None if !raw => {
            return Err(io::Error::other(
                "Required millie-native.jinja was not found. Restore it beside Millie or set MILLIE_LLAMACPP_CHAT_TEMPLATE to its path.",
            ));
        }
        _ => {}
    }

    let inherited_mask = ["CUDA_VISIBLE_DEVICES", "GGML_VK_VISIBLE_DEVICES"]
        .iter()
        .any(|k| std::env::var_os(k).is_some());
    if launch_config.gpu_explicit && launch_config.gpu.is_empty() {
        // CPU-only: give llama-server no offload device at all. `--n-gpu-layers
        // 0` alone keeps the main model's layers on the CPU, but the vision
        // tower (mmproj) offloads on a separate switch and would still land on
        // whatever device the backend auto-detects. On a Vulkan build that is
        // the integrated GPU -- e.g. the Radeon in a Ryzen APU, whose Vulkan
        // heap is only ~512 MiB -- and the mmproj allocation overflows it and
        // aborts the server. `--device none` (llama.cpp: "none = don't
        // offload") keeps the model and KV off the GPU -- but NOT the vision
        // tower: the clip backend ignores the device list and grabs a GPU (and
        // even an integrated one) on its own via ggml_backend_init_by_type, so
        // a vision-enabled CPU profile (e.g. 11GB cpu-full at 22 GB) would
        // still put the mmproj on that iGPU and OOM. `--no-mmproj-offload`
        // (mmproj_use_gpu = false) forces the tower onto the CPU too. Together
        // they keep everything -- model, KV, and mmproj -- on the CPU.
        cmd.arg("--device").arg("none");
        cmd.arg("--n-gpu-layers").arg("0");
        cmd.arg("--no-mmproj-offload");
    } else if inherited_mask && !launch_config.gpu_explicit {
        // The user masked devices in their own environment and did not ask
        // for a specific index; respect their mask rather than overriding it.
        cmd.arg("--n-gpu-layers").arg("999");
    } else {
        // Select the GPU(s) by name via llama-server's own --device flag.
        // A GPU index means "the Nth device in `llama-server --list-devices`",
        // which is what the user sees. This avoids CUDA_VISIBLE_DEVICES /
        // GGML_VK_VISIBLE_DEVICES, whose numeric indices renumber the device
        // list and compose unpredictably when both are set.
        cmd.arg("--n-gpu-layers").arg("999");
        let want: Vec<u32> = if launch_config.gpu_explicit {
            launch_config.gpu.clone()
        } else {
            // No explicit choice: the built-in default is the first device.
            vec![0]
        };
        let names = list_device_names(&launch_config.server_bin);
        if names.is_empty() {
            if launch_config.gpu_explicit {
                return Err(io::Error::other(
                    "could not list GPU devices from llama-server to honor the requested \
                     GPU index; run `llama-server --list-devices` to check the GPU backend"
                        .to_string(),
                ));
            }
            // Default path with no listable devices: let llama-server pick.
        } else {
            let mut selected = Vec::new();
            for &idx in &want {
                match names.get(idx as usize) {
                    Some(name) => selected.push(name.clone()),
                    None if launch_config.gpu_explicit => {
                        return Err(io::Error::other(format!(
                            "GPU index {idx} is out of range: llama-server lists {} device(s): {}. \
                             Pick an index from `llama-server --list-devices`.",
                            names.len(),
                            names.join(", ")
                        )));
                    }
                    None => {} // default device absent: fall through
                }
            }
            if !selected.is_empty() {
                cmd.arg("--device").arg(selected.join(","));
            }
        }
    }

    // The supervisor owns this child and handles owner-pipe closure. Keep its
    // output separate from the supervisor's readiness protocol.
    cmd.stdin(Stdio::null());
    match std::fs::File::create(server_log_path()) {
        Ok(log) => {
            let log2 = log.try_clone();
            cmd.stdout(Stdio::from(log));
            match log2 {
                Ok(l) => {
                    cmd.stderr(Stdio::from(l));
                }
                Err(_) => {
                    cmd.stderr(Stdio::null());
                }
            }
        }
        Err(err) => {
            tracing::warn!(
                "could not create {}: {err}; server output discarded",
                server_log_path().display()
            );
            cmd.stdout(Stdio::null());
            cmd.stderr(Stdio::null());
        }
    }
    cmd.kill_on_drop(true);

    tracing::info!(
        "launching llama-server: model={} port={} gpu={:?}",
        launch_config.model_path.display(),
        launch_config.port,
        launch_config.gpu
    );

    // Startup failures and owner exit terminate and reap this child.
    cmd.spawn()
}

/// Make sure `path` exists, downloading it from `url` when it does not.
/// Downloads stream to `<path>.part` with HTTP-Range resume across
/// interrupted runs, verify against `sha256` when given, then move into
/// place. No-op when the file exists; missing-file-and-no-url is left for
/// `spawn_server` to report.
async fn ensure_local_file(
    path: &std::path::Path,
    url: Option<&str>,
    sha256: Option<&str>,
    label: &str,
) -> io::Result<()> {
    let client = reqwest::Client::builder()
        .connect_timeout(download_timeout("MILLIE_DOWNLOAD_CONNECT_TIMEOUT_SECS", 30))
        .read_timeout(download_timeout("MILLIE_DOWNLOAD_IDLE_TIMEOUT_SECS", 60))
        .build()
        .map_err(io::Error::other)?;
    ensure_local_file_with_client(path, url, sha256, label, &client).await
}

async fn ensure_local_file_with_client(
    path: &Path,
    url: Option<&str>,
    sha256: Option<&str>,
    label: &str,
    client: &reqwest::Client,
) -> io::Result<()> {
    use sha2::Digest;

    if path.exists() {
        return Ok(());
    }
    let Some(url) = url else {
        return Ok(());
    };
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let _download_lock =
        lifecycle::lock_file(&PathBuf::from(format!("{}.download.lock", path.display()))).await?;
    if path.exists() {
        return Ok(());
    }
    let part = PathBuf::from(format!("{}.part", path.display()));

    // Hash whatever a previous interrupted run already fetched, then resume
    // after it.
    let mut hasher = sha2::Sha256::new();
    let mut downloaded: u64 = 0;
    if part.exists() {
        use std::io::Read;
        let mut existing = std::fs::File::open(&part)?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            let n = existing.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            downloaded += n as u64;
        }
    }

    eprintln!("millie: downloading {label} ({url})");

    let mut out = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .open(&part)?;
    out.set_len(downloaded)?;
    {
        use std::io::Seek;
        out.seek(std::io::SeekFrom::Start(downloaded))?;
    }

    // Consumer links drop long transfers routinely, so one mid-body error
    // must not kill a multi-GB fetch. On any transport error, back off and
    // re-request from the current offset with a Range header; give up only
    // after several consecutive attempts that made no progress at all.
    let mut stalls: u32 = 0;
    let mut total: Option<u64> = None;
    let mut last_report = Instant::now();
    'download: loop {
        let attempt_start = downloaded;
        let mut request = client.get(url);
        if downloaded > 0 {
            request = request.header("Range", format!("bytes={downloaded}-"));
        }
        let transport_err = match request.send().await {
            Err(e) => Some(format!("request failed: {e}")),
            Ok(mut response) => {
                let status = response.status();
                let range_total = download_range::resumed_total(
                    status.as_u16(),
                    response
                        .headers()
                        .get("content-range")
                        .and_then(|v| v.to_str().ok()),
                    downloaded,
                )?;
                if status.as_u16() == 416 {
                    break 'download;
                }
                if status.as_u16() != 200 && status.as_u16() != 206 {
                    return Err(io::Error::other(format!(
                        "{label} download failed ({status}): {url}; partial download preserved"
                    )));
                }
                if downloaded > 0 && status.as_u16() != 206 {
                    downloaded = 0;
                    hasher = sha2::Sha256::new();
                    out.set_len(0)?;
                    use std::io::Seek;
                    out.seek(std::io::SeekFrom::Start(0))?;
                    total = None;
                }
                if range_total.is_some() {
                    if total.is_some() && total != range_total {
                        return Err(io::Error::other(
                            "Download size changed during resume; partial download preserved",
                        ));
                    }
                    total = range_total;
                } else if total.is_none() {
                    total = response.content_length().map(|len| len + downloaded);
                }
                let response_end = if status.as_u16() == 206 {
                    response
                        .headers()
                        .get("content-range")
                        .and_then(|v| v.to_str().ok())
                        .and_then(|v| v.split_once('/'))
                        .and_then(|(bounds, _)| bounds.split_once('-'))
                        .and_then(|(_, end)| end.parse::<u64>().ok())
                        .and_then(|end| end.checked_add(1))
                } else {
                    total
                };
                loop {
                    match response.chunk().await {
                        Ok(Some(chunk)) => {
                            if response_end.is_some_and(|end| {
                                downloaded.saturating_add(chunk.len() as u64) > end
                            }) {
                                return Err(io::Error::other(
                                    "Download body exceeds its advertised range; partial download preserved",
                                ));
                            }
                            use std::io::Write;
                            out.write_all(&chunk)?;
                            hasher.update(&chunk);
                            downloaded += chunk.len() as u64;
                            if last_report.elapsed() >= Duration::from_secs(3) {
                                last_report = Instant::now();
                                match total {
                                    Some(total) => eprintln!(
                                        "millie: {label} {:.1} / {:.1} GB",
                                        downloaded as f64 / 1e9,
                                        total as f64 / 1e9
                                    ),
                                    None => eprintln!(
                                        "millie: {label} {:.1} GB",
                                        downloaded as f64 / 1e9
                                    ),
                                }
                            }
                        }
                        Ok(None) => {
                            if total.is_some_and(|total| downloaded > total) {
                                return Err(io::Error::other(
                                    "Download exceeded its advertised size; partial download preserved",
                                ));
                            }
                            if total.is_some_and(|total| downloaded < total) {
                                if downloaded == attempt_start {
                                    break Some("response ended without progress".to_string());
                                }
                                continue 'download;
                            }
                            break 'download;
                        }
                        Err(e) => break Some(format!("interrupted: {e}")),
                    }
                }
            }
        };
        if let Some(err) = transport_err {
            if downloaded > attempt_start {
                stalls = 0;
            } else {
                stalls += 1;
            }
            if stalls >= 5 {
                return Err(io::Error::other(format!(
                    "{label} download stalled after {downloaded} bytes ({err}); rerun to resume"
                )));
            }
            let delay = std::cmp::min(
                Duration::from_millis(500u64 << stalls.min(6)),
                Duration::from_secs(30),
            );
            eprintln!(
                "millie: {label} download {err} at {downloaded} bytes; retrying in {:.1}s",
                delay.as_secs_f32()
            );
            tokio::time::sleep(delay).await;
        }
    }
    out.sync_all()?;
    drop(out);

    let digest = format!("{:x}", hasher.finalize());
    if let Some(expected) = sha256
        && !digest.eq_ignore_ascii_case(expected)
    {
        std::fs::remove_file(&part).ok();
        return Err(io::Error::other(format!(
            "{label} download checksum mismatch: expected {expected}, got {digest};                  partial file removed, rerun to retry"
        )));
    }
    std::fs::rename(&part, path)?;
    eprintln!("millie: {label} ready at {}", path.display());
    Ok(())
}

async fn wait_until_healthy(base_url: &str, child: &mut tokio::process::Child) -> io::Result<()> {
    let deadline = Instant::now() + startup_timeout();
    while Instant::now() < deadline {
        if is_server_healthy(base_url).await {
            return Ok(());
        }
        // Fail fast the moment the server process dies (out of memory, bad
        // device, ...) instead of waiting out the full timeout, and carry
        // its final output into the error.
        if let Ok(Some(status)) = child.try_wait() {
            return Err(io::Error::other(format!(
                "llama-server exited during startup ({status}). Last lines of {}:\n{}",
                server_log_path().display(),
                server_log_tail(15)
            )));
        }
        sleep(STARTUP_POLL_INTERVAL).await;
    }
    Err(io::Error::other(format!(
        "llama-server did not become ready at {base_url} within the startup timeout; its output is in {}",
        server_log_path().display()
    )))
}

/// Where the launched server's stdout/stderr go, truncated at each launch.
/// The CLI layer resolves the real Millie home and exports it as
/// MILLIE_LLAMACPP_SERVER_LOG; the env-based guessing below is only the
/// fallback for direct crate use.
fn server_log_path() -> PathBuf {
    if let Some(path) = std::env::var("MILLIE_LLAMACPP_SERVER_LOG")
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        return PathBuf::from(path);
    }
    let home = std::env::var("MILLIE_HOME")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|h| PathBuf::from(h).join(".millie"))
        })
        .or_else(|| {
            // Windows: HOME is normally unset; the profile dir fills in.
            std::env::var("USERPROFILE")
                .ok()
                .map(|h| PathBuf::from(h).join(".millie"))
        })
        .unwrap_or_else(std::env::temp_dir);
    home.join("llama-server.log")
}

#[cfg(test)]
/// Serializes every test that mutates the MILLIE_LLAMACPP_* environment
/// variables (process-global state shared by chat.rs and format.rs tests).
pub(crate) static TEST_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Refuse a conflicting running model before asking for or transferring weights.
pub async fn check_running_model(model_path: &Path, vision: bool) -> io::Result<()> {
    lifecycle::preflight(
        codex_model_provider_info::llamacpp_port(),
        model_path,
        vision,
    )
    .await
}

fn download_timeout(key: &str, default_secs: u64) -> Duration {
    Duration::from_secs(
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(default_secs),
    )
}

#[cfg(test)]
mod download_resume_tests {
    use super::PromptCacheWarmup;
    use super::ensure_local_file;
    use std::io::Read;
    use std::io::Write;
    use std::net::TcpListener;

    #[tokio::test]
    async fn failed_warmup_does_not_save_cache() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::path("/completion"))
            .respond_with(wiremock::ResponseTemplate::new(503))
            .expect(1)
            .mount(&server)
            .await;
        wiremock::Mock::given(wiremock::matchers::path("/slots/0"))
            .respond_with(wiremock::ResponseTemplate::new(200))
            .expect(0)
            .mount(&server)
            .await;
        assert_eq!(
            super::warm_prompt_cache(&server.uri(), "system").await,
            PromptCacheWarmup::NotSaved
        );
    }

    #[tokio::test]
    async fn cache_save_accepts_the_runtime_response_with_zero_n_written() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::path("/completion"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_body_json(serde_json::json!({"content": "", "stop": true})),
            )
            .expect(1)
            .mount(&server)
            .await;
        // server_task_result_slot_save_load::to_json(), as populated by
        // SERVER_TASK_TYPE_PROMPT_CACHE_SAVE: n_saved counts states, not tokens.
        wiremock::Mock::given(wiremock::matchers::path("/slots/0"))
            .and(wiremock::matchers::query_param("action", "cache-save"))
            .respond_with(
                wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "id_slot": 0,
                    "filename": "prompt-cache.bin",
                    "n_saved": 1,
                    "n_written": 0,
                    "timings": {"save_ms": 0}
                })),
            )
            .expect(1)
            .mount(&server)
            .await;
        assert_eq!(
            super::warm_prompt_cache(&server.uri(), "system").await,
            PromptCacheWarmup::Saved
        );
    }

    #[tokio::test]
    async fn cache_save_rejects_empty_malformed_and_http_error_responses() {
        for response in [
            wiremock::ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"n_saved": 0, "n_written": 0})),
            wiremock::ResponseTemplate::new(200)
                .set_body_json(serde_json::json!({"success": true})),
            wiremock::ResponseTemplate::new(200).set_body_json(serde_json::json!({"n_saved": "1"})),
            wiremock::ResponseTemplate::new(200).set_body_string("not json"),
            wiremock::ResponseTemplate::new(503).set_body_json(serde_json::json!({"n_saved": 1})),
        ] {
            let server = wiremock::MockServer::start().await;
            wiremock::Mock::given(wiremock::matchers::path("/completion"))
                .respond_with(wiremock::ResponseTemplate::new(200))
                .expect(1)
                .mount(&server)
                .await;
            wiremock::Mock::given(wiremock::matchers::path("/slots/0"))
                .and(wiremock::matchers::query_param("action", "cache-save"))
                .respond_with(response)
                .expect(1)
                .mount(&server)
                .await;
            assert_eq!(
                super::warm_prompt_cache(&server.uri(), "system").await,
                PromptCacheWarmup::NotSaved
            );
        }
    }

    #[tokio::test]
    async fn http_error_preserves_partial_download() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .respond_with(wiremock::ResponseTemplate::new(503))
            .mount(&server)
            .await;
        let dir = test_dir("http-error");
        let path = dir.join("payload.bin");
        let part = dir.join("payload.bin.part");
        std::fs::write(&part, b"already downloaded").unwrap();
        assert!(
            ensure_local_file(&path, Some(&server.uri()), None, "test")
                .await
                .is_err()
        );
        assert_eq!(std::fs::read(&part).unwrap(), b"already downloaded");
        assert!(!path.exists());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn successful_ignored_range_restarts_download() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::header("Range", "bytes=3-"))
            .respond_with(wiremock::ResponseTemplate::new(200).set_body_bytes(b"complete"))
            .expect(1)
            .mount(&server)
            .await;
        let dir = test_dir("ignored-range");
        let path = dir.join("payload.bin");
        std::fs::write(dir.join("payload.bin.part"), b"old").unwrap();
        ensure_local_file(&path, Some(&server.uri()), None, "test")
            .await
            .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"complete");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn concurrent_downloaders_fetch_once() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("GET"))
            .respond_with(
                wiremock::ResponseTemplate::new(200)
                    .set_delay(std::time::Duration::from_millis(100))
                    .set_body_bytes(b"complete"),
            )
            .expect(1)
            .mount(&server)
            .await;
        let dir = test_dir("concurrent");
        let path = dir.join("payload.bin");
        let url = server.uri();
        let (first, second) = tokio::join!(
            ensure_local_file(&path, Some(&url), None, "test"),
            ensure_local_file(&path, Some(&url), None, "test"),
        );
        first.unwrap();
        second.unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"complete");
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn stalled_download_resumes_after_inactivity_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = std::thread::spawn(move || {
            let (mut first, _) = listener.accept().unwrap();
            let mut request = [0; 4096];
            first.read(&mut request).unwrap();
            first
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\npart")
                .unwrap();
            // Keep this connection open and silent until the client times out.
            let (mut resumed, _) = listener.accept().unwrap();
            let n = resumed.read(&mut request).unwrap();
            assert!(
                String::from_utf8_lossy(&request[..n])
                    .to_ascii_lowercase()
                    .contains("range: bytes=4-")
            );
            resumed.write_all(b"HTTP/1.1 206 Partial Content\r\nContent-Length: 4\r\nContent-Range: bytes 4-7/8\r\nConnection: close\r\n\r\nrest").unwrap();
        });
        let dir = test_dir("stalled");
        let path = dir.join("payload.bin");
        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(1))
            .read_timeout(std::time::Duration::from_millis(100))
            .build()
            .unwrap();
        tokio::time::timeout(
            std::time::Duration::from_secs(5),
            super::ensure_local_file_with_client(&path, Some(&url), None, "test", &client),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(std::fs::read(&path).unwrap(), b"partrest");
        server.join().unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }

    /// Minimal HTTP file server that drops every connection after sending at
    /// most `limit` body bytes (while claiming the full remaining length),
    /// and answers 416 for a Range at or past EOF -- the two failure shapes
    /// seen on real consumer links.
    fn spawn_flaky_server(payload: Vec<u8>, limit: usize) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut s) = stream else { continue };
                let mut req = Vec::new();
                let mut buf = [0u8; 4096];
                loop {
                    let n = s.read(&mut buf).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    req.extend_from_slice(&buf[..n]);
                    if req.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let req = String::from_utf8_lossy(&req).to_ascii_lowercase();
                let size = payload.len();
                let mut start = 0usize;
                for line in req.lines() {
                    if let Some(v) = line.strip_prefix("range: bytes=") {
                        start = v.trim().trim_end_matches('-').parse().unwrap_or(0);
                    }
                }
                if start > 0 && start >= size {
                    let _ = write!(
                        s,
                        "HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */{size}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    continue;
                }
                let remaining = size - start;
                let serve = remaining.min(limit);
                if start > 0 {
                    let end = size - 1;
                    let _ = write!(
                        s,
                        "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {start}-{end}/{size}\r\nContent-Length: {remaining}\r\nConnection: close\r\n\r\n"
                    );
                } else {
                    let _ = write!(
                        s,
                        "HTTP/1.1 200 OK\r\nContent-Length: {remaining}\r\nConnection: close\r\n\r\n"
                    );
                }
                let _ = s.write_all(&payload[start..start + serve]);
                // Dropping the stream here cuts the connection mid-body
                // whenever serve < remaining.
            }
        });
        format!("http://{addr}")
    }

    fn test_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("millie-dl-test-{}-{tag}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn download_resumes_across_dropped_connections() {
        let payload: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        use sha2::Digest;
        let sha = format!("{:x}", sha2::Sha256::digest(&payload));
        // 1 MB per connection: the 3 MB fetch needs several resumed attempts.
        let url = spawn_flaky_server(payload.clone(), 1_000_000);
        let dir = test_dir("resume");
        let path = dir.join("payload.bin");
        ensure_local_file(&path, Some(&format!("{url}/f.bin")), Some(&sha), "test")
            .await
            .expect("download should survive dropped connections");
        assert_eq!(std::fs::read(&path).unwrap(), payload);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn completed_part_finalizes_via_416() {
        let payload: Vec<u8> = (0..500_000u32).map(|i| (i % 199) as u8).collect();
        use sha2::Digest;
        let sha = format!("{:x}", sha2::Sha256::digest(&payload));
        let url = spawn_flaky_server(payload.clone(), usize::MAX);
        let dir = test_dir("finalize416");
        let path = dir.join("payload.bin");
        // A previous run fetched everything but died before the rename.
        std::fs::write(format!("{}.part", path.display()), &payload).unwrap();
        ensure_local_file(&path, Some(&format!("{url}/f.bin")), Some(&sha), "test")
            .await
            .expect("complete .part should finalize via the checksum");
        assert_eq!(std::fs::read(&path).unwrap(), payload);
        std::fs::remove_dir_all(&dir).ok();
    }
}
