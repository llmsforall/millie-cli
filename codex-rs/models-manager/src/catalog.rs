//! The Millie model catalog, read at runtime.
//!
//! Model names, display names and capabilities are not compiled into the
//! binary: they come from a JSON file with the same shape as the catalog
//! upstream ships, resolved once at startup. Adding a model or renaming one
//! is an edit to that file, not a new build. Lookup order:
//! 1. the path in the `MILLIE_MODELS` environment variable,
//! 2. `<MILLIE_HOME>/millie-models.json`,
//! 3. `millie-models.json` next to the running executable (release bundles),
//! 4. `models-manager/millie-models.json` in a source checkout (development only).
//! A missing catalog is a hard error, like a missing system prompt.

use std::path::Path;
use std::path::PathBuf;

use codex_protocol::openai_models::ModelInfo;
use codex_protocol::openai_models::ModelsResponse;

use crate::system_prompt::PERSONALITY_PLACEHOLDER;

pub const MODELS_ENV: &str = "MILLIE_MODELS";
pub const MODELS_FILE_NAME: &str = "millie-models.json";
#[cfg(any(debug_assertions, test))]
const DEV_CHECKOUT_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/", "millie-models.json");

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedCatalog {
    pub models: Vec<ModelInfo>,
    pub path: PathBuf,
}

/// Resolve and parse the catalog for the given `MILLIE_HOME`.
pub fn resolve_model_catalog(codex_home: &Path) -> std::io::Result<ResolvedCatalog> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(explicit) = std::env::var_os(MODELS_ENV) {
        let explicit = PathBuf::from(explicit);
        if !explicit.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "{MODELS_ENV} is set to {} but no such file exists",
                    explicit.display()
                ),
            ));
        }
        candidates.push(explicit);
    }
    candidates.push(codex_home.join(MODELS_FILE_NAME));
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    {
        candidates.push(exe_dir.join(MODELS_FILE_NAME));
    }
    #[cfg(any(debug_assertions, test))]
    candidates.push(PathBuf::from(DEV_CHECKOUT_PATH));

    for path in &candidates {
        match std::fs::read_to_string(path) {
            Ok(text) => {
                let response: ModelsResponse = serde_json::from_str(&text).map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("model catalog {} does not parse: {e}", path.display()),
                    )
                })?;
                if response.models.is_empty() {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        format!("model catalog {} lists no models", path.display()),
                    ));
                }
                tracing::info!(path = %path.display(), n = response.models.len(), "model catalog loaded");
                return Ok(ResolvedCatalog {
                    models: response.models,
                    path: path.clone(),
                });
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => {
                return Err(std::io::Error::new(
                    err.kind(),
                    format!("failed to read model catalog {}: {err}", path.display()),
                ));
            }
        }
    }
    let looked_in = candidates
        .iter()
        .map(|p| format!("  {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "no model catalog found. Looked for {MODELS_FILE_NAME} at:\n{looked_in}\n\
             Set {MODELS_ENV}=<file> or place the file in one of those locations."
        ),
    ))
}

/// The catalog entries with the served system prompt installed as every
/// entry's instructions (the file carries no prompt text of its own).
pub fn millie_models(system_prompt: &str, codex_home: &Path) -> std::io::Result<Vec<ModelInfo>> {
    let mut models = resolve_model_catalog(codex_home)?.models;
    for model in &mut models {
        let default_personality = model
            .model_messages
            .as_ref()
            .and_then(|messages| messages.get_personality_message(None))
            .unwrap_or_default();
        model.base_instructions =
            system_prompt.replace(PERSONALITY_PLACEHOLDER, &default_personality);
        if let Some(messages) = model.model_messages.as_mut() {
            messages.instructions_template = Some(system_prompt.to_string());
        }
    }
    Ok(models)
}

/// Slugs in catalog order; the first one is the default local model.
pub fn model_slugs(codex_home: &Path) -> std::io::Result<Vec<String>> {
    Ok(resolve_model_catalog(codex_home)?
        .models
        .into_iter()
        .map(|m| m.slug)
        .collect())
}

/// The default local model: the first catalog entry.
pub fn default_model_slug(codex_home: &Path) -> std::io::Result<String> {
    model_slugs(codex_home)?
        .into_iter()
        .next()
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "empty model catalog"))
}

/// Download metadata for one catalog model: where to fetch its GGUF files
/// and what the machine needs to run it. Parsed from the catalog file's
/// top-level `downloads` map (slug -> entry); binaries that predate this
/// section ignore it.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelDownload {
    /// Hugging Face repo id, e.g. "llmsforall/Millie-35B-A3B-11GB".
    pub repo: String,
    /// GGUF file name inside the repo (also the cached file name).
    pub model_file: String,
    #[serde(default)]
    pub model_sha256: Option<String>,
    /// Vision tower (mmproj) file in the same repo, when the model takes images.
    #[serde(default)]
    pub mmproj_file: Option<String>,
    #[serde(default)]
    pub mmproj_sha256: Option<String>,
    /// Approximate total download size in GB (model + vision tower), for
    /// display before asking the user to approve a download.
    #[serde(default)]
    pub download_gb: Option<f64>,
    /// Minimum DEDICATED memory in GB this model needs: a single GPU whose
    /// VRAM meets this runs it, while system RAM shared with the OS needs
    /// extra headroom on top (see `pick_local_model_slug`).
    #[serde(default)]
    pub min_dedicated_mem_gb: Option<f64>,
    /// Serving profiles in preference order: the first whose memory
    /// conditions this machine meets supplies the default launch settings.
    /// No profile matching means the model will not fit this machine.
    #[serde(default)]
    pub profiles: Vec<ModelProfile>,
    /// Per-model sampling defaults, editable without recompiling. Managed
    /// models require valid values from this entry or explicit user settings.
    #[serde(default)]
    pub sampling: Option<ModelSampling>,
    /// Prefer this model on Apple Silicon machines with this RAM capacity.
    #[serde(default)]
    pub recommended_apple_ram_gb: Option<f64>,
    /// Label this model as a tight fit on Apple Silicon with this RAM capacity.
    #[serde(default)]
    pub tight_apple_ram_gb: Option<f64>,
}

/// Per-model default sampling parameters, exported to the launch env at
/// bootstrap for every field the user's config/CLI left unset.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSampling {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<i64>,
    #[serde(default)]
    pub min_p: Option<f64>,
    #[serde(default)]
    pub thinking_budget: Option<u32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
}

/// One way of serving a model, tied to what the machine's memory allows:
/// e.g. everything on a big GPU, everything in ample RAM, or a hybrid
/// split. Conditions compare against total memory (VRAM of the largest
/// single GPU; total system RAM with OS headroom already baked into the
/// threshold values in the catalog).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ModelProfile {
    /// Name shown to the user and accepted by the `llamacpp.profile` pin.
    pub name: String,
    #[serde(default)]
    pub min_vram_gb: Option<f64>,
    #[serde(default)]
    pub min_ram_gb: Option<f64>,
    /// When true, this profile only applies on Apple Silicon. Used for the
    /// aggressive memory-mapped CPU tier, which is only a good experience on
    /// Macs (unified memory, Metal, guaranteed fast NVMe); other CPU-only
    /// machines fall through to a smaller model that loads fully into RAM.
    #[serde(default)]
    pub apple_silicon_only: bool,
    #[serde(default)]
    pub settings: ProfileSettings,
}

/// Launch settings a profile defaults. Every field is individually
/// overridable by config.toml / CLI flags; only unset ones fall through to
/// these values.
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct ProfileSettings {
    /// false = CPU-only serving even when a GPU is present.
    #[serde(default)]
    pub gpu: Option<bool>,
    #[serde(default)]
    pub ctx_size: Option<u32>,
    /// false = do not load the vision tower (text-only, saves memory).
    #[serde(default)]
    pub vision: Option<bool>,
    /// KV cache type for both K and V, e.g. "q8_0".
    #[serde(default)]
    pub kv_cache: Option<String>,
    /// false = load weights fully into RAM (--no-mmap).
    #[serde(default)]
    pub mmap: Option<bool>,
    /// MoE layers whose experts stay on the CPU (--n-cpu-moe); values past
    /// the layer count mean all of them.
    #[serde(default)]
    pub n_cpu_moe: Option<u32>,
}

/// What this machine has: VRAM of the largest single GPU and total system
/// RAM, in decimal GB. Either can be None when detection fails.
#[derive(Debug, Clone, Copy, Default)]
pub struct MachineMemory {
    pub vram_gb: Option<f64>,
    pub ram_gb: Option<f64>,
    /// Whether this is an Apple Silicon Mac (unified memory + Metal). Gates
    /// Apple-Silicon-only serving profiles.
    pub is_apple_silicon: bool,
}

/// Detect memory for the current device selection (may spawn subprocesses).
pub fn detected_machine_memory() -> MachineMemory {
    MachineMemory {
        vram_gb: max_gpu_vram_gb(),
        ram_gb: detected_total_mem_gb(),
        is_apple_silicon: cfg!(all(target_os = "macos", target_arch = "aarch64")),
    }
}

fn profile_fits(profile: &ModelProfile, mem: MachineMemory) -> bool {
    if let Ok(selection) = std::env::var("MILLIE_LLAMACPP_GPU") {
        if selection.eq_ignore_ascii_case("none") && profile.settings.gpu != Some(false) {
            return false;
        }
        if !selection.trim().is_empty()
            && !selection.eq_ignore_ascii_case("none")
            && profile.settings.gpu == Some(false)
        {
            return false;
        }
    }
    if profile.apple_silicon_only && !mem.is_apple_silicon {
        return false;
    }
    if let Some(need) = profile.min_vram_gb
        && !mem.vram_gb.is_some_and(|have| have >= need)
    {
        return false;
    }
    if let Some(need) = profile.min_ram_gb
        && !mem.ram_gb.is_some_and(|have| have >= need)
    {
        return false;
    }
    true
}

/// The serving profile to use for `download` on a machine with `mem`:
/// the pinned one when `pin` names one (fit is not checked -- a pin is an
/// explicit override), otherwise the first whose conditions the machine
/// meets. None means the model will not fit (or the catalog predates
/// profiles).
pub fn resolve_profile<'a>(
    download: &'a ModelDownload,
    mem: MachineMemory,
    pin: Option<&str>,
) -> Option<&'a ModelProfile> {
    if let Some(pin) = pin {
        match download.profiles.iter().find(|p| p.name == pin) {
            Some(p) => return Some(p),
            None => {
                eprintln!(
                    "millie: no serving profile named `{pin}` for this model (available: {}); using machine detection",
                    download
                        .profiles
                        .iter()
                        .map(|p| p.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
    }
    download.profiles.iter().find(|p| profile_fits(p, mem))
}

/// Whether this machine can run `download` at all: some profile fits, or --
/// for catalogs without profiles -- the dedicated-memory bound is met.
/// None when nothing can be determined.
pub fn model_fits(download: &ModelDownload, mem: MachineMemory) -> Option<bool> {
    if !download.profiles.is_empty() {
        return Some(download.profiles.iter().any(|p| profile_fits(p, mem)));
    }
    match (
        download.min_dedicated_mem_gb,
        mem.vram_gb
            .or_else(|| mem.ram_gb.map(|r| r - SYSTEM_RAM_HEADROOM_GB)),
    ) {
        (Some(need), Some(budget)) => Some(need <= budget),
        _ => None,
    }
}

/// The catalog's `downloads` section, keyed by model slug. Empty when the
/// catalog has none.
pub fn resolve_model_downloads(
    codex_home: &Path,
) -> std::io::Result<std::collections::HashMap<String, ModelDownload>> {
    let path = resolve_model_catalog(codex_home)?.path;
    let raw = std::fs::read_to_string(&path)?;
    #[derive(serde::Deserialize)]
    struct DownloadsOnly {
        #[serde(default)]
        downloads: std::collections::HashMap<String, ModelDownload>,
    }
    let parsed: DownloadsOnly = serde_json::from_str(&raw).map_err(|e| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "model catalog {} downloads section does not parse: {e}",
                path.display()
            ),
        )
    })?;
    Ok(parsed.downloads)
}

/// Total system RAM in GB, if it can be determined on this platform.
pub fn detected_total_mem_gb() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
        for line in meminfo.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().ok()?;
                return Some(kb / (1024.0 * 1024.0));
            }
        }
        None
    }
    #[cfg(target_os = "macos")]
    {
        let out = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        parse_ram_bytes(&String::from_utf8_lossy(&out.stdout))
    }
    #[cfg(target_os = "windows")]
    {
        let out = std::process::Command::new("wmic")
            .args(["computersystem", "get", "TotalPhysicalMemory", "/value"])
            .output()
            .ok()?;
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        parse_ram_bytes(text.split('=').nth(1)?)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        None
    }
}

// Use GiB consistently with Linux MemTotal and advertised RAM capacities.
#[cfg(any(target_os = "macos", target_os = "windows", test))]
fn parse_ram_bytes(value: &str) -> Option<f64> {
    let bytes: f64 = value.trim().parse().ok()?;
    Some(bytes / (1024.0 * 1024.0 * 1024.0))
}

/// Selected GPU memory in GB. Asks the llama-server binary first
/// (`--list-devices` covers every backend it was built with -- NVIDIA, AMD,
/// and Intel via Vulkan), then falls back to nvidia-smi.
///
/// Apple Silicon reports None deliberately: its "VRAM" is the same unified
/// memory as system RAM, so GPU-conditioned serving profiles (sized for a
/// dedicated card next to separate RAM) would double-count it. Macs resolve
/// through RAM-conditioned profiles and still get Metal acceleration; the
/// profile only governs the memory budget.
fn max_gpu_vram_gb() -> Option<f64> {
    if cfg!(all(target_os = "macos", target_arch = "aarch64"))
        || std::env::var("MILLIE_LLAMACPP_GPU").is_ok_and(|v| v.eq_ignore_ascii_case("none"))
    {
        return None;
    }
    llama_server_max_device_mib().map(|mib| mib * 1024.0 * 1024.0 / 1e9)
}

/// Selected device memory in MiB from `llama-server --list-devices`, whose device
/// lines look like `  CUDA0: <name> (97247 MiB, 96672 MiB free)`.
fn llama_server_max_device_mib() -> Option<f64> {
    // Same resolution order as the launcher: explicit env, then the
    // llama-server shipped next to the millie executable, then PATH.
    let server_bin = std::env::var("MILLIE_LLAMACPP_SERVER_BIN")
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
    let out = std::process::Command::new(server_bin)
        .arg("--list-devices")
        .output()
        .ok()?;
    let text =
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr);
    let devices = parse_device_memory(&text);
    selected_device_memory(
        &devices,
        &std::env::var("MILLIE_LLAMACPP_GPU").unwrap_or_else(|_| "0".into()),
    )
}

fn parse_device_memory(text: &str) -> Vec<Option<f64>> {
    let mut devices = Vec::new();
    for line in text.lines() {
        let Some((name, _)) = line.trim().split_once(':') else {
            continue;
        };
        if !name.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
            || !name.chars().last().is_some_and(|c| c.is_ascii_digit())
            || !name.chars().all(|c| c.is_ascii_alphanumeric())
        {
            continue;
        }
        let memory = line
            .rsplit_once('(')
            .and_then(|(_, rest)| rest.split_once(" MiB"))
            .and_then(|(value, _)| value.trim().parse::<f64>().ok())
            .filter(|value| value.is_finite() && *value > 0.0);
        devices.push(memory);
    }
    devices
}

/// Conservative per-device capacity for the exact device indices passed to the launcher.
fn selected_device_memory(devices: &[Option<f64>], selection: &str) -> Option<f64> {
    if selection.eq_ignore_ascii_case("none") {
        return None;
    }
    let selection = if selection.trim().is_empty() {
        "0"
    } else {
        selection
    };
    let selected: Option<Vec<f64>> = selection
        .split(',')
        .map(|index| {
            devices
                .get(index.trim().parse::<usize>().ok()?)
                .copied()
                .flatten()
        })
        .collect();
    selected?.into_iter().reduce(f64::min)
}

/// Headroom subtracted from system RAM before comparing against a model's
/// dedicated-memory requirement: RAM is shared with the OS, drivers, and
/// everything else, so a 16 GB machine does not have 16 GB for weights. A
/// dedicated GPU's VRAM is compared as-is.
const SYSTEM_RAM_HEADROOM_GB: f64 = 4.0;

/// The memory in GB this machine can dedicate to model weights: the larger
/// of a single GPU's VRAM and system RAM minus OS headroom. None when
/// nothing could be detected. Detection spawns subprocesses; it runs once
/// per process.
pub fn detected_budget_gb() -> Option<f64> {
    static BUDGET: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *BUDGET.get_or_init(|| {
        let vram = max_gpu_vram_gb();
        let shared = detected_total_mem_gb().map(|ram| ram - SYSTEM_RAM_HEADROOM_GB);
        match (vram, shared) {
            (Some(v), Some(s)) => Some(v.max(s)),
            (v, s) => v.or(s),
        }
    })
}

/// The local model this machine should serve by default: the largest catalog
/// model whose dedicated-memory requirement fits either a single GPU's VRAM
/// or system RAM minus OS headroom. Falls back to the catalog's first entry
/// (with a warning when the requirement is known not to fit) when nothing
/// fits or memory cannot be detected.
pub fn pick_local_model_slug(codex_home: &Path) -> std::io::Result<String> {
    let models = resolve_model_catalog(codex_home)?.models;
    // Explicit pin, e.g. for tests or a machine the detection misjudges.
    if let Ok(pinned) = std::env::var("MILLIE_DEFAULT_MODEL") {
        let pinned = pinned.trim();
        if models.iter().any(|m| m.slug == pinned) {
            return Ok(pinned.to_string());
        }
        if !pinned.is_empty() {
            eprintln!(
                "millie: MILLIE_DEFAULT_MODEL `{pinned}` is not in the model catalog; ignoring"
            );
        }
    }
    let downloads = resolve_model_downloads(codex_home).unwrap_or_default();
    let mem = detected_machine_memory();
    if let Some(model) = models.iter().find(|m| {
        downloads.get(&m.slug).is_some_and(|d| {
            apple_ram_matches(d.recommended_apple_ram_gb, mem) && model_fits(d, mem) == Some(true)
        })
    }) {
        return Ok(model.slug.clone());
    }
    // The largest fitting model wins; among fitting models "largest" is the
    // highest dedicated-memory requirement. When nothing fits (or nothing
    // could be detected), fall back to the largest model outright -- the
    // interactive chooser is what tells the user it will not fit, and the
    // smaller model is a reduced experience that should never be arrived at
    // silently.
    let size = |slug: &str| {
        downloads
            .get(slug)
            .and_then(|d| d.min_dedicated_mem_gb)
            .unwrap_or(0.0)
    };
    let mut best: Option<&str> = None;
    for model in &models {
        let Some(download) = downloads.get(&model.slug) else {
            continue;
        };
        if model_fits(download, mem) == Some(true)
            && best.is_none_or(|b| size(&model.slug) > size(b))
        {
            best = Some(&model.slug);
        }
    }
    match best {
        Some(slug) => Ok(slug.to_string()),
        None => models
            .iter()
            .map(|m| m.slug.as_str())
            .max_by(|a, b| size(a).total_cmp(&size(b)))
            .map(str::to_string)
            .ok_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "empty model catalog")
            }),
    }
}

fn apple_ram_matches(capacity: Option<f64>, mem: MachineMemory) -> bool {
    mem.is_apple_silicon
        && capacity
            .zip(mem.ram_gb)
            .is_some_and(|(want, actual)| (want - actual).abs() < 0.75)
}

/// Whether the catalog calls this model a tight fit on this Mac.
pub fn model_fits_tightly(download: &ModelDownload, mem: MachineMemory) -> bool {
    apple_ram_matches(download.tight_apple_ram_gb, mem) && model_fits(download, mem) == Some(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_descriptions_and_unknown_memory_preserve_indices() {
        let devices = parse_device_memory(
            "Devices:\n CUDA0: Example (TM) GPU (8192 MiB, 7000 MiB free)\n Vulkan0: Unknown (TM)\n Vulkan1: Next GPU (24576 MiB, 23000 MiB free)\n",
        );
        assert_eq!(devices, vec![Some(8192.0), None, Some(24576.0)]);
        assert_eq!(selected_device_memory(&devices, "1"), None);
        assert_eq!(selected_device_memory(&devices, "2"), Some(24576.0));
        assert_eq!(selected_device_memory(&devices, "0,1"), None);
    }

    #[test]
    fn device_memory_tracks_selection_not_largest_gpu() {
        let devices = [Some(8192.0), Some(24576.0), Some(16384.0)];
        assert_eq!(selected_device_memory(&devices, ""), Some(8192.0));
        assert_eq!(selected_device_memory(&devices, "1"), Some(24576.0));
        assert_eq!(selected_device_memory(&devices, "0,2"), Some(8192.0));
        assert_eq!(selected_device_memory(&devices, "none"), None);
        assert_eq!(selected_device_memory(&devices, "7"), None);
    }

    #[test]
    fn mac_16gb_recommendation_is_catalog_driven() {
        let downloads = shipped_catalog_downloads();
        let memory = MachineMemory {
            vram_gb: None,
            // A 16GB Mac's actual `sysctl -n hw.memsize` response.
            ram_gb: parse_ram_bytes("17179869184\n"),
            is_apple_silicon: true,
        };
        let preferred: Vec<_> = downloads
            .iter()
            .filter(|(_, d)| apple_ram_matches(d.recommended_apple_ram_gb, memory))
            .map(|(name, _)| name.as_str())
            .collect();
        assert_eq!(preferred, vec!["millie-35B-A3B-9GB"]);
        assert!(model_fits_tightly(
            &downloads["millie-35B-A3B-11GB"],
            memory
        ));
        assert!(!model_fits_tightly(
            &downloads["millie-35B-A3B-9GB"],
            memory
        ));
    }

    fn shipped_catalog_downloads() -> std::collections::HashMap<String, ModelDownload> {
        #[derive(serde::Deserialize)]
        struct DownloadsOnly {
            downloads: std::collections::HashMap<String, ModelDownload>,
        }
        let raw = std::fs::read_to_string(DEV_CHECKOUT_PATH).expect("dev catalog readable");
        serde_json::from_str::<DownloadsOnly>(&raw)
            .expect("dev catalog parses")
            .downloads
    }

    fn mem(vram_gb: Option<f64>, ram_gb: Option<f64>) -> MachineMemory {
        MachineMemory {
            vram_gb,
            ram_gb,
            is_apple_silicon: false,
        }
    }

    fn mac(ram_gb: Option<f64>) -> MachineMemory {
        // Apple Silicon reports no discrete VRAM (unified memory).
        MachineMemory {
            vram_gb: None,
            ram_gb,
            is_apple_silicon: true,
        }
    }

    #[test]
    fn shipped_catalog_profiles_parse_and_ladder_resolves() {
        let downloads = shipped_catalog_downloads();
        let big = downloads.get("millie-35B-A3B-11GB").expect("11GB entry");
        assert_eq!(
            big.profiles
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>(),
            vec![
                "metal-full",
                "metal-16gb",
                "gpu-full",
                "cpu-full",
                "hybrid",
                "deep-hybrid",
                "cpu-compact"
            ],
        );

        let pick = |m: MachineMemory| resolve_profile(big, m, None).map(|p| p.name.as_str());
        // A 16 GB GPU reports ~17 decimal GB; a 16 GB RAM machine ~17.
        assert_eq!(pick(mem(Some(17.0), Some(34.0))), Some("gpu-full"));
        // Ample-RAM CPU machine loads fully into RAM (no-mmap).
        assert_eq!(pick(mem(None, Some(34.0))), Some("cpu-full"));
        // Apple Silicon reports no VRAM, so Macs resolve via the
        // apple_silicon_only Metal profiles: ample unified memory gets the
        // full config, a 16 GB Mac the memory-mapped 16 GB config.
        assert_eq!(pick(mac(Some(25.0))), Some("metal-full"));
        assert_eq!(pick(mac(Some(17.0))), Some("metal-16gb"));
        // A 16 GB non-Mac CPU machine cannot cache 11 GB of weights: the
        // mmap CPU tier starts at 20 GB RAM, so this machine routes to the
        // 9GB model instead.
        assert_eq!(pick(mem(None, Some(17.0))), None);
        assert_eq!(model_fits(big, mem(None, Some(17.0))), Some(false));
        assert_eq!(model_fits(big, mac(Some(17.0))), Some(true));
        // 8 GB gaming GPU next to 16 GB RAM (hybrid is GPU-backed, not gated).
        assert_eq!(pick(mem(Some(8.6), Some(17.0))), Some("hybrid"));
        // 4 GB GPU next to 16 GB RAM.
        assert_eq!(pick(mem(Some(4.2), Some(17.0))), Some("deep-hybrid"));
        // Below the floor: nothing fits.
        assert_eq!(pick(mem(Some(4.2), Some(12.0))), None);
        assert_eq!(model_fits(big, mem(None, Some(12.0))), Some(false));
    }

    #[test]
    fn nine_gb_is_the_default_on_a_16gb_non_mac_cpu_machine() {
        // cpu-compact is not Apple-gated, but the 11GB's starts at 20 GB
        // RAM; on a 16 GB Linux machine without a GPU the 9GB fits (fully
        // in RAM, no offload) and wins the pick over the 7GB.
        let downloads = shipped_catalog_downloads();
        let small = downloads.get("millie-35B-A3B-7GB").expect("7GB entry");
        let mid = downloads.get("millie-35B-A3B-9GB").expect("9GB entry");
        let big = downloads.get("millie-35B-A3B-11GB").expect("11GB entry");
        let m = mem(None, Some(17.0));
        assert_eq!(model_fits(big, m), Some(false));
        assert_eq!(model_fits(mid, m), Some(true));
        let p = resolve_profile(mid, m, None).expect("9GB profile");
        assert_eq!(p.name, "cpu-compact");
        assert_eq!(p.settings.mmap, Some(false));
        assert_eq!(p.settings.n_cpu_moe, None);
        assert_eq!(model_fits(small, m), Some(true));
        // 7GB never memory-maps: it always fits in RAM at its floor.
        for prof in &small.profiles {
            assert_eq!(prof.settings.mmap, Some(false), "profile {}", prof.name);
        }
    }

    #[test]
    fn seven_gb_model_profiles_default_32k() {
        let downloads = shipped_catalog_downloads();
        let small = downloads.get("millie-35B-A3B-7GB").expect("7GB entry");
        for profile in &small.profiles {
            assert_eq!(
                profile.settings.ctx_size,
                Some(32768),
                "profile {}",
                profile.name
            );
        }
    }

    #[test]
    fn pinned_profile_wins_even_when_it_does_not_fit() {
        let downloads = shipped_catalog_downloads();
        let big = downloads.get("millie-35B-A3B-11GB").expect("11GB entry");
        let picked = resolve_profile(big, mem(None, Some(12.0)), Some("cpu-compact"));
        assert_eq!(picked.map(|p| p.name.as_str()), Some("cpu-compact"));
        // An unknown pin falls back to detection.
        let picked = resolve_profile(big, mem(None, Some(34.0)), Some("no-such-profile"));
        assert_eq!(picked.map(|p| p.name.as_str()), Some("cpu-full"));
    }

    #[test]
    fn undetectable_memory_fits_nothing() {
        let downloads = shipped_catalog_downloads();
        let big = downloads.get("millie-35B-A3B-11GB").expect("11GB entry");
        assert!(resolve_profile(big, mem(None, None), None).is_none());
        assert_eq!(model_fits(big, mem(None, None)), Some(false));
    }
}

#[cfg(test)]
mod sampling_validation_tests {
    #[test]
    fn sampling_typos_fail_and_deliberate_zero_survives() {
        assert!(serde_json::from_str::<super::ModelSampling>(r#"{"temperture":1.0}"#).is_err());
        let sampling: super::ModelSampling =
            serde_json::from_str(r#"{"temperature":1.0,"min_p":0.0}"#).unwrap();
        assert_eq!(sampling.min_p, Some(0.0));
    }
}
