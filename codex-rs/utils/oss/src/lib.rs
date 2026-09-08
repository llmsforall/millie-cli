mod installation;
pub mod model_updates;
mod sampling_validation;
mod selection;
pub use selection::select_startup_model;
// OSS provider utilities shared between TUI and exec.

use codex_core::config::Config;
use codex_model_provider_info::LLAMACPP_OSS_PROVIDER_ID;
use codex_model_provider_info::LMSTUDIO_OSS_PROVIDER_ID;
use codex_model_provider_info::OLLAMA_OSS_PROVIDER_ID;
use codex_model_provider_info::VLLM_OSS_PROVIDER_ID;

/// Returns the default model for a given OSS provider.
pub fn get_default_model_for_oss_provider(
    provider_id: &str,
    codex_home: &std::path::Path,
) -> Option<String> {
    match provider_id {
        LMSTUDIO_OSS_PROVIDER_ID => Some(codex_lmstudio::DEFAULT_OSS_MODEL.to_string()),
        OLLAMA_OSS_PROVIDER_ID => Some(codex_ollama::DEFAULT_OSS_MODEL.to_string()),
        // llama.cpp has no named-model-registry concept: the model is a
        // fixed local .gguf file (`llamacpp.model_path`), so this name is a
        // display/config label, not something to resolve or pull. Without it
        // the UI falls back to codex's built-in default model name.
        // llama.cpp picks the largest catalog model this machine's memory
        // fits (falling back to the catalog's first entry); vLLM serves
        // whatever the catalog lists first.
        LLAMACPP_OSS_PROVIDER_ID => {
            codex_models_manager::catalog::pick_local_model_slug(codex_home).ok()
        }
        VLLM_OSS_PROVIDER_ID => codex_models_manager::catalog::default_model_slug(codex_home).ok(),
        _ => None,
    }
}

/// Ensures the specified OSS provider is ready (models downloaded, service reachable).
pub async fn ensure_oss_provider_ready(
    provider_id: &str,
    config: &Config,
) -> Result<(), std::io::Error> {
    match provider_id {
        LMSTUDIO_OSS_PROVIDER_ID => {
            codex_lmstudio::ensure_oss_ready(config)
                .await
                .map_err(|e| std::io::Error::other(format!("OSS setup failed: {e}")))?;
        }
        OLLAMA_OSS_PROVIDER_ID => {
            codex_ollama::ensure_responses_supported(&config.model_provider).await?;
            codex_ollama::ensure_oss_ready(config)
                .await
                .map_err(|e| std::io::Error::other(format!("OSS setup failed: {e}")))?;
        }
        LLAMACPP_OSS_PROVIDER_ID => {
            let port = codex_model_provider_info::llamacpp_port();
            for base in [
                config.model_provider.base_url.clone(),
                std::env::var("MILLIE_OSS_BASE_URL").ok(),
            ]
            .into_iter()
            .flatten()
            {
                let root = codex_model_provider_info::local_server_root(&base);
                if root != format!("http://127.0.0.1:{port}")
                    && root != format!("http://localhost:{port}")
                {
                    return Err(std::io::Error::other(format!(
                        "Managed llama-server uses local port {port}, but a conflicting API URL is configured: {base}. Set llamacpp.port and remove the conflicting model provider base_url or MILLIE_OSS_BASE_URL. For an external vLLM server use the vllm provider and vllm.base_url."
                    )));
                }
            }
            prepare_llamacpp_download_env(config).await?;
            codex_llamacpp::ensure_oss_ready()
                .await
                .map_err(|e| std::io::Error::other(format!("llama.cpp setup failed: {e}")))?;
        }
        VLLM_OSS_PROVIDER_ID => {
            // Same config-first sampling as the llamacpp provider: millie's
            // vLLM transport reads the same MILLIE_LLAMACPP_* sampling env, so
            // export the catalog's per-model block here too -- otherwise a
            // vLLM-backed run silently falls to the compiled fallbacks.
            apply_model_sampling_for_config(config)?;
            let base_url = config.model_provider.base_url.clone().unwrap_or_else(|| {
                format!(
                    "http://localhost:{}/v1",
                    codex_model_provider_info::DEFAULT_VLLM_PORT
                )
            });
            codex_llamacpp::ensure_remote_ready(&base_url).await?;
        }
        _ => {
            // Unknown provider, skip setup
        }
    }
    Ok(())
}

fn env_is_unset(key: &str) -> bool {
    std::env::var(key)
        .map(|v| v.trim().is_empty())
        .unwrap_or(true)
}

/// "org/repo/file.gguf" -> (repo id, file path inside the repo).
fn split_hf_spec(spec: &str) -> Option<(String, String)> {
    let mut parts = spec.trim().trim_start_matches('/').splitn(3, '/');
    let org = parts.next()?;
    let repo = parts.next()?;
    let file = parts.next()?;
    if org.is_empty() || repo.is_empty() || file.is_empty() {
        return None;
    }
    Some((format!("{org}/{repo}"), file.to_string()))
}

/// When no explicit `llamacpp.model_path` is configured, resolve which model
/// to serve, make sure any multi-gigabyte download has user approval, and
/// export where the files live in the download cache (`$MILLIE_HOME/models/`)
/// plus the URLs and checksums `codex_llamacpp::ensure_oss_ready` needs to
/// fetch anything missing.
///
/// Cached files need no approval. Missing catalog files require `--download`
/// (`MILLIE_DOWNLOAD_APPROVED=1`) or a separate interactive download confirmation.
/// Remembering a model selection never implicitly approves a download. A running
/// conflicting model is refused before either approval or transfer.
async fn prepare_llamacpp_download_env(config: &Config) -> Result<(), std::io::Error> {
    if !env_is_unset("MILLIE_LLAMACPP_MODEL_PATH") {
        return Ok(());
    }
    let codex_home = config.codex_home.as_path();
    let models_dir = codex_home.join("models");

    if let Ok(spec) = std::env::var("MILLIE_LLAMACPP_MODEL_HF")
        && !spec.trim().is_empty()
    {
        match split_hf_spec(&spec) {
            Some((repo, file)) => {
                let path = codex_llamacpp::model_updates::hf_cache_path(&models_dir, &repo, &file);
                let sources =
                    codex_llamacpp::model_updates::active_files(&models_dir, &repo, &file)?
                        .unwrap_or_else(|| {
                            vec![codex_llamacpp::model_updates::ModelSource {
                                repo,
                                file,
                                path,
                                sha256: None,
                                revision: None,
                            }]
                        });
                let vision = codex_llamacpp::parse_vision_setting(
                    std::env::var("MILLIE_LLAMACPP_VISION").ok().as_deref(),
                ) != Some(false)
                    && !env_is_unset("MILLIE_LLAMACPP_MMPROJ_PATH");
                codex_llamacpp::check_running_model(&sources[0].path, vision).await?;
                // Explicit custom-source configuration approves its download.
                let sources = installation::install_missing(&models_dir, sources, false).await?;
                installation::export_sources(&sources)?;
                return Ok(());
            }
            None => {
                eprintln!(
                    "millie: llamacpp.model_hf is not of the form org/repo/file.gguf: {spec}"
                );
            }
        }
    }

    let downloads = codex_models_manager::catalog::resolve_model_downloads(codex_home)?;
    if downloads.is_empty() {
        return Ok(());
    }
    let slug = config.model.clone().unwrap_or_default();
    if !downloads.contains_key(&slug) {
        return Err(std::io::Error::other(format!(
            "Unknown model `{slug}`. Run `millie --model select` to choose a model."
        )));
    }
    let Some(download) = downloads.get(&slug) else {
        return Ok(());
    };
    apply_serving_profile(download, config.model_context_window)?;
    apply_model_sampling(download)?;
    let sources = installation::catalog_sources(&models_dir, download, /*record*/ true).await?;
    let vision = codex_llamacpp::parse_vision_setting(
        std::env::var("MILLIE_LLAMACPP_VISION").ok().as_deref(),
    ) != Some(false)
        && download.mmproj_file.is_some();
    codex_llamacpp::check_running_model(&sources[0].path, vision).await?;
    installation::require_known_vision_revision(&sources, vision)?;
    if installation::missing(&sources, vision) && !download_approved(codex_home, &slug) {
        let missing = if std::io::IsTerminal::is_terminal(&std::io::stdin())
            && std::io::IsTerminal::is_terminal(&std::io::stderr())
        {
            let remote = codex_llamacpp::model_updates::check_downloads(&sources).await?;
            remote
                .iter()
                .enumerate()
                .filter(|(index, f)| (*index == 0 || vision) && !f.source.path.is_file())
                .map(|(_, f)| format!("{} ({:.2} GB)", f.source.file, f.size as f64 / 1e9))
                .collect::<Vec<_>>()
        } else {
            sources
                .iter()
                .enumerate()
                .filter(|(index, f)| (*index == 0 || vision) && !f.path.is_file())
                .map(|(_, f)| f.file.clone())
                .collect::<Vec<_>>()
        };
        selection::approve_download(&slug, &missing)?;
    }
    let sources = installation::install_missing(&models_dir, sources, vision).await?;
    installation::export_sources(&sources)?;

    // Export the resolved system prompt for this model so the launcher can
    // warm and persist the base prompt cache. This is exactly the
    // base_instructions the chat transport sends (the shared system-prompt
    // template with the model's default personality filled in), so the warmed
    // token prefix matches real requests. Best-effort: any failure just skips
    // the warm-up.
    if let Ok(sp) = codex_models_manager::system_prompt::resolve_system_prompt(codex_home)
        && let Ok(models) = codex_models_manager::catalog::millie_models(&sp.text, codex_home)
        && let Some(model) = models.iter().find(|m| m.slug == slug)
        && !model.base_instructions.trim().is_empty()
    {
        // SAFETY: startup path, before anything reading this is spawned.
        unsafe {
            std::env::set_var("MILLIE_LLAMACPP_SYSTEM_PROMPT", &model.base_instructions);
        }
    }
    Ok(())
}

/// Fill launch env vars from the machine's serving profile, then print one
/// line saying what was resolved. Only vars the config/CLI layer left unset
/// are touched, so the precedence is CLI flag > config.toml > profile >
/// built-in default, per individual setting. A `llamacpp.profile` pin
/// (exported as MILLIE_LLAMACPP_PROFILE) picks the profile by name instead
/// of machine detection.
fn apply_serving_profile(
    download: &codex_models_manager::catalog::ModelDownload,
    user_context_window: Option<i64>,
) -> std::io::Result<()> {
    let mem = codex_models_manager::catalog::detected_machine_memory();
    let pin = std::env::var("MILLIE_LLAMACPP_PROFILE").ok();
    let pin = pin.as_deref().map(str::trim).filter(|v| !v.is_empty());
    if let Some(pin) = pin
        && !download.profiles.iter().any(|profile| profile.name == pin)
    {
        return Err(std::io::Error::other(format!(
            "Unknown serving profile `{pin}`. Available profiles: {}",
            download
                .profiles
                .iter()
                .map(|p| p.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    let Some(profile) = codex_models_manager::catalog::resolve_profile(download, mem, pin) else {
        if !download.profiles.is_empty() {
            eprintln!(
                "millie: no serving profile fits this machine (VRAM {}, RAM {}); the model will likely not load. Settings can still be forced in config.toml.",
                mem.vram_gb
                    .map(|v| format!("~{v:.0} GB"))
                    .unwrap_or_else(|| "unknown".into()),
                mem.ram_gb
                    .map(|v| format!("~{v:.0} GB"))
                    .unwrap_or_else(|| "unknown".into()),
            );
        }
        return Ok(());
    };
    let s = &profile.settings;
    // Each export happens only when the config/CLI layer left the var
    // unset; `desc` collects what the user actually got, tagging values
    // that came from config rather than the profile. `show` renders the raw
    // env value readably (booleans as on/off).
    let mut desc: Vec<String> = Vec::new();
    let mut apply =
        |key: &str, profile_value: Option<String>, label: &str, show: fn(&str) -> String| {
            if env_is_unset(key) {
                if let Some(value) = profile_value {
                    // SAFETY: startup path, before anything reading these is spawned.
                    unsafe { std::env::set_var(key, &value) };
                    desc.push(format!("{label} {}", show(&value)));
                }
            } else if let Ok(configured) = std::env::var(key) {
                desc.push(format!("{label} {} (config)", show(&configured)));
            }
        };
    let verbatim = |v: &str| v.to_string();
    let on_off = |v: &str| match v.trim() {
        "1" | "true" | "on" => "on".to_string(),
        _ => "off".to_string(),
    };
    // On Apple Silicon the profile's gpu on/off is not applied: memory is
    // unified, so "CPU-budget" profiles should still run Metal-accelerated.
    // An explicit `gpu = []` in config.toml still forces CPU-only.
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        apply(
            "MILLIE_LLAMACPP_GPU",
            s.gpu.map(|on| {
                if on {
                    "0".to_string()
                } else {
                    "none".to_string()
                }
            }),
            "gpu",
            verbatim,
        );
    }
    // The user's one context knob is `model_context_window`; the profile's
    // ctx_size is the machine-fit cap, not a setting of its own. The server
    // context becomes min of the two (each optional), unless the user forced
    // `llamacpp.ctx_size` explicitly (env already set -> apply() keeps it).
    // After launch the confirmed server value is fed back to the agent side
    // via MILLIE_LLAMACPP_RESOLVED_CTX, so the two can never disagree.
    let ctx_pref = effective_ctx_pref(s.ctx_size, user_context_window);
    apply(
        "MILLIE_LLAMACPP_CTX_SIZE",
        ctx_pref.map(|v| v.to_string()),
        "ctx",
        verbatim,
    );
    apply(
        "MILLIE_LLAMACPP_VISION",
        s.vision
            .map(|on| if on { "1".to_string() } else { "0".to_string() }),
        "vision",
        on_off,
    );
    apply(
        "MILLIE_LLAMACPP_KV_CACHE",
        s.kv_cache.clone(),
        "kv",
        verbatim,
    );
    apply(
        "MILLIE_LLAMACPP_MMAP",
        s.mmap
            .map(|on| if on { "1".to_string() } else { "0".to_string() }),
        "mmap",
        on_off,
    );
    apply(
        "MILLIE_LLAMACPP_N_CPU_MOE",
        s.n_cpu_moe.map(|v| v.to_string()),
        "cpu-experts",
        verbatim,
    );
    let origin = if pin.is_some() { "pinned" } else { "detected" };
    eprintln!(
        "millie: serving profile {} ({origin}): {}",
        profile.name,
        desc.join(", ")
    );
    Ok(())
}

/// Whether downloading `slug` is already approved without asking: the
/// `MILLIE_DOWNLOAD_APPROVED` env (exec's `--download`), or the user having
/// written this model into config.toml themselves.
fn download_approved(codex_home: &std::path::Path, slug: &str) -> bool {
    if std::env::var("MILLIE_DOWNLOAD_APPROVED")
        .map(|v| v.trim() == "1")
        .unwrap_or(false)
    {
        return true;
    }
    let _ = (codex_home, slug);
    false
}

fn fit_label(fits: Option<bool>) -> &'static str {
    match fits {
        Some(true) => " -- fits this machine",
        Some(false) => " -- will not fit this machine  (not recommended)",
        None => "",
    }
}

/// One-time terminal chooser: lists the catalog models with download sizes
/// and whether they fit this machine, persists the choice to config.toml,
/// and returns the chosen slug. Errors (cleanly refusing the run) when the
/// user cancels.
fn choose_model_interactively(
    codex_home: &std::path::Path,
    config_path: &std::path::Path,
    recommended: &str,
) -> Result<String, std::io::Error> {
    let models = codex_models_manager::catalog::resolve_model_catalog(codex_home)?.models;
    let downloads = codex_models_manager::catalog::resolve_model_downloads(codex_home)?;
    let mem = codex_models_manager::catalog::detected_machine_memory();

    let entries: Vec<_> = models
        .iter()
        .filter_map(|m| downloads.get(&m.slug).map(|d| (m, d)))
        .collect();
    if entries.is_empty() {
        return Err(std::io::Error::other(
            "model catalog has no downloadable models",
        ));
    }

    eprintln!();
    eprintln!("Select the model for this and future launches.");
    match (mem.vram_gb, mem.ram_gb) {
        (Some(v), Some(r)) => eprintln!("Detected memory: ~{v:.0} GB GPU, ~{r:.0} GB RAM."),
        (None, Some(r)) => eprintln!("Detected memory: no GPU found, ~{r:.0} GB RAM."),
        _ => {}
    }
    let mut default_index: Option<usize> = None;
    let mut any_fits = false;
    for (i, (model, download)) in entries.iter().enumerate() {
        let fits = codex_models_manager::catalog::model_fits(download, mem);
        if fits == Some(true) {
            any_fits = true;
        }
        let recommended_marker = if model.slug == recommended && fits == Some(true) {
            default_index = Some(i);
            "  (recommended)"
        } else {
            ""
        };
        let size = download
            .download_gb
            .map(|gb| format!("{gb:.1} GB download"))
            .unwrap_or_else(|| "size unknown".to_string());
        eprintln!(
            "  [{}] {} -- {}{}{}",
            i + 1,
            model.display_name,
            size,
            if codex_models_manager::catalog::model_fits_tightly(download, mem) {
                " -- fits, but tight"
            } else {
                fit_label(fits)
            },
            recommended_marker
        );
    }
    if !any_fits {
        eprintln!("No Millie model fits this machine's memory.");
    }
    match default_index {
        Some(i) => eprint!("Use which model? [{}]/number/q: ", i + 1),
        None => eprint!("Use which model? number/q: "),
    }

    let mut line = String::new();
    std::io::stdin().read_line(&mut line)?;
    let line = line.trim();
    let chosen_index = if line.is_empty() {
        match default_index {
            Some(i) => i,
            None => return Err(download_refusal(codex_home)),
        }
    } else if line.eq_ignore_ascii_case("q") {
        return Err(download_refusal(codex_home));
    } else {
        match line.parse::<usize>() {
            Ok(n) if n >= 1 && n <= entries.len() => n - 1,
            _ => return Err(download_refusal(codex_home)),
        }
    };
    let chosen = entries[chosen_index].0.slug.clone();

    codex_core::config::edit::ConfigEditsBuilder::for_config_path(config_path)
        .set_model(Some(&chosen), None)
        .apply_blocking()
        .map_err(|e| {
            std::io::Error::other(format!("could not save model choice to config.toml: {e}"))
        })?;
    eprintln!("Saved model = \"{chosen}\" to {}.", config_path.display());
    Ok(chosen)
}

/// The error a non-interactive (or cancelled) run gets instead of a silent
/// multi-gigabyte download.
fn download_refusal(codex_home: &std::path::Path) -> std::io::Error {
    let mut lines = vec!["a Millie model needs to be downloaded first. To approve:".to_string()];
    if let Ok(downloads) = codex_models_manager::catalog::resolve_model_downloads(codex_home) {
        let mem = codex_models_manager::catalog::detected_machine_memory();
        for (slug, d) in &downloads {
            let size = d
                .download_gb
                .map(|gb| format!("{gb:.1} GB"))
                .unwrap_or_else(|| "?".to_string());
            lines.push(format!(
                "  {slug}: {size}{}",
                fit_label(codex_models_manager::catalog::model_fits(d, mem))
            ));
        }
    }
    lines.push("pass --download to approve the selected model download".to_string());
    std::io::Error::other(lines.join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_default_model_for_provider_lmstudio() {
        let result = get_default_model_for_oss_provider(
            LMSTUDIO_OSS_PROVIDER_ID,
            std::path::Path::new("/nonexistent"),
        );
        assert_eq!(result.as_deref(), Some(codex_lmstudio::DEFAULT_OSS_MODEL));
    }

    #[test]
    fn test_get_default_model_for_provider_ollama() {
        let result = get_default_model_for_oss_provider(
            OLLAMA_OSS_PROVIDER_ID,
            std::path::Path::new("/nonexistent"),
        );
        assert_eq!(result.as_deref(), Some(codex_ollama::DEFAULT_OSS_MODEL));
    }

    #[test]
    fn test_get_default_model_for_provider_unknown() {
        let result = get_default_model_for_oss_provider(
            "unknown-provider",
            std::path::Path::new("/nonexistent"),
        );
        assert_eq!(result, None);
    }
}

/// Resolve the catalog entry for the configured (or machine-picked) model and
/// export its sampling block. Used by provider arms that do not go through
/// `prepare_llamacpp_download_env` (vLLM: the server is external, but millie's
/// sampling must still come from the same catalog). Quietly does nothing when
/// no catalog entry matches -- the backend's compiled-fallback warning then
/// makes the situation visible.
fn apply_model_sampling_for_config(config: &Config) -> std::io::Result<()> {
    let codex_home = config.codex_home.as_path();
    let downloads = codex_models_manager::catalog::resolve_model_downloads(codex_home)?;
    let slug = config.model.clone().unwrap_or_default();
    if let Some(download) = downloads.get(&slug) {
        apply_model_sampling(download)?;
    }
    Ok(())
}

/// Export the catalog's per-model sampling defaults into the launch env for
/// every field the config/CLI layer left unset (same precedence as serving
/// profiles: CLI flag > config.toml > catalog). Catalog-managed sampling
/// must be complete and valid; generic custom sources retain their fallback.
fn apply_model_sampling(
    download: &codex_models_manager::catalog::ModelDownload,
) -> std::io::Result<()> {
    sampling_validation::validate(download)?;
    let Some(s) = download.sampling.as_ref() else {
        return Ok(());
    };
    let mut desc: Vec<String> = Vec::new();
    let mut apply = |key: &str, value: Option<String>, label: &str| {
        if env_is_unset(key) {
            if let Some(v) = value {
                // SAFETY: startup path, before anything reading these is spawned.
                unsafe { std::env::set_var(key, &v) };
                desc.push(format!("{label} {v}"));
            }
        } else if let Ok(configured) = std::env::var(key) {
            desc.push(format!("{label} {configured} (config)"));
        }
    };
    apply(
        "MILLIE_LLAMACPP_TEMPERATURE",
        s.temperature.map(|v| v.to_string()),
        "temp",
    );
    apply(
        "MILLIE_LLAMACPP_TOP_P",
        s.top_p.map(|v| v.to_string()),
        "top_p",
    );
    apply(
        "MILLIE_LLAMACPP_TOP_K",
        s.top_k.map(|v| v.to_string()),
        "top_k",
    );
    apply(
        "MILLIE_LLAMACPP_MIN_P",
        s.min_p.map(|v| v.to_string()),
        "min_p",
    );
    apply(
        "MILLIE_LLAMACPP_THINKING_BUDGET",
        s.thinking_budget.map(|v| v.to_string()),
        "think-budget",
    );
    apply(
        "MILLIE_LLAMACPP_MAX_OUTPUT_TOKENS",
        s.max_output_tokens.map(|v| v.to_string()),
        "max-out",
    );
    if !desc.is_empty() {
        eprintln!("millie: model sampling (catalog): {}", desc.join(", "));
    }
    Ok(())
}

/// min of the machine-fit profile cap and the user's `model_context_window`,
/// treating each as optional. The profile cap protects small machines from a
/// large default window; the user's knob lowers (never raises past the cap)
/// what the server allocates.
fn effective_ctx_pref(profile_cap: Option<u32>, user_context_window: Option<i64>) -> Option<u32> {
    let user = user_context_window.and_then(|w| u32::try_from(w).ok());
    match (profile_cap, user) {
        (Some(p), Some(u)) => Some(p.min(u)),
        (p, u) => p.or(u),
    }
}

#[cfg(test)]
mod ctx_pref_tests {
    use super::effective_ctx_pref;

    #[test]
    fn profile_cap_clamps_large_user_window() {
        assert_eq!(effective_ctx_pref(Some(32768), Some(131072)), Some(32768));
    }

    #[test]
    fn user_window_lowers_below_cap() {
        assert_eq!(effective_ctx_pref(Some(131072), Some(65536)), Some(65536));
    }

    #[test]
    fn either_alone_passes_through() {
        assert_eq!(effective_ctx_pref(Some(32768), None), Some(32768));
        assert_eq!(effective_ctx_pref(None, Some(65536)), Some(65536));
        assert_eq!(effective_ctx_pref(None, None), None);
    }
}
