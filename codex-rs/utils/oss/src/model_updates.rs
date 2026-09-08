//! Model-management commands share startup's catalog and custom-source resolution.
use codex_llamacpp::model_updates::ModelSource;
use codex_llamacpp::model_updates::active_files;
use codex_llamacpp::model_updates::check_updates;
use codex_llamacpp::model_updates::hf_cache_path;
use codex_llamacpp::model_updates::install_update;
use std::io;
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub enum UpdateAction {
    Check,
    Install,
}

pub struct ModelSelection {
    pub explicit: Option<String>,
    pub remembered: Option<String>,
    pub model_hf: Option<String>,
    pub model_path: Option<std::path::PathBuf>,
}

pub async fn run(
    home: &Path,
    selection: ModelSelection,
    action: UpdateAction,
) -> io::Result<String> {
    let models = home.join("models");
    let model = selection.explicit.as_deref();
    let explicit_path = std::env::var("MILLIE_LLAMACPP_MODEL_PATH")
        .ok()
        .filter(|v| !v.trim().is_empty());
    if model.is_none() && (selection.model_path.is_some() || explicit_path.is_some()) {
        return Err(io::Error::other(
            "A manual GGUF path has no tracked update source. Name a catalog model after the command, or configure llamacpp.model_hf instead.",
        ));
    }
    let custom = selection
        .model_hf
        .or_else(|| std::env::var("MILLIE_LLAMACPP_MODEL_HF").ok());
    let mut sources = if let Some(spec) = custom.filter(|_| model.is_none()) {
        let (repo, file) = super::split_hf_spec(&spec).ok_or_else(|| {
            io::Error::other("Invalid llamacpp.model_hf; expected org/repo/file.gguf")
        })?;
        vec![ModelSource {
            path: hf_cache_path(&models, &repo, &file),
            repo,
            file,
            sha256: None,
            revision: None,
        }]
    } else {
        let selected = model.or(selection.remembered.as_deref()).ok_or_else(|| {
            io::Error::other(
                "No model selected. Specify a catalog model after check-updates or update.",
            )
        })?;
        let downloads = codex_models_manager::catalog::resolve_model_downloads(home)?;
        let entry = downloads
            .get(selected)
            .ok_or_else(|| io::Error::other(format!("Unknown catalog model `{selected}`")))?;
        super::installation::catalog_sources(
            &models,
            entry,
            matches!(action, UpdateAction::Install),
        )
        .await?
    };
    if let Some(active) = active_files(&models, &sources[0].repo, &sources[0].file)? {
        for source in &mut sources {
            if let Some(installed) = active.iter().find(|v| v.file == source.file) {
                *source = installed.clone();
            }
        }
    }
    let remote = check_updates(&sources).await?;
    let mut lines = Vec::new();
    let mut changed = false;
    for file in &remote {
        let state = if !file.source.path.is_file() {
            changed = true;
            "not installed"
        } else if file.source.sha256.as_deref() == Some(&file.sha256) {
            "up to date"
        } else {
            changed = true;
            "update available (or installed revision unrecorded)"
        };
        lines.push(format!(
            "{}/{}: {state}; {:.2} GB; revision {}",
            file.source.repo,
            file.source.file,
            file.size as f64 / 1e9,
            file.revision
        ));
    }
    if matches!(action, UpdateAction::Install) && changed {
        install_update(&models, &remote).await?;
        lines.push("Update verified and installed. It will be used on the next model-server launch. Existing model files and running sessions were preserved. Sampling settings are unchanged.".to_string());
    } else if matches!(action, UpdateAction::Check) && changed {
        lines.push("Run millie models update (with the same model/profile options) to download and install this revision.".to_string());
    }
    Ok(lines.join("\n"))
}
