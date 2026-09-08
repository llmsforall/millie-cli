//! Resolve installed sources offline; fetch revision metadata only for approved missing files.
use codex_llamacpp::model_updates::ModelSource;
use codex_llamacpp::model_updates::active_files;
use codex_llamacpp::model_updates::check_updates;
use codex_llamacpp::model_updates::file_url;
use codex_llamacpp::model_updates::hf_cache_path;
use codex_llamacpp::model_updates::install_initial;
use std::io;
use std::path::Path;

pub(crate) async fn catalog_sources(
    models: &Path,
    download: &codex_models_manager::catalog::ModelDownload,
    record: bool,
) -> io::Result<Vec<ModelSource>> {
    if let Some(active) = active_files(models, &download.repo, &download.model_file)? {
        return Ok(active);
    }
    let mut sources = Vec::new();
    for file in std::iter::once(&download.model_file).chain(download.mmproj_file.iter()) {
        let cached = hf_cache_path(models, &download.repo, file);
        let existing = models.join(file);
        let path = if existing.is_file() { existing } else { cached };
        sources.push(ModelSource {
            repo: download.repo.clone(),
            file: file.clone(),
            path,
            sha256: None,
            revision: None,
        });
    }
    if record && sources[0].path.is_file() {
        codex_llamacpp::model_updates::record_local_files(models, &sources).await?;
        return active_files(models, &download.repo, &download.model_file)?
            .ok_or_else(|| io::Error::other("Missing local installation record"));
    }
    Ok(sources)
}

pub(crate) fn missing(sources: &[ModelSource], vision: bool) -> bool {
    sources
        .iter()
        .enumerate()
        .any(|(index, file)| (index == 0 || vision) && !file.path.is_file())
}

pub(crate) fn require_known_vision_revision(
    sources: &[ModelSource],
    vision: bool,
) -> io::Result<()> {
    if vision
        && sources[0].path.is_file()
        && sources[0].revision.is_none()
        && missing(sources, vision)
    {
        return Err(io::Error::other(
            "The installed model has no recorded remote revision and its vision tower is missing. Existing model files were preserved. Use --no-vision to keep using it, provide a matching local tower, or explicitly run millie models update to install a matching model and tower.",
        ));
    }
    Ok(())
}

pub(crate) async fn install_missing(
    models: &Path,
    sources: Vec<ModelSource>,
    vision: bool,
) -> io::Result<Vec<ModelSource>> {
    if !missing(&sources, vision) || sources[0].revision.is_some() {
        // An installed revision supplies immutable repair URLs below. Never check main on ordinary launches.
        return Ok(sources);
    }
    require_known_vision_revision(&sources, vision)?;
    let remote = check_updates(&sources).await?;
    install_initial(models, &remote, vision).await?;
    active_files(models, &sources[0].repo, &sources[0].file)?
        .ok_or_else(|| io::Error::other("Model installation did not publish its revision"))
}

pub(crate) fn export_sources(sources: &[ModelSource]) -> io::Result<()> {
    let endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    for (index, source) in sources.iter().enumerate() {
        let prefix = if index == 0 {
            "MILLIE_LLAMACPP_MODEL"
        } else {
            "MILLIE_LLAMACPP_MMPROJ"
        };
        let url = file_url(
            &endpoint,
            &source.repo,
            source.revision.as_deref().unwrap_or("main"),
            &source.file,
        )?;
        // SAFETY: provider bootstrap, before launching the server or issuing requests.
        unsafe {
            std::env::set_var(format!("{prefix}_PATH"), &source.path);
            std::env::set_var(format!("{prefix}_URL"), url.as_str());
            if let Some(sha256) = &source.sha256 {
                std::env::set_var(format!("{prefix}_SHA256"), sha256);
            } else {
                std::env::remove_var(format!("{prefix}_SHA256"));
            }
        }
    }
    Ok(())
}
