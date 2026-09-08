//! Explicit, resumable model updates. Files are immutable; one manifest switches a complete set.
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use std::io;
use std::path::Path;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct ModelSource {
    pub repo: String,
    pub file: String,
    pub path: PathBuf,
    pub sha256: Option<String>,
    pub revision: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstalledFile {
    pub file: String,
    pub sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstalledRevision {
    pub revision: String,
    pub files: Vec<InstalledFile>,
}

fn source_key(repo: &str, file: &str) -> String {
    format!("{:x}", Sha256::digest(format!("{repo}\0{file}")))
}

/// Cache custom files by complete repository identity, never just their basename.
pub fn hf_cache_path(models: &Path, repo: &str, file: &str) -> PathBuf {
    models
        .join("huggingface")
        .join(source_key(repo, file))
        .join("model.gguf")
}
fn update_dir(models: &Path, repo: &str, file: &str) -> PathBuf {
    models.join("updates").join(source_key(repo, file))
}
fn valid_hash(value: &str, len: usize) -> bool {
    value.len() == len && value.bytes().all(|c| c.is_ascii_hexdigit())
}

#[derive(Serialize, Deserialize)]
struct LocalFile {
    file: String,
    path: PathBuf,
}

/// Remember existing file locations without reading, copying, or claiming a remote revision.
pub async fn record_local_files(models: &Path, sources: &[ModelSource]) -> io::Result<()> {
    let first = sources
        .first()
        .ok_or_else(|| io::Error::other("No local model files"))?;
    let dir = update_dir(models, &first.repo, &first.file);
    std::fs::create_dir_all(&dir)?;
    let _guard = super::lifecycle::lock_file(&dir.join("update.lock")).await?;
    if dir.join("active.json").is_file() || dir.join("local.json").is_file() {
        return Ok(());
    }
    let files: Vec<_> = sources
        .iter()
        .map(|s| LocalFile {
            file: s.file.clone(),
            path: s.path.clone(),
        })
        .collect();
    let bytes = serde_json::to_vec_pretty(&files).map_err(io::Error::other)?;
    let temp = dir.join("local.json.new");
    std::fs::write(&temp, bytes)?;
    std::fs::rename(temp, dir.join("local.json"))?;
    Ok(())
}

/// Resolve installed paths offline; unversioned local files carry no checksum or revision.
pub fn active_files(models: &Path, repo: &str, file: &str) -> io::Result<Option<Vec<ModelSource>>> {
    let dir = update_dir(models, repo, file);
    let bytes = match std::fs::read(dir.join("active.json")) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let bytes = match std::fs::read(dir.join("local.json")) {
                Ok(bytes) => bytes,
                Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
                Err(e) => return Err(e),
            };
            let files: Vec<LocalFile> = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
            if files.first().map(|f| f.file.as_str()) != Some(file) {
                return Err(io::Error::other("Invalid local model installation record"));
            }
            return Ok(Some(
                files
                    .into_iter()
                    .map(|f| ModelSource {
                        repo: repo.to_owned(),
                        file: f.file,
                        path: f.path,
                        sha256: None,
                        revision: None,
                    })
                    .collect(),
            ));
        }
        Err(e) => return Err(e),
    };
    let active: InstalledRevision = serde_json::from_slice(&bytes).map_err(io::Error::other)?;
    if !valid_hash(&active.revision, 40) || active.files.is_empty() || active.files[0].file != file
    {
        return Err(io::Error::other(
            "Invalid installed model revision; rerun millie models update",
        ));
    }
    let mut files = Vec::new();
    for item in active.files {
        if !valid_hash(&item.sha256, 64) {
            return Err(io::Error::other("Invalid installed model checksum"));
        }
        let path = dir.join("files").join(&item.sha256);
        files.push(ModelSource {
            repo: repo.to_string(),
            file: item.file,
            path,
            sha256: Some(item.sha256),
            revision: Some(active.revision.clone()),
        });
    }
    Ok(Some(files))
}

#[derive(Clone, Debug)]
pub struct RemoteFile {
    pub source: ModelSource,
    pub revision: String,
    pub sha256: String,
    pub size: u64,
    pub url: String,
}

pub fn file_url(base: &str, repo: &str, revision: &str, file: &str) -> io::Result<reqwest::Url> {
    let mut url = reqwest::Url::parse(base).map_err(io::Error::other)?;
    let repo_parts: Vec<_> = repo.split('/').collect();
    if repo_parts.len() != 2
        || repo_parts
            .iter()
            .chain(file.split('/').collect::<Vec<_>>().iter())
            .any(|p| p.is_empty() || *p == "." || *p == "..")
    {
        return Err(io::Error::other(
            "Invalid Hugging Face repository or file path",
        ));
    }
    url.path_segments_mut()
        .map_err(|_| io::Error::other("Invalid Hugging Face base URL"))?
        .pop_if_empty()
        .extend(repo_parts)
        .push("resolve")
        .push(revision)
        .extend(file.split('/'));
    Ok(url)
}

/// Fetch metadata only. Hugging Face's resolve HEAD exposes the LFS SHA256 and commit.
pub async fn check_updates(sources: &[ModelSource]) -> io::Result<Vec<RemoteFile>> {
    let endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    check_at(&endpoint, sources).await
}
/// Read download sizes for the installed revision, or current main for a new installation.
pub async fn check_downloads(sources: &[ModelSource]) -> io::Result<Vec<RemoteFile>> {
    let endpoint =
        std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".to_string());
    let revision = sources
        .first()
        .and_then(|s| s.revision.as_deref())
        .unwrap_or("main");
    check_revision_at(&endpoint, sources, revision).await
}

async fn check_at(base: &str, sources: &[ModelSource]) -> io::Result<Vec<RemoteFile>> {
    check_revision_at(base, sources, "main").await
}

async fn check_revision_at(
    base: &str,
    sources: &[ModelSource],
    revision: &str,
) -> io::Result<Vec<RemoteFile>> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(io::Error::other)?;
    let mut revision = revision.to_owned();
    let mut result = Vec::new();
    for source in sources {
        let response = client
            .head(file_url(base, &source.repo, &revision, &source.file)?)
            .send()
            .await
            .map_err(io::Error::other)?;
        if !response.status().is_success() && !response.status().is_redirection() {
            return Err(io::Error::other(format!(
                "Cannot check {}/{}: {}",
                source.repo,
                source.file,
                response.status()
            )));
        }
        let header = |key| {
            response
                .headers()
                .get(key)
                .and_then(|v| v.to_str().ok())
                .map(|v| v.trim_matches('"').to_string())
        };
        let commit = header("x-repo-commit")
            .filter(|v| valid_hash(v, 40))
            .ok_or_else(|| {
                io::Error::other("Hugging Face did not provide an immutable revision")
            })?;
        if revision != "main" && revision != commit {
            return Err(io::Error::other(
                "Hugging Face revision changed during check",
            ));
        }
        revision = commit;
        let sha256 = header("x-linked-etag")
            .or_else(|| header("etag"))
            .filter(|v| valid_hash(v, 64))
            .ok_or_else(|| {
                io::Error::other("Hugging Face did not provide a SHA256 checksum for the model")
            })?;
        let size = header("x-linked-size")
            .or_else(|| header("content-length"))
            .and_then(|v| v.parse().ok())
            .filter(|v: &u64| *v > 0)
            .ok_or_else(|| io::Error::other("Hugging Face did not provide the model size"))?;
        result.push(RemoteFile {
            source: source.clone(),
            revision: revision.clone(),
            sha256,
            size,
            url: file_url(base, &source.repo, &revision, &source.file)?.to_string(),
        });
    }
    Ok(result)
}

/// Download and verify every file before atomically publishing a new active manifest.
/// Keeps existing files (including files mapped into a running server) untouched.
pub async fn install_update(models: &Path, files: &[RemoteFile]) -> io::Result<()> {
    install_files(models, files, true, /*initial*/ false).await
}

/// Install required startup files, recording the pinned vision revision even in text-only mode.
pub async fn install_initial(models: &Path, files: &[RemoteFile], vision: bool) -> io::Result<()> {
    install_files(models, files, vision, /*initial*/ true).await
}

async fn install_files(
    models: &Path,
    files: &[RemoteFile],
    vision: bool,
    initial: bool,
) -> io::Result<()> {
    let first = files
        .first()
        .ok_or_else(|| io::Error::other("No model files to update"))?;
    if files
        .iter()
        .any(|f| f.revision != first.revision || f.source.repo != first.source.repo)
    {
        return Err(io::Error::other(
            "An update must use one repository revision",
        ));
    }
    let dir = update_dir(models, &first.source.repo, &first.source.file);
    std::fs::create_dir_all(dir.join("files"))?;
    let _guard = super::lifecycle::lock_file(&dir.join("update.lock")).await?;
    let current = active_files(models, &first.source.repo, &first.source.file)?;
    let current_revision = current
        .as_ref()
        .and_then(|files| files[0].revision.as_deref());
    if current_revision != first.source.revision.as_deref()
        && !(initial && current_revision == Some(first.revision.as_str()))
    {
        return Err(io::Error::other(
            "Another model update completed during this check; rerun the update command to use the latest installed revision",
        ));
    }
    let mut installed = Vec::new();
    for (index, file) in files.iter().enumerate() {
        if !valid_hash(&file.sha256, 64) || !valid_hash(&file.revision, 40) {
            return Err(io::Error::other("Invalid update checksum or revision"));
        }
        installed.push(InstalledFile {
            file: file.source.file.clone(),
            sha256: file.sha256.clone(),
        });
        if index > 0 && !vision {
            continue;
        }
        let destination = dir.join("files").join(&file.sha256);
        if destination.is_file() {
            if checksum(&destination).await? != file.sha256 {
                return Err(io::Error::other(format!(
                    "Cached update {} failed verification; move that damaged file aside and retry",
                    destination.display()
                )));
            }
        } else if file.source.path.is_file() && checksum(&file.source.path).await? == file.sha256 {
            // Both files are retained. Hard links avoid duplicating unchanged multi-GB weights.
            if std::fs::hard_link(&file.source.path, &destination).is_err() {
                let reuse_part = destination.with_extension("reuse");
                std::fs::copy(&file.source.path, &reuse_part)?;
                if checksum(&reuse_part).await? != file.sha256 {
                    return Err(io::Error::other(
                        "Source changed while reusing model weights",
                    ));
                }
                std::fs::rename(reuse_part, &destination)?;
            }
        }
        super::ensure_local_file(
            &destination,
            Some(&file.url),
            Some(&file.sha256),
            &file.source.file,
        )
        .await?;
    }
    let active = InstalledRevision {
        revision: first.revision.clone(),
        files: installed,
    };
    let bytes = serde_json::to_vec_pretty(&active).map_err(io::Error::other)?;
    let temp = dir.join("active.json.new");
    let mut output = std::fs::File::create(&temp)?;
    use std::io::Write;
    output.write_all(&bytes)?;
    output.sync_all()?;
    drop(output);
    std::fs::rename(temp, dir.join("active.json"))?;
    Ok(())
}

async fn checksum(path: &Path) -> io::Result<String> {
    let path = path.to_owned();
    tokio::task::spawn_blocking(move || {
        use std::io::Read;
        let mut file = std::fs::File::open(path)?;
        let mut digest = Sha256::new();
        let mut buffer = vec![0; 1024 * 1024];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        Ok(format!("{:x}", digest.finalize()))
    })
    .await
    .map_err(io::Error::other)?
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::Mock;
    use wiremock::MockServer;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::method;
    use wiremock::matchers::path;
    fn source(file: &str) -> ModelSource {
        ModelSource {
            repo: "org/model".into(),
            file: file.into(),
            path: PathBuf::from("missing"),
            sha256: None,
            revision: None,
        }
    }
    #[test]
    fn cache_identity_includes_repository_and_full_path() {
        let dir = Path::new("models");
        let first = hf_cache_path(dir, "a/model", "model.gguf");
        assert_ne!(first, hf_cache_path(dir, "b/model", "model.gguf"));
        assert_ne!(first, hf_cache_path(dir, "a/model", "sub/model.gguf"));
        assert!(file_url("https://huggingface.co", "a/model", "main", "../model.gguf").is_err());
    }
    #[tokio::test]
    async fn check_is_metadata_only_and_pins_all_files_to_one_revision() {
        let server = MockServer::start().await;
        let commit = "a".repeat(40);
        for file in ["model.gguf", "vision.gguf"] {
            let revision = if file == "model.gguf" {
                "main"
            } else {
                &commit
            };
            Mock::given(method("HEAD"))
                .and(path(format!("/org/model/resolve/{revision}/{file}")))
                .respond_with(
                    ResponseTemplate::new(302)
                        .insert_header("x-repo-commit", commit.as_str())
                        .insert_header("x-linked-etag", "b".repeat(64))
                        .insert_header("x-linked-size", "8"),
                )
                .expect(1)
                .mount(&server)
                .await;
        }
        let files = check_at(
            &server.uri(),
            &[source("model.gguf"), source("vision.gguf")],
        )
        .await
        .unwrap();
        assert_eq!(files.len(), 2);
        assert!(
            files
                .iter()
                .all(|v| v.revision == commit && v.url.contains(&commit))
        );
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }
    #[tokio::test]
    async fn failed_update_keeps_previous_revision_and_success_switches_the_whole_set() {
        let server = MockServer::start().await;
        let dir = std::env::temp_dir().join(format!("millie-update-{}", std::process::id()));
        let old_bytes = b"old model";
        let old_hash = format!("{:x}", Sha256::digest(old_bytes));
        let active_dir = update_dir(&dir, "org/model", "model.gguf");
        std::fs::create_dir_all(active_dir.join("files")).unwrap();
        std::fs::write(active_dir.join("files").join(&old_hash), old_bytes).unwrap();
        let old = serde_json::to_vec(&InstalledRevision {
            revision: "a".repeat(40),
            files: vec![InstalledFile {
                file: "model.gguf".into(),
                sha256: old_hash,
            }],
        })
        .unwrap();
        std::fs::write(active_dir.join("active.json"), &old).unwrap();
        let mut files = Vec::new();
        for (file, bytes) in [
            ("model.gguf", b"new model".as_slice()),
            ("vision.gguf", b"new vision".as_slice()),
        ] {
            let mut previous_source = source(file);
            previous_source.revision = Some("a".repeat(40));
            files.push(RemoteFile {
                source: previous_source,
                revision: "b".repeat(40),
                sha256: format!("{:x}", Sha256::digest(bytes)),
                size: bytes.len() as u64,
                url: format!("{}/{file}", server.uri()),
            });
        }
        Mock::given(path("/model.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"new model"))
            .expect(1)
            .mount(&server)
            .await;
        let failure = Mock::given(path("/vision.gguf"))
            .respond_with(ResponseTemplate::new(503))
            .mount_as_scoped(&server)
            .await;
        assert!(install_update(&dir, &files).await.is_err());
        assert_eq!(std::fs::read(active_dir.join("active.json")).unwrap(), old);
        drop(failure);
        Mock::given(path("/vision.gguf"))
            .respond_with(ResponseTemplate::new(200).set_body_bytes(b"new vision"))
            .expect(1)
            .mount(&server)
            .await;
        install_update(&dir, &files).await.unwrap();
        let active = active_files(&dir, "org/model", "model.gguf")
            .unwrap()
            .unwrap();
        assert_eq!(active.len(), 2);
        assert_eq!(std::fs::read(&active[0].path).unwrap(), b"new model");
        assert_eq!(std::fs::read(&active[1].path).unwrap(), b"new vision");
        assert!(active_dir.join("files").read_dir().unwrap().count() >= 3);
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(test)]
mod installation_tests {
    use super::*;
    use wiremock::Mock;
    use wiremock::MockServer;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::method;
    use wiremock::matchers::path;

    #[tokio::test]
    async fn initial_download_uses_remote_hash_and_pins_undownloaded_vision() {
        let server = MockServer::start().await;
        let models = std::env::temp_dir().join(format!(
            "millie-initial-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let revision = "c".repeat(40);
        let mut sources = Vec::new();
        for (index, (name, bytes)) in [
            ("model.gguf", b"new weights".as_slice()),
            ("vision.gguf", b"new tower".as_slice()),
        ]
        .into_iter()
        .enumerate()
        {
            let sha = format!("{:x}", Sha256::digest(bytes));
            Mock::given(method("HEAD"))
                .and(path(format!(
                    "/org/model/resolve/{}/{}",
                    if index == 0 { "main" } else { &revision },
                    name
                )))
                .respond_with(
                    ResponseTemplate::new(302)
                        .insert_header("x-repo-commit", revision.as_str())
                        .insert_header("x-linked-etag", sha.as_str())
                        .insert_header("x-linked-size", bytes.len().to_string()),
                )
                .expect(1)
                .mount(&server)
                .await;
            Mock::given(method("GET"))
                .and(path(format!("/org/model/resolve/{revision}/{name}")))
                .respond_with(ResponseTemplate::new(200).set_body_bytes(bytes))
                .expect(1)
                .mount(&server)
                .await;
            sources.push(ModelSource {
                repo: "org/model".into(),
                file: name.into(),
                path: hf_cache_path(&models, "org/model", name),
                sha256: Some("a".repeat(64)),
                revision: None,
            });
        }
        let remote = check_at(&server.uri(), &sources).await.unwrap();
        install_initial(&models, &remote, /*vision*/ false)
            .await
            .unwrap();
        // A simultaneous cold launch may have resolved metadata before this installation completed.
        install_initial(&models, &remote, /*vision*/ false)
            .await
            .unwrap();
        let installed = active_files(&models, "org/model", "model.gguf")
            .unwrap()
            .unwrap();
        assert_eq!(std::fs::read(&installed[0].path).unwrap(), b"new weights");
        assert!(!installed[1].path.exists());
        assert!(
            installed
                .iter()
                .all(|f| f.revision.as_deref() == Some(revision.as_str()))
        );
        assert_eq!(
            installed[1].sha256,
            Some(format!("{:x}", Sha256::digest(b"new tower")))
        );
        assert!(
            !server
                .received_requests()
                .await
                .unwrap()
                .iter()
                .any(|r| r.method.as_str() == "GET" && r.url.path().ends_with("vision.gguf"))
        );
        // Enabling vision later uses the recorded revision, not whatever main now contains.
        let tower_url = file_url(
            &server.uri(),
            &installed[1].repo,
            installed[1].revision.as_deref().unwrap(),
            &installed[1].file,
        )
        .unwrap();
        crate::ensure_local_file(
            &installed[1].path,
            Some(tower_url.as_str()),
            installed[1].sha256.as_deref(),
            "vision",
        )
        .await
        .unwrap();
        assert_eq!(std::fs::read(&installed[1].path).unwrap(), b"new tower");
        assert_eq!(std::fs::read(&installed[0].path).unwrap(), b"new weights");
        std::fs::remove_dir_all(models).unwrap();
    }

    #[tokio::test]
    async fn local_installation_preserves_paths_without_reading_weights() {
        let models = std::env::temp_dir().join(format!(
            "millie-local-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&models).unwrap();
        let path = models.join("model.gguf");
        // Existing files retain their location and have no asserted remote identity.
        let file = std::fs::File::create(&path).unwrap();
        file.set_len(8 * 1024 * 1024).unwrap();
        let source = ModelSource {
            repo: "org/model".into(),
            file: "model.gguf".into(),
            path: path.clone(),
            sha256: None,
            revision: None,
        };
        record_local_files(&models, &[source]).await.unwrap();
        let files = active_files(&models, "org/model", "model.gguf")
            .unwrap()
            .unwrap();
        assert_eq!(files[0].path, path);
        assert_eq!(files[0].revision, None);
        assert_eq!(files[0].sha256, None);
        assert!(!hf_cache_path(&models, "org/model", "model.gguf").exists());
        std::fs::remove_dir_all(models).unwrap();
    }
}
