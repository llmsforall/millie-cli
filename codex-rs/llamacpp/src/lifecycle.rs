//! Cross-process startup coordination and first-session server ownership.

use std::fs::File;
use std::io;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::time::Duration;

use tokio::io::AsyncBufReadExt;
use tokio::process::Command;

use super::LlamaCppLaunchConfig;

pub const SUPERVISOR_ARG: &str = "__millie_model_server";

/// Acquire an OS-managed lock without blocking the async executor. Lock files
/// are never deleted: unlinking a locked inode would allow competing owners.
pub(crate) async fn lock_file(path: &Path) -> io::Result<File> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(path)?;
    let deadline = tokio::time::Instant::now() + super::startup_timeout();
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(file),
            Err(std::fs::TryLockError::WouldBlock) => {}
            Err(std::fs::TryLockError::Error(error)) => return Err(error),
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(io::Error::other(
                "Another Millie startup or download is still in progress; retry after it finishes.",
            ));
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn endpoint_lock(port: u16) -> std::path::PathBuf {
    // Port ownership is machine-wide, including clients with different homes.
    #[cfg(unix)]
    let root = std::path::PathBuf::from("/tmp");
    #[cfg(windows)]
    let root = std::env::var_os("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| std::path::PathBuf::from(r"C:\ProgramData"));
    #[cfg(not(any(unix, windows)))]
    let root = std::env::temp_dir();
    root.join(format!("millie-llama-server-{port}.lock"))
}

pub(crate) async fn connect(config: &LlamaCppLaunchConfig) -> io::Result<bool> {
    let mut command = Command::new(std::env::current_exe()?);
    command
        .arg(SUPERVISOR_ARG)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());
    #[cfg(windows)]
    command.creation_flags(0x00000208); // DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    let mut child = command.spawn()?;
    let pipe = child
        .stdin
        .take()
        .ok_or_else(|| io::Error::other("Missing owner pipe"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| io::Error::other("Missing supervisor response"))?;
    let mut reader = tokio::io::BufReader::new(stdout);
    let mut line = String::new();
    let response = tokio::time::timeout(
        super::startup_timeout() * 2 + Duration::from_secs(40),
        reader.read_line(&mut line),
    )
    .await;
    match response {
        Ok(Ok(n)) if n > 0 => {}
        _ => {
            return Err(io::Error::other(
                "Model server startup stopped or timed out. Retry this session to attempt loading again.",
            ));
        }
    }
    let result: serde_json::Value = serde_json::from_str(&line).map_err(io::Error::other)?;
    if let Some(error) = result.get("error").and_then(serde_json::Value::as_str) {
        return Err(io::Error::other(error.to_owned()));
    }
    if result.get("owned").and_then(serde_json::Value::as_bool) == Some(true) {
        // The OS closes this pipe even on SIGKILL or terminal closure. The
        // independent supervisor then kills and reaps its own server child.
        static OWNERS: std::sync::Mutex<Vec<tokio::process::ChildStdin>> =
            std::sync::Mutex::new(Vec::new());
        OWNERS
            .lock()
            .map_err(|_| io::Error::other("Owner registry poisoned"))?
            .push(pipe);
    }
    tokio::spawn(async move {
        let _ = child.wait().await;
    });
    super::export_resolved_ctx(
        super::server_ctx_size(&config.base_url())
            .await
            .or(config.ctx_size),
    );
    Ok(result.get("owned").and_then(serde_json::Value::as_bool) == Some(true))
}

/// Entry point for the small server supervisor, dispatched before CLI parsing.
pub async fn run_supervisor() -> io::Result<()> {
    let (closed_tx, mut closed_rx) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let mut byte = [0];
        while std::io::stdin().read(&mut byte).is_ok_and(|n| n > 0) {}
        let _ = closed_tx.send(());
    });
    let config = LlamaCppLaunchConfig::from_env()?;
    let path = endpoint_lock(config.port);
    let gate = tokio::select! {
        result = lock_file(&path) => result?,
        _ = &mut closed_rx => return Ok(()),
    };
    let startup = async {
        if closed_rx.try_recv().is_ok() {
            return Err(io::Error::other("Owning session exited during startup"));
        }
        let client = reqwest::Client::builder()
            .timeout(super::HEALTH_CHECK_TIMEOUT)
            .build()
            .map_err(io::Error::other)?;
        match client
            .get(format!("{}/health", config.base_url()))
            .send()
            .await
        {
            Ok(response) => {
                if !response.status().is_success() {
                    // Managed cold starts hold the gate until ready. Waiting here
                    // would hold up an existing owner's shutdown if its server
                    // became unhealthy; external loading servers are left alone.
                    return Err(io::Error::other(
                        "Existing model server is loading or unhealthy. Retry after it is ready; no replacement was launched.",
                    ));
                }
                validate_model(&config).await?;
                Ok(None)
            }
            Err(error) if error.is_connect() => {
                eprintln!(
                    "millie: loading the model server; log: {}",
                    super::server_log_path().display()
                );
                let mut child = super::spawn_server(&config)?;
                let base_url = config.base_url();
                let readiness = tokio::select! {
                    result = super::wait_until_healthy(&base_url, &mut child) => result,
                    _ = &mut closed_rx => Err(io::Error::other("Owning session exited during startup")),
                };
                if let Err(error) = readiness {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    return Err(error);
                }
                if let Err(error) = validate_model(&config).await {
                    let _ = child.kill().await;
                    let _ = child.wait().await;
                    return Err(error);
                }
                Ok(Some(child))
            }
            Err(error) => Err(io::Error::other(format!(
                "Could not determine model server status: {error}. No replacement was launched."
            ))),
        }
    };
    let child = match startup.await {
        Ok(value) => value,
        Err(error) => {
            println!("{}", serde_json::json!({"error": error.to_string()}));
            std::io::stdout().flush()?;
            return Ok(());
        }
    };
    let Some(mut child) = child else {
        println!("{}", serde_json::json!({"owned": false}));
        std::io::stdout().flush()?;
        return Ok(());
    };
    println!("{}", serde_json::json!({"owned": true}));
    if let Err(error) = std::io::stdout().flush() {
        let _ = child.kill().await;
        let _ = child.wait().await;
        return Err(error);
    }
    drop(gate);
    let persistent = std::env::var("MILLIE_LLAMACPP_KEEP_ALIVE").as_deref() == Ok("1");
    if persistent {
        child.wait().await?;
    } else {
        tokio::select! {
            _ = &mut closed_rx => {
                let _gate = lock_file(&path).await?;
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
            result = child.wait() => { result?; }
        }
    }
    Ok(())
}

async fn validate_model(config: &LlamaCppLaunchConfig) -> io::Result<()> {
    validate_model_path(
        &config.base_url(),
        &config.model_path,
        config.vision != Some(false) && config.mmproj_path.is_some(),
    )
    .await
}

async fn validate_model_path(base_url: &str, model_path: &Path, vision: bool) -> io::Result<()> {
    let props: serde_json::Value = reqwest::Client::builder()
        .timeout(super::HEALTH_CHECK_TIMEOUT)
        .build()
        .map_err(io::Error::other)?
        .get(format!("{base_url}/props"))
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(io::Error::other)?
        .json()
        .await
        .map_err(io::Error::other)?;
    let actual_vision = props
        .pointer("/modalities/vision")
        .and_then(serde_json::Value::as_bool)
        .ok_or_else(|| {
            io::Error::other(
                "The existing server does not report its vision mode; cannot safely attach.",
            )
        })?;
    if actual_vision != vision {
        return Err(io::Error::other(format!(
            "A model server with different vision mode is already running (running: {}, requested: {}). Close existing Millie sessions before changing vision mode. An independently launched server must be stopped by its owner.",
            if actual_vision { "on" } else { "off" },
            if vision { "on" } else { "off" }
        )));
    }
    let actual = props.get("model_path").and_then(serde_json::Value::as_str)
        .ok_or_else(|| io::Error::other("The existing server does not report its model path; cannot safely verify the selected model."))?;
    let actual = Path::new(actual);
    if actual
        .canonicalize()
        .unwrap_or_else(|_| actual.to_path_buf())
        != model_path
            .canonicalize()
            .unwrap_or_else(|_| model_path.to_path_buf())
    {
        return Err(io::Error::other(format!(
            "A different model is already running at {} ({}). Close existing Millie sessions before launching another model. An independently launched server must be stopped by its owner.",
            base_url,
            actual.display()
        )));
    }
    Ok(())
}

/// Check under the same gate as attachment/shutdown, then release it before downloads.
pub(crate) async fn preflight(port: u16, model_path: &Path, vision: bool) -> io::Result<()> {
    let _gate = lock_file(&endpoint_lock(port)).await?;
    let base_url = format!("http://127.0.0.1:{port}");
    let client = reqwest::Client::builder()
        .timeout(super::HEALTH_CHECK_TIMEOUT)
        .build()
        .map_err(io::Error::other)?;
    match client.get(format!("{base_url}/health")).send().await {
        Ok(response) if response.status().is_success() => {
            validate_model_path(&base_url, model_path, vision).await
        }
        Ok(_) => Err(io::Error::other(
            "Existing model server is loading or unhealthy. Retry after it is ready; no download or replacement was started.",
        )),
        Err(error) if error.is_connect() => Ok(()),
        Err(error) => Err(io::Error::other(format!(
            "Could not determine model server status: {error}; no download was started."
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::Mock;
    use wiremock::MockServer;
    use wiremock::ResponseTemplate;
    use wiremock::matchers::path;

    #[tokio::test]
    async fn attachment_requires_the_same_vision_mode_in_both_directions() {
        let server = MockServer::start().await;
        for actual in [false, true] {
            let response = Mock::given(path("/props"))
                .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({
                    "model_path": "/same/model.gguf", "modalities": { "vision": actual }
                })))
                .mount_as_scoped(&server)
                .await;
            assert!(
                validate_model_path(&server.uri(), Path::new("/same/model.gguf"), actual)
                    .await
                    .is_ok()
            );
            let error = validate_model_path(&server.uri(), Path::new("/same/model.gguf"), !actual)
                .await
                .unwrap_err();
            assert!(error.to_string().contains("different vision mode"));
            drop(response);
        }
    }
}
