//! Local, opt-in log of every model call and tool call, written as JSON lines.
//!
//! Nothing here is sent anywhere. The log exists so a run can be reconstructed
//! exactly for local debugging: every model call
//! carries the full request, the raw generated text as the model produced it,
//! the parsed items, and a key that is unique within the run:
//!
//!   (run_id, thread_id, turn_id, call_ordinal)
//!
//! `call_ordinal` counts model calls per thread, so the many model calls that
//! make up one user turn are distinguishable. Tool records carry the key of the
//! model call whose response contained the tool call plus the tool's
//! `call_id`, so they join to it without guessing. Keys are asserted unique: a
//! duplicate is written as an `anomaly` record, never merged.
//!
//! Enabled by `[model_call_log] enabled = true` in config.toml or the
//! `MILLIE_MODEL_CALL_LOG=1` environment variable; the directory is
//! `[model_call_log] dir`, `MILLIE_MODEL_CALL_LOG_DIR`, or `$MILLIE_HOME/model-calls`.

use serde_json::Value;
use serde_json::json;
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

pub const ENV_ENABLED: &str = "MILLIE_MODEL_CALL_LOG";
pub const ENV_DIR: &str = "MILLIE_MODEL_CALL_LOG_DIR";

/// Identity of one model call within a run.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelCallKey {
    pub thread_id: String,
    pub turn_id: String,
    pub call_ordinal: u64,
}

impl ModelCallKey {
    fn to_json(&self) -> Value {
        json!({
            "thread_id": self.thread_id,
            "turn_id": self.turn_id,
            "call_ordinal": self.call_ordinal,
        })
    }
}

struct Logger {
    run_id: String,
    path: PathBuf,
    file: Mutex<std::fs::File>,
    seen_keys: Mutex<HashSet<ModelCallKey>>,
}

static LOGGER: OnceLock<Option<Logger>> = OnceLock::new();
static CONFIGURED: OnceLock<(bool, PathBuf)> = OnceLock::new();

/// Record the configuration chosen by config loading. The environment still
/// wins over it in both directions. Called once; later calls are ignored.
pub fn configure(enabled: bool, dir: PathBuf) {
    let _ = CONFIGURED.set((enabled, dir));
}

fn env_flag(name: &str) -> Option<bool> {
    let v = std::env::var(name).ok()?;
    match v.trim() {
        "1" | "true" | "yes" | "on" => Some(true),
        "0" | "false" | "no" | "off" | "" => Some(false),
        _ => Some(true),
    }
}

fn init() -> Option<Logger> {
    let (cfg_enabled, cfg_dir) = CONFIGURED.get().cloned().unwrap_or((false, PathBuf::new()));
    let enabled = env_flag(ENV_ENABLED).unwrap_or(cfg_enabled);
    if !enabled {
        return None;
    }
    let dir = std::env::var_os(ENV_DIR)
        .map(PathBuf::from)
        .filter(|p| !p.as_os_str().is_empty())
        .or_else(|| (!cfg_dir.as_os_str().is_empty()).then(|| cfg_dir.clone()))
        .or_else(|| {
            std::env::var_os("MILLIE_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".millie")))
                .map(|home| home.join("model-calls"))
        })?;
    if let Err(err) = std::fs::create_dir_all(&dir) {
        tracing::warn!("model call log: cannot create {}: {err}", dir.display());
        return None;
    }
    let run_id = format!("{}-{}", timestamp_ms(), std::process::id());
    let path = dir.join(format!("model-calls-{run_id}.jsonl"));
    let file = match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(f) => f,
        Err(err) => {
            tracing::warn!("model call log: cannot open {}: {err}", path.display());
            return None;
        }
    };
    tracing::info!(path = %path.display(), "model call log enabled");
    Some(Logger {
        run_id,
        path,
        file: Mutex::new(file),
        seen_keys: Mutex::new(HashSet::new()),
    })
}

fn logger() -> Option<&'static Logger> {
    LOGGER.get_or_init(init).as_ref()
}

/// Whether logging is active for this process.
pub fn enabled() -> bool {
    logger().is_some()
}

/// Path of the log file, if logging is active.
pub fn path() -> Option<&'static Path> {
    logger().map(|l| l.path.as_path())
}

fn write(logger: &Logger, kind: &str, mut record: Value) {
    if let Some(obj) = record.as_object_mut() {
        obj.insert("type".into(), json!(kind));
        obj.insert("run_id".into(), json!(logger.run_id));
        obj.insert("ts_ms".into(), json!(timestamp_ms()));
    }
    match serde_json::to_string(&record) {
        Ok(line) => {
            if let Ok(mut f) = logger.file.lock() {
                let _ = writeln!(f, "{line}");
            }
        }
        Err(err) => tracing::debug!("model call log: encode failed: {err}"),
    }
}

/// One model call: the request as sent, the raw generation, the parsed items.
#[allow(clippy::too_many_arguments)]
pub fn record_model_call(
    key: &ModelCallKey,
    transport: &str,
    request: &Value,
    raw_generation: Option<&str>,
    reasoning: &str,
    content: &str,
    tool_calls: &[Value],
    finish_reason: Option<&str>,
    usage: Option<&Value>,
    error: Option<&str>,
) {
    let Some(logger) = logger() else { return };
    if let Ok(mut seen) = logger.seen_keys.lock()
        && !seen.insert(key.clone())
    {
        write(
            logger,
            "anomaly",
            json!({
                "key": key.to_json(),
                "message": "duplicate model call key; record kept separately, not merged",
            }),
        );
    }
    write(
        logger,
        "model_call",
        json!({
            "key": key.to_json(),
            "transport": transport,
            "request": request,
            "raw_generation": raw_generation,
            "response": {
                "reasoning": reasoning,
                "content": content,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason,
                "usage": usage,
            },
            "error": error,
        }),
    );
}

/// One tool call: which model call produced it, what was asked, what came back.
pub fn record_tool_call(
    key: &ModelCallKey,
    call_id: &str,
    tool_name: &str,
    arguments: &str,
    output_preview: &str,
    success: bool,
    aborted: bool,
) {
    let Some(logger) = logger() else { return };
    write(
        logger,
        "tool_call",
        json!({
            "key": key.to_json(),
            "call_id": call_id,
            "tool": tool_name,
            "arguments": arguments,
            "output_preview": output_preview,
            "success": success,
            "aborted": aborted,
        }),
    );
}

/// A free-form record tied to a model call (corrections, repeat warnings, ...).
/// `payload` must carry everything needed to reconstruct what happened, in raw
/// form: for a correction, the original and corrected tool calls verbatim.
pub fn record_event(kind: &str, key: &ModelCallKey, call_id: Option<&str>, payload: Value) {
    let Some(logger) = logger() else { return };
    write(
        logger,
        kind,
        json!({
            "key": key.to_json(),
            "call_id": call_id,
            "payload": payload,
        }),
    );
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_json_has_all_parts() {
        let k = ModelCallKey {
            thread_id: "t".into(),
            turn_id: "u".into(),
            call_ordinal: 3,
        };
        let v = k.to_json();
        assert_eq!(v["thread_id"], "t");
        assert_eq!(v["turn_id"], "u");
        assert_eq!(v["call_ordinal"], 3);
    }

    #[test]
    fn env_flag_parses_common_spellings() {
        unsafe { std::env::set_var("MILLIE_MODEL_CALL_LOG_TEST_FLAG", "off") };
        assert_eq!(env_flag("MILLIE_MODEL_CALL_LOG_TEST_FLAG"), Some(false));
        unsafe { std::env::set_var("MILLIE_MODEL_CALL_LOG_TEST_FLAG", "1") };
        assert_eq!(env_flag("MILLIE_MODEL_CALL_LOG_TEST_FLAG"), Some(true));
        unsafe { std::env::remove_var("MILLIE_MODEL_CALL_LOG_TEST_FLAG") };
        assert_eq!(env_flag("MILLIE_MODEL_CALL_LOG_TEST_FLAG"), None);
    }
}
