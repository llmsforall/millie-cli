use bytes::Bytes;
use http::HeaderMap;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use crate::Request;
use crate::RequestBody;

static LOGGER: OnceLock<Option<Arc<LlmTrafficLogger>>> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

struct LlmTrafficLogger {
    run_id: String,
    file: Mutex<std::fs::File>,
    clean_file: Mutex<std::fs::File>,
    pending_requests: Mutex<HashMap<u64, Value>>,
    exchange_metadata: Mutex<HashMap<u64, Value>>,
    turn_indices: Mutex<HashMap<String, u64>>,
    next_turn_indices: Mutex<HashMap<String, u64>>,
}

#[derive(Clone)]
pub struct HttpStreamResponseRecorder {
    inner: Arc<Mutex<HttpStreamResponseState>>,
}

struct HttpStreamResponseState {
    exchange_id: u64,
    body: Vec<u8>,
}

pub struct WebsocketResponseRecorder {
    exchange_id: u64,
    output_text: String,
    completed: Option<Value>,
    output_items: Vec<Value>,
}

pub fn next_exchange_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

pub fn should_log_url(url: &str) -> bool {
    url.contains("/responses")
        || url.contains("/chat/completions")
        || url.contains("/completions")
        || url.contains("/realtime/calls")
}

pub fn log_http_request(exchange_id: u64, req: &Request) {
    let body = match req.body.as_ref() {
        Some(RequestBody::Json(value)) => value.clone(),
        Some(RequestBody::Raw(bytes)) => bytes_body(bytes),
        None => Value::Null,
    };
    write_record(json!({
        "type": "llm_request",
        "id": exchange_id,
        "transport": "http",
        "method": req.method.as_str(),
        "url": redact_url(&req.url),
        "headers": headers_to_json(&req.headers),
        "request": body,
    }));
    remember_exchange_metadata(exchange_id, metadata_from_headers(&req.headers));
    remember_request(exchange_id, body);
}

pub fn log_http_response_headers(exchange_id: u64, status: u16, headers: &HeaderMap) {
    write_record(json!({
        "type": "llm_http_response_headers",
        "id": exchange_id,
        "status": status,
        "headers": headers_to_json(headers),
    }));
}

pub fn log_http_response_body(exchange_id: u64, body: &Bytes) {
    let response = bytes_body(body);
    let output_text = output_text_from_value(&response);
    let mut record = json!({
        "type": "llm_response",
        "id": exchange_id,
        "transport": "http",
        "response": response,
    });
    if !output_text.is_empty() {
        record["output_text"] = Value::String(output_text.clone());
    }
    write_record(record);
    write_clean_exchange(
        exchange_id,
        "http",
        Some(response),
        Some(output_text),
        None,
        None,
    );
}

pub fn http_stream_response_recorder(exchange_id: u64) -> HttpStreamResponseRecorder {
    HttpStreamResponseRecorder {
        inner: Arc::new(Mutex::new(HttpStreamResponseState {
            exchange_id,
            body: Vec::new(),
        })),
    }
}

impl HttpStreamResponseRecorder {
    pub fn append(&self, chunk: &Bytes) {
        if let Ok(mut state) = self.inner.lock() {
            state.body.extend_from_slice(chunk);
        }
    }
}

impl Drop for HttpStreamResponseRecorder {
    fn drop(&mut self) {
        if Arc::strong_count(&self.inner) != 1 {
            return;
        }
        if let Ok(state) = self.inner.lock() {
            let body = Bytes::from(state.body.clone());
            let output_text = output_text_from_stream_body(&body);
            let mut record = json!({
                "type": "llm_response",
                "id": state.exchange_id,
                "transport": "http_stream",
                "response": stream_body(&body),
            });
            if !output_text.is_empty() {
                record["output_text"] = Value::String(output_text.clone());
            }
            write_record(record);
            write_clean_exchange(
                state.exchange_id,
                "http_stream",
                Some(stream_body(&body)),
                Some(output_text),
                None,
                None,
            );
        }
    }
}

pub fn log_http_stream_response_aborted(exchange_id: u64, message: impl Into<String>) {
    let message = message.into();
    write_record(json!({
        "type": "llm_response",
        "id": exchange_id,
        "transport": "http_stream",
        "error": message.clone(),
    }));
    write_clean_exchange(exchange_id, "http_stream", None, None, None, Some(message));
}

pub fn log_error(exchange_id: u64, message: impl Into<String>) {
    let message = message.into();
    write_record(json!({
        "type": "llm_error",
        "id": exchange_id,
        "error": message.clone(),
    }));
    write_clean_exchange(exchange_id, "unknown", None, None, None, Some(message));
}

pub fn log_websocket_connect(exchange_id: u64, url: &str, headers: &HeaderMap) {
    write_record(json!({
        "type": "llm_connection",
        "id": exchange_id,
        "transport": "websocket",
        "phase": "connect",
        "url": redact_url(url),
        "headers": headers_to_json(headers),
    }));
    remember_exchange_metadata(exchange_id, metadata_from_headers(headers));
}

pub fn log_websocket_connected(exchange_id: u64, status: u16, headers: &HeaderMap) {
    write_record(json!({
        "type": "llm_connection",
        "id": exchange_id,
        "transport": "websocket",
        "phase": "connected",
        "status": status,
        "headers": headers_to_json(headers),
    }));
}

pub fn log_websocket_outgoing(exchange_id: u64, text: &str) {
    let request = text_body(text);
    write_record(json!({
        "type": "llm_request",
        "id": exchange_id,
        "transport": "websocket",
        "request": request,
    }));
    remember_request(exchange_id, request);
}

pub fn websocket_response_recorder(exchange_id: u64) -> WebsocketResponseRecorder {
    WebsocketResponseRecorder {
        exchange_id,
        output_text: String::new(),
        completed: None,
        output_items: Vec::new(),
    }
}

impl WebsocketResponseRecorder {
    pub fn ingest(&mut self, text: &str) {
        let Ok(value) = serde_json::from_str::<Value>(text) else {
            return;
        };
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match kind {
            "response.output_text.delta" => {
                if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                    self.output_text.push_str(delta);
                }
            }
            "response.output_text.done" => {
                if let Some(text) = value.get("text").and_then(Value::as_str) {
                    self.output_text.clear();
                    self.output_text.push_str(text);
                }
            }
            "response.output_item.done" => {
                if let Some(item) = value.get("item") {
                    self.output_items.push(item.clone());
                }
            }
            "response.completed" => {
                self.completed = Some(value);
            }
            _ => {}
        }
    }

    pub fn abort(&mut self, message: impl Into<String>) {
        self.completed = Some(json!({
            "type": "error",
            "error": message.into(),
        }));
    }
}

impl Drop for WebsocketResponseRecorder {
    fn drop(&mut self) {
        let completed_for_clean = self.completed.clone();
        let mut record = json!({
            "type": "llm_response",
            "id": self.exchange_id,
            "transport": "websocket",
            "output_text": self.output_text,
            "output_items": self.output_items,
        });
        if let Some(completed) = self.completed.take() {
            record["completed"] = completed;
        }
        write_record(record);
        write_clean_exchange(
            self.exchange_id,
            "websocket",
            completed_for_clean,
            Some(self.output_text.clone()),
            Some(Value::Array(self.output_items.clone())),
            None,
        );
    }
}

fn write_record(mut record: Value) {
    let Some(logger) = logger() else {
        return;
    };
    record["run_id"] = json!(logger.run_id.as_str());
    record["ts_ms"] = json!(timestamp_ms());
    match serde_json::to_string(&record) {
        Ok(line) => {
            if let Ok(mut file) = logger.file.lock() {
                let _ = writeln!(file, "{line}");
            }
        }
        Err(err) => {
            tracing::debug!("failed to encode LLM traffic log record: {err}");
        }
    }
}

fn remember_request(exchange_id: u64, request: Value) {
    let Some(logger) = logger() else {
        return;
    };
    if let Ok(mut pending) = logger.pending_requests.lock() {
        pending.insert(exchange_id, sanitize_for_clean_log(request));
    }
}

fn remember_exchange_metadata(exchange_id: u64, metadata: Option<Value>) {
    let Some(metadata) = metadata else {
        return;
    };
    let Some(logger) = logger() else {
        return;
    };
    let enriched = enrich_turn_metadata(&logger, metadata);
    if let Ok(mut exchange_metadata) = logger.exchange_metadata.lock() {
        exchange_metadata.insert(exchange_id, enriched);
    }
}

fn write_clean_exchange(
    exchange_id: u64,
    transport: &str,
    response: Option<Value>,
    output_text: Option<String>,
    output_items: Option<Value>,
    error: Option<String>,
) {
    let Some(logger) = logger() else {
        return;
    };

    let request = logger
        .pending_requests
        .lock()
        .ok()
        .and_then(|mut pending| pending.remove(&exchange_id));
    let metadata = logger
        .exchange_metadata
        .lock()
        .ok()
        .and_then(|mut metadata| metadata.remove(&exchange_id));

    if request.is_none() && response.is_none() && output_items.is_none() && error.is_none() {
        return;
    }

    let sanitized_response = response.map(sanitize_for_clean_log);
    let sanitized_output_items = output_items.map(sanitize_for_clean_log);
    let model = request
        .as_ref()
        .and_then(|value| value.get("model"))
        .cloned()
        .unwrap_or(Value::Null);
    let usage = sanitized_response
        .as_ref()
        .and_then(extract_usage)
        .unwrap_or(Value::Null);

    let mut record = json!({
        "type": "model_call",
        "run_id": logger.run_id.as_str(),
        "id": exchange_id,
        "transport": transport,
        "model": model,
        "metadata": metadata.unwrap_or(Value::Null),
        "request": request.unwrap_or(Value::Null),
        "response": {
            "output_text": output_text.unwrap_or_default(),
            "output_items": sanitized_output_items.unwrap_or(Value::Null),
            "raw": sanitized_response.unwrap_or(Value::Null),
            "usage": usage,
        },
        "error": error,
    });
    record["ts_ms"] = json!(timestamp_ms());

    match serde_json::to_string(&record) {
        Ok(line) => {
            if let Ok(mut file) = logger.clean_file.lock() {
                let _ = writeln!(file, "{line}");
            }
        }
        Err(err) => {
            tracing::debug!("failed to encode clean LLM log record: {err}");
        }
    }
}

fn logger() -> Option<Arc<LlmTrafficLogger>> {
    LOGGER.get_or_init(init_logger).clone()
}

fn init_logger() -> Option<Arc<LlmTrafficLogger>> {
    if !env_enabled("MILLIE_LLM_TRAFFIC_LOG") && !env_enabled("MILLIE_LLM_LOG") {
        return None;
    }

    let dir = std::env::var_os("MILLIE_LLM_TRAFFIC_LOG_DIR")
        .or_else(|| std::env::var_os("MILLIE_LLM_LOG_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|| default_log_dir().unwrap_or_else(std::env::temp_dir));

    if let Err(err) = std::fs::create_dir_all(&dir) {
        tracing::warn!(
            "failed to create LLM traffic log dir {}: {err}",
            dir.display()
        );
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755));
    }

    let run_id = std::env::var("MILLIE_LLM_RUN_ID")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .map(|value| safe_file_component(value.trim()))
        .unwrap_or_else(default_run_id);

    let path = dir.join(format!("codex-llm-traffic-{run_id}.jsonl"));
    let file = match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(file) => file,
        Err(err) => {
            tracing::warn!("failed to open LLM traffic log {}: {err}", path.display());
            return None;
        }
    };
    let clean_path = dir.join(format!("codex-llm-model-calls-{run_id}.jsonl"));
    let clean_file = match OpenOptions::new()
        .create(true)
        .append(true)
        .open(&clean_path)
    {
        Ok(file) => file,
        Err(err) => {
            tracing::warn!(
                "failed to open clean LLM log {}: {err}",
                clean_path.display()
            );
            return None;
        }
    };

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = file.set_permissions(std::fs::Permissions::from_mode(0o644));
        let _ = clean_file.set_permissions(std::fs::Permissions::from_mode(0o644));
    }

    Some(Arc::new(LlmTrafficLogger {
        run_id,
        file: Mutex::new(file),
        clean_file: Mutex::new(clean_file),
        pending_requests: Mutex::new(HashMap::new()),
        exchange_metadata: Mutex::new(HashMap::new()),
        turn_indices: Mutex::new(HashMap::new()),
        next_turn_indices: Mutex::new(HashMap::new()),
    }))
}

fn env_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|value| {
            let normalized = value.trim().to_ascii_lowercase();
            matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn default_log_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".millie").join("llm-traffic"))
}

fn timestamp_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis())
        .unwrap_or_default()
}

fn default_run_id() -> String {
    format!("{}-p{}", timestamp_ms(), std::process::id())
}

fn safe_file_component(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_' | '.') {
            out.push(ch);
        } else {
            out.push('-');
        }
    }
    let trimmed = out.trim_matches('-');
    if trimmed.is_empty() {
        default_run_id()
    } else {
        trimmed.to_string()
    }
}

fn metadata_from_headers(headers: &HeaderMap) -> Option<Value> {
    let value = headers.get("x-codex-turn-metadata")?;
    let text = value.to_str().ok()?;
    serde_json::from_str::<Value>(text).ok()
}

fn enrich_turn_metadata(logger: &Arc<LlmTrafficLogger>, mut metadata: Value) -> Value {
    let conversation_id = metadata
        .get("thread_id")
        .or_else(|| metadata.get("session_id"))
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let turn_id = metadata
        .get("turn_id")
        .and_then(Value::as_str)
        .filter(|turn_id| !turn_id.is_empty())
        .map(ToString::to_string);

    let (Some(conversation_id), Some(turn_id)) = (conversation_id, turn_id) else {
        return metadata;
    };

    let key = format!("{conversation_id}:{turn_id}");
    let mut turn_index = None;
    if let Ok(mut turn_indices) = logger.turn_indices.lock() {
        if let Some(existing) = turn_indices.get(&key).copied() {
            turn_index = Some(existing);
        } else if let Ok(mut next_turn_indices) = logger.next_turn_indices.lock() {
            let next = next_turn_indices
                .entry(conversation_id.clone())
                .or_insert(0);
            let assigned = *next;
            *next += 1;
            turn_indices.insert(key, assigned);
            turn_index = Some(assigned);
        }
    }

    if let Some(turn_index) = turn_index {
        if let Some(object) = metadata.as_object_mut() {
            object.insert(
                "conversation_id".to_string(),
                Value::String(conversation_id),
            );
            object.insert("turn_index".to_string(), json!(turn_index));
        }
    }

    metadata
}

fn sanitize_for_clean_log(value: Value) -> Value {
    match value {
        Value::Array(items) => {
            Value::Array(items.into_iter().map(sanitize_for_clean_log).collect())
        }
        Value::Object(object) => {
            let mut out = serde_json::Map::new();
            for (key, value) in object {
                if key == "encrypted_content" {
                    out.insert(
                        key,
                        Value::String(format!(
                            "[omitted encrypted_content: {} chars]",
                            value_len(&value)
                        )),
                    );
                } else {
                    out.insert(key, sanitize_for_clean_log(value));
                }
            }
            Value::Object(out)
        }
        other => other,
    }
}

fn value_len(value: &Value) -> usize {
    match value {
        Value::String(text) => text.len(),
        other => serde_json::to_string(other)
            .map(|text| text.len())
            .unwrap_or(0),
    }
}

fn extract_usage(value: &Value) -> Option<Value> {
    if let Some(usage) = value.get("usage") {
        return Some(usage.clone());
    }
    if let Some(response) = value.get("response") {
        if let Some(usage) = response.get("usage") {
            return Some(usage.clone());
        }
    }
    if let Some(completed_response) = value
        .get("completed")
        .and_then(|completed| completed.get("response"))
    {
        if let Some(usage) = completed_response.get("usage") {
            return Some(usage.clone());
        }
    }
    None
}

fn headers_to_json(headers: &HeaderMap) -> Value {
    let mut object = serde_json::Map::new();
    for (name, value) in headers {
        let value = if should_redact_header(name.as_str()) {
            "[REDACTED]".to_string()
        } else {
            value
                .to_str()
                .map(ToString::to_string)
                .unwrap_or_else(|_| format!("<non-utf8: {} bytes>", value.as_bytes().len()))
        };
        object.insert(name.as_str().to_string(), Value::String(value));
    }
    Value::Object(object)
}

fn should_redact_header(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name == "authorization"
        || name == "proxy-authorization"
        || name == "cookie"
        || name == "set-cookie"
        || name.contains("api-key")
        || name.contains("token")
}

fn redact_url(url: &str) -> String {
    let Some((base, query)) = url.split_once('?') else {
        return url.to_string();
    };
    let redacted_query = query
        .split('&')
        .map(|part| {
            let Some((key, _value)) = part.split_once('=') else {
                return part.to_string();
            };
            if should_redact_query_param(key) {
                format!("{key}=[REDACTED]")
            } else {
                part.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join("&");
    format!("{base}?{redacted_query}")
}

fn should_redact_query_param(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    name.contains("key")
        || name.contains("token")
        || name.contains("secret")
        || name.contains("signature")
        || name == "sig"
}

fn bytes_body(bytes: &Bytes) -> Value {
    match std::str::from_utf8(bytes) {
        Ok(text) => text_body(text),
        Err(_) => json!({
            "encoding": "binary",
            "bytes": bytes.len(),
        }),
    }
}

fn stream_body(bytes: &Bytes) -> Value {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return json!({
            "encoding": "binary",
            "bytes": bytes.len(),
        });
    };

    let mut events = Vec::new();
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        match serde_json::from_str::<Value>(data) {
            Ok(value) => events.push(value),
            Err(_) => events.push(Value::String(data.to_string())),
        }
    }

    if events.is_empty() {
        text_body(text)
    } else {
        Value::Array(events)
    }
}

fn output_text_from_stream_body(bytes: &Bytes) -> String {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return String::new();
    };
    let mut output = String::new();
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data.is_empty() || data == "[DONE]" {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<Value>(data) {
            if value.get("type").and_then(Value::as_str) == Some("response.output_text.done") {
                if let Some(text) = value.get("text").and_then(Value::as_str) {
                    output.clear();
                    output.push_str(text);
                }
            } else {
                output.push_str(&output_text_from_value(&value));
            }
        }
    }
    output
}

fn output_text_from_value(value: &Value) -> String {
    let mut output = String::new();

    if let Some(text) = value.get("output_text").and_then(Value::as_str) {
        output.push_str(text);
    }

    match value.get("type").and_then(Value::as_str) {
        Some("response.output_text.delta") => {
            if let Some(delta) = value.get("delta").and_then(Value::as_str) {
                output.push_str(delta);
            }
        }
        Some("response.output_text.done") => {
            if let Some(text) = value.get("text").and_then(Value::as_str) {
                output.clear();
                output.push_str(text);
            }
        }
        _ => {}
    }

    if let Some(choices) = value.get("choices").and_then(Value::as_array) {
        for choice in choices {
            if let Some(content) = choice
                .get("delta")
                .and_then(|delta| delta.get("content"))
                .and_then(Value::as_str)
            {
                output.push_str(content);
            }
            if let Some(content) = choice
                .get("message")
                .and_then(|message| message.get("content"))
                .and_then(Value::as_str)
            {
                output.push_str(content);
            }
            if let Some(text) = choice.get("text").and_then(Value::as_str) {
                output.push_str(text);
            }
        }
    }

    if let Some(output_items) = value.get("output").and_then(Value::as_array) {
        for item in output_items {
            if let Some(content) = item.get("content").and_then(Value::as_array) {
                for part in content {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        output.push_str(text);
                    }
                }
            }
        }
    }

    output
}

fn text_body(text: &str) -> Value {
    serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderValue;

    #[test]
    fn redacts_sensitive_headers() {
        let mut headers = HeaderMap::new();
        headers.insert(
            http::header::AUTHORIZATION,
            HeaderValue::from_static("Bearer secret"),
        );
        headers.insert(
            http::header::COOKIE,
            HeaderValue::from_static("session=secret"),
        );
        headers.insert("x-safe", HeaderValue::from_static("visible"));

        let value = headers_to_json(&headers);

        assert_eq!(value["authorization"], "[REDACTED]");
        assert_eq!(value["cookie"], "[REDACTED]");
        assert_eq!(value["x-safe"], "visible");
    }

    #[test]
    fn redacts_sensitive_query_params() {
        let redacted = redact_url("https://example.test/v1/responses?api_key=secret&safe=value");

        assert_eq!(
            redacted,
            "https://example.test/v1/responses?api_key=[REDACTED]&safe=value"
        );
    }
}
