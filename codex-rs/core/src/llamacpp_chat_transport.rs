//! Chat-completions transport for the local `llamacpp` provider: sends the
//! conversation as OpenAI messages to `llama-server`'s
//! `/v1/chat/completions`, where the model's own chat template
//! (`millie-native.jinja`) renders the prompt and the server's PEG parser +
//! lazy grammar handle tool-call extraction and enforcement. This replaces
//! the raw `/completion` path's client-side rendering and parsing; the raw
//! path remains available via `CODEX_LLAMACPP_TRANSPORT=raw`.
//!
//! Streaming contract with the agent loop is identical to the raw
//! transport: `OutputItemAdded` opens an item, deltas render live,
//! `OutputItemDone` (same id) finalizes. Tool calls surface as
//! `OutputItemDone(FunctionCall)` once the response finishes.

use futures::StreamExt;
use serde_json::json;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

use codex_protocol::error::CodexErr;
use codex_protocol::error::Result;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;

use codex_api::ResponseEvent;
use codex_llamacpp::unflatten_tool_name;

use crate::client_common::Prompt;
use crate::client_common::ResponseStream;

fn chat_endpoint(base_url: &str) -> String {
    let root = codex_model_provider_info::local_server_root(base_url);
    format!("{}/v1/chat/completions", root.trim_end_matches('/'))
}

/// Accumulates one tool call from streamed `tool_calls` deltas (OpenAI
/// stream shape: fragments keyed by `index`, with `id`/`function.name`
/// arriving once and `function.arguments` arriving in pieces).
#[derive(Default)]
struct ToolCallAcc {
    id: String,
    name: String,
    arguments: String,
}

/// Which local server the chat transport talks to. Both speak OpenAI chat
/// completions; they differ in the extra request fields (thinking budget,
/// loop guard) and in who runs the repetition stop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Backend {
    /// The llama.cpp fork's llama-server: extra fields are top-level request
    /// fields and the server runs the repetition stop itself.
    LlamaServer,
    /// A compatible vLLM server: extra fields travel in
    /// `vllm_xargs` (penalty spans as token ids), and the repetition stop
    /// runs here on the streamed deltas.
    Vllm,
}

pub(crate) fn is_local_backend(provider_name: &str) -> bool {
    provider_name == codex_model_provider_info::LLAMACPP_OSS_PROVIDER_ID
        || provider_name == codex_model_provider_info::VLLM_OSS_PROVIDER_ID
}

fn tokenize_endpoint(base_url: &str) -> String {
    let root = codex_model_provider_info::local_server_root(base_url);
    format!("{root}/tokenize")
}

/// Token ids of each span through the server's tokenizer (vLLM `/tokenize`),
/// cached per span text. Spans that fail to tokenize are dropped.
pub(crate) async fn tokenize_spans(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    spans: &[String],
) -> Vec<Vec<u64>> {
    let timeout_secs = std::env::var("MILLIE_TOKENIZE_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(10);
    match tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        tokenize_spans_inner(client, base_url, model, spans),
    )
    .await
    {
        Ok(ids) => ids,
        Err(_) => {
            tracing::warn!(
                "Penalty tokenization timed out; continuing without optional penalty spans"
            );
            Vec::new()
        }
    }
}

async fn tokenize_spans_inner(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    spans: &[String],
) -> Vec<Vec<u64>> {
    use std::sync::Mutex;
    use std::sync::OnceLock;
    static CACHE: OnceLock<Mutex<std::collections::HashMap<String, Vec<u64>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(std::collections::HashMap::new()));
    let mut out = Vec::new();
    for span in spans {
        let key = format!(
            "{}\0{model}\0{span}",
            codex_model_provider_info::local_server_root(base_url)
        );
        if let Some(ids) = cache.lock().ok().and_then(|c| c.get(&key).cloned()) {
            out.push(ids);
            continue;
        }
        let resp = client
            .post(tokenize_endpoint(base_url))
            // Both server dialects in one body: vLLM reads `prompt` /
            // `add_special_tokens`, llama-server reads `content` /
            // `add_special`; each ignores the other's fields.
            .json(&json!({
                "model": model,
                "prompt": span,
                "add_special_tokens": false,
                "content": span,
                "add_special": false,
            }))
            .send()
            .await;
        let ids: Option<Vec<u64>> = match resp {
            Ok(r) if r.status().is_success() => {
                r.json::<serde_json::Value>().await.ok().and_then(|v| {
                    v.get("tokens")
                        .and_then(|t| t.as_array())
                        .map(|a| a.iter().filter_map(serde_json::Value::as_u64).collect())
                })
            }
            _ => None,
        };
        match ids {
            Some(ids) if !ids.is_empty() => {
                if let Ok(mut c) = cache.lock() {
                    c.insert(key, ids.clone());
                }
                out.push(ids);
            }
            _ => tracing::warn!(
                "vllm: could not tokenize a penalty span ({} chars); dropped",
                span.len()
            ),
        }
    }
    out
}

pub(crate) async fn stream_chat_completions(
    base_url: &str,
    prompt: &Prompt,
    log_key: crate::model_call_log::ModelCallKey,
    stats_key: (String, u64),
    backend: Backend,
) -> Result<ResponseStream> {
    let transport_name = match backend {
        Backend::LlamaServer => "llamacpp-chat",
        Backend::Vllm => "vllm-chat",
    };
    let input = prompt.get_formatted_input();
    let messages = codex_llamacpp::build_chat_messages(&prompt.base_instructions.text, &input);
    let tools = codex_llamacpp::chat_tools_json(&prompt.tools);
    let transport_name = match backend {
        Backend::LlamaServer => transport_name,
        Backend::Vllm => "vllm-chat",
    };
    let client = reqwest::Client::new();

    let sampling = codex_llamacpp::sampling_params_from_env();
    let mut body = json!({
        "messages": messages,
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": codex_llamacpp::max_output_tokens_from_env(),
        "thinking_budget": codex_llamacpp::thinking_budget_from_env(),
        "temperature": sampling.temperature,
        "min_p": sampling.min_p,
        "top_k": sampling.top_k,
        "top_p": sampling.top_p,
    });
    // Client-side repetition stop (vLLM only; llama-server runs its own).
    let mut client_repeat_stop: Option<crate::repetition::RepeatStopParams> = None;
    match backend {
        Backend::LlamaServer => {
            if let Some(serde_json::Value::Object(extra)) = prompt.llamacpp_request_extra.as_ref() {
                for (k, v) in extra {
                    body[k] = v.clone();
                }
            }
            // Pin the recurrent checkpoint at the end of the user turn, on the
            // turn's first model call only (mid-tool-loop calls must not pin --
            // see `is_turn_start`). The server-side jinja appends the same
            // `<|im_start|>assistant\n<think>\n` generation prompt the raw path
            // uses, so the same suffix token count places the pin right before
            // the assistant header. The server's OAI chat endpoint forwards
            // unknown fields to the native param parser, so the field works
            // here exactly as on /completion. This is the DEFAULT transport --
            // the raw path carries the same logic but only runs under
            // MILLIE_LLAMACPP_TRANSPORT=raw.
            if crate::llamacpp_transport::is_turn_start(&input) {
                if let Some(from_end) =
                    codex_llamacpp::generation_prompt_suffix_tokens(base_url).await
                {
                    tracing::debug!(
                        "turn start: requesting pinned checkpoint {from_end} tokens before prompt end"
                    );
                    body["pin_checkpoint_from_end"] = json!(from_end);
                }
            } else {
                tracing::debug!("not a turn start: no checkpoint pin on this completion");
            }
        }
        Backend::Vllm => {
            let model = codex_llamacpp::vllm_model_from_env();
            body["model"] = json!(model);
            let mut xargs = serde_json::Map::new();
            if let Some(tb) = body
                .as_object_mut()
                .and_then(|o| o.remove("thinking_budget"))
                && tb.as_u64().unwrap_or(0) > 0
            {
                xargs.insert("thinking_budget".to_string(), tb);
            }
            if let Some(serde_json::Value::Object(extra)) = prompt.llamacpp_request_extra.as_ref() {
                if let Some(rs) = extra.get("repeat_stop") {
                    client_repeat_stop = Some(crate::repetition::RepeatStopParams::from_json(rs));
                }
                if let Some(np) = extra.get("ngram_penalty") {
                    let spans: Vec<String> = np
                        .get("spans")
                        .and_then(|s| s.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|x| x.as_str().map(str::to_string))
                                .collect()
                        })
                        .unwrap_or_default();
                    let ids = tokenize_spans(&client, base_url, &model, &spans).await;
                    if !ids.is_empty() {
                        let joined = ids
                            .iter()
                            .map(|s| s.iter().map(u64::to_string).collect::<Vec<_>>().join(","))
                            .collect::<Vec<_>>()
                            .join(";");
                        xargs.insert("ngram_penalty_spans".to_string(), json!(joined));
                        for key in ["start_n", "scale", "max"] {
                            if let Some(v) = np.get(key) {
                                xargs.insert(format!("ngram_penalty_{key}"), v.clone());
                            }
                        }
                    }
                }
            }
            if !xargs.is_empty() {
                body["vllm_xargs"] = serde_json::Value::Object(xargs);
            }
        }
    }
    if !tools.is_empty() {
        body["tools"] = json!(tools);
        body["tool_choice"] = json!("auto");
        body["parallel_tool_calls"] = json!(true);
    }
    // When the model-call log is on, ask llama-server to echo the unparsed
    // generation on the final chunk so the log holds the model's exact output.
    let logging = crate::model_call_log::enabled();
    if logging && backend == Backend::LlamaServer {
        body["include_raw_generation"] = json!(true);
    }
    let logged_request = logging.then(|| body.clone());
    if let Ok(dump_path) = std::env::var("MILLIE_LLAMACPP_DUMP_REQUEST") {
        let _ = std::fs::write(
            &dump_path,
            serde_json::to_string_pretty(&body).unwrap_or_default(),
        );
    }

    // Same connection-race retry ladder as the raw transport.
    let mut resp = None;
    for (attempt, delay_ms) in [(0u32, 0u64), (1, 500), (2, 1000), (3, 2000)] {
        if delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }
        match client
            .post(chat_endpoint(base_url))
            .json(&body)
            .send()
            .await
        {
            Ok(r) => {
                resp = Some(r);
                break;
            }
            Err(e) if e.is_connect() && attempt < 3 => {
                tracing::warn!("llama-server connection failed (attempt {attempt}), retrying: {e}");
                continue;
            }
            Err(e) => {
                return Err(CodexErr::Fatal(format!(
                    "llama-server chat/completions request failed: {e}"
                )));
            }
        }
    }
    let resp = resp.expect("retry loop either set resp or returned");
    let status = resp.status();
    if !status.is_success() {
        let err_body = resp.text().await.unwrap_or_default();
        return Err(CodexErr::Fatal(format!(
            "llama-server chat/completions failed ({status}): {err_body}"
        )));
    }

    let debug = std::env::var("MILLIE_LLAMACPP_DEBUG").is_ok();

    let (tx_event, rx_event) = mpsc::channel(64);
    let consumer_dropped = CancellationToken::new();
    let cancelled = consumer_dropped.clone();
    tokio::spawn(async move {
        tokio::select! {
            biased;
            _ = cancelled.cancelled() => {},
            _ = async move {
                let _ = tx_event.send(Ok(ResponseEvent::Created)).await;

                let reasoning_id = format!("rs_{}", uuid::Uuid::new_v4());
                let message_id = format!("msg_{}", uuid::Uuid::new_v4());

                let mut reasoning_added = false;
                let mut message_added = false;
                let mut reasoning_acc = String::new();
                let mut content_acc = String::new();
                let mut tool_calls: Vec<ToolCallAcc> = Vec::new();
                let mut usage: Option<serde_json::Value> = None;

                let mut finish_reason: Option<String> = None;
                let mut terminal_received = false;
                let mut raw_generation: Option<String> = None;
                let mut repetition_stats: Option<serde_json::Value> = None;
                // client-side repetition stop state: one unit per streamed delta
                let mut units: Vec<u64> = Vec::new();
                let mut unit_texts: Vec<String> = Vec::new();

                let mut sse_buf: Vec<u8> = Vec::new();
                let mut byte_stream = resp.bytes_stream();

                'stream: while let Some(next) = byte_stream.next().await {
                    let bytes = match next {
                        Ok(b) => b,
                        Err(e) => {
                            if let Some(request) = logged_request.as_ref() {
                                crate::model_call_log::record_model_call(
                                    &log_key, transport_name, request, None, &reasoning_acc,
                                    &content_acc, &[], None, None,
                                    Some(&format!("stream error: {e}")),
                                );
                            }
                            let _ = tx_event
                                .send(Err(CodexErr::Fatal(format!(
                                    "llama-server stream error: {e}"
                                ))))
                                .await;
                            return;
                        }
                    };
                    sse_buf.extend_from_slice(&bytes);

                    while let Some((sep, width)) = sse_buf.windows(2).position(|w| w == b"\n\n").map(|p| (p, 2))
                        .or_else(|| sse_buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| (p, 4))) {
                        let event_bytes: Vec<u8> = sse_buf.drain(..sep + width).collect();
                        let event_text = String::from_utf8_lossy(&event_bytes);
                        for line in event_text.lines() {
                            let Some(data) = line.strip_prefix("data:") else {
                                continue;
                            };
                            if data.trim() == "[DONE]" {
                                terminal_received = true;
                                break 'stream;
                            }
                            let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) else {
                                continue;
                            };
                            if let Some(error) = chunk.get("error").filter(|value| !value.is_null()) {
                                let _ = tx_event.send(Err(CodexErr::Fatal(format!(
                                    "llama-server stream error: {error}. Generation was interrupted."
                                )))).await;
                                return;
                            }
                            if let Some(u) = chunk.get("usage").filter(|u| !u.is_null()) {
                                usage = Some(u.clone());
                            }
                            if let Some(raw) = chunk.get("raw_generation").and_then(|v| v.as_str()) {
                                raw_generation = Some(raw.to_string());
                            }
                            if let Some(stats) = chunk.get("repetition").filter(|v| v.is_object()) {
                                repetition_stats = Some(stats.clone());
                            }
                            let Some(choice) = chunk.get("choices").and_then(|c| c.get(0)) else {
                                continue;
                            };
                            if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                                finish_reason = Some(reason.to_string());
                            }
                            let Some(delta) = choice.get("delta") else {
                                continue;
                            };

                            // llama-server streams `reasoning_content`; vLLM streams `reasoning`.
                            let reasoning_delta = delta
                                .get("reasoning_content")
                                .or_else(|| delta.get("reasoning"))
                                .and_then(|v| v.as_str())
                                .filter(|t| !t.is_empty());
                            let mut new_units: Vec<&str> = Vec::new();
                            if let Some(text) = reasoning_delta {
                                new_units.push(text);
                            }
                            if let Some(text) = delta.get("content").and_then(|v| v.as_str()).filter(|t| !t.is_empty()) {
                                new_units.push(text);
                            }
                            if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                                for frag in calls {
                                    if let Some(args) = frag.get("function").and_then(|f| f.get("arguments")).and_then(|v| v.as_str())
                                        && !args.is_empty()
                                    {
                                        new_units.push(args);
                                    }
                                }
                            }
                            let mut stop_for_repetition = false;
                            if let Some(p) = client_repeat_stop.as_ref() {
                                for u in &new_units {
                                    units.push(crate::repetition::hash_unit(u));
                                    unit_texts.push((*u).to_string());
                                }
                                if !new_units.is_empty() && units.len() >= p.min_tokens && units.len().is_multiple_of(p.stride) {
                                    let st = crate::repetition::detect(&units, p);
                                    if st.flagged {
                                        let gram_text: String = unit_texts[st.gram_pos..(st.gram_pos + st.gram_len).min(unit_texts.len())].concat();
                                        repetition_stats = Some(json!({
                                            "flagged": true, "window": st.window, "gram_len": st.gram_len,
                                            "count": st.count, "coverage": st.coverage, "gram_text": gram_text,
                                            "n_generated": units.len(), "detector": "client",
                                        }));
                                        stop_for_repetition = true;
                                    }
                                }
                            }

                            if let Some(text) = reasoning_delta {
                                reasoning_acc.push_str(text);
                                if !reasoning_added {
                                    reasoning_added = true;
                                    if tx_event
                                        .send(Ok(ResponseEvent::OutputItemAdded(
                                            ResponseItem::Reasoning {
                                                id: reasoning_id.clone(),
                                                summary: Vec::new(),
                                                content: Some(Vec::new()),
                                                encrypted_content: None,
                                            },
                                        )))
                                        .await
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                                if tx_event
                                    .send(Ok(ResponseEvent::ReasoningContentDelta {
                                        delta: text.to_string(),
                                        content_index: 0,
                                    }))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                            }

                            if let Some(text) = delta.get("content").and_then(|v| v.as_str())
                                && !text.is_empty()
                            {
                                content_acc.push_str(text);
                                if !message_added {
                                    message_added = true;
                                    if tx_event
                                        .send(Ok(ResponseEvent::OutputItemAdded(ResponseItem::Message {
                                            id: Some(message_id.clone()),
                                            role: "assistant".to_string(),
                                            content: Vec::new(),
                                            phase: None,
                                        })))
                                        .await
                                        .is_err()
                                    {
                                        return;
                                    }
                                }
                                if tx_event
                                    .send(Ok(ResponseEvent::OutputTextDelta(text.to_string())))
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                            }

                            if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                                for frag in calls {
                                    let idx =
                                        frag.get("index").and_then(serde_json::Value::as_u64).unwrap_or(0) as usize;
                                    while tool_calls.len() <= idx {
                                        tool_calls.push(ToolCallAcc::default());
                                    }
                                    let acc = &mut tool_calls[idx];
                                    if let Some(id) = frag.get("id").and_then(|v| v.as_str())
                                        && !id.is_empty()
                                    {
                                        acc.id = id.to_string();
                                    }
                                    if let Some(f) = frag.get("function") {
                                        if let Some(name) = f.get("name").and_then(|v| v.as_str())
                                            && !name.is_empty()
                                        {
                                            acc.name = name.to_string();
                                        }
                                        if let Some(args) = f.get("arguments").and_then(|v| v.as_str()) {
                                            acc.arguments.push_str(args);
                                        }
                                    }
                                }
                            }
                            if stop_for_repetition {
                                terminal_received = true;
                                finish_reason = Some(crate::pathology::REPETITION_FINISH_REASON.to_string());
                                // dropping the response body aborts the request on the server
                                break 'stream;
                            }
                        }
                    }
                }

                drop(byte_stream);
                if !terminal_received || finish_reason.is_none() {
                    let _ = tx_event.send(Err(CodexErr::Fatal(
                        "llama-server stream ended before completion. Generation was interrupted.".into()
                    ))).await;
                    return;
                }

                if debug {
                    eprintln!(
                        "=== CHAT RESPONSE ===\nreasoning: {reasoning_acc}\ncontent: {content_acc}\ncalls: {}\n=== END ===",
                        tool_calls.len()
                    );
                }

                // A tool call whose arguments are not complete JSON (cut off at the
                // generation cap, or malformed) must never become a FunctionCall: it
                // would not execute correctly, and the server rejects a history that
                // contains it. Keep its text in the assistant message instead, which
                // is what the model actually produced, and let the turn continue with
                // an error prompt.
                let (complete_calls, malformed_calls): (Vec<&ToolCallAcc>, Vec<&ToolCallAcc>) = tool_calls
                    .iter()
                    .filter(|c| !c.name.is_empty())
                    .partition(|c| tool_call_arguments_complete(&c.arguments));
                for call in &malformed_calls {
                    content_acc.push_str(&render_incomplete_tool_call(&call.name, &call.arguments));
                }
                if !malformed_calls.is_empty() && finish_reason.as_deref() != Some("length") {
                    finish_reason = Some(MALFORMED_TOOL_CALL_FINISH_REASON.to_string());
                }

                // Final items (source of truth for conversation history).
                if !reasoning_acc.trim().is_empty()
                    && tx_event
                        .send(Ok(ResponseEvent::OutputItemDone(ResponseItem::Reasoning {
                            id: reasoning_id.clone(),
                            summary: Vec::new(),
                            content: Some(vec![ReasoningItemContent::ReasoningText {
                                text: reasoning_acc.trim().to_string(),
                            }]),
                            encrypted_content: None,
                        })))
                        .await
                        .is_err()
                {
                    return;
                }
                if !content_acc.trim().is_empty()
                    && tx_event
                        .send(Ok(ResponseEvent::OutputItemDone(ResponseItem::Message {
                            id: Some(message_id.clone()),
                            role: "assistant".to_string(),
                            content: vec![ContentItem::OutputText {
                                text: content_acc.trim_start().to_string(),
                            }],
                            phase: None,
                        })))
                        .await
                        .is_err()
                {
                    return;
                }

                // Call ids must be unique across the whole session (TUI groups exec
                // cells by call id); prefer the model's round-tripped id.
                static CALL_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                // A terminal marker does not make a capped or loop-stopped tool batch safe.
                let truncated_calls = matches!(finish_reason.as_deref(), Some("length") | Some(crate::pathology::REPETITION_FINISH_REASON));
                for call in complete_calls.into_iter().filter(|_| !truncated_calls) {
                    let (namespace, name) = match unflatten_tool_name(&call.name) {
                        Some((ns, n)) => (Some(ns), n),
                        None => (None, call.name.clone()),
                    };
                    let call_id = if call.id.is_empty() {
                        format!(
                            "call_{}",
                            CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        )
                    } else {
                        call.id.clone()
                    };
                    let item = ResponseItem::FunctionCall {
                        id: None,
                        name,
                        namespace,
                        arguments: call.arguments.clone(),
                        call_id,
                    };
                    if tx_event
                        .send(Ok(ResponseEvent::OutputItemDone(item)))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }

                let token_usage = usage
                    .as_ref()
                    .map(|u| {
                        let input = u.get("prompt_tokens").and_then(serde_json::Value::as_i64).unwrap_or(0);
                        let output = u
                            .get("completion_tokens")
                            .and_then(serde_json::Value::as_i64)
                            .unwrap_or(0);
                        let cached = u
                            .get("prompt_tokens_details")
                            .and_then(|d| d.get("cached_tokens"))
                            .and_then(serde_json::Value::as_i64)
                            .unwrap_or(0);
                        codex_protocol::protocol::TokenUsage {
                            input_tokens: input,
                            cached_input_tokens: cached,
                            output_tokens: output,
                            reasoning_output_tokens: 0,
                            total_tokens: input + output,
                        }
                    })
                    .unwrap_or_else(|| codex_protocol::protocol::TokenUsage {
                        input_tokens: 0,
                        cached_input_tokens: 0,
                        output_tokens: 0,
                        reasoning_output_tokens: 0,
                        total_tokens: 0,
                    });
                let truncated = finish_reason.as_deref() == Some("length")
                    || finish_reason.as_deref() == Some(MALFORMED_TOOL_CALL_FINISH_REASON)
                    || finish_reason.as_deref() == Some(crate::pathology::REPETITION_FINISH_REASON);
                if let Some(stats) = repetition_stats.clone() {
                    crate::pathology::store_repetition_stats(&stats_key.0, stats_key.1, stats);
                }
                if let Some(request) = logged_request.as_ref() {
                    let calls: Vec<serde_json::Value> = tool_calls
                        .iter()
                        .map(|c| {
                            json!({
                                "id": c.id,
                                "name": c.name,
                                "arguments": c.arguments,
                                "complete": tool_call_arguments_complete(&c.arguments),
                            })
                        })
                        .collect();
                    if let Some(stats) = repetition_stats.as_ref() {
                        crate::model_call_log::record_event("repetition_stats", &log_key, None, stats.clone());
                    }
                    crate::model_call_log::record_model_call(
                        &log_key,
                        transport_name,
                        request,
                        raw_generation.as_deref(),
                        &reasoning_acc,
                        &content_acc,
                        &calls,
                        finish_reason.as_deref(),
                        usage.as_ref(),
                        None,
                    );
                }
                let _ = tx_event
                    .send(Ok(ResponseEvent::Completed {
                        response_id: uuid::Uuid::new_v4().to_string(),
                        token_usage: Some(token_usage),
                        end_turn: Some(!truncated),
                        finish_reason,
                    }))
                    .await;
            } => {},
        }
    });

    Ok(ResponseStream {
        rx_event,
        consumer_dropped,
    })
}

/// Finish reason reported when a completion contained a tool call whose
/// arguments were not complete JSON and the completion was not otherwise
/// truncated. The turn loop treats it like a truncation: the call is not
/// executed and the model is asked to make it again.
pub const MALFORMED_TOOL_CALL_FINISH_REASON: &str = "malformed_tool_call";

/// Whether a tool call's accumulated `arguments` string is complete JSON.
/// A call cut off at the generation cap leaves a prefix of a JSON object
/// here, which is neither executable nor acceptable to the server as history.
fn tool_call_arguments_complete(arguments: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(arguments).is_ok()
}

/// Text kept in the assistant message for a tool call that could not be
/// executed, so the model's own output survives in history verbatim.
fn render_incomplete_tool_call(name: &str, arguments: &str) -> String {
    format!("\n\n[incomplete tool call, not executed] {name} {arguments}")
}

#[cfg(test)]
mod tool_call_completeness_tests {
    use super::*;

    #[test]
    fn complete_json_arguments_pass() {
        assert!(tool_call_arguments_complete(
            r#"{"path":"a.py","old_str":"x","new_str":"y"}"#
        ));
        assert!(tool_call_arguments_complete("{}"));
    }

    #[test]
    fn truncated_arguments_fail() {
        assert!(!tool_call_arguments_complete(
            r#"{"path":"a.py","old_str":"def f():\n    retu"#
        ));
        assert!(!tool_call_arguments_complete(""));
        assert!(!tool_call_arguments_complete(r#"{"path":"#));
    }

    #[test]
    fn incomplete_call_rendering_names_the_tool_and_keeps_the_text() {
        let r = render_incomplete_tool_call("str_replace", r#"{"path":"a"#);
        assert!(r.contains("str_replace"));
        assert!(r.contains(r#"{"path":"a"#));
        assert!(r.contains("not executed"));
    }
}
