//! Raw-completions transport for the local `llamacpp` provider: builds the
//! model's native raw-text prompt (via `codex_llamacpp::render_prompt`),
//! streams it from `llama-server`'s `/completion` endpoint directly
//! (bypassing llama-server's own chat/Responses-shaped tool-aware endpoints,
//! which break on this model's chat template -- see codex-llamacpp crate
//! docs), and parses the raw output back into `ResponseItem`s the rest of
//! the agent loop already understands.
//!
//! Streaming contract with the agent loop (see `session/turn.rs`):
//! `OutputItemAdded(item)` marks the item active and streaming, then
//! `ReasoningContentDelta`/`OutputTextDelta` events render live, then
//! `OutputItemDone(item)` (SAME id as the Added item) finalizes it. Tool
//! calls are not streamed -- they surface as `OutputItemDone(FunctionCall)`
//! once the completion finishes and the call text can be parsed whole.

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
use codex_llamacpp::arguments_to_json_string;
use codex_llamacpp::unflatten_tool_name;

use crate::client_common::Prompt;
use crate::client_common::ResponseStream;

/// Splits a leading `<think>...</think>` block (if present at the very
/// start of the completion, which is where this model always puts it) from
/// the rest of the content, mirroring the reverse operation the chat
/// template performs when re-serializing a prior turn's
/// `reasoning_content`.
fn split_think_block(text: &str) -> (Option<String>, String) {
    // The generation prompt primes `<think>`, so generation begins INSIDE the
    // reasoning region: the output is `REASONING</think>CONTENT`, with no
    // opening tag. A leading `<think>` may still appear if the model emits one
    // anyway -- tolerate it. Everything up to `</think>` is reasoning; if the
    // block never closes, it is all reasoning (still thinking).
    let body = text
        .trim_start()
        .strip_prefix("<think>")
        .unwrap_or(text.trim_start());
    match body.find("</think>") {
        Some(end) => {
            let reasoning = body[..end].trim().to_string();
            let content = body[end + "</think>".len()..].trim_start().to_string();
            (Some(reasoning), content)
        }
        None => (Some(body.trim().to_string()), String::new()),
    }
}

fn completion_endpoint(base_url: &str) -> String {
    // base_url is registered as e.g. "http://127.0.0.1:8090/v1"; the raw
    // /completion endpoint (not part of the OpenAI-compatible surface) sits
    // at the server root.
    let root = codex_model_provider_info::local_server_root(base_url);
    format!("{}/completion", root.trim_end_matches('/'))
}

/// Incremental splitter for the model's raw output shape:
/// `<think>REASONING</think> TEXT <tool_call>...` . Tracks which region the
/// accumulated text is in and yields safe-to-emit deltas, holding back a
/// small tail so region markers split across network chunks are never
/// half-emitted as visible text.
struct RegionSplitter {
    acc: String,
    state: Region,
    /// Byte offset into `acc` up to which deltas have been emitted.
    emitted: usize,
    /// Whether a stray leading `<think>` has been consumed (start-of-stream
    /// only). Generation already begins inside the think region.
    skipped_open: bool,
}

#[derive(PartialEq)]
enum Region {
    /// Generation starts here: the prompt primes `<think>`, so output opens
    /// inside the reasoning region (no leading tag). A stray leading
    /// `<think>` the model emits anyway is skipped.
    Think,
    Text,
    /// A `<tool_call>` was seen; nothing further streams as visible text.
    ToolCalls,
}

enum RegionDelta {
    Reasoning(String),
    Text(String),
}

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";

/// Length of the longest suffix of `s` that is a prefix of `marker`.
fn partial_marker_len(s: &str, marker: &str) -> usize {
    let max = marker.len().min(s.len());
    for len in (1..=max).rev() {
        if marker
            .as_bytes()
            .starts_with(&s.as_bytes()[s.len() - len..])
        {
            return len;
        }
    }
    0
}

impl RegionSplitter {
    fn new() -> Self {
        Self {
            acc: String::new(),
            state: Region::Think,
            emitted: 0,
            skipped_open: false,
        }
    }

    fn push(&mut self, chunk: &str, out: &mut Vec<RegionDelta>) {
        self.acc.push_str(chunk);
        loop {
            match self.state {
                Region::Think => {
                    // Skip a stray leading `<think>` the model emits despite
                    // the prompt already priming one. Only at the very start,
                    // before any reasoning has been emitted.
                    if !self.skipped_open && self.emitted == 0 {
                        let lead = self.acc.trim_start();
                        if let Some(_rest) = lead.strip_prefix(THINK_OPEN) {
                            self.emitted = self.acc.len() - lead.len() + THINK_OPEN.len();
                            self.skipped_open = true;
                        } else if THINK_OPEN.as_bytes().starts_with(lead.as_bytes())
                            && !lead.is_empty()
                        {
                            // Could still become "<think>" once more bytes arrive.
                            return;
                        } else {
                            self.skipped_open = true;
                        }
                    }
                    if let Some(pos) = self.acc[self.emitted..].find(THINK_CLOSE) {
                        let reasoning = &self.acc[self.emitted..self.emitted + pos];
                        if !reasoning.is_empty() {
                            out.push(RegionDelta::Reasoning(reasoning.to_string()));
                        }
                        self.emitted += pos + THINK_CLOSE.len();
                        self.state = Region::Text;
                        continue;
                    }
                    let hold = partial_marker_len(&self.acc, THINK_CLOSE);
                    let safe_end = self.acc.len() - hold;
                    if safe_end > self.emitted {
                        out.push(RegionDelta::Reasoning(
                            self.acc[self.emitted..safe_end].to_string(),
                        ));
                        self.emitted = safe_end;
                    }
                    return;
                }
                Region::Text => {
                    if let Some(pos) = self.acc[self.emitted..].find(TOOL_OPEN) {
                        let text = &self.acc[self.emitted..self.emitted + pos];
                        if !text.is_empty() {
                            out.push(RegionDelta::Text(text.to_string()));
                        }
                        self.emitted += pos;
                        self.state = Region::ToolCalls;
                        continue;
                    }
                    let hold = partial_marker_len(&self.acc, TOOL_OPEN);
                    let safe_end = self.acc.len() - hold;
                    if safe_end > self.emitted {
                        out.push(RegionDelta::Text(
                            self.acc[self.emitted..safe_end].to_string(),
                        ));
                        self.emitted = safe_end;
                    }
                    return;
                }
                Region::ToolCalls => return,
            }
        }
    }

    /// Flush any held-back tail at end of stream (a partial marker that never
    /// completed is real content).
    fn finish(&mut self, out: &mut Vec<RegionDelta>) {
        if self.emitted < self.acc.len() {
            let tail = self.acc[self.emitted..].to_string();
            match self.state {
                Region::Think => out.push(RegionDelta::Reasoning(tail)),
                Region::Text => out.push(RegionDelta::Text(tail)),
                Region::ToolCalls => {}
            }
            self.emitted = self.acc.len();
        }
    }
}

/// Parses the COMPLETE raw completion into final `ResponseItem`s. This is
/// the source of truth for conversation history; the streamed deltas are
/// display-only.
fn parse_final_items(
    raw_content: &str,
    prompt: &Prompt,
    reasoning_id: &str,
    message_id: &str,
) -> Vec<ResponseItem> {
    let param_types = codex_llamacpp::tool_param_types(&prompt.tools);
    let (reasoning, content_after_think) = split_think_block(raw_content);
    let extracted = codex_llamacpp::extract_tool_calls(&content_after_think);

    let mut items: Vec<ResponseItem> = Vec::new();
    if let Some(reasoning_text) = reasoning.filter(|t| !t.is_empty()) {
        items.push(ResponseItem::Reasoning {
            id: reasoning_id.to_string(),
            summary: Vec::new(),
            content: Some(vec![ReasoningItemContent::ReasoningText {
                text: reasoning_text,
            }]),
            encrypted_content: None,
        });
    }
    if let Some(content_text) = extracted.content.filter(|t| !t.trim().is_empty()) {
        items.push(ResponseItem::Message {
            id: Some(message_id.to_string()),
            role: "assistant".to_string(),
            content: vec![ContentItem::OutputText { text: content_text }],
            phase: None,
        });
    }
    // Call ids must be unique across the whole session, not just within one
    // response: the TUI tracks/groups exec cells by call id, so a repeated
    // "call_0" from a later turn collapses into the earlier turn's cell
    // instead of rendering as a new command.
    static CALL_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    for call in extracted.tool_calls.iter() {
        let (namespace, name) = match unflatten_tool_name(&call.name) {
            Some((ns, n)) => (Some(ns), n),
            None => (None, call.name.clone()),
        };
        let arguments = arguments_to_json_string(call, |p| {
            param_types.get(&call.name).and_then(|m| m.get(p)).cloned()
        });
        items.push(ResponseItem::FunctionCall {
            id: None,
            name,
            namespace,
            arguments,
            call_id: call.call_id.clone().unwrap_or_else(|| {
                format!(
                    "call_{}",
                    CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                )
            }),
        });
    }
    items
}

fn token_usage_from_final(final_chunk: &serde_json::Value) -> codex_protocol::protocol::TokenUsage {
    // Field semantics per llama-server's server-task.cpp / server-context.cpp
    // (NOT what the names suggest): `tokens_evaluated` is the FULL prompt
    // size (slot.task->n_tokens()); `timings.cache_n` is the prefix reused
    // from cache; `tokens_cached` is the slot's post-generation token count
    // -- useless for accounting.
    let tokens_evaluated = final_chunk
        .get("tokens_evaluated")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    let cache_n = final_chunk
        .get("timings")
        .and_then(|t| t.get("cache_n"))
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0)
        .max(0);
    let tokens_predicted = final_chunk
        .get("tokens_predicted")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    codex_protocol::protocol::TokenUsage {
        input_tokens: tokens_evaluated,
        cached_input_tokens: cache_n,
        output_tokens: tokens_predicted,
        reasoning_output_tokens: 0,
        total_tokens: tokens_evaluated + tokens_predicted,
    }
}

/// Is this the turn's first model call? True when nothing the model produced
/// (assistant text, reasoning, tool calls) and no tool activity appears after
/// the last user message. Context items recorded after the user message --
/// skills/plugins injections, developer blocks, compaction markers -- do not
/// end the turn-start window; a mid-tool-loop call always has a call/output
/// item after the user message and returns false.
pub(crate) fn is_turn_start(input: &[ResponseItem]) -> bool {
    for item in input.iter().rev() {
        match item {
            ResponseItem::Message { role, .. } => {
                if role == "user" {
                    return true;
                }
                if role == "assistant" {
                    return false;
                }
                // developer/system context blocks: keep scanning.
            }
            ResponseItem::Reasoning { .. }
            | ResponseItem::LocalShellCall { .. }
            | ResponseItem::FunctionCall { .. }
            | ResponseItem::ToolSearchCall { .. }
            | ResponseItem::FunctionCallOutput { .. }
            | ResponseItem::CustomToolCall { .. }
            | ResponseItem::CustomToolCallOutput { .. }
            | ResponseItem::ToolSearchOutput { .. }
            | ResponseItem::WebSearchCall { .. }
            | ResponseItem::ImageGenerationCall { .. } => return false,
            ResponseItem::Compaction { .. }
            | ResponseItem::CompactionTrigger
            | ResponseItem::ContextCompaction { .. }
            | ResponseItem::Other => {}
        }
    }
    false
}

pub(crate) async fn stream_raw_completions(
    base_url: &str,
    prompt: &Prompt,
) -> Result<ResponseStream> {
    let input = prompt.get_formatted_input();
    let rendered =
        codex_llamacpp::render_prompt(&prompt.base_instructions.text, &input, &prompt.tools);

    let client = reqwest::Client::new();
    let sampling = codex_llamacpp::sampling_params_from_env();
    // An optional lazy grammar constrains tool-call syntax after its trigger.
    // Grammar constraints can affect token selection; enable them explicitly
    // when the raw transport requires constrained tool syntax.
    let tool_grammar = if std::env::var("MILLIE_LLAMACPP_TOOL_GRAMMAR").as_deref() != Ok("1") {
        None
    } else {
        codex_llamacpp::tool_call_grammar(&codex_llamacpp::flattened_tools_json(&prompt.tools))
    };
    let mut body = json!({
        "prompt": rendered,
        "n_predict": codex_llamacpp::max_output_tokens_from_env(),
        "thinking_budget": codex_llamacpp::thinking_budget_from_env(),
        "cache_prompt": true,
        "stream": true,
        "temperature": sampling.temperature,
        "min_p": sampling.min_p,
        "top_k": sampling.top_k,
        "top_p": sampling.top_p,
    });
    // Images: the server's multimodal shape nests the payloads inside a
    // JSON-object prompt ({"prompt_string", "multimodal_data"}), one base64
    // entry per media marker in the string. Only used when images exist;
    // the server errors on it unless the vision tower (--mmproj) is loaded.
    let media = codex_llamacpp::collect_media(&input);
    if !media.is_empty() {
        let prompt_string = body["prompt"].take();
        body["prompt"] = json!({
            "prompt_string": prompt_string,
            "multimodal_data": media,
        });
    }
    if let Some(g) = tool_grammar {
        body["grammar"] = json!(g);
        body["grammar_lazy"] = json!(true);
        body["grammar_triggers"] = json!([{"type": 1, "value": "<tool_call>"}]);
        body["preserved_tokens"] = json!(["<tool_call>"]);
    }
    // Pin the recurrent checkpoint at the end of the user turn, but only on the
    // turn's first model call. Mid-tool-loop calls must NOT pin (a mid-loop pin
    // would overwrite the turn-boundary one and break the next turn's strip).
    // Note the last input item is usually NOT the user message itself: the turn
    // start records skills/plugins injections and hook context blocks after it,
    // so `is_turn_start` scans backwards past context items. The offset is the
    // token length of the trailing
    // generation-prompt suffix render_prompt appended, so the pin lands right
    // before the assistant header -- after all user-side content, injections
    // included. This lets the next turn strip past thinking without a full
    // recurrent reset even after a long tool loop.
    if is_turn_start(&input) {
        if let Some(from_end) = codex_llamacpp::generation_prompt_suffix_tokens(base_url).await {
            tracing::debug!(
                "turn start: requesting pinned checkpoint {from_end} tokens before prompt end"
            );
            body["pin_checkpoint_from_end"] = json!(from_end);
        }
    } else {
        tracing::debug!("not a turn start: no checkpoint pin on this completion");
    }
    // Connection-class failures get a short retry ladder: covers the launch race
    // (server passes the health check but refuses the very first connection) and
    // transient hiccups, without masking real request errors.
    let mut resp = None;
    for (attempt, delay_ms) in [(0u32, 0u64), (1, 500), (2, 1000), (3, 2000)] {
        if delay_ms > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        }
        match client
            .post(completion_endpoint(base_url))
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
                    "llama-server /completion request failed: {e}"
                )));
            }
        }
    }
    let resp = resp.expect("retry loop either set resp or returned");
    let status = resp.status();
    if !status.is_success() {
        let err_body = resp.text().await.unwrap_or_default();
        return Err(CodexErr::Fatal(format!(
            "llama-server /completion failed ({status}): {err_body}"
        )));
    }

    let debug = std::env::var("MILLIE_LLAMACPP_DEBUG").is_ok();
    let rendered_for_debug = debug.then(|| rendered.clone());
    let prompt = prompt.clone();

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

                let mut splitter = RegionSplitter::new();
                let mut reasoning_added = false;
                let mut message_added = false;
                let mut final_chunk: Option<serde_json::Value> = None;

                let mut sse_buf: Vec<u8> = Vec::new();
                let mut byte_stream = resp.bytes_stream();

                'stream: while let Some(next) = byte_stream.next().await {
                    let bytes = match next {
                        Ok(b) => b,
                        Err(e) => {
                            let _ = tx_event
                                .send(Err(CodexErr::Fatal(format!(
                                    "llama-server stream error: {e}"
                                ))))
                                .await;
                            return;
                        }
                    };
                    sse_buf.extend_from_slice(&bytes);

                    // SSE events are separated by a blank line; each carries one
                    // `data: {...}` payload.
                    while let Some((sep, width)) = sse_buf.windows(2).position(|w| w == b"\n\n").map(|p| (p, 2))
                        .or_else(|| sse_buf.windows(4).position(|w| w == b"\r\n\r\n").map(|p| (p, 4))) {
                        let event_bytes: Vec<u8> = sse_buf.drain(..sep + width).collect();
                        let event_text = String::from_utf8_lossy(&event_bytes);
                        for line in event_text.lines() {
                            let Some(data) = line.strip_prefix("data:") else {
                                continue;
                            };
                            let Ok(chunk) = serde_json::from_str::<serde_json::Value>(data) else {
                                continue;
                            };
                            if let Some(error) = chunk.get("error").filter(|value| !value.is_null()) {
                                let _ = tx_event.send(Err(CodexErr::Fatal(format!(
                                    "llama-server stream error: {error}. Generation was interrupted."
                                )))).await;
                                return;
                            }
                            let is_final = chunk.get("stop").and_then(serde_json::Value::as_bool).unwrap_or(false);
                            if let Some(delta) = chunk.get("content").and_then(|v| v.as_str())
                                && !delta.is_empty()
                            {
                                let mut deltas = Vec::new();
                                splitter.push(delta, &mut deltas);
                                if !emit_region_deltas(
                                    &tx_event,
                                    deltas,
                                    &reasoning_id,
                                    &message_id,
                                    &mut reasoning_added,
                                    &mut message_added,
                                )
                                .await
                                {
                                    return; // consumer dropped
                                }
                            }
                            if is_final {
                                final_chunk = Some(chunk);
                                break 'stream;
                            }
                        }
                    }
                }

                drop(byte_stream);
                let Some(final_chunk) = final_chunk else {
                    let _ = tx_event.send(Err(CodexErr::Fatal(
                        "llama-server stream ended before completion. Generation was interrupted.".into()
                    ))).await;
                    return;
                };
                let finish_reason = match final_chunk.get("stop_type").and_then(|v| v.as_str()) {
                    Some("limit") => Some("length".to_string()),
                    Some("repetition") => Some(crate::pathology::REPETITION_FINISH_REASON.to_string()),
                    _ => Some("stop".to_string()),
                };
                let truncated = matches!(finish_reason.as_deref(), Some("length") | Some(crate::pathology::REPETITION_FINISH_REASON));

                // Flush any held-back tail (display-only; final items re-parse whole).
                let mut deltas = Vec::new();
                splitter.finish(&mut deltas);
                let _ = emit_region_deltas(
                    &tx_event,
                    deltas,
                    &reasoning_id,
                    &message_id,
                    &mut reasoning_added,
                    &mut message_added,
                )
                .await;

                let raw_content = splitter.acc.as_str();
                if let Some(rendered) = rendered_for_debug {
                    eprintln!(
                        "=== RENDERED PROMPT ===\n{rendered}\n=== RAW RESPONSE ===\n{raw_content}\n=== END ==="
                    );
                }

                for item in parse_final_items(raw_content, &prompt, &reasoning_id, &message_id) {
                    if truncated && matches!(item, ResponseItem::FunctionCall { .. } | ResponseItem::CustomToolCall { .. }) {
                        continue;
                    }
                    if tx_event
                        .send(Ok(ResponseEvent::OutputItemDone(item)))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }

                let token_usage = token_usage_from_final(&final_chunk);
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

/// Sends the streamed display deltas, lazily opening the reasoning/message
/// items with `OutputItemAdded` so the agent loop marks them active before
/// any delta arrives. Returns false if the consumer went away.
async fn emit_region_deltas(
    tx_event: &mpsc::Sender<Result<ResponseEvent>>,
    deltas: Vec<RegionDelta>,
    reasoning_id: &str,
    message_id: &str,
    reasoning_added: &mut bool,
    message_added: &mut bool,
) -> bool {
    for delta in deltas {
        let events: Vec<ResponseEvent> = match delta {
            RegionDelta::Reasoning(text) => {
                let mut evs = Vec::new();
                if !*reasoning_added {
                    *reasoning_added = true;
                    evs.push(ResponseEvent::OutputItemAdded(ResponseItem::Reasoning {
                        id: reasoning_id.to_string(),
                        summary: Vec::new(),
                        content: Some(Vec::new()),
                        encrypted_content: None,
                    }));
                }
                evs.push(ResponseEvent::ReasoningContentDelta {
                    delta: text,
                    content_index: 0,
                });
                evs
            }
            RegionDelta::Text(text) => {
                let mut evs = Vec::new();
                if !*message_added {
                    *message_added = true;
                    evs.push(ResponseEvent::OutputItemAdded(ResponseItem::Message {
                        id: Some(message_id.to_string()),
                        role: "assistant".to_string(),
                        content: Vec::new(),
                        phase: None,
                    }));
                }
                evs.push(ResponseEvent::OutputTextDelta(text));
                evs
            }
        };
        for ev in events {
            if tx_event.send(Ok(ev)).await.is_err() {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod turn_start_tests {
    use super::*;

    fn msg(role: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: role.to_string(),
            content: Vec::new(),
            phase: None,
        }
    }

    fn call() -> ResponseItem {
        ResponseItem::FunctionCall {
            id: None,
            name: "read_file".to_string(),
            namespace: None,
            arguments: "{}".to_string(),
            call_id: "c1".to_string(),
        }
    }

    #[test]
    fn user_message_last_is_turn_start() {
        assert!(is_turn_start(&[msg("user")]));
    }

    #[test]
    fn context_items_after_user_message_still_turn_start() {
        // Real sessions record skills/plugins injections and developer context
        // blocks AFTER the user message; those must not suppress the pin.
        assert!(is_turn_start(&[
            msg("user"),
            msg("developer"),
            ResponseItem::Other,
        ]));
    }

    #[test]
    fn mid_tool_loop_is_not_turn_start() {
        assert!(!is_turn_start(&[msg("user"), msg("assistant"), call()]));
    }

    #[test]
    fn assistant_reply_after_user_is_not_turn_start() {
        assert!(!is_turn_start(&[msg("user"), msg("assistant")]));
    }

    #[test]
    fn tool_call_then_context_item_is_not_turn_start() {
        // A trailing context item must not mask mid-loop tool activity.
        assert!(!is_turn_start(&[msg("user"), call(), ResponseItem::Other]));
    }

    #[test]
    fn no_user_message_is_not_turn_start() {
        assert!(!is_turn_start(&[msg("developer")]));
        assert!(!is_turn_start(&[]));
    }
}
