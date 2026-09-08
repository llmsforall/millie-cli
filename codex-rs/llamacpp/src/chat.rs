//! OpenAI-messages builder for the llama.cpp chat-completions transport.
//!
//! The server renders these messages through the model's own
//! chat template (`millie-native.jinja`) and parses tool calls server-side
//! (PEG parser + lazy grammar), so this builder's only job is to express the
//! conversation in the OpenAI shape the template expects. It mirrors the
//! turn-folding semantics of `render_prompt` in `format.rs` (the raw
//! `/completion` path), with one deliberate difference: tool-call arguments
//! are passed through as the ORIGINAL JSON string from the agent loop, never
//! re-serialized, so key order and number formatting cannot drift from what
//! the model produced.

use codex_protocol::models::ResponseItem;
use codex_tools::ToolSpec;
use serde_json::json;

use crate::format::content_items_to_text_plain;
use crate::format::flatten_tool_name;
use crate::format::function_call_output_text;
use crate::format::media_payload;

/// One pending assistant message being assembled (mirrors `Turn::Assistant`).
struct AssistantMsg {
    content: String,
    reasoning: Option<String>,
    tool_calls: Vec<serde_json::Value>,
}

impl AssistantMsg {
    fn into_json(self) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("role".into(), json!("assistant"));
        if self.content.trim().is_empty() && !self.tool_calls.is_empty() {
            m.insert("content".into(), serde_json::Value::Null);
        } else {
            m.insert("content".into(), json!(self.content));
        }
        if let Some(r) = self.reasoning {
            // The template itself decides whether a <think> block renders
            // (only after the last user turn), so attaching reasoning to
            // older messages is harmless.
            m.insert("reasoning_content".into(), json!(r));
        }
        if !self.tool_calls.is_empty() {
            m.insert("tool_calls".into(), json!(self.tool_calls));
        }
        serde_json::Value::Object(m)
    }
}

/// A chat message's `content` value: a plain string normally, or an
/// OpenAI-style content-part array when the message carries images (the
/// server converts `image_url` parts into media tokens itself).
fn content_to_chat_value(items: &[codex_protocol::models::ContentItem]) -> serde_json::Value {
    use codex_protocol::models::ContentItem;
    let has_image = items.iter().any(|item| {
        matches!(item, ContentItem::InputImage { image_url, .. } if media_payload(image_url.as_str()).is_some())
    });
    if !has_image {
        return json!(content_items_to_text_plain(items));
    }
    let mut parts: Vec<serde_json::Value> = Vec::new();
    let mut text = String::new();
    for item in items {
        match item {
            ContentItem::InputText { text: t } | ContentItem::OutputText { text: t } => {
                text.push_str(t);
            }
            ContentItem::InputImage { image_url, .. } => {
                if media_payload(image_url.as_str()).is_some() {
                    if !text.is_empty() {
                        parts.push(json!({"type": "text", "text": std::mem::take(&mut text)}));
                    }
                    parts.push(json!({"type": "image_url", "image_url": {"url": image_url}}));
                }
            }
        }
    }
    if !text.is_empty() {
        parts.push(json!({"type": "text", "text": text}));
    }
    json!(parts)
}

/// Builds the `messages` array for `/v1/chat/completions`, including the
/// merged system message and (on fresh conversations) the format primer.
pub fn build_chat_messages(instructions: &str, input: &[ResponseItem]) -> Vec<serde_json::Value> {
    let mut messages: Vec<serde_json::Value> = Vec::new();

    // System preamble: explicit instructions win; a leading system message
    // in the input is the fallback (same precedence as render_prompt).
    let leading_system_text = match input.first() {
        Some(ResponseItem::Message { role, content, .. }) if role == "system" => {
            Some(content_items_to_text_plain(content))
        }
        _ => None,
    };
    let system_text = if !instructions.trim().is_empty() {
        instructions.trim().to_string()
    } else {
        leading_system_text.clone().unwrap_or_default()
    };
    if !system_text.is_empty() {
        messages.push(json!({"role": "system", "content": system_text}));
    }

    // Keep a fixed tool-call example at the start of the conversation to
    // establish the expected call format. Preserve it across turns so the
    // prompt prefix remains cacheable.
    if std::env::var("MILLIE_LLAMACPP_FORMAT_PRIMER").as_deref() != Ok("0") {
        // Render the assistant call as XML and the result as a
        // <tool_response> block inside a user turn.
        messages.push(
            json!({"role": "user", "content": "Before we start: confirm the working directory."}),
        );
        messages.push(json!({
            "role": "assistant",
            "content": serde_json::Value::Null,
            "tool_calls": [{
                "id": "call_formatPrimer0001",
                "type": "function",
                "function": {"name": "exec_command", "arguments": "{\"cmd\":\"pwd\"}"},
            }],
        }));
        messages.push(json!({"role": "tool", "tool_call_id": "call_formatPrimer0001", "content": "(working directory confirmed)"}));
        messages.push(json!({"role": "assistant", "content": "Working directory confirmed."}));
    }

    let mut pending: Option<AssistantMsg> = None;
    let flush = |pending: &mut Option<AssistantMsg>, messages: &mut Vec<serde_json::Value>| {
        if let Some(a) = pending.take() {
            messages.push(a.into_json());
        }
    };

    let mut saw_first_system = false;
    for item in input {
        match item {
            ResponseItem::Message { role, content, .. } => {
                match role.as_str() {
                    "system" => {
                        // Leading system already folded into the preamble;
                        // later system turns have no position in the native format.
                        if !saw_first_system {
                            saw_first_system = true;
                        }
                        flush(&mut pending, &mut messages);
                    }
                    "assistant" => {
                        flush(&mut pending, &mut messages);
                        pending = Some(AssistantMsg {
                            content: content_items_to_text_plain(content),
                            reasoning: None,
                            tool_calls: Vec::new(),
                        });
                    }
                    _ => {
                        flush(&mut pending, &mut messages);
                        messages.push(
                            json!({"role": "user", "content": content_to_chat_value(content)}),
                        );
                    }
                }
            }
            ResponseItem::Reasoning { content, .. } => {
                use codex_protocol::models::ReasoningItemContent;
                let text = content
                    .as_ref()
                    .map(|items| {
                        items
                            .iter()
                            .map(|c| match c {
                                ReasoningItemContent::ReasoningText { text }
                                | ReasoningItemContent::Text { text } => text.as_str(),
                            })
                            .collect::<Vec<_>>()
                            .join("")
                    })
                    .unwrap_or_default();
                flush(&mut pending, &mut messages);
                pending = Some(AssistantMsg {
                    content: String::new(),
                    reasoning: Some(text),
                    tool_calls: Vec::new(),
                });
            }
            ResponseItem::FunctionCall {
                name,
                namespace,
                arguments,
                call_id,
                ..
            } => {
                let flat_name = match namespace {
                    Some(ns) => flatten_tool_name(ns, name),
                    None => name.clone(),
                };
                let call = json!({
                    "id": call_id,
                    "type": "function",
                    "function": {"name": flat_name, "arguments": arguments},
                });
                // Merge into the immediately preceding assistant message if
                // one is open (parallel/back-to-back calls from one turn).
                match pending.as_mut() {
                    Some(a) => a.tool_calls.push(call),
                    None => {
                        pending = Some(AssistantMsg {
                            content: String::new(),
                            reasoning: None,
                            tool_calls: vec![call],
                        });
                    }
                }
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                flush(&mut pending, &mut messages);
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": function_call_output_text(&output.body),
                }));
            }
            // Item kinds with no equivalent in the native format are
            // skipped, matching render_prompt.
            _ => {}
        }
    }
    flush(&mut pending, &mut messages);

    messages
}

/// Builds the OpenAI `tools` array from the flattened tool list.
pub fn chat_tools_json(tools: &[ToolSpec]) -> Vec<serde_json::Value> {
    crate::flattened_tools_json(tools)
        .into_iter()
        .map(|t| json!({"type": "function", "function": t}))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::models::ContentItem;
    use codex_protocol::models::FunctionCallOutputBody;
    use codex_protocol::models::FunctionCallOutputPayload;

    fn user_msg(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: text.to_string(),
            }],
            phase: None,
        }
    }

    // Env mutation is process-global: serialize with every other env-mutating
    // test in the crate (format.rs pins the same variables).
    use crate::TEST_ENV_LOCK as PRIMER_ENV_LOCK;
    struct PrimerOff(std::sync::MutexGuard<'static, ()>);
    impl PrimerOff {
        fn new() -> Self {
            let g = PRIMER_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
            // SAFETY: serialized by the lock; tests only.
            unsafe { std::env::set_var("MILLIE_LLAMACPP_FORMAT_PRIMER", "0") };
            PrimerOff(g)
        }
    }
    impl Drop for PrimerOff {
        fn drop(&mut self) {
            // SAFETY: serialized by the lock held in self.0; tests only.
            unsafe { std::env::remove_var("MILLIE_LLAMACPP_FORMAT_PRIMER") };
        }
    }

    #[test]
    fn call_and_result_round_trip_with_original_argument_bytes() {
        let _guard = PrimerOff::new();
        let args = "{\"cmd\":\"ls\",\"workdir\":\"/workspace\"}";
        let input = vec![
            user_msg("do it"),
            ResponseItem::FunctionCall {
                id: None,
                name: "exec_command".to_string(),
                namespace: None,
                arguments: args.to_string(),
                call_id: "call_1".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call_1".to_string(),
                output: FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::Text("done".to_string()),
                    success: Some(true),
                },
            },
        ];
        let msgs = build_chat_messages("SYS", &input);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[1]["role"], "user");
        let call_msg = &msgs[2];
        assert_eq!(call_msg["role"], "assistant");
        assert!(call_msg["content"].is_null());
        assert_eq!(call_msg["tool_calls"][0]["function"]["arguments"], args);
        assert_eq!(call_msg["tool_calls"][0]["id"], "call_1");
        let tool_msg = &msgs[3];
        assert_eq!(tool_msg["role"], "tool");
        assert_eq!(tool_msg["tool_call_id"], "call_1");
        assert_eq!(tool_msg["content"], "done");
    }

    #[test]
    fn primer_present_regardless_of_history_for_stable_prefix() {
        // The primer must NOT come and go with history contents: dropping it
        // once a real call exists shifts every later token and diverges the
        // server's prompt cache mid-conversation (invalidating the pinned
        // turn-boundary checkpoint). It is rendered unconditionally.
        let _g = PRIMER_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let input = vec![user_msg("hi")];
        let msgs = build_chat_messages("SYS", &input);
        // system + 4 primer messages + user
        assert_eq!(msgs.len(), 6);
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[2]["tool_calls"][0]["id"], "call_formatPrimer0001");

        let seeded = vec![
            user_msg("hi"),
            ResponseItem::FunctionCall {
                id: None,
                name: "exec_command".to_string(),
                namespace: None,
                arguments: "{}".to_string(),
                call_id: "call_9".to_string(),
            },
        ];
        let msgs = build_chat_messages("SYS", &seeded);
        assert!(msgs.iter().any(|m| {
            m["tool_calls"]
                .as_array()
                .is_some_and(|c| c[0]["id"] == "call_formatPrimer0001")
        }));
    }

    #[test]
    fn reasoning_folds_into_following_call() {
        let _guard = PrimerOff::new();
        use codex_protocol::models::ReasoningItemContent;
        let input = vec![
            user_msg("go"),
            ResponseItem::Reasoning {
                id: "r1".to_string(),
                summary: Vec::new(),
                content: Some(vec![ReasoningItemContent::ReasoningText {
                    text: "thinking...".to_string(),
                }]),
                encrypted_content: None,
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "exec_command".to_string(),
                namespace: None,
                arguments: "{}".to_string(),
                call_id: "call_2".to_string(),
            },
        ];
        let msgs = build_chat_messages("SYS", &input);
        let a = &msgs[2];
        assert_eq!(a["role"], "assistant");
        assert_eq!(a["reasoning_content"], "thinking...");
        assert_eq!(a["tool_calls"][0]["id"], "call_2");
    }
}
