//! Repeated-tool-call guard.
//!
//! A model that issues the same tool call again and again is usually stuck.
//! This module recognises an exact repeat (same tool, same canonical
//! arguments) -- either identical to the immediately previous call or at least
//! three times within the last ten calls -- and, depending on
//! `[tools] repeat_guard`, appends a warning to the tool's output (`warn`),
//! only records it in the model-call log (`log`), or does nothing (`off`).
//! The call itself always runs; nothing is blocked.

use codex_config::types::RepeatGuardMode;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ResponseInputItem;
use codex_tools::ToolOutput;
use codex_tools::ToolPayload;
use serde_json::Value as JsonValue;
use std::collections::VecDeque;

/// How many recent calls the "3 of the last 10" rule looks at.
pub(crate) const WINDOW: usize = 10;
/// Repeats within the window that trigger the guard.
pub(crate) const REPEAT_THRESHOLD: usize = 3;

/// One remembered tool call.
#[derive(Debug, Clone)]
pub(crate) struct RecentToolCall {
    pub(crate) call_id: String,
    pub(crate) signature: String,
    /// The call's arguments as text, for the loop guard's penalty spans.
    pub(crate) arguments: String,
}

/// What the guard concluded for one call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RepeatVerdict {
    /// Same as the immediately previous tool call.
    pub(crate) immediate_repeat: bool,
    /// How many of the last `WINDOW` calls (before this one) were identical.
    pub(crate) repeats_in_window: usize,
}

impl RepeatVerdict {
    pub(crate) fn triggered(&self) -> bool {
        self.immediate_repeat || self.repeats_in_window + 1 >= REPEAT_THRESHOLD
    }
}

/// Canonical form of a tool call: tool name plus arguments with JSON keys
/// sorted recursively, so two calls that differ only in key order or
/// whitespace between tokens compare equal. Non-JSON payloads are trimmed.
pub(crate) fn signature(tool_name: &str, payload: &ToolPayload) -> String {
    let args = payload.log_payload();
    let canonical = match serde_json::from_str::<JsonValue>(&args) {
        Ok(value) => canonical_json(&value),
        Err(_) => args.trim().to_string(),
    };
    format!("{tool_name}\u{0}{canonical}")
}

fn canonical_json(value: &JsonValue) -> String {
    fn sort(value: &JsonValue) -> JsonValue {
        match value {
            JsonValue::Object(map) => {
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                let mut out = serde_json::Map::new();
                for k in keys {
                    out.insert(k.clone(), sort(&map[k]));
                }
                JsonValue::Object(out)
            }
            JsonValue::Array(items) => JsonValue::Array(items.iter().map(sort).collect()),
            other => other.clone(),
        }
    }
    serde_json::to_string(&sort(value)).unwrap_or_default()
}

/// Judge `signature` against the remembered calls (excluding `call_id`
/// itself, in case it was already recorded), then remember it.
pub(crate) fn judge_and_remember(
    recent: &mut VecDeque<RecentToolCall>,
    call_id: &str,
    signature: &str,
    arguments: &str,
) -> RepeatVerdict {
    let prior: Vec<&RecentToolCall> = recent.iter().filter(|c| c.call_id != call_id).collect();
    let immediate_repeat = prior.last().is_some_and(|c| c.signature == signature);
    let repeats_in_window = prior
        .iter()
        .rev()
        .take(WINDOW)
        .filter(|c| c.signature == signature)
        .count();
    recent.push_back(RecentToolCall {
        call_id: call_id.to_string(),
        signature: signature.to_string(),
        arguments: arguments.to_string(),
    });
    while recent.len() > WINDOW + 1 {
        recent.pop_front();
    }
    RepeatVerdict {
        immediate_repeat,
        repeats_in_window,
    }
}

/// The text appended to a repeated call's output in `warn` mode.
pub(crate) fn warning_text(tool_name: &str, arguments: &str, verdict: &RepeatVerdict) -> String {
    let shown = if arguments.chars().count() > 400 {
        let cut: String = arguments.chars().take(400).collect();
        format!("{cut}...")
    } else {
        arguments.to_string()
    };
    let count = verdict.repeats_in_window + 1;
    let when = if verdict.immediate_repeat {
        "and it is the same as your previous call"
    } else {
        "within your last ten tool calls"
    };
    format!(
        "\n\nWARNING: this is call #{count} of the identical tool call `{tool_name}` with arguments {shown} {when}. \
         Repeating an identical call rarely changes the outcome. Do something different unless you have a \
         clear reason to repeat it: read the file or output again, change the anchor or the command, or take a \
         different approach."
    )
}

/// Which mode to apply, from config.
pub(crate) fn mode(config: &crate::config::Config) -> RepeatGuardMode {
    config.repeat_guard
}

/// A tool output with a warning appended to its model-visible text.
pub(crate) struct WarnedToolOutput {
    pub(crate) inner: Box<dyn ToolOutput>,
    pub(crate) suffix: String,
}

impl ToolOutput for WarnedToolOutput {
    fn log_preview(&self) -> String {
        format!("{}{}", self.inner.log_preview(), self.suffix)
    }
    fn success_for_logging(&self) -> bool {
        self.inner.success_for_logging()
    }
    fn to_response_item(&self, call_id: &str, payload: &ToolPayload) -> ResponseInputItem {
        let mut item = self.inner.to_response_item(call_id, payload);
        match &mut item {
            ResponseInputItem::FunctionCallOutput { output, .. }
            | ResponseInputItem::CustomToolCallOutput { output, .. } => {
                append_to_body(&mut output.body, &self.suffix);
            }
            _ => {}
        }
        item
    }
    fn post_tool_use_id(&self, call_id: &str) -> String {
        self.inner.post_tool_use_id(call_id)
    }
    fn post_tool_use_input(&self, payload: &ToolPayload) -> Option<JsonValue> {
        self.inner.post_tool_use_input(payload)
    }
    fn post_tool_use_response(&self, call_id: &str, payload: &ToolPayload) -> Option<JsonValue> {
        self.inner.post_tool_use_response(call_id, payload)
    }
    fn code_mode_result(&self, payload: &ToolPayload) -> JsonValue {
        self.inner.code_mode_result(payload)
    }
}

fn append_to_body(body: &mut FunctionCallOutputBody, suffix: &str) {
    match body {
        FunctionCallOutputBody::Text(text) => text.push_str(suffix),
        FunctionCallOutputBody::ContentItems(items) => {
            items.push(FunctionCallOutputContentItem::InputText {
                text: suffix.to_string(),
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn payload(args: &str) -> ToolPayload {
        ToolPayload::Function {
            arguments: args.to_string(),
        }
    }

    #[test]
    fn signature_ignores_key_order_and_whitespace() {
        let a = signature("str_replace", &payload(r#"{"path":"a.py","old_str":"x","new_str":"y"}"#));
        let b = signature("str_replace", &payload(r#"{ "new_str": "y", "old_str": "x", "path": "a.py" }"#));
        assert_eq!(a, b);
        let c = signature("str_replace", &payload(r#"{"path":"b.py","old_str":"x","new_str":"y"}"#));
        assert_ne!(a, c);
        assert_ne!(a, signature("create_file", &payload(r#"{"path":"a.py","old_str":"x","new_str":"y"}"#)));
    }

    #[test]
    fn immediate_repeat_triggers() {
        let mut recent = VecDeque::new();
        let v1 = judge_and_remember(&mut recent, "c1", "sig", "");
        assert!(!v1.triggered());
        let v2 = judge_and_remember(&mut recent, "c2", "sig", "");
        assert!(v2.immediate_repeat && v2.triggered());
    }

    #[test]
    fn three_in_window_triggers_even_when_not_adjacent() {
        let mut recent = VecDeque::new();
        judge_and_remember(&mut recent, "c1", "sig", "");
        judge_and_remember(&mut recent, "c2", "other", "");
        let v = judge_and_remember(&mut recent, "c3", "sig", "");
        assert!(!v.triggered(), "second occurrence, not adjacent: no warning yet");
        judge_and_remember(&mut recent, "c4", "other", "");
        let v = judge_and_remember(&mut recent, "c5", "sig", "");
        assert_eq!(v.repeats_in_window, 2);
        assert!(v.triggered());
    }

    #[test]
    fn window_forgets_old_calls() {
        let mut recent = VecDeque::new();
        judge_and_remember(&mut recent, "c0", "sig", "");
        for i in 1..=WINDOW {
            judge_and_remember(&mut recent, &format!("x{i}"), "other", "");
        }
        let v = judge_and_remember(&mut recent, "c9", "sig", "");
        assert_eq!(v.repeats_in_window, 0);
        assert!(!v.triggered());
    }

    #[test]
    fn current_call_already_recorded_is_not_counted_against_itself() {
        let mut recent = VecDeque::new();
        recent.push_back(RecentToolCall {
            call_id: "c1".into(),
            signature: "sig".into(), arguments: String::new(),
        });
        let v = judge_and_remember(&mut recent, "c1", "sig", "");
        assert!(!v.triggered());
    }

    #[test]
    fn warning_names_the_call_and_the_count() {
        let v = RepeatVerdict {
            immediate_repeat: true,
            repeats_in_window: 1,
        };
        let w = warning_text("shell", r#"{"command":"ls"}"#, &v);
        assert!(w.contains("call #2"));
        assert!(w.contains("`shell`"));
        assert!(w.contains(r#"{"command":"ls"}"#));
        assert!(w.contains("previous call"));
    }
}
