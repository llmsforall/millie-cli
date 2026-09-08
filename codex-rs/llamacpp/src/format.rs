//! Raw-text prompt construction matching this model family's native format,
//! translated directly from the model's own `chat_template.jinja` (tool
//! declaration block, `<|im_start|>role...<|im_end|>` turn wrapping,
//! `<tool_call>`/`<function=>`/`<parameter=>` serialization, and
//! `<tool_response>`-wrapped tool results) rather than relying on
//! llama-server's template-derived auto-parser, which does not handle this
//! template correctly (see crate-level notes).
//!
//! Tool calls and results use the XML wire format described below.

use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ResponseItem;
use codex_tools::ResponsesApiNamespaceTool;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use serde_json::json;

// XML tool calls use <function=NAME> and <parameter=k> with raw values.
// Tool results use <tool_response> blocks inside user turns.

const NAMESPACE_SEPARATOR: &str = "__";

/// Join a namespace and tool name with `__` (`files.read` ->
/// `files__read`) to match the tool-execution layer.
pub fn flatten_tool_name(namespace: &str, name: &str) -> String {
    format!("{namespace}{NAMESPACE_SEPARATOR}{name}")
}

pub fn unflatten_tool_name(flat_name: &str) -> Option<(String, String)> {
    flat_name
        .split_once(NAMESPACE_SEPARATOR)
        .map(|(ns, name)| (ns.to_string(), name.to_string()))
}

fn responses_api_tool_to_json(tool: &ResponsesApiTool) -> serde_json::Value {
    json!({
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    })
}

/// Flattens the tool list (functions + namespaced sub-tools) into the flat
/// `[{"name", "description", "parameters"}, ...]` shape the template's
/// `tool | tojson` expects, one entry per callable function.
/// Public view of the flattened tool list, for building the tool-call
/// grammar against exactly what the prompt advertises.
pub fn flattened_tools_json(tools: &[ToolSpec]) -> Vec<serde_json::Value> {
    flatten_tools(tools)
}

fn flatten_tools(tools: &[ToolSpec]) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    for tool in tools {
        match tool {
            ToolSpec::Function(f) => out.push(responses_api_tool_to_json(f)),
            ToolSpec::Namespace(ns) => {
                for nested in &ns.tools {
                    let ResponsesApiNamespaceTool::Function(f) = nested;
                    let mut v = responses_api_tool_to_json(f);
                    if let Some(obj) = v.as_object_mut() {
                        obj.insert(
                            "name".to_string(),
                            json!(flatten_tool_name(&ns.name, &f.name)),
                        );
                    }
                    out.push(v);
                }
            }
            // This item has no equivalent in the native tool-call format.
            ToolSpec::ToolSearch { .. }
            | ToolSpec::ImageGeneration { .. }
            | ToolSpec::WebSearch { .. }
            | ToolSpec::Freeform(_) => {}
        }
    }
    out
}

/// The exact tools-declaration block from the template (lines 45-53 of
/// `chat_template.jinja`), reproduced verbatim.
/// Map of flattened tool name -> parameter name -> declared JSON-schema type,
/// used by the transport to coerce raw string arguments back to typed JSON
/// (see `arguments_to_json_string` / `coerce_param_value`).
pub fn tool_param_types(
    tools: &[ToolSpec],
) -> std::collections::HashMap<String, std::collections::HashMap<String, String>> {
    let mut res = std::collections::HashMap::new();
    for t in flatten_tools(tools) {
        let Some(name) = t.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let mut params = std::collections::HashMap::new();
        if let Some(props) = t
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object())
        {
            for (k, v) in props {
                if let Some(ty) = v.get("type").and_then(|x| x.as_str()) {
                    params.insert(k.clone(), ty.to_string());
                }
            }
        }
        res.insert(name.to_string(), params);
    }
    res
}

fn render_tools_block(tools: &[ToolSpec]) -> String {
    let flat = flatten_tools(tools);
    let mut s = String::new();
    s.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
    for tool in &flat {
        s.push('\n');
        s.push_str(&serde_json::to_string(tool).unwrap_or_default());
    }
    s.push_str("\n</tools>");
    s.push_str(
        "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n\
value_1\n</parameter>\n<parameter=example_parameter_2>\n\
This is the value for the second parameter\nthat can span\nmultiple lines\n\
</parameter>\n</function>\n</tool_call>\n\n\
<IMPORTANT>\nReminder:\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
- Required parameters MUST be specified\n\
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
</IMPORTANT>",
    );
    s
}

/// The server's multimodal marker (`mtmd_default_marker()`): each occurrence
/// in the prompt is substituted, in order, by one `multimodal_data` entry.
pub const MEDIA_MARKER: &str = "<__media__>";

/// The raw base64 payload of an image data URL, when the image can actually
/// be shipped. Markers and payloads must stay 1:1: an image renders a marker
/// exactly when this returns Some, and `collect_media` uses the same test.
pub(crate) fn media_payload(image_url: &str) -> Option<&str> {
    let rest = image_url.strip_prefix("data:")?;
    let (meta, payload) = rest.split_once(',')?;
    if !meta.ends_with(";base64") || payload.is_empty() {
        return None;
    }
    Some(payload)
}

/// A literal marker inside text would desync the server's marker/payload
/// pairing, so break it up.
fn push_sanitized_text(out: &mut String, text: &str) {
    if text.contains(MEDIA_MARKER) {
        out.push_str(&text.replace(MEDIA_MARKER, "<_media_>"));
    } else {
        out.push_str(text);
    }
}

/// Text-only flattening: images contribute nothing. For the chat path,
/// where images travel as separate `image_url` content parts, and for
/// contexts that cannot carry media.
pub(crate) fn content_items_to_text_plain(items: &[ContentItem]) -> String {
    let mut out = String::new();
    for item in items {
        if let ContentItem::InputText { text } | ContentItem::OutputText { text } = item {
            push_sanitized_text(&mut out, text);
        }
    }
    out
}

/// Raw-completions flattening: each shippable image renders a media marker
/// (the payloads travel via `collect_media`).
pub(crate) fn content_items_to_text(items: &[ContentItem]) -> String {
    let mut out = String::new();
    for item in items {
        match item {
            ContentItem::InputText { text } | ContentItem::OutputText { text } => {
                push_sanitized_text(&mut out, text);
            }
            ContentItem::InputImage { image_url, .. } => {
                if media_payload(image_url).is_some() {
                    out.push_str(MEDIA_MARKER);
                }
            }
        }
    }
    out
}

/// Base64 payloads for every image the rendered prompt marks, in marker
/// order (the renderer walks `input` in the same order). Send these as the
/// request's `multimodal_data` alongside the prompt.
pub fn collect_media(input: &[ResponseItem]) -> Vec<String> {
    let mut media = Vec::new();
    for item in input {
        match item {
            ResponseItem::Message { content, .. } => {
                for c in content {
                    if let ContentItem::InputImage { image_url, .. } = c {
                        if let Some(payload) = media_payload(image_url) {
                            media.push(payload.to_string());
                        }
                    }
                }
            }
            ResponseItem::FunctionCallOutput { output, .. } => {
                if let FunctionCallOutputBody::ContentItems(items) = &output.body {
                    for c in items {
                        if let FunctionCallOutputContentItem::InputImage { image_url, .. } = c {
                            if let Some(payload) = media_payload(image_url) {
                                media.push(payload.to_string());
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    media
}

pub(crate) fn function_call_output_text(body: &FunctionCallOutputBody) -> String {
    match body {
        FunctionCallOutputBody::Text(text) => {
            let mut out = String::new();
            push_sanitized_text(&mut out, text);
            out
        }
        FunctionCallOutputBody::ContentItems(items) => {
            let mut segments: Vec<String> = Vec::new();
            for item in items {
                match item {
                    FunctionCallOutputContentItem::InputText { text }
                        if !text.trim().is_empty() =>
                    {
                        let mut s = String::new();
                        push_sanitized_text(&mut s, text);
                        segments.push(s);
                    }
                    FunctionCallOutputContentItem::InputImage { image_url, .. } => {
                        if media_payload(image_url).is_some() {
                            segments.push(MEDIA_MARKER.to_string());
                        }
                    }
                    _ => {}
                }
            }
            segments.join("\n")
        }
    }
}

/// A single call within one assistant turn, already flattened to the
/// `name`/`arguments` shape the `<function=NAME>`/`<parameter=NAME>`
/// serialization needs. `arguments` holds the JSON object's top-level
/// key/value pairs in DOCUMENT order (each becomes one `<parameter=key>`
/// block); a plain `serde_json::Value` object would alphabetize the keys
/// and diverge byte-wise from the order the model generated.
struct FlatToolCall {
    name: String,
    arguments: Vec<(String, serde_json::Value)>,
    call_id: String,
}

fn serialize_tool_calls(calls: &[FlatToolCall], content_is_nonempty: bool) -> String {
    let mut s = String::new();
    for (i, call) in calls.iter().enumerate() {
        if i == 0 {
            if content_is_nonempty {
                s.push_str("\n\n<tool_call>\n<function=");
            } else {
                s.push_str("<tool_call>\n<function=");
            }
        } else {
            s.push_str("\n<tool_call>\n<function=");
        }
        s.push_str(&call.name);
        s.push_str(">\n");
        {
            for (name, value) in &call.arguments {
                s.push_str("<parameter=");
                s.push_str(name);
                s.push_str(">\n");
                let value_text = match value {
                    serde_json::Value::String(text) => text.clone(),
                    other => other.to_string(),
                };
                s.push_str(&value_text);
                s.push_str("\n</parameter>\n");
            }
        }
        s.push_str("</function>\n</tool_call>");
    }
    s
}

/// One resolved turn boundary as the template sees it: an original-role
/// message, or a `FunctionCall`/`FunctionCallOutput` pair folded into the
/// assistant/tool structure the template expects.
enum Turn {
    System(String),
    User(String),
    /// Assistant turn: plain content plus zero or more tool calls emitted
    /// in the same turn (parallel calls).
    Assistant {
        content: String,
        reasoning: Option<String>,
        tool_calls: Vec<FlatToolCall>,
    },
    /// One tool result. Consecutive `ToolResult`s get batched into a single
    /// `<|im_start|>user` block by the renderer, matching
    /// `loop.previtem.role != "tool"` in the template.
    ToolResult {
        call_id: String,
        output: String,
    },
}

/// Converts the flat `ResponseItem` list into `Turn`s, pairing each
/// `FunctionCall` with its later `FunctionCallOutput` by `call_id` where
/// possible is unnecessary here since the template only cares about
/// *emission order*, not pairing -- it renders whatever it's given, in
/// order. We fold consecutive `FunctionCall`s that share an assistant
/// message boundary (i.e. arrive back-to-back with no intervening
/// output-carrying item) into one `Assistant` turn's `tool_calls`, matching
/// how the agent loop actually emits parallel calls from one turn.
fn to_turns(input: &[ResponseItem]) -> Vec<Turn> {
    let mut turns = Vec::new();
    for item in input {
        match item {
            ResponseItem::Message { role, content, .. } => {
                let text = content_items_to_text(content);
                match role.as_str() {
                    "system" => turns.push(Turn::System(text)),
                    "user" => turns.push(Turn::User(text)),
                    "assistant" => turns.push(Turn::Assistant {
                        content: text,
                        reasoning: None,
                        tool_calls: Vec::new(),
                    }),
                    _ => turns.push(Turn::User(text)),
                }
            }
            ResponseItem::Reasoning { content, .. } => {
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
                // Attach to a following assistant turn if one hasn't
                // started yet; otherwise carry as its own empty-content
                // assistant turn so the reasoning isn't dropped.
                turns.push(Turn::Assistant {
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
                let args_pairs = crate::parse::ordered_object_pairs(arguments).unwrap_or_default();
                let call = FlatToolCall {
                    name: flat_name,
                    arguments: args_pairs,
                    call_id: call_id.clone(),
                };
                // Merge into the immediately preceding assistant turn if
                // there is one (this is how parallel/back-to-back calls
                // from one turn are represented), else start a new one.
                if let Some(Turn::Assistant { tool_calls, .. }) = turns.last_mut() {
                    tool_calls.push(call);
                } else {
                    turns.push(Turn::Assistant {
                        content: String::new(),
                        reasoning: None,
                        tool_calls: vec![call],
                    });
                }
            }
            ResponseItem::FunctionCallOutput { call_id, output } => {
                turns.push(Turn::ToolResult {
                    call_id: call_id.clone(),
                    output: function_call_output_text(&output.body),
                });
            }
            // Other item kinds (local shell, MCP, custom tool, tool
            // search) have no equivalent in the native format;
            // skipped rather than guessed at.
            _ => {}
        }
    }
    turns
}

/// Renders `instructions` + `input` + `tools` into the exact raw text this
/// model expects, ending with `<|im_start|>assistant\n` so a raw
/// `/completion` call continues generation from there.
pub fn render_prompt(instructions: &str, input: &[ResponseItem], tools: &[ToolSpec]) -> String {
    let mut out = String::new();

    let turns = to_turns(input);

    // last_query_index: index (within `turns`) of the last turn that is a
    // *real* user turn, i.e. not a ToolResult. Assistant turns after this
    // point get a rendered <think> block; earlier ones (mid-history) don't.
    // Mirrors `ns.last_query_index` in the template.
    let last_query_index = turns
        .iter()
        .enumerate()
        .rev()
        .find(|(_, t)| matches!(t, Turn::User(_)))
        .map(|(i, _)| i);

    // System + tools preamble (template lines 45-65).
    let has_tools = !tools.is_empty();
    let leading_system_text = match turns.first() {
        Some(Turn::System(text)) => Some(text.clone()),
        _ => None,
    };
    // Keep a fixed tool-call example at the start of the conversation to
    // establish the expected call format. Preserve it across turns so the
    // prompt prefix remains cacheable. MILLIE_LLAMACPP_FORMAT_PRIMER=0 disables.
    fn render_format_primer(out: &mut String) {
        out.push_str(
            "<|im_start|>user\nBefore we start: confirm the working directory.<|im_end|>\n",
        );
        out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n<function=exec_command>\n<parameter=cmd>\npwd\n</parameter>\n</function>\n</tool_call><|im_end|>\n");
        out.push_str("<|im_start|>user\n<tool_response>\n(working directory confirmed)\n</tool_response><|im_end|>\n");
        out.push_str("<|im_start|>assistant\n<think>\n\n</think>\n\nWorking directory confirmed.<|im_end|>\n");
    }

    // Tool declarations are omitted by default. Set
    // MILLIE_LLAMACPP_TOOLS_BLOCK=1 to advertise additional tools.
    let render_block = std::env::var("MILLIE_LLAMACPP_TOOLS_BLOCK").as_deref() == Ok("1");
    if has_tools {
        out.push_str("<|im_start|>system\n");
        if render_block {
            out.push_str(&render_tools_block(tools));
        }
        let combined_system = if !instructions.trim().is_empty() {
            instructions.to_string()
        } else {
            leading_system_text.clone().unwrap_or_default()
        };
        if !combined_system.trim().is_empty() {
            out.push_str("\n\n");
            out.push_str(combined_system.trim());
        }
        out.push_str("<|im_end|>\n");
    } else if !instructions.trim().is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(instructions.trim());
        out.push_str("<|im_end|>\n");
    } else if let Some(text) = &leading_system_text {
        out.push_str("<|im_start|>system\n");
        out.push_str(text.trim());
        out.push_str("<|im_end|>\n");
    }

    // Preserve the primer across turns so the prompt prefix remains cacheable.
    if has_tools && std::env::var("MILLIE_LLAMACPP_FORMAT_PRIMER").as_deref() != Ok("0") {
        render_format_primer(&mut out);
    }

    let mut i = 0;
    while i < turns.len() {
        match &turns[i] {
            Turn::System(_) => {
                // System messages are folded into the preamble; later system
                // turns are not supported by this template.
            }
            Turn::User(text) => {
                out.push_str("<|im_start|>user\n");
                out.push_str(text);
                out.push_str("<|im_end|>\n");
            }
            Turn::Assistant {
                content,
                reasoning,
                tool_calls,
            } => {
                let render_reasoning = last_query_index.is_none_or(|idx| i > idx);
                out.push_str("<|im_start|>assistant\n");
                if render_reasoning {
                    out.push_str("<think>\n");
                    out.push_str(reasoning.as_deref().unwrap_or("").trim());
                    out.push_str("\n</think>\n\n");
                }
                out.push_str(content);
                if !tool_calls.is_empty() {
                    out.push_str(&serialize_tool_calls(
                        tool_calls,
                        !content.trim().is_empty(),
                    ));
                }
                out.push_str("<|im_end|>\n");
            }
            Turn::ToolResult { output, .. } => {
                // Batch consecutive tool results into one user turn containing
                // <tool_response> sections.
                out.push_str("<|im_start|>user");
                out.push_str("\n<tool_response>\n");
                out.push_str(output);
                out.push_str("\n</tool_response>");
                let mut j = i + 1;
                while let Some(Turn::ToolResult {
                    output: next_text, ..
                }) = turns.get(j)
                {
                    out.push_str("\n<tool_response>\n");
                    out.push_str(next_text);
                    out.push_str("\n</tool_response>");
                    j += 1;
                }
                out.push_str("<|im_end|>\n");
                i = j - 1;
            }
        }
        i += 1;
    }

    // Open the reasoning block after the assistant header.
    // Matches the chat template's generation prompt. The raw-path parser
    // (`split_think_block` in the transport) therefore treats the output as
    // opening inside the reasoning region.
    out.push_str(GENERATION_PROMPT_SUFFIX);
    out
}

/// The trailing tokens `render_prompt` appends after the history: the assistant
/// header plus the primed `<think>`. On a turn-start request these are exactly
/// the tokens that follow the user message, so their token count is the offset
/// (from the prompt end) at which the server should pin the turn-boundary
/// recurrent checkpoint. Keep in sync with the suffix pushed in `render_prompt`.
pub const GENERATION_PROMPT_SUFFIX: &str = "<|im_start|>assistant\n<think>\n";

#[cfg(test)]
mod tests {
    #[test]
    fn images_render_markers_and_collect_media_in_order() {
        use codex_protocol::models::*;
        let png = "data:image/png;base64,AAAA";
        let jpg = "data:image/jpeg;base64,BBBB";
        let input = vec![
            ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content: vec![
                    ContentItem::InputText {
                        text: "look: ".to_string(),
                    },
                    ContentItem::InputImage {
                        image_url: png.to_string(),
                        detail: None,
                    },
                ],
                phase: None,
            },
            ResponseItem::FunctionCallOutput {
                call_id: "c1".to_string(),
                output: FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::ContentItems(vec![
                        FunctionCallOutputContentItem::InputText {
                            text: "shot:".to_string(),
                        },
                        FunctionCallOutputContentItem::InputImage {
                            image_url: jpg.to_string(),
                            detail: None,
                        },
                    ]),
                    success: Some(true),
                },
            },
        ];
        let prompt = crate::render_prompt("sys", &input, &[]);
        assert_eq!(
            prompt.matches(crate::format::MEDIA_MARKER).count(),
            2,
            "{prompt}"
        );
        let media = crate::collect_media(&input);
        assert_eq!(media, vec!["AAAA".to_string(), "BBBB".to_string()]);
    }

    use super::*;
    use codex_protocol::models::FunctionCallOutputPayload;

    fn user(text: &str) -> ResponseItem {
        ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: text.to_string(),
            }],
            phase: None,
        }
    }

    #[test]
    fn renders_tools_block_and_trailing_assistant_marker() {
        // this golden asserts the legacy tools-block + bare render: pin the
        // xml dialect, re-enable the block, and disable the primer
        let _g = pin_wire();
        unsafe {
            std::env::set_var("MILLIE_LLAMACPP_TOOLS_BLOCK", "1");
            std::env::set_var("MILLIE_LLAMACPP_FORMAT_PRIMER", "0");
        }
        let tools = vec![ToolSpec::Function(ResponsesApiTool {
            name: "fetch_url".to_string(),
            description: "Fetch a URL.".to_string(),
            strict: false,
            defer_loading: None,
            parameters: serde_json::from_value(json!({
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            }))
            .unwrap(),
            output_schema: None,
        })];
        let input = vec![user("hello")];
        let rendered = render_prompt("You are helpful.", &input, &tools);
        assert!(rendered.starts_with("<|im_start|>system\n# Tools"));
        assert!(rendered.contains("\"name\":\"fetch_url\""));
        assert!(rendered.contains("You are helpful."));
        assert!(rendered.contains("<|im_start|>user\nhello<|im_end|>\n"));
        // The generation prompt primes the think block (the model does not
        // emit the opening tag itself).
        assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n"));
    }

    // Same lock as chat.rs's primer tests: both mutate MILLIE_LLAMACPP_FORMAT_PRIMER.
    use crate::TEST_ENV_LOCK as WIRE_ENV_LOCK;

    // Serializes tests that mutate MILLIE_LLAMACPP_TOOLS_BLOCK /
    // _FORMAT_PRIMER (there is no longer a wire-format env var).
    struct WireGuard(std::sync::MutexGuard<'static, ()>);
    fn pin_wire() -> WireGuard {
        WireGuard(WIRE_ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner()))
    }
    impl Drop for WireGuard {
        fn drop(&mut self) {
            unsafe {
                std::env::remove_var("MILLIE_LLAMACPP_TOOLS_BLOCK");
                std::env::remove_var("MILLIE_LLAMACPP_FORMAT_PRIMER");
            }
        }
    }

    #[test]
    fn render_matches_native_xml_shape() {
        // The one wire format is the native XML dialect: <function=>/
        // <parameter=> calls with raw values, tool results as <tool_response>
        // inside a user turn, no call_id. Preserve the native serialization contract.
        let _g = pin_wire();
        unsafe { std::env::set_var("MILLIE_LLAMACPP_FORMAT_PRIMER", "0") };
        let items = vec![
            ResponseItem::FunctionCall {
                id: None,
                name: "exec_command".to_string(),
                namespace: None,
                arguments: "{\"cmd\":\"ls\"}".to_string(),
                call_id: "call_x".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "call_x".to_string(),
                output: codex_protocol::models::FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::Text("ok".to_string()),
                    success: None,
                },
            },
        ];
        let out = render_prompt("sys", &items, &[]);
        assert!(
            out.contains(
                "<tool_call>\n<function=exec_command>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>"
            ),
            "call render: {out}"
        );
        assert!(
            out.contains("<|im_start|>user\n<tool_response>\nok\n</tool_response><|im_end|>"),
            "result render: {out}"
        );
        assert!(
            !out.contains("call_id"),
            "no call_id in the native format: {out}"
        );
    }

    #[test]
    fn batches_consecutive_tool_results_into_one_user_block() {
        let _g = pin_wire();
        let input = vec![
            user("do two things"),
            ResponseItem::FunctionCall {
                id: None,
                name: "a".to_string(),
                namespace: None,
                arguments: "{}".to_string(),
                call_id: "1".to_string(),
            },
            ResponseItem::FunctionCall {
                id: None,
                name: "b".to_string(),
                namespace: None,
                arguments: "{}".to_string(),
                call_id: "2".to_string(),
            },
            ResponseItem::FunctionCallOutput {
                call_id: "1".to_string(),
                output: FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::Text("result a".to_string()),
                    success: Some(true),
                },
            },
            ResponseItem::FunctionCallOutput {
                call_id: "2".to_string(),
                output: FunctionCallOutputPayload {
                    body: FunctionCallOutputBody::Text("result b".to_string()),
                    success: Some(true),
                },
            },
        ];
        let rendered = render_prompt("", &input, &[]);
        let user_blocks = rendered.matches("<|im_start|>user").count();
        // one for "do two things", one batched block for both tool results
        assert_eq!(user_blocks, 2);
        assert!(rendered.contains(
            "<|im_start|>user\n<tool_response>\nresult a\n</tool_response>\n<tool_response>\nresult b\n</tool_response><|im_end|>\n"
        ));
    }
}
