//! Prints a realistic multi-turn rendered prompt (tool call already made,
//! tool result already returned) so it can be manually fed to a running
//! llama-server via raw /completion to validate the continuation format.
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputBody;
use codex_protocol::models::FunctionCallOutputPayload;
use codex_protocol::models::ResponseItem;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;

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

fn main() {
    let tools = vec![ToolSpec::Function(ResponsesApiTool {
        name: "fetch_url".to_string(),
        description: "Fetch the contents of a URL.".to_string(),
        strict: false,
        defer_loading: None,
        parameters: serde_json::from_value(serde_json::json!({
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to fetch."}},
            "required": ["url"],
        }))
        .unwrap(),
        output_schema: None,
    })];

    let input = vec![
        user("What's on the front page of https://example.com right now? Use your tools to check."),
        ResponseItem::FunctionCall {
            id: None,
            name: "fetch_url".to_string(),
            namespace: None,
            arguments: serde_json::json!({"url": "https://example.com"}).to_string(),
            call_id: "call_1".to_string(),
        },
        ResponseItem::FunctionCallOutput {
            call_id: "call_1".to_string(),
            output: FunctionCallOutputPayload {
                body: FunctionCallOutputBody::Text(
                    "Example Domain\n\nThis domain is for use in illustrative examples in documents.".to_string(),
                ),
                success: Some(true),
            },
        },
    ];

    let rendered =
        codex_llamacpp::render_prompt("You are a helpful research assistant.", &input, &tools);
    print!("{rendered}");
}
