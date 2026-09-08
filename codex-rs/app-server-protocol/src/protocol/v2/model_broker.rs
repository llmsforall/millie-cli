use codex_protocol::config_types::ReasoningSummary;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort;
use codex_protocol::protocol::TokenUsage;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;
use serde_json::Value as JsonValue;
use ts_rs::TS;

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ModelBrokerCompleteParams {
    pub thread_id: String,
    pub calls: Vec<ModelBrokerCall>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ModelBrokerCall {
    #[ts(optional = nullable)]
    pub id: Option<String>,
    #[ts(optional = nullable)]
    pub instructions: Option<String>,
    pub messages: Vec<ModelBrokerMessage>,
    #[ts(optional = nullable)]
    pub model: Option<String>,
    #[ts(optional = nullable)]
    pub reasoning_effort: Option<ReasoningEffort>,
    #[ts(optional = nullable)]
    pub reasoning_summary: Option<ReasoningSummary>,
    #[ts(optional = nullable)]
    pub service_tier: Option<Option<String>>,
    #[ts(optional = nullable)]
    pub output_schema: Option<JsonValue>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<JsonValue>,
    #[ts(optional = nullable)]
    pub tool_choice: Option<String>,
    #[ts(optional = nullable)]
    pub parallel_tool_calls: Option<bool>,
    #[ts(optional = nullable)]
    pub previous_response_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub replay_items: Vec<ResponseItem>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub input_items: Vec<ResponseInputItem>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ModelBrokerMessage {
    pub role: String,
    pub content: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub images: Vec<ModelBrokerImage>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ModelBrokerImage {
    pub image_url: String,
    #[ts(optional = nullable)]
    pub detail: Option<ModelBrokerImageDetail>,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, JsonSchema, TS)]
#[serde(rename_all = "lowercase")]
#[ts(export_to = "v2/")]
pub enum ModelBrokerImageDetail {
    Auto,
    Low,
    High,
    Original,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ModelBrokerCompleteResponse {
    pub results: Vec<ModelBrokerResult>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ModelBrokerResult {
    #[ts(optional = nullable)]
    pub id: Option<String>,
    pub text: String,
    #[ts(optional = nullable)]
    pub response_id: Option<String>,
    #[ts(optional = nullable)]
    pub token_usage: Option<TokenUsage>,
    pub model: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub output_items: Vec<ResponseItem>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub function_calls: Vec<ModelBrokerFunctionCall>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, JsonSchema, TS)]
#[serde(rename_all = "camelCase")]
#[ts(export_to = "v2/")]
pub struct ModelBrokerFunctionCall {
    #[ts(optional = nullable)]
    pub id: Option<String>,
    pub name: String,
    #[ts(optional = nullable)]
    pub namespace: Option<String>,
    pub arguments: String,
    pub call_id: String,
}
