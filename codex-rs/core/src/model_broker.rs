use codex_protocol::config_types::ReasoningSummary as ReasoningSummaryConfig;
use codex_protocol::models::ImageDetail;
use codex_protocol::models::ResponseInputItem;
use codex_protocol::models::ResponseItem;
use codex_protocol::openai_models::ReasoningEffort as ReasoningEffortConfig;
use codex_protocol::protocol::TokenUsage;
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct ModelBrokerRequest {
    pub calls: Vec<ModelBrokerCall>,
}

#[derive(Debug, Clone)]
pub struct ModelBrokerCall {
    pub id: Option<String>,
    pub instructions: Option<String>,
    pub messages: Vec<ModelBrokerMessage>,
    pub model: Option<String>,
    pub reasoning_effort: Option<ReasoningEffortConfig>,
    pub reasoning_summary: Option<ReasoningSummaryConfig>,
    pub service_tier: Option<Option<String>>,
    pub output_schema: Option<Value>,
    pub tools: Vec<Value>,
    pub tool_choice: Option<String>,
    pub parallel_tool_calls: Option<bool>,
    pub previous_response_id: Option<String>,
    pub replay_items: Vec<ResponseItem>,
    pub input_items: Vec<ResponseInputItem>,
}

#[derive(Debug, Clone)]
pub struct ModelBrokerMessage {
    pub role: String,
    pub content: String,
    pub images: Vec<ModelBrokerImage>,
}

#[derive(Debug, Clone)]
pub struct ModelBrokerImage {
    pub image_url: String,
    pub detail: Option<ImageDetail>,
}

#[derive(Debug, Clone)]
pub struct ModelBrokerResponse {
    pub results: Vec<ModelBrokerResult>,
}

#[derive(Debug, Clone)]
pub struct ModelBrokerResult {
    pub id: Option<String>,
    pub text: String,
    pub response_id: Option<String>,
    pub token_usage: Option<TokenUsage>,
    pub model: String,
    pub output_items: Vec<ResponseItem>,
    pub function_calls: Vec<ModelBrokerFunctionCall>,
}

#[derive(Debug, Clone)]
pub struct ModelBrokerFunctionCall {
    pub id: Option<String>,
    pub name: String,
    pub namespace: Option<String>,
    pub arguments: String,
    pub call_id: String,
}
