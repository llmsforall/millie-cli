use super::*;

#[derive(Clone)]
pub(crate) struct ModelBrokerRequestProcessor {
    thread_manager: Arc<ThreadManager>,
}

impl ModelBrokerRequestProcessor {
    pub(crate) fn new(thread_manager: Arc<ThreadManager>) -> Self {
        Self { thread_manager }
    }

    pub(crate) async fn complete(
        &self,
        params: ModelBrokerCompleteParams,
    ) -> Result<ModelBrokerCompleteResponse, JSONRPCErrorError> {
        let thread_id = ThreadId::from_string(&params.thread_id)
            .map_err(|err| invalid_request(format!("invalid thread id: {err}")))?;
        let thread = self
            .thread_manager
            .get_thread(thread_id)
            .await
            .map_err(|_| invalid_request(format!("thread not found: {thread_id}")))?;

        let response = thread
            .model_broker_complete(core_request_from_api(params))
            .await
            .map_err(|err| internal_error(err.to_string()))?;

        Ok(ModelBrokerCompleteResponse {
            results: response
                .results
                .into_iter()
                .map(|result| ModelBrokerResult {
                    id: result.id,
                    text: result.text,
                    response_id: result.response_id,
                    token_usage: result.token_usage,
                    model: result.model,
                    output_items: result.output_items,
                    function_calls: result
                        .function_calls
                        .into_iter()
                        .map(|call| ModelBrokerFunctionCall {
                            id: call.id,
                            name: call.name,
                            namespace: call.namespace,
                            arguments: call.arguments,
                            call_id: call.call_id,
                        })
                        .collect(),
                })
                .collect(),
        })
    }
}

fn core_request_from_api(
    params: ModelBrokerCompleteParams,
) -> codex_core::model_broker::ModelBrokerRequest {
    codex_core::model_broker::ModelBrokerRequest {
        calls: params
            .calls
            .into_iter()
            .map(|call| codex_core::model_broker::ModelBrokerCall {
                id: call.id,
                instructions: call.instructions,
                messages: call
                    .messages
                    .into_iter()
                    .map(|message| codex_core::model_broker::ModelBrokerMessage {
                        role: message.role,
                        content: message.content,
                        images: message
                            .images
                            .into_iter()
                            .map(|image| codex_core::model_broker::ModelBrokerImage {
                                image_url: image.image_url,
                                detail: image.detail.map(core_image_detail_from_api),
                            })
                            .collect(),
                    })
                    .collect(),
                model: call.model,
                reasoning_effort: call.reasoning_effort,
                reasoning_summary: call.reasoning_summary,
                service_tier: call.service_tier,
                output_schema: call.output_schema,
                tools: call.tools,
                tool_choice: call.tool_choice,
                parallel_tool_calls: call.parallel_tool_calls,
                previous_response_id: call.previous_response_id,
                replay_items: call.replay_items,
                input_items: call.input_items,
            })
            .collect(),
    }
}

fn core_image_detail_from_api(
    detail: ModelBrokerImageDetail,
) -> codex_protocol::models::ImageDetail {
    match detail {
        ModelBrokerImageDetail::Auto => codex_protocol::models::ImageDetail::Auto,
        ModelBrokerImageDetail::Low => codex_protocol::models::ImageDetail::Low,
        ModelBrokerImageDetail::High => codex_protocol::models::ImageDetail::High,
        ModelBrokerImageDetail::Original => codex_protocol::models::ImageDetail::Original,
    }
}
