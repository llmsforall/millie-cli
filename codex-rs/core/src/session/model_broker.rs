use super::*;
use crate::client_common::Prompt;
use crate::client_common::ResponseEvent;
use crate::model_broker::ModelBrokerCall;
use crate::model_broker::ModelBrokerFunctionCall;
use crate::model_broker::ModelBrokerMessage;
use crate::model_broker::ModelBrokerRequest;
use crate::model_broker::ModelBrokerResponse;
use crate::model_broker::ModelBrokerResult;
use codex_protocol::models::ContentItem;
use codex_protocol::models::ResponseItem;
use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use futures::StreamExt;
use futures::future::try_join_all;
use serde_json::Value;

impl Session {
    pub(crate) async fn model_broker_complete(
        &self,
        request: ModelBrokerRequest,
    ) -> CodexResult<ModelBrokerResponse> {
        let results = try_join_all(
            request
                .calls
                .into_iter()
                .map(|call| self.model_broker_complete_one(call)),
        )
        .await?;
        Ok(ModelBrokerResponse { results })
    }

    async fn model_broker_complete_one(
        &self,
        call: ModelBrokerCall,
    ) -> CodexResult<ModelBrokerResult> {
        let session_configuration = {
            let state = self.state.lock().await;
            state.session_configuration.clone()
        };

        let requested_model = call
            .model
            .as_deref()
            .unwrap_or_else(|| session_configuration.collaboration_mode.model());
        let mut per_call_config = (*session_configuration.original_config_do_not_use).clone();
        per_call_config.model = Some(requested_model.to_string());
        let model_info = self
            .services
            .models_manager
            .get_model_info(requested_model, &per_call_config.to_models_manager_config())
            .await;

        let mut prompt = Prompt {
            input: Vec::new(),
            tools: broker_tools_from_values(&call.tools)?,
            parallel_tool_calls: call.parallel_tool_calls.unwrap_or(false),
            tool_choice: call.tool_choice.clone(),
            previous_response_id: call.previous_response_id.clone(),
            base_instructions: BaseInstructions {
                text: call
                    .instructions
                    .clone()
                    .unwrap_or_else(|| session_configuration.base_instructions.clone()),
            },
            personality: session_configuration.personality,
            llamacpp_request_extra: None,
            output_schema: call.output_schema.clone(),
            output_schema_strict: true,
        };

        for message in &call.messages {
            append_broker_message(&mut prompt, message)?;
        }
        prompt.input.extend(call.replay_items);
        prompt
            .input
            .extend(call.input_items.into_iter().map(ResponseItem::from));

        let reasoning_effort = call
            .reasoning_effort
            .or_else(|| session_configuration.collaboration_mode.reasoning_effort())
            .or(model_info.default_reasoning_level);
        let reasoning_summary = call
            .reasoning_summary
            .or(session_configuration.model_reasoning_summary)
            .unwrap_or(model_info.default_reasoning_summary);
        let service_tier = call
            .service_tier
            .unwrap_or_else(|| session_configuration.service_tier.clone());
        let turn_metadata_header =
            crate::turn_metadata::build_turn_metadata_header(&session_configuration.cwd, None)
                .await;
        let inference_trace = self.services.rollout_thread_trace.inference_trace_context(
            "model_broker",
            model_info.slug.as_str(),
            session_configuration.provider.name.as_str(),
        );

        let mut client_session = self.services.model_client.new_session();
        let mut stream = client_session
            .stream(
                &prompt,
                &model_info,
                &self.services.session_telemetry,
                reasoning_effort,
                reasoning_summary,
                service_tier,
                turn_metadata_header.as_deref(),
                &inference_trace,
            )
            .await?;

        let mut text = String::new();
        let mut completed_text = String::new();
        let mut response_id = None;
        let mut token_usage = None;
        let mut output_items = Vec::new();
        let mut function_calls = Vec::new();
        while let Some(event) = stream.next().await {
            match event? {
                ResponseEvent::OutputTextDelta(delta) => text.push_str(&delta),
                ResponseEvent::OutputItemDone(item) => {
                    match &item {
                        ResponseItem::Message { role, content, .. } if role == "assistant" => {
                            completed_text.push_str(&content_items_text(content));
                        }
                        ResponseItem::FunctionCall {
                            id,
                            name,
                            namespace,
                            arguments,
                            call_id,
                        } => {
                            function_calls.push(ModelBrokerFunctionCall {
                                id: id.clone(),
                                name: name.clone(),
                                namespace: namespace.clone(),
                                arguments: arguments.clone(),
                                call_id: call_id.clone(),
                            });
                        }
                        _ => {}
                    }
                    output_items.push(item);
                }
                ResponseEvent::Completed {
                    response_id: id,
                    token_usage: usage,
                    ..
                } => {
                    response_id = Some(id);
                    token_usage = usage;
                }
                _ => {}
            }
        }

        if text.is_empty() {
            text = completed_text;
        }

        Ok(ModelBrokerResult {
            id: call.id,
            text,
            response_id,
            token_usage,
            model: model_info.slug,
            output_items,
            function_calls,
        })
    }
}

fn broker_tools_from_values(values: &[Value]) -> CodexResult<Vec<ToolSpec>> {
    values
        .iter()
        .map(broker_tool_from_value)
        .collect::<CodexResult<Vec<_>>>()
}

fn broker_tool_from_value(value: &Value) -> CodexResult<ToolSpec> {
    let object = value
        .as_object()
        .ok_or_else(|| CodexErr::InvalidRequest("broker tool must be an object".to_string()))?;
    if object.get("type").and_then(Value::as_str) != Some("function") {
        return Err(CodexErr::InvalidRequest(
            "model broker currently supports only function tools".to_string(),
        ));
    }

    let function = object
        .get("function")
        .and_then(Value::as_object)
        .unwrap_or(object);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| CodexErr::InvalidRequest("function tool missing name".to_string()))?
        .to_string();
    let description = function
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let parameters_value = function
        .get("parameters")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}}));
    let parameters: JsonSchema = serde_json::from_value(parameters_value).map_err(|err| {
        CodexErr::InvalidRequest(format!(
            "invalid parameters for function tool {name}: {err}"
        ))
    })?;
    let strict = function
        .get("strict")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    Ok(ToolSpec::Function(ResponsesApiTool {
        name,
        description,
        strict,
        defer_loading: None,
        parameters,
        output_schema: None,
    }))
}

fn append_broker_message(prompt: &mut Prompt, message: &ModelBrokerMessage) -> CodexResult<()> {
    match message.role.as_str() {
        "system" | "developer" => {
            if !message.content.trim().is_empty() {
                prompt.base_instructions.text.push_str("\n\n");
                prompt.base_instructions.text.push_str(&message.content);
            }
        }
        "user" => {
            let mut content = vec![ContentItem::InputText {
                text: message.content.clone(),
            }];
            content.extend(message.images.iter().map(|image| ContentItem::InputImage {
                image_url: image.image_url.clone(),
                detail: image.detail,
            }));
            prompt.input.push(ResponseItem::Message {
                id: None,
                role: "user".to_string(),
                content,
                phase: None,
            });
        }
        "assistant" => {
            prompt.input.push(ResponseItem::Message {
                id: None,
                role: "assistant".to_string(),
                content: vec![ContentItem::OutputText {
                    text: message.content.clone(),
                }],
                phase: None,
            });
        }
        other => {
            return Err(CodexErr::InvalidRequest(format!(
                "unsupported model broker message role: {other}"
            )));
        }
    }

    Ok(())
}

fn content_items_text(content: &[ContentItem]) -> String {
    content
        .iter()
        .filter_map(|item| match item {
            ContentItem::OutputText { text } | ContentItem::InputText { text } => {
                Some(text.as_str())
            }
            ContentItem::InputImage { .. } => None,
        })
        .collect()
}
