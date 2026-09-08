use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Instant;

use tokio::sync::RwLock;
use tokio::task::JoinError;
use tokio_util::either::Either;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use tracing::Instrument;
use tracing::instrument;
use tracing::trace_span;

use crate::function_tool::FunctionCallError;
use crate::session::session::Session;
use crate::session::turn_context::TurnContext;
use crate::tools::context::AbortedToolOutput;
use crate::tools::context::SharedTurnDiffTracker;
use crate::tools::context::ToolPayload;
use crate::tools::lifecycle::notify_tool_aborted;
use crate::tools::registry::AnyToolResult;
use crate::tools::registry::ToolArgumentDiffConsumer;
use crate::tools::router::ToolCall;
use crate::tools::router::ToolCallSource;
use crate::tools::router::ToolRouter;
use codex_protocol::error::CodexErr;
use codex_protocol::models::ResponseInputItem;

#[derive(Clone)]
pub(crate) struct ToolCallRuntime {
    router: Arc<ToolRouter>,
    session: Arc<Session>,
    turn_context: Arc<TurnContext>,
    tracker: SharedTurnDiffTracker,
    parallel_execution: Arc<RwLock<()>>,
}

impl ToolCallRuntime {
    pub(crate) fn new(
        router: Arc<ToolRouter>,
        session: Arc<Session>,
        turn_context: Arc<TurnContext>,
        tracker: SharedTurnDiffTracker,
    ) -> Self {
        Self {
            router,
            session,
            turn_context,
            tracker,
            parallel_execution: Arc::new(RwLock::new(())),
        }
    }

    pub(crate) fn create_diff_consumer(
        &self,
        tool_name: &codex_tools::ToolName,
    ) -> Option<Box<dyn ToolArgumentDiffConsumer>> {
        self.router.create_diff_consumer(tool_name)
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) fn handle_tool_call(
        self,
        call: ToolCall,
        cancellation_token: CancellationToken,
    ) -> impl std::future::Future<Output = Result<ResponseInputItem, CodexErr>> {
        let error_call = call.clone();
        let future =
            self.handle_tool_call_with_source(call, ToolCallSource::Direct, cancellation_token);
        async move {
            match future.await {
                Ok(response) => Ok(response.into_response()),
                Err(FunctionCallError::Fatal(message)) => Err(CodexErr::Fatal(message)),
                Err(other) => Ok(Self::failure_response(error_call, other)),
            }
        }
        .in_current_span()
    }

    #[instrument(level = "trace", skip_all)]
    pub(crate) fn handle_tool_call_with_source(
        self,
        call: ToolCall,
        source: ToolCallSource,
        cancellation_token: CancellationToken,
    ) -> impl std::future::Future<Output = Result<AnyToolResult, FunctionCallError>> {
        let supports_parallel = self.router.tool_supports_parallel(&call);
        let router = Arc::clone(&self.router);
        let session = Arc::clone(&self.session);
        let turn = Arc::clone(&self.turn_context);
        let tracker = Arc::clone(&self.tracker);
        let lock = Arc::clone(&self.parallel_execution);
        let invocation_cancellation_token = cancellation_token.clone();
        let wait_for_runtime_cancellation = self.router.tool_waits_for_runtime_cancellation(&call);
        let started = Instant::now();
        let abort_session = Arc::clone(&session);
        let abort_source = source.clone();
        let abort_turn = Arc::clone(&turn);
        let log_session = Arc::clone(&session);
        let log_turn = Arc::clone(&turn);
        let terminal_outcome_reached = Arc::new(AtomicBool::new(false));
        let dispatch_terminal_outcome_reached = Arc::clone(&terminal_outcome_reached);
        let dispatch_call = call.clone();

        let dispatch_span = trace_span!(
            "dispatch_tool_call_with_code_mode_result",
            otel.name = %call.tool_name,
            tool_name = %call.tool_name,
            call_id = call.call_id.as_str(),
            aborted = false,
        );
        let abort_dispatch_span = dispatch_span.clone();

        async move {
            // Repeat guard: judged against the session's recent tool calls at
            // dispatch; the verdict is applied to the output once it exists.
            let guard_mode = crate::tools::repeat_guard::mode(&log_turn.config);
            let mut pathology_params = codex_llamacpp::pathology_params_from_env();
            // The resample layer only exists for local providers (the turn
            // driver consumes its signal under the same gate); a remote
            // provider must never claim.
            if !crate::llamacpp_chat_transport::is_local_backend(&log_turn.provider.info().name) {
                pathology_params.resamples = 0;
            }
            let signature = crate::tools::repeat_guard::signature(&call.tool_name.to_string(), &call.payload);
            let verdict = if matches!(guard_mode, codex_config::types::RepeatGuardMode::Off)
                && pathology_params.resamples == 0
            {
                None
            } else {
                log_session
                    .recent_tool_calls
                    .lock()
                    .ok()
                    .map(|mut recent| {
                        let verdict = crate::tools::repeat_guard::judge_and_remember(
                            &mut recent,
                            &call.call_id,
                            &signature,
                            &call.payload.log_payload(),
                        );
                        let spans: Vec<String> = recent
                            .iter()
                            .rev()
                            .take(crate::tools::repeat_guard::WINDOW)
                            .map(|c| c.arguments.clone())
                            .collect();
                        (verdict, spans)
                    })
                    .filter(|(v, _)| v.triggered())
                    .map(|(verdict, spans)| {
                        // Resample layer first: while budget remains, skip
                        // executing this call entirely and signal the turn
                        // driver to discard the completion and re-sample.
                        // The penalty machinery below stays untouched.
                        let resample = log_session
                            .resample
                            .lock()
                            .ok()
                            .map(|mut r| r.request_repeated_call(pathology_params.resamples))
                            .unwrap_or(false);
                        if resample || matches!(guard_mode, codex_config::types::RepeatGuardMode::Off) {
                            return (verdict, resample);
                        }
                        // Loop guard: a repeated call is a failure; the last
                        // ten calls' text is penalized on the next completion.
                        let params = &pathology_params;
                        let state_summary = log_session.pathology.lock().ok().map(|mut st| {
                            st.on_failure(
                                crate::pathology::Trigger::RepeatedToolCall,
                                spans,
                                params,
                            );
                            st.summary()
                        });
                        if crate::model_call_log::enabled() {
                            let key = crate::model_call_log::ModelCallKey {
                                thread_id: log_session.services.model_client.thread_id_string(),
                                turn_id: log_turn.sub_id.clone(),
                                call_ordinal: log_session.services.model_client.last_model_call_ordinal(),
                            };
                            crate::model_call_log::record_event(
                                "pathology",
                                &key,
                                Some(&call.call_id),
                                serde_json::json!({
                                    "trigger": crate::pathology::Trigger::RepeatedToolCall.as_str(),
                                    "state": state_summary,
                                }),
                            );
                        }
                        (verdict, false)
                    })
            };
            let (verdict, resample_skip) = match verdict {
                Some((v, skip)) => (Some(v), skip),
                None => {
                    // A resample requested by an earlier call in this batch
                    // also skips later calls, without consuming more budget.
                    let already = log_session
                        .resample
                        .lock()
                        .ok()
                        .map(|r| r.is_requested())
                        .unwrap_or(false);
                    (None, already)
                }
            };
            let resample_skip = resample_skip
                || log_session
                    .resample
                    .lock()
                    .ok()
                    .map(|r| r.is_requested())
                    .unwrap_or(false);
            if resample_skip {
                if crate::model_call_log::enabled() {
                    let key = crate::model_call_log::ModelCallKey {
                        thread_id: log_session.services.model_client.thread_id_string(),
                        turn_id: log_turn.sub_id.clone(),
                        call_ordinal: log_session.services.model_client.last_model_call_ordinal(),
                    };
                    crate::model_call_log::record_event(
                        "pathology",
                        &key,
                        Some(&call.call_id),
                        serde_json::json!({
                            "action": "resample_skip_call",
                            "tool": call.tool_name.to_string(),
                            "signature": signature,
                        }),
                    );
                }
                return Err(FunctionCallError::RespondToModel(
                    "(call not executed: completion discarded for resampling)".to_string(),
                ));
            }
            let mut handle: AbortOnDropHandle<Result<AnyToolResult, FunctionCallError>> =
                AbortOnDropHandle::new(tokio::spawn(async move {
                    let _guard = if supports_parallel {
                        Either::Left(lock.read().await)
                    } else {
                        Either::Right(lock.write().await)
                    };

                    router
                        .dispatch_tool_call_with_terminal_outcome(
                            session,
                            turn,
                            invocation_cancellation_token,
                            tracker,
                            dispatch_call,
                            source,
                            dispatch_terminal_outcome_reached,
                        )
                        .instrument(dispatch_span.clone())
                        .await
                }));
            let apply_guard = |mut result: AnyToolResult| -> AnyToolResult {
                let Some(verdict) = verdict.as_ref() else {
                    return result;
                };
                let warning = crate::tools::repeat_guard::warning_text(
                    &call.tool_name.to_string(),
                    &call.payload.log_payload(),
                    verdict,
                );
                if crate::model_call_log::enabled() {
                    let key = crate::model_call_log::ModelCallKey {
                        thread_id: log_session.services.model_client.thread_id_string(),
                        turn_id: log_turn.sub_id.clone(),
                        call_ordinal: log_session.services.model_client.last_model_call_ordinal(),
                    };
                    crate::model_call_log::record_event(
                        "repeat",
                        &key,
                        Some(&call.call_id),
                        serde_json::json!({
                            "mode": format!("{guard_mode:?}").to_lowercase(),
                            "tool": call.tool_name.to_string(),
                            "arguments": call.payload.log_payload(),
                            "signature": signature,
                            "immediate_repeat": verdict.immediate_repeat,
                            "repeats_in_window": verdict.repeats_in_window,
                            "warning_appended": matches!(guard_mode, codex_config::types::RepeatGuardMode::Warn),
                            "warning_text": warning,
                        }),
                    );
                }
                if matches!(guard_mode, codex_config::types::RepeatGuardMode::Warn) {
                    let inner = std::mem::replace(
                        &mut result.result,
                        Box::new(crate::tools::context::FunctionToolOutput::from_text(String::new(), None)),
                    );
                    result.result = Box::new(crate::tools::repeat_guard::WarnedToolOutput {
                        inner,
                        suffix: warning,
                    });
                }
                result
            };
            let log_result = |result: &AnyToolResult, aborted: bool| {
                if crate::model_call_log::enabled() {
                    let key = crate::model_call_log::ModelCallKey {
                        thread_id: log_session.services.model_client.thread_id_string(),
                        turn_id: log_turn.sub_id.clone(),
                        call_ordinal: log_session.services.model_client.last_model_call_ordinal(),
                    };
                    crate::model_call_log::record_tool_call(
                        &key,
                        &result.call_id,
                        &call.tool_name.to_string(),
                        &result.payload.log_payload(),
                        &result.result.log_preview(),
                        result.result.success_for_logging(),
                        aborted,
                    );
                }
            };
            tokio::select! {
                res = &mut handle => {
                    let res = res.map_err(Self::tool_task_join_error)?;
                    let res = res.map(apply_guard);
                    if let Ok(result) = &res {
                        log_result(result, false);
                    }
                    res
                }
                _ = cancellation_token.cancelled() => {
                    if terminal_outcome_reached.load(Ordering::Acquire) || handle.is_finished() {
                        handle.await.map_err(Self::tool_task_join_error)?
                    } else {
                        let secs = started.elapsed().as_secs_f32().max(0.1);
                        abort_dispatch_span.record("aborted", true);
                        if wait_for_runtime_cancellation {
                            if terminal_outcome_reached.swap(true, Ordering::AcqRel) {
                                return handle.await.map_err(Self::tool_task_join_error)?;
                            }
                            // The abort owns the terminal outcome; await only so
                            // the runtime can finish process teardown.
                            match handle.await {
                                Ok(_) => {}
                                Err(err) if err.is_cancelled() => {}
                                Err(err) => return Err(Self::tool_task_join_error(err)),
                            }
                        } else {
                            handle.abort();
                            match handle.await {
                                Ok(result) => return result,
                                Err(err) if err.is_cancelled() => {}
                                Err(err) => return Err(Self::tool_task_join_error(err)),
                            }
                        }
                        let response = Self::aborted_response(&call, secs);
                        log_result(&response, true);
                        notify_tool_aborted(
                            abort_session.as_ref(),
                            abort_turn.as_ref(),
                            call.call_id.as_str(),
                            &call.tool_name,
                            abort_source,
                        )
                        .await;
                        Ok(response)
                    }
                },
            }
        }
        .in_current_span()
    }
}

impl ToolCallRuntime {
    fn tool_task_join_error(err: JoinError) -> FunctionCallError {
        FunctionCallError::Fatal(format!("tool task failed to receive: {err:?}"))
    }

    fn failure_response(call: ToolCall, err: FunctionCallError) -> ResponseInputItem {
        let message = err.to_string();
        match call.payload {
            ToolPayload::ToolSearch { .. } => ResponseInputItem::ToolSearchOutput {
                call_id: call.call_id,
                status: "completed".to_string(),
                execution: "client".to_string(),
                tools: Vec::new(),
            },
            ToolPayload::Custom { .. } => ResponseInputItem::CustomToolCallOutput {
                call_id: call.call_id,
                name: None,
                output: codex_protocol::models::FunctionCallOutputPayload {
                    body: codex_protocol::models::FunctionCallOutputBody::Text(message),
                    success: Some(false),
                },
            },
            _ => ResponseInputItem::FunctionCallOutput {
                call_id: call.call_id,
                output: codex_protocol::models::FunctionCallOutputPayload {
                    body: codex_protocol::models::FunctionCallOutputBody::Text(message),
                    success: Some(false),
                },
            },
        }
    }

    fn aborted_response(call: &ToolCall, secs: f32) -> AnyToolResult {
        AnyToolResult {
            call_id: call.call_id.clone(),
            payload: call.payload.clone(),
            result: Box::new(AbortedToolOutput {
                message: Self::abort_message(call, secs),
            }),
            post_tool_use_payload: None,
        }
    }

    fn abort_message(call: &ToolCall, secs: f32) -> String {
        if call.tool_name.namespace.is_none()
            && matches!(
                call.tool_name.name.as_str(),
                "shell_command" | "unified_exec"
            )
        {
            format!("Wall time: {secs:.1} seconds\naborted by user")
        } else {
            format!("aborted by user after {secs:.1}s")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    use crate::tools::context::FunctionToolOutput;
    use crate::tools::context::ToolInvocation;
    use crate::tools::registry::CoreToolRuntime;
    use crate::tools::registry::ToolExecutor;
    use crate::tools::registry::ToolRegistry;
    use crate::turn_diff_tracker::TurnDiffTracker;
    use codex_extension_api::ToolCallOutcome;
    use codex_protocol::models::FunctionCallOutputBody;
    use codex_protocol::models::FunctionCallOutputPayload;
    use pretty_assertions::assert_eq;
    use tokio::sync::Notify;
    use tokio::sync::oneshot;

    struct ImmediateHandler {
        tool_name: codex_tools::ToolName,
    }

    #[async_trait::async_trait]
    impl ToolExecutor<ToolInvocation> for ImmediateHandler {
        fn tool_name(&self) -> codex_tools::ToolName {
            self.tool_name.clone()
        }

        fn spec(&self) -> codex_tools::ToolSpec {
            codex_tools::ToolSpec::Function(codex_tools::ResponsesApiTool {
                name: self.tool_name.name.clone(),
                description: "Immediate test tool.".to_string(),
                strict: false,
                defer_loading: None,
                parameters: codex_tools::JsonSchema::default(),
                output_schema: None,
            })
        }

        async fn handle(
            &self,
            _invocation: ToolInvocation,
        ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
            Ok(Box::new(FunctionToolOutput::from_text(
                "ok".to_string(),
                Some(true),
            )))
        }
    }

    impl CoreToolRuntime for ImmediateHandler {}

    struct CancellationCleanupHandler {
        tool_name: codex_tools::ToolName,
        started: std::sync::Mutex<Option<oneshot::Sender<()>>>,
        cleanup_started: std::sync::Mutex<Option<oneshot::Sender<()>>>,
        allow_cleanup: Arc<Notify>,
    }

    #[async_trait::async_trait]
    impl ToolExecutor<ToolInvocation> for CancellationCleanupHandler {
        fn tool_name(&self) -> codex_tools::ToolName {
            self.tool_name.clone()
        }

        fn spec(&self) -> codex_tools::ToolSpec {
            codex_tools::ToolSpec::Function(codex_tools::ResponsesApiTool {
                name: self.tool_name.name.clone(),
                description: "Cancellation cleanup test tool.".to_string(),
                strict: false,
                defer_loading: None,
                parameters: codex_tools::JsonSchema::default(),
                output_schema: None,
            })
        }

        async fn handle(
            &self,
            invocation: ToolInvocation,
        ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
            let started = self
                .started
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            if let Some(started) = started {
                let _ = started.send(());
            }
            invocation.cancellation_token.cancelled().await;
            let cleanup_started = self
                .cleanup_started
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            if let Some(cleanup_started) = cleanup_started {
                let _ = cleanup_started.send(());
            }
            self.allow_cleanup.notified().await;
            Ok(Box::new(FunctionToolOutput::from_text(
                "cleanup complete".to_string(),
                Some(false),
            )))
        }
    }

    impl CoreToolRuntime for CancellationCleanupHandler {
        fn waits_for_runtime_cancellation(&self) -> bool {
            true
        }
    }

    struct FinishRecorder {
        records: Arc<std::sync::Mutex<Vec<ToolCallOutcome>>>,
    }

    impl codex_extension_api::ToolLifecycleContributor for FinishRecorder {
        fn on_tool_finish<'a>(
            &'a self,
            input: codex_extension_api::ToolFinishInput<'a>,
        ) -> codex_extension_api::ToolLifecycleFuture<'a> {
            let records = Arc::clone(&self.records);
            let outcome = input.outcome;
            Box::pin(async move {
                records
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(outcome);
            })
        }
    }

    struct BlockingFinishContributor {
        records: Arc<std::sync::Mutex<Vec<ToolCallOutcome>>>,
        finish_started: std::sync::Mutex<Option<oneshot::Sender<()>>>,
        allow_finish: Arc<Notify>,
    }

    impl codex_extension_api::ToolLifecycleContributor for BlockingFinishContributor {
        fn on_tool_finish<'a>(
            &'a self,
            input: codex_extension_api::ToolFinishInput<'a>,
        ) -> codex_extension_api::ToolLifecycleFuture<'a> {
            let records = Arc::clone(&self.records);
            let allow_finish = Arc::clone(&self.allow_finish);
            let finish_started = self
                .finish_started
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            let outcome = input.outcome;
            Box::pin(async move {
                if let Some(finish_started) = finish_started {
                    let _ = finish_started.send(());
                }
                allow_finish.notified().await;
                records
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(outcome);
            })
        }
    }

    #[tokio::test]
    async fn cancellation_after_handler_finishes_preserves_completed_lifecycle()
    -> anyhow::Result<()> {
        let (mut session, turn_context) = crate::session::tests::make_session_and_context().await;
        let records = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (finish_started_tx, finish_started_rx) = oneshot::channel();
        let allow_finish = Arc::new(Notify::new());
        let mut builder =
            codex_extension_api::ExtensionRegistryBuilder::<crate::config::Config>::new();
        builder.tool_lifecycle_contributor(Arc::new(BlockingFinishContributor {
            records: Arc::clone(&records),
            finish_started: std::sync::Mutex::new(Some(finish_started_tx)),
            allow_finish: Arc::clone(&allow_finish),
        }));
        session.services.extensions = Arc::new(builder.build());

        let session = Arc::new(session);
        let turn_context = Arc::new(turn_context);
        let tool_name = codex_tools::ToolName::plain("test_tool");
        let handler = Arc::new(ImmediateHandler {
            tool_name: tool_name.clone(),
        }) as Arc<dyn CoreToolRuntime>;
        let router = Arc::new(ToolRouter::from_parts(
            ToolRegistry::from_tools([handler]),
            Vec::new(),
        ));
        let tracker = Arc::new(tokio::sync::Mutex::new(TurnDiffTracker::new()));
        let runtime = ToolCallRuntime::new(router, session, turn_context, tracker);
        let cancellation_token = CancellationToken::new();
        let call = ToolCall {
            tool_name,
            call_id: "call-1".to_string(),
            payload: ToolPayload::Function {
                arguments: "{}".to_string(),
            },
        };

        let response_task =
            tokio::spawn(runtime.handle_tool_call(call, cancellation_token.clone()));
        tokio::time::timeout(Duration::from_secs(1), finish_started_rx)
            .await
            .expect("timed out waiting for lifecycle notification to start")
            .expect("lifecycle notification should start");
        cancellation_token.cancel();
        tokio::time::sleep(Duration::from_millis(10)).await;
        allow_finish.notify_waiters();

        let response = tokio::time::timeout(Duration::from_secs(1), response_task)
            .await
            .expect("timed out waiting for tool response")
            .expect("tool response task should join")?;
        let expected_response = ResponseInputItem::FunctionCallOutput {
            call_id: "call-1".to_string(),
            output: FunctionCallOutputPayload {
                body: FunctionCallOutputBody::Text("ok".to_string()),
                success: Some(true),
            },
        };
        assert_eq!(expected_response, response);

        let actual = records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain(..)
            .collect::<Vec<_>>();
        assert_eq!(vec![ToolCallOutcome::Completed { success: true }], actual);

        Ok(())
    }

    #[tokio::test]
    async fn cancellation_waiting_for_runtime_cleanup_emits_only_aborted_lifecycle()
    -> anyhow::Result<()> {
        let (mut session, turn_context) = crate::session::tests::make_session_and_context().await;
        let records = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut builder =
            codex_extension_api::ExtensionRegistryBuilder::<crate::config::Config>::new();
        builder.tool_lifecycle_contributor(Arc::new(FinishRecorder {
            records: Arc::clone(&records),
        }));
        session.services.extensions = Arc::new(builder.build());

        let session = Arc::new(session);
        let turn_context = Arc::new(turn_context);
        let tool_name = codex_tools::ToolName::plain("cleanup_tool");
        let (started_tx, started_rx) = oneshot::channel();
        let (cleanup_started_tx, cleanup_started_rx) = oneshot::channel();
        let allow_cleanup = Arc::new(Notify::new());
        let handler = Arc::new(CancellationCleanupHandler {
            tool_name: tool_name.clone(),
            started: std::sync::Mutex::new(Some(started_tx)),
            cleanup_started: std::sync::Mutex::new(Some(cleanup_started_tx)),
            allow_cleanup: Arc::clone(&allow_cleanup),
        }) as Arc<dyn CoreToolRuntime>;
        let router = Arc::new(ToolRouter::from_parts(
            ToolRegistry::from_tools([handler]),
            Vec::new(),
        ));
        let tracker = Arc::new(tokio::sync::Mutex::new(TurnDiffTracker::new()));
        let runtime = ToolCallRuntime::new(router, session, turn_context, tracker);
        let cancellation_token = CancellationToken::new();
        let call = ToolCall {
            tool_name,
            call_id: "call-1".to_string(),
            payload: ToolPayload::Function {
                arguments: "{}".to_string(),
            },
        };

        let response_task =
            tokio::spawn(runtime.handle_tool_call(call, cancellation_token.clone()));
        started_rx.await.expect("handler should start");
        cancellation_token.cancel();
        cleanup_started_rx
            .await
            .expect("handler should start cleanup");
        tokio::time::sleep(Duration::from_millis(10)).await;
        allow_cleanup.notify_one();

        let response = tokio::time::timeout(Duration::from_secs(1), response_task)
            .await
            .expect("timed out waiting for tool response")
            .expect("tool response task should join")?;
        let ResponseInputItem::FunctionCallOutput { output, .. } = response else {
            anyhow::bail!("cancelled tool should return function output");
        };
        let FunctionCallOutputBody::Text(text) = output.body else {
            anyhow::bail!("cancelled tool output should be text");
        };
        assert!(text.contains("aborted by user"));

        let actual = records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .drain(..)
            .collect::<Vec<_>>();
        assert_eq!(vec![ToolCallOutcome::Aborted], actual);

        Ok(())
    }
}
