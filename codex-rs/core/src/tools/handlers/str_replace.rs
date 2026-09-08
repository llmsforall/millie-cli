use std::io::ErrorKind;

use codex_exec_server::CreateDirectoryOptions;
use codex_exec_server::RemoveOptions;
use codex_tools::ToolName;
use codex_tools::ToolSpec;
use codex_utils_absolute_path::AbsolutePathBuf;
use serde::Deserialize;

use crate::function_tool::FunctionCallError;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::tools::context::boxed_tool_output;
use crate::tools::handlers::parse_arguments;
use crate::tools::handlers::resolve_tool_environment;
use crate::tools::handlers::str_replace_correct;
use crate::tools::handlers::str_replace_spec::CREATE_FILE_TOOL_NAME;
use crate::tools::handlers::str_replace_spec::DELETE_FILE_TOOL_NAME;
use crate::tools::handlers::str_replace_spec::STR_REPLACE_TOOL_NAME;
use crate::tools::handlers::str_replace_spec::create_create_file_tool;
use crate::tools::handlers::str_replace_spec::create_delete_file_tool;
use crate::tools::handlers::str_replace_spec::create_str_replace_tool;
use crate::tools::registry::CoreToolRuntime;
use crate::tools::registry::ToolExecutor;

#[derive(Default)]
pub struct CreateFileHandler;

#[derive(Default)]
pub struct StrReplaceHandler;

#[derive(Default)]
pub struct DeleteFileHandler;

#[derive(Deserialize)]
struct PathArgs {
    path: String,
}

#[derive(Deserialize)]
struct CreateFileArgs {
    path: String,
    content: String,
}

#[derive(Deserialize)]
struct StrReplaceArgs {
    path: String,
    old_str: String,
    new_str: String,
}

struct ParsedInvocation {
    turn: std::sync::Arc<crate::session::turn_context::TurnContext>,
    session: std::sync::Arc<crate::session::session::Session>,
    call_id: String,
    arguments: String,
}

fn function_arguments(
    invocation: ToolInvocation,
    tool_name: &str,
) -> Result<ParsedInvocation, FunctionCallError> {
    let ToolInvocation {
        turn,
        session,
        call_id,
        payload,
        ..
    } = invocation;
    let ToolPayload::Function { arguments } = payload else {
        return Err(FunctionCallError::RespondToModel(format!(
            "{tool_name} handler received unsupported payload"
        )));
    };
    Ok(ParsedInvocation {
        turn,
        session,
        call_id,
        arguments,
    })
}

/// Key of the model call that produced this tool call, for the log.
fn log_key(
    session: &crate::session::session::Session,
    turn: &crate::session::turn_context::TurnContext,
) -> crate::model_call_log::ModelCallKey {
    crate::model_call_log::ModelCallKey {
        thread_id: session.services.model_client.thread_id_string(),
        turn_id: turn.sub_id.clone(),
        call_ordinal: session.services.model_client.last_model_call_ordinal(),
    }
}

/// Corrector tunables from config.
/// Append the configured edit reminder (`[tools] edit_reminder`) to a
/// successful edit result. Rides on every edit, so compaction cannot lose it.
fn with_edit_reminder(turn: &crate::session::turn_context::TurnContext, text: String) -> String {
    match turn.config.edit_reminder.as_deref() {
        Some(reminder) if !reminder.is_empty() => format!("{text}\n\n{reminder}"),
        _ => text,
    }
}

fn correction_config(
    settings: &codex_config::types::StrReplaceAutocorrectSettings,
) -> str_replace_correct::CorrectionConfig {
    str_replace_correct::CorrectionConfig {
        posterior_threshold: settings.posterior_threshold,
        max_distance_fraction: settings.max_distance_fraction,
        h0_prior: settings.h0_prior,
        max_old_str_chars: settings.max_old_str_chars,
        max_file_bytes: settings.max_file_bytes,
        ..str_replace_correct::CorrectionConfig::default()
    }
}

fn resolve_path(cwd: &AbsolutePathBuf, path: &str) -> AbsolutePathBuf {
    cwd.join(path)
}

#[async_trait::async_trait]
impl ToolExecutor<ToolInvocation> for CreateFileHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(CREATE_FILE_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        create_create_file_tool()
    }

    async fn handle(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ParsedInvocation { turn, arguments, .. } =
            function_arguments(invocation, CREATE_FILE_TOOL_NAME)?;
        let CreateFileArgs { path, content } = parse_arguments(&arguments)?;
        let Some(turn_environment) = resolve_tool_environment(turn.as_ref(), None)? else {
            return Err(FunctionCallError::RespondToModel(
                "create_file is unavailable in this session".to_string(),
            ));
        };
        let cwd = turn_environment.cwd.clone();
        let fs = turn_environment.environment.get_filesystem();
        let sandbox = turn.file_system_sandbox_context(/*additional_permissions*/ None, &cwd);
        let target = resolve_path(&cwd, &path);

        match fs.get_metadata(&target, Some(&sandbox)).await {
            Ok(_) => {
                return Ok(boxed_tool_output(
                    crate::tools::context::FunctionToolOutput::from_text(
                        format!("create_file failed: {path} already exists"),
                        Some(false),
                    ),
                ));
            }
            Err(err) if err.kind() == ErrorKind::NotFound => {}
            Err(err) => return Err(FunctionCallError::RespondToModel(format!("error: {err}"))),
        }

        if let Some(parent) = target.parent() {
            let parent = AbsolutePathBuf::from_absolute_path(parent.to_path_buf())
                .map_err(|err| FunctionCallError::RespondToModel(format!("error: {err}")))?;
            fs.create_directory(
                &parent,
                CreateDirectoryOptions { recursive: true },
                Some(&sandbox),
            )
            .await
            .map_err(|err| FunctionCallError::RespondToModel(format!("error: {err}")))?;
        }
        let mut content = content;
        if !content.ends_with('\n') {
            content.push('\n');
        }
        fs.write_file(&target, content.into_bytes(), Some(&sandbox))
            .await
            .map_err(|err| FunctionCallError::RespondToModel(format!("error: {err}")))?;

        Ok(boxed_tool_output(
            crate::tools::context::FunctionToolOutput::from_text(
                with_edit_reminder(&turn, format!("Created {path}")),
                Some(true),
            ),
        ))
    }
}

impl CoreToolRuntime for CreateFileHandler {}

#[async_trait::async_trait]
impl ToolExecutor<ToolInvocation> for StrReplaceHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(STR_REPLACE_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        create_str_replace_tool()
    }

    async fn handle(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ParsedInvocation {
            turn,
            session,
            call_id,
            arguments,
        } = function_arguments(invocation, STR_REPLACE_TOOL_NAME)?;
        let StrReplaceArgs {
            path,
            old_str,
            new_str,
        } = parse_arguments(&arguments)?;
        // An identity edit is definitionally useless: it changes nothing but
        // reads as progress. Fail it like any other bad edit, unless
        // `[tools] allow_null_edit` restores the old no-op-success behavior.
        if old_str == new_str && !turn.config.allow_null_edit {
            return Ok(boxed_tool_output(
                crate::tools::context::FunctionToolOutput::from_text(
                    "str_replace failed: old_str and new_str are identical, so this edit changes nothing and was not applied."
                        .to_string(),
                    Some(false),
                ),
            ));
        }
        let Some(turn_environment) = resolve_tool_environment(turn.as_ref(), None)? else {
            return Err(FunctionCallError::RespondToModel(
                "str_replace is unavailable in this session".to_string(),
            ));
        };
        let cwd = turn_environment.cwd.clone();
        let fs = turn_environment.environment.get_filesystem();
        let sandbox = turn.file_system_sandbox_context(/*additional_permissions*/ None, &cwd);
        let target = resolve_path(&cwd, &path);

        let text = match fs.read_file_text(&target, Some(&sandbox)).await {
            Ok(text) => text,
            Err(err) if err.kind() == ErrorKind::NotFound => {
                return Ok(boxed_tool_output(
                    crate::tools::context::FunctionToolOutput::from_text(
                        format!("str_replace failed: {path} not found"),
                        Some(false),
                    ),
                ));
            }
            Err(err) => return Err(FunctionCallError::RespondToModel(format!("error: {err}"))),
        };
        let count = text.matches(&old_str).count();
        if count == 0 {
            // Conservative autocorrect (see str_replace_correct.rs). `off`
            // does nothing; `log` records the would-be correction and still
            // fails; `correct` applies it and tells the model what changed.
            use codex_config::types::StrReplaceAutocorrectMode as Mode;
            let settings = turn.config.str_replace_autocorrect;
            let failure = "str_replace failed: old_str not found in file".to_string();
            if settings.mode == Mode::Off {
                return Ok(boxed_tool_output(
                    crate::tools::context::FunctionToolOutput::from_text(failure, Some(false)),
                ));
            }
            let outcome = str_replace_correct::suggest(
                &text,
                &old_str,
                &new_str,
                &correction_config(&settings),
            );
            let key = log_key(&session, &turn);
            let mode_name = format!("{:?}", settings.mode).to_lowercase();
            match outcome {
                str_replace_correct::Outcome::Declined(d) => {
                    if crate::model_call_log::enabled() {
                        crate::model_call_log::record_event(
                            "correction",
                            &key,
                            Some(&call_id),
                            serde_json::json!({
                                "tool": STR_REPLACE_TOOL_NAME,
                                "mode": mode_name,
                                "original": {"path": path, "old_str": old_str, "new_str": new_str},
                                "decision": "declined",
                                "reason": d.reason,
                                "best_distance": d.best_distance,
                                "runner_up_distance": d.runner_up_distance,
                                "posterior": d.posterior,
                                "applied": false,
                            }),
                        );
                    }
                    return Ok(boxed_tool_output(
                        crate::tools::context::FunctionToolOutput::from_text(failure, Some(false)),
                    ));
                }
                str_replace_correct::Outcome::Corrected(c) => {
                    let effective_new =
                        c.adjusted_new_str.clone().unwrap_or_else(|| new_str.clone());
                    let diff = str_replace_correct::short_diff(&old_str, &c.matched, 12);
                    let apply = settings.mode == Mode::Correct;
                    if crate::model_call_log::enabled() {
                        crate::model_call_log::record_event(
                            "correction",
                            &key,
                            Some(&call_id),
                            serde_json::json!({
                                "tool": STR_REPLACE_TOOL_NAME,
                                "mode": mode_name,
                                "original": {"path": path, "old_str": old_str, "new_str": new_str},
                                "corrected": {"path": path, "old_str": c.matched, "new_str": effective_new},
                                "decision": "corrected",
                                "tier": c.tier,
                                "distance": c.distance,
                                "posterior": c.posterior,
                                "runner_up_distance": c.runner_up_distance,
                                "diff": diff,
                                "applied": apply,
                            }),
                        );
                    }
                    if !apply {
                        return Ok(boxed_tool_output(
                            crate::tools::context::FunctionToolOutput::from_text(failure, Some(false)),
                        ));
                    }
                    let replaced =
                        format!("{}{}{}", &text[..c.start], effective_new, &text[c.end..]);
                    fs.write_file(&target, replaced.into_bytes(), Some(&sandbox))
                        .await
                        .map_err(|err| FunctionCallError::RespondToModel(format!("error: {err}")))?;
                    let mut note = format!(
                        "Replaced in {path}\n\nnote: old_str did not match exactly and was corrected \
                         ({}, {} edit(s)); the replaced text differed from your old_str as follows:\n{}",
                        c.tier, c.distance, diff
                    );
                    if c.adjusted_new_str.is_some() {
                        note.push_str("\nnew_str was re-indented to match the file.");
                    }
                    return Ok(boxed_tool_output(
                        crate::tools::context::FunctionToolOutput::from_text(with_edit_reminder(&turn, note), Some(true)),
                    ));
                }
            }
        }
        if count > 1 {
            return Ok(boxed_tool_output(
                crate::tools::context::FunctionToolOutput::from_text(
                    format!("str_replace failed: old_str matched {count} locations"),
                    Some(false),
                ),
            ));
        }

        let replaced = text.replacen(&old_str, &new_str, 1);
        fs.write_file(&target, replaced.into_bytes(), Some(&sandbox))
            .await
            .map_err(|err| FunctionCallError::RespondToModel(format!("error: {err}")))?;
        Ok(boxed_tool_output(
            crate::tools::context::FunctionToolOutput::from_text(
                with_edit_reminder(&turn, format!("Replaced in {path}")),
                Some(true),
            ),
        ))
    }
}

impl CoreToolRuntime for StrReplaceHandler {}

#[async_trait::async_trait]
impl ToolExecutor<ToolInvocation> for DeleteFileHandler {
    fn tool_name(&self) -> ToolName {
        ToolName::plain(DELETE_FILE_TOOL_NAME)
    }

    fn spec(&self) -> ToolSpec {
        create_delete_file_tool()
    }

    async fn handle(
        &self,
        invocation: ToolInvocation,
    ) -> Result<Box<dyn crate::tools::context::ToolOutput>, FunctionCallError> {
        let ParsedInvocation { turn, arguments, .. } =
            function_arguments(invocation, DELETE_FILE_TOOL_NAME)?;
        let PathArgs { path } = parse_arguments(&arguments)?;
        let Some(turn_environment) = resolve_tool_environment(turn.as_ref(), None)? else {
            return Err(FunctionCallError::RespondToModel(
                "delete_file is unavailable in this session".to_string(),
            ));
        };
        let cwd = turn_environment.cwd.clone();
        let fs = turn_environment.environment.get_filesystem();
        let sandbox = turn.file_system_sandbox_context(/*additional_permissions*/ None, &cwd);
        let target = resolve_path(&cwd, &path);

        match fs.get_metadata(&target, Some(&sandbox)).await {
            Ok(_) => {}
            Err(err) if err.kind() == ErrorKind::NotFound => {
                return Ok(boxed_tool_output(
                    crate::tools::context::FunctionToolOutput::from_text(
                        format!("delete_file failed: {path} not found"),
                        Some(false),
                    ),
                ));
            }
            Err(err) => return Err(FunctionCallError::RespondToModel(format!("error: {err}"))),
        }

        fs.remove(
            &target,
            RemoveOptions {
                recursive: false,
                force: false,
            },
            Some(&sandbox),
        )
        .await
        .map_err(|err| FunctionCallError::RespondToModel(format!("error: {err}")))?;
        Ok(boxed_tool_output(
            crate::tools::context::FunctionToolOutput::from_text(
                format!("Deleted {path}"),
                Some(true),
            ),
        ))
    }
}

impl CoreToolRuntime for DeleteFileHandler {}

#[cfg(test)]
#[path = "str_replace_tests.rs"]
mod tests;
