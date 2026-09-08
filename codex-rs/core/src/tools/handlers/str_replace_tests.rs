#![allow(clippy::expect_used)]

use std::path::PathBuf;
use std::sync::Arc;

use super::*;
use codex_protocol::models::PermissionProfile;
use pretty_assertions::assert_eq;
use serde_json::json;
use tokio::sync::Mutex;

use crate::session::tests::make_session_and_context;
use crate::tools::context::ToolCallSource;
use crate::tools::context::ToolInvocation;
use crate::tools::context::ToolPayload;
use crate::turn_diff_tracker::TurnDiffTracker;

async fn call(
    handler: &dyn crate::tools::registry::CoreToolRuntime,
    tool_name: &str,
    args: serde_json::Value,
    setup: impl FnOnce(&PathBuf),
) -> (String, PathBuf) {
    let (session, mut turn) = make_session_and_context().await;
    turn.permission_profile = PermissionProfile::Disabled;
    // hermetic cwd: the default test session cwd is the PROCESS cwd, and
    // these tests were leaking nested/file.txt etc. into the repo and then
    // failing on their own leftovers on the next run
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let cwd = codex_utils_absolute_path::AbsolutePathBuf::try_from(tmp.path())
        .expect("absolute tempdir");
    turn.environments.turn_environments[0].cwd = cwd.clone();
    std::mem::forget(tmp); // keep the dir for post-call assertions
    let cwd = cwd.to_path_buf();
    setup(&cwd);
    let output = handler
        .handle(ToolInvocation {
            session: Arc::new(session),
            turn: Arc::new(turn),
            cancellation_token: tokio_util::sync::CancellationToken::new(),
            tracker: Arc::new(Mutex::new(TurnDiffTracker::new())),
            call_id: format!("call-{tool_name}"),
            tool_name: codex_tools::ToolName::plain(tool_name),
            source: ToolCallSource::Direct,
            payload: ToolPayload::Function {
                arguments: args.to_string(),
            },
        })
        .await
        .expect("tool call succeeds")
        .log_preview();
    (output, cwd)
}

#[tokio::test]
async fn create_file_result_strings_and_trailing_newline() {
    let handler = CreateFileHandler;

    let (output, cwd) = call(
        &handler,
        CREATE_FILE_TOOL_NAME,
        json!({"path": "nested/file.txt", "content": "hello"}),
        |_| {},
    )
    .await;
    assert_eq!(output, "Created nested/file.txt");
    assert_eq!(
        std::fs::read_to_string(cwd.join("nested/file.txt")).expect("read created file"),
        "hello\n"
    );

    let (output, cwd) = call(
        &handler,
        CREATE_FILE_TOOL_NAME,
        json!({"path": "exists.txt", "content": "again"}),
        |cwd| {
            std::fs::write(cwd.join("exists.txt"), "existing\n").expect("write fixture");
        },
    )
    .await;
    assert_eq!(output, "create_file failed: exists.txt already exists");
    assert_eq!(
        std::fs::read_to_string(cwd.join("exists.txt")).expect("read unchanged file"),
        "existing\n"
    );
}

#[tokio::test]
async fn delete_file_result_strings() {
    let handler = DeleteFileHandler;

    let (output, cwd) = call(
        &handler,
        DELETE_FILE_TOOL_NAME,
        json!({"path": "gone.txt"}),
        |cwd| {
            std::fs::write(cwd.join("gone.txt"), "bye").expect("write fixture");
        },
    )
    .await;
    assert_eq!(output, "Deleted gone.txt");
    assert!(!cwd.join("gone.txt").exists());

    let (output, _) = call(
        &handler,
        DELETE_FILE_TOOL_NAME,
        json!({"path": "missing.txt"}),
        |_| {},
    )
    .await;
    assert_eq!(output, "delete_file failed: missing.txt not found");
}

#[tokio::test]
async fn allow_null_edit_restores_identity_noop_success() {
    let handler = super::StrReplaceHandler;
    let (session, mut turn) = make_session_and_context().await;
    turn.permission_profile = PermissionProfile::Disabled;
    let mut config = (*turn.config).clone();
    config.allow_null_edit = true;
    turn.config = Arc::new(config);
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let cwd = codex_utils_absolute_path::AbsolutePathBuf::try_from(tmp.path())
        .expect("absolute tempdir");
    turn.environments.turn_environments[0].cwd = cwd.clone();
    std::fs::write(cwd.to_path_buf().join("same.txt"), "hello world").expect("write");
    let output = handler
        .handle(ToolInvocation {
            session: Arc::new(session),
            turn: Arc::new(turn),
            cancellation_token: tokio_util::sync::CancellationToken::new(),
            tracker: Arc::new(Mutex::new(TurnDiffTracker::new())),
            call_id: "call1".to_string(),
            tool_name: codex_tools::ToolName::plain(STR_REPLACE_TOOL_NAME),
            source: ToolCallSource::Direct,
            payload: ToolPayload::Function {
                arguments: json!({"path": "same.txt", "old_str": "hello", "new_str": "hello"})
                    .to_string(),
            },
        })
        .await
        .expect("handler")
        .log_preview();
    assert!(!output.contains("failed"), "{output}");
    assert_eq!(
        std::fs::read_to_string(cwd.to_path_buf().join("same.txt")).expect("read"),
        "hello world"
    );
}

#[tokio::test]
async fn identity_str_replace_fails_and_leaves_file_untouched() {
    let handler = super::StrReplaceHandler;
    let (output, cwd) = call(
        &handler,
        STR_REPLACE_TOOL_NAME,
        json!({"path": "same.txt", "old_str": "hello", "new_str": "hello"}),
        |cwd| std::fs::write(cwd.join("same.txt"), "hello world").expect("write"),
    )
    .await;
    assert_eq!(
        output,
        "str_replace failed: old_str and new_str are identical, so this edit changes nothing and was not applied."
    );
    assert_eq!(
        std::fs::read_to_string(cwd.join("same.txt")).expect("read"),
        "hello world"
    );
}

#[tokio::test]
async fn str_replace_result_strings_and_noop_failures() {
    let handler = StrReplaceHandler;

    let (output, _) = call(
        &handler,
        STR_REPLACE_TOOL_NAME,
        json!({"path": "missing.txt", "old_str": "x", "new_str": "y"}),
        |_| {},
    )
    .await;
    assert_eq!(output, "str_replace failed: missing.txt not found");

    let (output, cwd) = call(
        &handler,
        STR_REPLACE_TOOL_NAME,
        json!({"path": "sample.txt", "old_str": "absent", "new_str": "new"}),
        |cwd| {
            std::fs::write(cwd.join("sample.txt"), "one\ntwo\n").expect("write fixture");
        },
    )
    .await;
    assert_eq!(output, "str_replace failed: old_str not found in file");
    assert_eq!(
        std::fs::read_to_string(cwd.join("sample.txt")).expect("read unchanged"),
        "one\ntwo\n"
    );

    let (output, cwd) = call(
        &handler,
        STR_REPLACE_TOOL_NAME,
        json!({"path": "sample.txt", "old_str": "dup", "new_str": "new"}),
        |cwd| {
            std::fs::write(cwd.join("sample.txt"), "dup\ndup\n").expect("write fixture");
        },
    )
    .await;
    assert_eq!(output, "str_replace failed: old_str matched 2 locations");
    assert_eq!(
        std::fs::read_to_string(cwd.join("sample.txt")).expect("read unchanged"),
        "dup\ndup\n"
    );

    let (output, cwd) = call(
        &handler,
        STR_REPLACE_TOOL_NAME,
        json!({"path": "sample.txt", "old_str": "dup\ndup", "new_str": "ok"}),
        |cwd| {
            std::fs::write(cwd.join("sample.txt"), "dup\ndup\n").expect("write fixture");
        },
    )
    .await;
    assert_eq!(output, "Replaced in sample.txt");
    assert_eq!(
        std::fs::read_to_string(cwd.join("sample.txt")).expect("read replaced"),
        "ok\n"
    );
}

#[tokio::test]
async fn sequential_str_replace_calls_see_prior_changes() {
    let (session, mut turn) = make_session_and_context().await;
    turn.permission_profile = PermissionProfile::Disabled;
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let cwd_abs = codex_utils_absolute_path::AbsolutePathBuf::try_from(tmp.path())
        .expect("absolute tempdir");
    turn.environments.turn_environments[0].cwd = cwd_abs.clone();
    std::mem::forget(tmp);
    let cwd = cwd_abs.to_path_buf();
    std::fs::write(cwd.join("sample.txt"), "alpha\n").expect("write fixture");
    let session = Arc::new(session);
    let turn = Arc::new(turn);

    for (old_str, new_str) in [("alpha", "beta"), ("beta", "gamma")] {
        let output = StrReplaceHandler
            .handle(ToolInvocation {
                session: Arc::clone(&session),
                turn: Arc::clone(&turn),
                cancellation_token: tokio_util::sync::CancellationToken::new(),
                tracker: Arc::new(Mutex::new(TurnDiffTracker::new())),
                call_id: "call-str_replace".to_string(),
                tool_name: codex_tools::ToolName::plain(STR_REPLACE_TOOL_NAME),
                source: ToolCallSource::Direct,
                payload: ToolPayload::Function {
                    arguments: json!({
                        "path": "sample.txt",
                        "old_str": old_str,
                        "new_str": new_str,
                    })
                    .to_string(),
                },
            })
            .await
            .expect("tool call succeeds")
            .log_preview();
        assert_eq!(output, "Replaced in sample.txt");
    }

    assert_eq!(
        std::fs::read_to_string(cwd.join("sample.txt")).expect("read final"),
        "gamma\n"
    );
}

/// Like `call`, but with the autocorrect mode set on the turn's config.
async fn call_with_autocorrect(
    mode: codex_config::types::StrReplaceAutocorrectMode,
    args: serde_json::Value,
    setup: impl FnOnce(&PathBuf),
) -> (String, PathBuf) {
    let (session, mut turn) = make_session_and_context().await;
    turn.permission_profile = PermissionProfile::Disabled;
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let cwd = codex_utils_absolute_path::AbsolutePathBuf::try_from(tmp.path())
        .expect("absolute tempdir");
    turn.environments.turn_environments[0].cwd = cwd.clone();
    std::mem::forget(tmp);
    let mut config = (*turn.config).clone();
    config.str_replace_autocorrect.mode = mode;
    turn.config = Arc::new(config);
    let cwd = cwd.to_path_buf();
    setup(&cwd);
    let output = StrReplaceHandler
        .handle(ToolInvocation {
            session: Arc::new(session),
            turn: Arc::new(turn),
            cancellation_token: tokio_util::sync::CancellationToken::new(),
            tracker: Arc::new(Mutex::new(TurnDiffTracker::new())),
            call_id: "call-str_replace-autocorrect".to_string(),
            tool_name: codex_tools::ToolName::plain("str_replace"),
            source: ToolCallSource::Direct,
            payload: ToolPayload::Function {
                arguments: args.to_string(),
            },
        })
        .await
        .expect("tool call succeeds")
        .log_preview();
    (output, cwd)
}

const DRIFT_FILE: &str = "class Calc:\n    def mul(self, a, b):\n        return a * b\n";

#[tokio::test]
async fn autocorrect_correct_mode_applies_indentation_drift() {
    use codex_config::types::StrReplaceAutocorrectMode;
    let (out, cwd) = call_with_autocorrect(
        StrReplaceAutocorrectMode::Correct,
        json!({"path": "calc.py", "old_str": "def mul(self, a, b):\n    return a * b", "new_str": "def mul(self, a, b):\n    return a * b * 1"}),
        |cwd| std::fs::write(cwd.join("calc.py"), DRIFT_FILE).unwrap(),
    )
    .await;
    assert!(out.starts_with("Replaced in calc.py"), "{out}");
    assert!(out.contains("was corrected"), "{out}");
    let file = std::fs::read_to_string(cwd.join("calc.py")).unwrap();
    assert_eq!(file, "class Calc:\n    def mul(self, a, b):\n        return a * b * 1\n");
}

#[tokio::test]
async fn autocorrect_log_mode_keeps_the_original_failure() {
    use codex_config::types::StrReplaceAutocorrectMode;
    let (out, cwd) = call_with_autocorrect(
        StrReplaceAutocorrectMode::Log,
        json!({"path": "calc.py", "old_str": "def mul(self, a, b):\n    return a * b", "new_str": "X"}),
        |cwd| std::fs::write(cwd.join("calc.py"), DRIFT_FILE).unwrap(),
    )
    .await;
    assert_eq!(out, "str_replace failed: old_str not found in file");
    assert_eq!(std::fs::read_to_string(cwd.join("calc.py")).unwrap(), DRIFT_FILE);
}

#[tokio::test]
async fn autocorrect_declines_ambiguous_and_hallucinated() {
    use codex_config::types::StrReplaceAutocorrectMode;
    let file = "def f1(x):\n    return x + 1\n\ndef f2(x):\n    return x + 2\n";
    let (out, cwd) = call_with_autocorrect(
        StrReplaceAutocorrectMode::Correct,
        json!({"path": "m.py", "old_str": "def f9(x):\n    return x + 3", "new_str": "X"}),
        |cwd| std::fs::write(cwd.join("m.py"), file).unwrap(),
    )
    .await;
    assert_eq!(out, "str_replace failed: old_str not found in file");
    assert_eq!(std::fs::read_to_string(cwd.join("m.py")).unwrap(), file);
}


async fn call_edit_tool_with_reminder(
    tool: &str,
    reminder: Option<&str>,
    args: serde_json::Value,
    setup: impl FnOnce(&PathBuf),
) -> String {
    let (session, mut turn) = make_session_and_context().await;
    turn.permission_profile = PermissionProfile::Disabled;
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let cwd = codex_utils_absolute_path::AbsolutePathBuf::try_from(tmp.path()).expect("absolute tempdir");
    turn.environments.turn_environments[0].cwd = cwd.clone();
    std::mem::forget(tmp);
    let mut config = (*turn.config).clone();
    config.edit_reminder = reminder.map(str::to_string);
    turn.config = Arc::new(config);
    setup(&cwd.to_path_buf());
    let invocation = ToolInvocation {
        session: Arc::new(session),
        turn: Arc::new(turn),
        cancellation_token: tokio_util::sync::CancellationToken::new(),
        tracker: Arc::new(Mutex::new(TurnDiffTracker::new())),
        call_id: format!("call-{tool}-reminder"),
        tool_name: codex_tools::ToolName::plain(tool),
        source: ToolCallSource::Direct,
        payload: ToolPayload::Function { arguments: args.to_string() },
    };
    let output = if tool == "create_file" {
        CreateFileHandler.handle(invocation).await
    } else {
        StrReplaceHandler.handle(invocation).await
    }
    .expect("edit succeeds");
    output.log_preview()
}

#[tokio::test]
async fn edit_reminder_is_appended_to_successful_edits() {
    let out = call_edit_tool_with_reminder(
        "create_file",
        Some("TEST REMINDER"),
        serde_json::json!({"path": "a.txt", "content": "hello\n"}),
        |_| {},
    )
    .await;
    assert!(out.starts_with("Created a.txt"), "{out}");
    assert!(out.ends_with("TEST REMINDER"), "{out}");
    let out = call_edit_tool_with_reminder(
        "str_replace",
        Some("TEST REMINDER"),
        serde_json::json!({"path": "b.txt", "old_str": "hello", "new_str": "bye"}),
        |cwd| std::fs::write(cwd.join("b.txt"), "hello\n").expect("write"),
    )
    .await;
    assert!(out.starts_with("Replaced in b.txt"), "{out}");
    assert!(out.ends_with("TEST REMINDER"), "{out}");
}

#[tokio::test]
async fn edit_reminder_off_leaves_results_untouched() {
    let out = call_edit_tool_with_reminder(
        "str_replace",
        None,
        serde_json::json!({"path": "c.txt", "old_str": "hello", "new_str": "bye"}),
        |cwd| std::fs::write(cwd.join("c.txt"), "hello\n").expect("write"),
    )
    .await;
    assert_eq!(out, "Replaced in c.txt");
}
