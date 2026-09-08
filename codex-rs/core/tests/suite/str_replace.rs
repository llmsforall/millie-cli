#![cfg(not(target_os = "windows"))]
#![allow(clippy::expect_used)]

use std::fs;

use anyhow::Result;
use codex_protocol::models::PermissionProfile;
use codex_protocol::protocol::AskForApproval;
use core_test_support::responses::ev_assistant_message;
use core_test_support::responses::ev_completed;
use core_test_support::responses::ev_function_call;
use core_test_support::responses::ev_response_created;
use core_test_support::responses::mount_sse_once;
use core_test_support::responses::sse;
use core_test_support::responses::start_mock_server;
use core_test_support::skip_if_no_network;
use core_test_support::test_codex::test_codex;
use pretty_assertions::assert_eq;
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn str_replace_calls_in_one_reply_execute_in_order() -> Result<()> {
    skip_if_no_network!(Ok(()));

    let server = start_mock_server().await;
    let mut builder = test_codex();
    let test = builder.build(&server).await?;
    let path = test.workspace_path("ordered.txt");
    fs::write(&path, "one\n").expect("write fixture");

    let first_args = json!({
        "path": "ordered.txt",
        "old_str": "one",
        "new_str": "two",
    });
    let second_args = json!({
        "path": "ordered.txt",
        "old_str": "two",
        "new_str": "three",
    });

    mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-1"),
            ev_function_call(
                "replace-1",
                "str_replace",
                &serde_json::to_string(&first_args)?,
            ),
            ev_function_call(
                "replace-2",
                "str_replace",
                &serde_json::to_string(&second_args)?,
            ),
            ev_completed("resp-1"),
        ]),
    )
    .await;
    let final_mock = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-2"),
            ev_assistant_message("msg-1", "done"),
            ev_completed("resp-2"),
        ]),
    )
    .await;

    test.submit_turn_with_approval_and_permission_profile(
        "apply ordered replacements",
        AskForApproval::Never,
        PermissionProfile::Disabled,
    )
    .await?;

    let request = final_mock.single_request();
    assert_eq!(
        request.function_call_output_text("replace-1").as_deref(),
        Some("Replaced in ordered.txt")
    );
    assert_eq!(
        request.function_call_output_text("replace-2").as_deref(),
        Some("Replaced in ordered.txt")
    );
    assert_eq!(
        fs::read_to_string(path).expect("read updated file"),
        "three\n"
    );

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edit_tool_result_strings_preserve_compatibility() -> Result<()> {
    skip_if_no_network!(Ok(()));

    // Keep tool-result strings stable so callers can distinguish failures
    // from successful edits and decide whether to re-read or retry.
    // Pin every result character-for-character.
    let server = start_mock_server().await;
    let mut builder = test_codex();
    let test = builder.build(&server).await?;
    let existing = test.workspace_path("dup.txt");
    fs::write(&existing, "dup\ndup\n").expect("write fixture");

    let calls: Vec<(&str, &str, serde_json::Value)> = vec![
        (
            "c-new",
            "create_file",
            json!({"path": "made/new.txt", "content": "hi"}),
        ),
        (
            "c-dup",
            "create_file",
            json!({"path": "dup.txt", "content": "hi"}),
        ),
        (
            "s-missing",
            "str_replace",
            json!({"path": "absent.txt", "old_str": "a", "new_str": "b"}),
        ),
        (
            "s-none",
            "str_replace",
            json!({"path": "dup.txt", "old_str": "zebra", "new_str": "b"}),
        ),
        (
            "s-multi",
            "str_replace",
            json!({"path": "dup.txt", "old_str": "dup", "new_str": "b"}),
        ),
        ("d-missing", "delete_file", json!({"path": "absent.txt"})),
        ("d-ok", "delete_file", json!({"path": "made/new.txt"})),
    ];
    let mut events = vec![ev_response_created("resp-1")];
    for (id, name, args) in &calls {
        events.push(ev_function_call(id, name, &serde_json::to_string(args)?));
    }
    events.push(ev_completed("resp-1"));
    mount_sse_once(&server, sse(events)).await;
    let final_mock = mount_sse_once(
        &server,
        sse(vec![
            ev_response_created("resp-2"),
            ev_assistant_message("msg-1", "done"),
            ev_completed("resp-2"),
        ]),
    )
    .await;

    test.submit_turn_with_approval_and_permission_profile(
        "exercise every edit-tool result string",
        AskForApproval::Never,
        PermissionProfile::Disabled,
    )
    .await?;

    let request = final_mock.single_request();
    let expect = [
        ("c-new", "Created made/new.txt"),
        ("c-dup", "create_file failed: dup.txt already exists"),
        ("s-missing", "str_replace failed: absent.txt not found"),
        ("s-none", "str_replace failed: old_str not found in file"),
        ("s-multi", "str_replace failed: old_str matched 2 locations"),
        ("d-missing", "delete_file failed: absent.txt not found"),
        ("d-ok", "Deleted made/new.txt"),
    ];
    for (id, want) in expect {
        assert_eq!(
            request.function_call_output_text(id).as_deref(),
            Some(want),
            "result string for {id}"
        );
    }
    // create_file appends the trailing newline; failed calls changed nothing
    assert_eq!(
        fs::read_to_string(test.workspace_path("dup.txt")).expect("read unchanged"),
        "dup\ndup\n"
    );
    assert!(!test.workspace_path("made/new.txt").exists());

    Ok(())
}
