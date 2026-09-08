//! Network regressions shared by llama-server chat/raw and vLLM chat.
use crate::client_common::Prompt;
use crate::client_common::ResponseStream;
use crate::llamacpp_chat_transport::Backend;
use crate::llamacpp_chat_transport::stream_chat_completions;
use codex_api::ResponseEvent;
use codex_protocol::models::ResponseItem;
use futures::StreamExt;
use serde_json::json;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::time::Duration;

#[derive(Clone, Copy, Debug)]
enum Transport {
    Raw,
    Chat,
    Vllm,
}
const TRANSPORTS: [Transport; 3] = [Transport::Raw, Transport::Chat, Transport::Vllm];

async fn stream(transport: Transport, base: &str) -> ResponseStream {
    let prompt = Prompt::default();
    match transport {
        Transport::Raw => crate::llamacpp_transport::stream_raw_completions(base, &prompt)
            .await
            .unwrap(),
        Transport::Chat | Transport::Vllm => stream_chat_completions(
            base,
            &prompt,
            crate::model_call_log::ModelCallKey {
                thread_id: "transport-test".into(),
                turn_id: "turn".into(),
                call_ordinal: 0,
            },
            ("transport-test".into(), 0),
            if matches!(transport, Transport::Chat) {
                Backend::LlamaServer
            } else {
                Backend::Vllm
            },
        )
        .await
        .unwrap(),
    }
}
fn partial(transport: Transport) -> String {
    let chunk = match transport {
        Transport::Raw => {
            json!({"content":"thinking</think>partial answer <tool_call>{\"name\":\"shell\",\"arguments\":{}}", "stop":false})
        }
        Transport::Chat | Transport::Vllm => {
            json!({"choices":[{"delta":{"content":"partial answer", "tool_calls":[{"index":0,"id":"call_1","function":{"name":"shell","arguments":"{}"}}]}, "finish_reason":null}]})
        }
    };
    format!("data: {chunk}\n\n")
}
fn terminal(transport: Transport, reason: &str) -> String {
    match transport {
        Transport::Raw => format!(
            "data: {}\n\n",
            json!({"content":"", "stop":true,"stop_type":reason,"tokens_evaluated":10,"tokens_predicted":2})
        ),
        Transport::Chat | Transport::Vllm => format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"choices":[{"delta":{},"finish_reason":reason}]})
        ),
    }
}
async fn server(body: String, transport: Transport) -> wiremock::MockServer {
    let server = wiremock::MockServer::start().await;
    let path = if matches!(transport, Transport::Raw) {
        "/completion"
    } else {
        "/v1/chat/completions"
    };
    wiremock::Mock::given(wiremock::matchers::path(path))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .expect(1)
        .mount(&server)
        .await;
    server
}

#[tokio::test]
async fn errors_and_premature_eof_never_complete_or_finalize_tools() {
    for transport in TRANSPORTS {
        for tail in [
            "",
            "data: {\"error\":{\"message\":\"generation failed\"}}\n\n",
            "data: [DONE]\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
        ] {
            let server = server(format!("{}{tail}", partial(transport)), transport).await;
            let mut response = stream(transport, &format!("{}/v1/", server.uri())).await;
            let mut failed = false;
            while let Some(event) = response.next().await {
                match event {
                    Err(error) => {
                        assert!(error.to_string().contains("llama-server"));
                        failed = true;
                    }
                    Ok(ResponseEvent::Completed { .. } | ResponseEvent::OutputItemDone(_)) => {
                        panic!("{transport:?} finalized an interrupted stream")
                    }
                    _ => {}
                }
            }
            assert!(failed, "{transport:?} hid the failure");
        }
    }
}

#[tokio::test]
async fn valid_completion_and_stop_reasons_survive_with_trailing_api_slash() {
    for transport in TRANSPORTS {
        for (stop, expected, ended) in if matches!(transport, Transport::Raw) {
            [
                ("eos", "stop", true),
                ("limit", "length", false),
                (
                    "repetition",
                    crate::pathology::REPETITION_FINISH_REASON,
                    false,
                ),
            ]
        } else {
            [
                ("stop", "stop", true),
                ("length", "length", false),
                (
                    "repetition",
                    crate::pathology::REPETITION_FINISH_REASON,
                    false,
                ),
            ]
        } {
            let server = server(
                format!("{}{}", partial(transport), terminal(transport, stop))
                    .replace('\n', "\r\n"),
                transport,
            )
            .await;
            let mut response = stream(transport, &format!("{}/v1/", server.uri())).await;
            let mut completed = false;
            while let Some(event) = response.next().await {
                match event.unwrap() {
                    ResponseEvent::Completed {
                        finish_reason,
                        end_turn,
                        ..
                    } => {
                        assert_eq!(
                            (finish_reason.as_deref(), end_turn),
                            (Some(expected), Some(ended))
                        );
                        completed = true;
                    }
                    ResponseEvent::OutputItemDone(ResponseItem::FunctionCall { .. })
                        if matches!(transport, Transport::Raw) && !ended =>
                    {
                        panic!("truncated raw response executed a tool")
                    }
                    _ => {}
                }
            }
            assert!(completed);
        }
    }
}

// Read the whole HTTP request before testing EOF on its connection.
async fn read_request(socket: &mut tokio::net::TcpStream) {
    let mut request = Vec::new();
    loop {
        let byte = socket.read_u8().await.unwrap();
        request.push(byte);
        if request.ends_with(b"\r\n\r\n") {
            break;
        }
    }
    let headers = String::from_utf8_lossy(&request).to_ascii_lowercase();
    let len: usize = headers
        .lines()
        .find_map(|line| line.strip_prefix("content-length:"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    socket.read_exact(&mut vec![0; len]).await.unwrap();
}

#[tokio::test]
async fn dropping_consumer_closes_silent_and_buffered_tool_streams() {
    for transport in TRANSPORTS {
        for buffered in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}/v1", listener.local_addr().unwrap());
            let (sent, received) = tokio::sync::oneshot::channel();
            let serving = tokio::spawn(async move {
                let (mut socket, _) = listener.accept().await.unwrap();
                read_request(&mut socket).await;
                socket.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n").await.unwrap();
                if buffered {
                    let body = match transport {
                        Transport::Raw => {
                            "data: {\"content\":\"<tool_call>{\",\"stop\":false}\n\n".to_string()
                        }
                        _ => format!(
                            "data: {}\n\n",
                            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{"}}]}}]})
                        ),
                    };
                    socket
                        .write_all(format!("{:x}\r\n{body}\r\n", body.len()).as_bytes())
                        .await
                        .unwrap();
                }
                sent.send(()).unwrap();
                let mut byte = [0];
                assert_eq!(
                    tokio::time::timeout(Duration::from_secs(2), socket.read(&mut byte))
                        .await
                        .unwrap()
                        .unwrap(),
                    0,
                    "request connection did not close"
                );
                // Only accept the next request once the occupied request is released.
                let (mut next, _) = listener.accept().await.unwrap();
                read_request(&mut next).await;
                let body = terminal(transport, "stop");
                next.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).await.unwrap();
            });
            let response = stream(transport, &url).await;
            received.await.unwrap();
            drop(response);
            let mut next = tokio::time::timeout(Duration::from_secs(3), stream(transport, &url))
                .await
                .unwrap();
            let mut completed = false;
            while let Some(event) = next.next().await {
                completed |= matches!(event.unwrap(), ResponseEvent::Completed { .. });
            }
            assert!(completed);
            serving.await.unwrap();
        }
    }
}

#[tokio::test]
async fn stalled_penalty_tokenization_is_bounded() {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::path("/tokenize"))
        .respond_with(
            wiremock::ResponseTemplate::new(200)
                .set_delay(Duration::from_secs(30))
                .set_body_json(json!({"tokens":[1,2]})),
        )
        .mount(&server)
        .await;
    let ids = tokio::time::timeout(
        Duration::from_secs(12),
        crate::llamacpp_chat_transport::tokenize_spans(
            &reqwest::Client::new(),
            &server.uri(),
            "model",
            &["stalled unique span".into()],
        ),
    )
    .await
    .unwrap();
    assert!(ids.is_empty());
}

#[tokio::test]
async fn intentional_vllm_repetition_stop_does_not_require_server_done() {
    let body = (0..24)
        .map(|i| {
            format!(
                "data: {}\n\n",
                json!({"choices":[{"delta":{"content":(["a","b","c"][i%3])},"finish_reason":null}]})
            )
        })
        .collect::<String>();
    let server = server(body, Transport::Vllm).await;
    let prompt = Prompt {
        llamacpp_request_extra: Some(
            json!({"repeat_stop":{"windows":[12],"min_tokens":12,"min_repeats":3,"coverage":0.8,"stride":1}}),
        ),
        ..Prompt::default()
    };
    let mut response = stream_chat_completions(
        &server.uri(),
        &prompt,
        crate::model_call_log::ModelCallKey {
            thread_id: "repeat-test".into(),
            turn_id: "turn".into(),
            call_ordinal: 0,
        },
        ("repeat-test".into(), 0),
        Backend::Vllm,
    )
    .await
    .unwrap();
    let mut completed = false;
    while let Some(event) = response.next().await {
        if let ResponseEvent::Completed {
            finish_reason,
            end_turn,
            ..
        } = event.unwrap()
        {
            assert_eq!(
                (finish_reason.as_deref(), end_turn),
                (
                    Some(crate::pathology::REPETITION_FINISH_REASON),
                    Some(false)
                )
            );
            completed = true;
        }
    }
    assert!(completed);
}

#[tokio::test]
async fn capped_and_repeated_chat_batches_never_finalize_tools() {
    for transport in [Transport::Chat, Transport::Vllm] {
        for reason in ["length", "repetition"] {
            let server = server(
                format!("{}{}", partial(transport), terminal(transport, reason)),
                transport,
            )
            .await;
            let mut response = stream(transport, &server.uri()).await;
            let mut completed = false;
            while let Some(event) = response.next().await {
                match event.unwrap() {
                    ResponseEvent::OutputItemDone(
                        ResponseItem::FunctionCall { .. } | ResponseItem::CustomToolCall { .. },
                    ) => panic!("truncated tool call finalized"),
                    ResponseEvent::Completed { end_turn, .. } => {
                        assert_eq!(end_turn, Some(false));
                        completed = true;
                    }
                    _ => {}
                }
            }
            assert!(completed);
        }
    }
}
