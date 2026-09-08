use crate::default_client::CodexHttpClient;
use crate::default_client::CodexRequestBuilder;
use crate::error::TransportError;
use crate::llm_traffic_log;
use crate::request::Request;
use crate::request::RequestBody;
use crate::request::Response;
use async_trait::async_trait;
use bytes::Bytes;
use futures::StreamExt;
use futures::stream::BoxStream;
use http::HeaderMap;
use http::Method;
use http::StatusCode;
use tracing::Level;
use tracing::enabled;
use tracing::trace;

pub type ByteStream = BoxStream<'static, Result<Bytes, TransportError>>;

pub struct StreamResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub bytes: ByteStream,
}

#[async_trait]
pub trait HttpTransport: Send + Sync {
    async fn execute(&self, req: Request) -> Result<Response, TransportError>;
    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError>;
}

#[derive(Clone, Debug)]
pub struct ReqwestTransport {
    client: CodexHttpClient,
}

impl ReqwestTransport {
    pub fn new(client: reqwest::Client) -> Self {
        Self {
            client: CodexHttpClient::new(client),
        }
    }

    fn build(&self, req: Request) -> Result<CodexRequestBuilder, TransportError> {
        let prepared = req.prepare_body_for_send().map_err(TransportError::Build)?;

        let Request {
            method,
            url,
            headers: _,
            body: _,
            compression: _,
            timeout,
        } = req;

        let mut builder = self.client.request(
            Method::from_bytes(method.as_str().as_bytes()).unwrap_or(Method::GET),
            &url,
        );

        if let Some(timeout) = timeout {
            builder = builder.timeout(timeout);
        }

        builder = builder.headers(prepared.headers);
        if let Some(body) = prepared.body {
            builder = builder.body(body);
        }
        Ok(builder)
    }

    fn map_error(err: reqwest::Error) -> TransportError {
        if err.is_timeout() {
            TransportError::Timeout
        } else {
            TransportError::Network(err.to_string())
        }
    }
}

fn request_body_for_trace(req: &Request) -> String {
    match req.body.as_ref() {
        Some(RequestBody::Json(body)) => body.to_string(),
        Some(RequestBody::Raw(body)) => format!("<raw body: {} bytes>", body.len()),
        None => String::new(),
    }
}

#[async_trait]
impl HttpTransport for ReqwestTransport {
    async fn execute(&self, req: Request) -> Result<Response, TransportError> {
        let exchange_id = llm_traffic_log::next_exchange_id();
        let log_llm = llm_traffic_log::should_log_url(&req.url);
        if log_llm {
            llm_traffic_log::log_http_request(exchange_id, &req);
        }
        if enabled!(Level::TRACE) {
            trace!(
                "{} to {}: {}",
                req.method,
                req.url,
                request_body_for_trace(&req)
            );
        }

        let url = req.url.clone();
        let builder = self.build(req)?;
        let resp = match builder.send().await {
            Ok(resp) => resp,
            Err(err) => {
                if log_llm {
                    llm_traffic_log::log_error(exchange_id, err.to_string());
                }
                return Err(Self::map_error(err));
            }
        };
        let status = resp.status();
        let headers = resp.headers().clone();
        if log_llm {
            llm_traffic_log::log_http_response_headers(exchange_id, status.as_u16(), &headers);
        }
        let bytes = match resp.bytes().await {
            Ok(bytes) => bytes,
            Err(err) => {
                if log_llm {
                    llm_traffic_log::log_error(exchange_id, err.to_string());
                }
                return Err(Self::map_error(err));
            }
        };
        if log_llm {
            llm_traffic_log::log_http_response_body(exchange_id, &bytes);
        }
        if !status.is_success() {
            let body = String::from_utf8(bytes.to_vec()).ok();
            return Err(TransportError::Http {
                status,
                url: Some(url),
                headers: Some(headers),
                body,
            });
        }
        Ok(Response {
            status,
            headers,
            body: bytes,
        })
    }

    async fn stream(&self, req: Request) -> Result<StreamResponse, TransportError> {
        let exchange_id = llm_traffic_log::next_exchange_id();
        let log_llm = llm_traffic_log::should_log_url(&req.url);
        if log_llm {
            llm_traffic_log::log_http_request(exchange_id, &req);
        }
        if enabled!(Level::TRACE) {
            trace!(
                "{} to {}: {}",
                req.method,
                req.url,
                request_body_for_trace(&req)
            );
        }

        let url = req.url.clone();
        let builder = self.build(req)?;
        let resp = match builder.send().await {
            Ok(resp) => resp,
            Err(err) => {
                if log_llm {
                    llm_traffic_log::log_error(exchange_id, err.to_string());
                }
                return Err(Self::map_error(err));
            }
        };
        let status = resp.status();
        let headers = resp.headers().clone();
        if log_llm {
            llm_traffic_log::log_http_response_headers(exchange_id, status.as_u16(), &headers);
        }
        if !status.is_success() {
            let body = resp.text().await.ok();
            if log_llm {
                if let Some(body) = body.as_ref() {
                    llm_traffic_log::log_http_response_body(
                        exchange_id,
                        &Bytes::from(body.clone()),
                    );
                }
            }
            return Err(TransportError::Http {
                status,
                url: Some(url),
                headers: Some(headers),
                body,
            });
        }
        let response_recorder =
            log_llm.then(|| llm_traffic_log::http_stream_response_recorder(exchange_id));
        let stream_recorder = response_recorder.clone();
        let stream = resp.bytes_stream().map(move |result| match result {
            Ok(bytes) => {
                if let Some(recorder) = stream_recorder.as_ref() {
                    recorder.append(&bytes);
                }
                Ok(bytes)
            }
            Err(err) => {
                if log_llm {
                    llm_traffic_log::log_error(exchange_id, err.to_string());
                    llm_traffic_log::log_http_stream_response_aborted(exchange_id, err.to_string());
                }
                Err(Self::map_error(err))
            }
        });
        Ok(StreamResponse {
            status,
            headers,
            bytes: Box::pin(stream),
        })
    }
}
