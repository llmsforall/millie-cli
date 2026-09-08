use std::collections::BTreeMap;
use std::collections::HashMap;
use std::path::PathBuf;

use codex_utils_absolute_path::AbsolutePathBuf;
use serde::Deserialize;
use serde::Serialize;

pub(crate) fn resolve_exporter(exporter: &OtelExporter) -> OtelExporter {
    match exporter {
        // Upstream's built-in metrics exporter posted to a third-party
        // analytics endpoint. This build never sends metrics anywhere unless
        // the user configures an explicit OTLP exporter; the variant is kept
        // only so existing configs still parse.
        OtelExporter::Statsig => OtelExporter::None,
        _ => exporter.clone(),
    }
}

/// Validates configured span attributes before they are attached to exported spans.
pub fn validate_span_attributes(attributes: &BTreeMap<String, String>) -> std::io::Result<()> {
    if attributes.keys().any(String::is_empty) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "configured span attribute key must not be empty",
        ));
    }

    Ok(())
}

#[derive(Clone, Debug)]
pub struct OtelSettings {
    pub environment: String,
    pub service_name: String,
    pub service_version: String,
    pub codex_home: PathBuf,
    pub exporter: OtelExporter,
    pub trace_exporter: OtelExporter,
    pub metrics_exporter: OtelExporter,
    pub runtime_metrics: bool,
    pub span_attributes: BTreeMap<String, String>,
    pub tracestate: BTreeMap<String, BTreeMap<String, String>>,
}

/// Resolved Statsig metrics settings that another process can use to recreate
/// the built-in metrics exporter configuration without receiving generic
/// exporter credentials in-process.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatsigMetricsSettings {
    pub environment: String,
}

#[derive(Clone, Debug)]
pub enum OtelHttpProtocol {
    /// HTTP protocol with binary protobuf
    Binary,
    /// HTTP protocol with JSON payload
    Json,
}

#[derive(Clone, Debug, Default)]
pub struct OtelTlsConfig {
    pub ca_certificate: Option<AbsolutePathBuf>,
    pub client_certificate: Option<AbsolutePathBuf>,
    pub client_private_key: Option<AbsolutePathBuf>,
}

#[derive(Clone, Debug)]
pub enum OtelExporter {
    None,
    /// Statsig metrics ingestion exporter using Codex-internal defaults.
    ///
    /// This is intended for metrics only.
    Statsig,
    OtlpGrpc {
        endpoint: String,
        headers: HashMap<String, String>,
        tls: Option<OtelTlsConfig>,
    },
    OtlpHttp {
        endpoint: String,
        headers: HashMap<String, String>,
        protocol: OtelHttpProtocol,
        tls: Option<OtelTlsConfig>,
    },
}

#[cfg(test)]
mod tests {
    use super::OtelExporter;
    use super::resolve_exporter;

    #[test]
    fn statsig_exporter_resolves_to_none() {
        assert!(matches!(
            resolve_exporter(&OtelExporter::Statsig),
            OtelExporter::None
        ));
    }
}
