//! Catalog-managed models must resolve every sampling value before serving.
use codex_models_manager::catalog::ModelDownload;
use std::io;

pub(super) fn validate(download: &ModelDownload) -> io::Result<()> {
    let s = download.sampling.as_ref();
    let fields = [
        (
            "TEMPERATURE",
            "temperature",
            s.and_then(|s| s.temperature).map(|v| v.to_string()),
        ),
        (
            "TOP_P",
            "top_p",
            s.and_then(|s| s.top_p).map(|v| v.to_string()),
        ),
        (
            "TOP_K",
            "top_k",
            s.and_then(|s| s.top_k).map(|v| v.to_string()),
        ),
        (
            "MIN_P",
            "min_p",
            s.and_then(|s| s.min_p).map(|v| v.to_string()),
        ),
        (
            "THINKING_BUDGET",
            "thinking_budget",
            s.and_then(|s| s.thinking_budget).map(|v| v.to_string()),
        ),
        (
            "MAX_OUTPUT_TOKENS",
            "max_output_tokens",
            s.and_then(|s| s.max_output_tokens).map(|v| v.to_string()),
        ),
    ];
    for (suffix, name, catalog) in fields {
        let configured = std::env::var(format!("MILLIE_LLAMACPP_{suffix}")).ok();
        let value = configured.filter(|v| !v.trim().is_empty()).or(catalog);
        validate_value(name, value.as_deref())?;
    }
    Ok(())
}

fn validate_value(name: &str, value: Option<&str>) -> io::Result<()> {
    let valid = value.is_some_and(|v| {
        let v = v.trim();
        match name {
            "temperature" => v.parse::<f32>().is_ok_and(|v| v.is_finite() && v >= 0.0),
            "top_p" | "min_p" => v
                .parse::<f32>()
                .is_ok_and(|v| v.is_finite() && (0.0..=1.0).contains(&v)),
            "top_k" => v.parse::<i32>().is_ok_and(|v| v >= 0),
            "thinking_budget" => v.parse::<u32>().is_ok(),
            "max_output_tokens" => v.parse::<u32>().is_ok_and(|v| v > 0),
            _ => false,
        }
    });
    if !valid {
        return Err(io::Error::other(format!(
            "Missing or invalid sampling setting `{name}` for the selected catalog model. Set it in the model's sampling block in millie-models.json or [llamacpp] in config.toml. Compiled sampling defaults are not used for catalog-managed models."
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_value;

    #[test]
    fn missing_and_invalid_catalog_settings_fail() {
        for name in [
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "thinking_budget",
            "max_output_tokens",
        ] {
            assert!(validate_value(name, None).is_err());
            assert!(validate_value(name, Some("bad")).is_err());
        }
        for (name, value) in [
            ("temperature", "NaN"),
            ("temperature", "-1"),
            ("top_p", "1.1"),
            ("min_p", "-0.1"),
            ("top_k", "1.5"),
            ("thinking_budget", "-1"),
            ("max_output_tokens", "0"),
        ] {
            assert!(validate_value(name, Some(value)).is_err());
        }
    }

    #[test]
    fn deliberate_zero_values_are_preserved() {
        for name in ["temperature", "min_p", "top_k", "thinking_budget"] {
            assert!(validate_value(name, Some("0")).is_ok());
        }
        assert!(validate_value("top_p", Some("0.95")).is_ok());
        assert!(validate_value("max_output_tokens", Some("8192")).is_ok());
    }
}
