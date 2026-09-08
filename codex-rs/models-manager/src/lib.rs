pub(crate) mod cache;
pub mod catalog;
pub mod collaboration_mode_presets;
pub(crate) mod config;
pub mod manager;
pub mod model_info;
pub mod model_presets;
pub mod system_prompt;
pub mod test_support;

pub use codex_app_server_protocol::AuthMode;
pub use config::ModelsManagerConfig;

/// Load the bundled model catalog shipped with `codex-models-manager`.
pub fn bundled_models_response()
-> std::result::Result<codex_protocol::openai_models::ModelsResponse, serde_json::Error> {
    serde_json::from_str(include_str!("../models.json"))
}

/// The Millie catalog as tests and tooling see it: read from the source
/// checkout (`catalog::resolve_model_catalog` with no `MILLIE_HOME` file), with
/// the given system prompt installed. Panics with the resolver's message if
/// no catalog is found. Production code goes through `catalog::millie_models`.
pub fn bundled_millie_models(system_prompt: &str) -> Vec<codex_protocol::openai_models::ModelInfo> {
    let nowhere = std::path::Path::new("/nonexistent-millie-home");
    catalog::millie_models(system_prompt, nowhere).expect("model catalog resolves from the checkout")
}


/// Convert the client version string to a whole version string (e.g. "1.2.3-alpha.4" -> "1.2.3").
pub fn client_version_to_whole() -> String {
    format!(
        "{}.{}.{}",
        env!("CARGO_PKG_VERSION_MAJOR"),
        env!("CARGO_PKG_VERSION_MINOR"),
        env!("CARGO_PKG_VERSION_PATCH")
    )
}
