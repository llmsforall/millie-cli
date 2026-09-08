//! Explicit model update commands; loading configuration never launches a server.
use clap::Args;
use clap::Subcommand;
use codex_core::config::find_codex_home;
use codex_core::config::load_config_as_toml_with_cli_and_load_options;
use codex_utils_cli::CliConfigOverrides;

#[derive(Debug, Args)]
pub(crate) struct ModelsCommand {
    #[command(subcommand)]
    command: ModelCommand,
}
#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Check the tracked Hugging Face files without downloading model weights.
    CheckUpdates { model: Option<String> },
    /// Download and verify new weights; use them on the next server launch.
    Update { model: Option<String> },
}

pub(crate) async fn run(
    command: ModelsCommand,
    overrides: CliConfigOverrides,
    interactive: &codex_tui::Cli,
) -> anyhow::Result<()> {
    let home = find_codex_home()?;
    let config = load_config_as_toml_with_cli_and_load_options(
        &home,
        /*cwd*/ None,
        overrides.parse_overrides().map_err(anyhow::Error::msg)?,
        codex_config::ConfigLoadOptions {
            loader_overrides: super::loader_overrides_for_profile(
                interactive.config_profile_v2.as_ref(),
            )?,
            strict_config: interactive.strict_config,
        },
    )
    .await?;
    let (model, action) = match command.command {
        ModelCommand::CheckUpdates { model } => {
            (model, codex_exec::model_updates::UpdateAction::Check)
        }
        ModelCommand::Update { model } => (model, codex_exec::model_updates::UpdateAction::Install),
    };
    let local = config.llamacpp.unwrap_or_default();
    let selection = codex_exec::model_updates::ModelSelection {
        explicit: model.or_else(|| interactive.model.clone()),
        remembered: config.model,
        model_hf: local.model_hf,
        model_path: interactive
            .llama_model_path
            .clone()
            .or_else(|| local.model_path.map(|p| p.to_path_buf())),
    };
    let result = codex_exec::model_updates::run(&home, selection, action).await?;
    println!("{result}");
    Ok(())
}
