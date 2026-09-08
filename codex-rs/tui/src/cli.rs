use clap::Args;
use clap::FromArgMatches;
use clap::Parser;
use codex_utils_cli::ApprovalModeCliArg;
use codex_utils_cli::CliConfigOverrides;
use codex_utils_cli::SharedCliOptions;

#[derive(Parser, Clone, Debug)]
#[command(version)]
pub struct Cli {
    /// Optional user prompt to start the session.
    #[arg(value_name = "PROMPT", value_hint = clap::ValueHint::Other)]
    pub prompt: Option<String>,

    /// Error out when config.toml contains fields that are not recognized by this version of Millie.
    #[arg(long = "strict-config", default_value_t = false)]
    pub strict_config: bool,

    // Internal controls set by the top-level `codex resume` subcommand.
    // These are not exposed as user flags on the base `codex` command.
    #[clap(skip)]
    pub resume_picker: bool,

    #[clap(skip)]
    pub resume_last: bool,

    /// Internal: resume a specific recorded session by id (UUID). Set by the
    /// top-level `codex resume <SESSION_ID>` wrapper; not exposed as a public flag.
    #[clap(skip)]
    pub resume_session_id: Option<String>,

    /// Internal: show all sessions (disables cwd filtering and shows CWD column).
    #[clap(skip)]
    pub resume_show_all: bool,

    /// Internal: include non-interactive sessions in resume listings.
    #[clap(skip)]
    pub resume_include_non_interactive: bool,

    // Internal controls set by the top-level `codex fork` subcommand.
    // These are not exposed as user flags on the base `codex` command.
    #[clap(skip)]
    pub fork_picker: bool,

    #[clap(skip)]
    pub fork_last: bool,

    /// Internal: fork a specific recorded session by id (UUID). Set by the
    /// top-level `codex fork <SESSION_ID>` wrapper; not exposed as a public flag.
    #[clap(skip)]
    pub fork_session_id: Option<String>,

    /// Internal: show all sessions (disables cwd filtering and shows CWD column).
    #[clap(skip)]
    pub fork_show_all: bool,

    #[clap(flatten)]
    pub shared: TuiSharedCliOptions,

    /// Configure when the model requires human approval before executing a command.
    #[arg(long = "ask-for-approval", short = 'a')]
    pub approval_policy: Option<ApprovalModeCliArg>,

    /// Enable live web search. When enabled, the native Responses `web_search` tool is available to the model (no per‑call approval).
    #[arg(long = "search", default_value_t = false)]
    pub web_search: bool,

    /// Disable alternate screen mode
    ///
    /// Runs the TUI in inline mode, preserving terminal scrollback history.
    #[arg(long = "no-alt-screen", default_value_t = false)]
    pub no_alt_screen: bool,

    /// Approve downloading missing files for the selected model.
    #[arg(long = "download", default_value_t = false)]
    pub download: bool,

    /// Serve a specific serving profile (e.g. "cpu-compact") instead of the
    /// one machine detection picks. Overrides `llamacpp.profile` in
    /// config.toml. (`--profile` selects a config profile, a different thing.)
    #[arg(long = "serving-profile", value_name = "NAME")]
    pub serving_profile: Option<String>,

    /// Path to the .gguf model file to serve via the local `llamacpp` OSS
    /// provider. Overrides `llamacpp.model_path` in config.toml.
    #[arg(long = "llama-model-path", value_name = "FILE")]
    pub llama_model_path: Option<std::path::PathBuf>,

    /// GPU indices for the local server, e.g. `--llama-gpu 1` or
    /// `--llama-gpu 0,1`. Overrides `llamacpp.gpu` in config.toml.
    #[arg(long = "llama-gpu", value_name = "INDICES", value_delimiter = ',')]
    pub llama_gpu: Option<Vec<u32>>,

    /// Port for `llama-server` to listen on. Overrides `llamacpp.port`.
    #[arg(long = "llama-port", value_name = "PORT")]
    pub llama_port: Option<u16>,

    /// Force the vision tower (image input) on, even where the serving
    /// profile leaves it off to save memory.
    #[arg(long = "vision", default_value_t = false, conflicts_with = "no_vision")]
    pub vision: bool,

    /// Force text-only serving (no image input), saving the vision tower's
    /// memory.
    #[arg(long = "no-vision", default_value_t = false)]
    pub no_vision: bool,

    #[clap(skip)]
    pub config_overrides: CliConfigOverrides,
}

impl std::ops::Deref for Cli {
    type Target = SharedCliOptions;

    fn deref(&self) -> &Self::Target {
        &self.shared.0
    }
}

impl std::ops::DerefMut for Cli {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.shared.0
    }
}

#[derive(Clone, Debug, Default)]
pub struct TuiSharedCliOptions(SharedCliOptions);

impl TuiSharedCliOptions {
    pub fn into_inner(self) -> SharedCliOptions {
        self.0
    }
}

impl std::ops::Deref for TuiSharedCliOptions {
    type Target = SharedCliOptions;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for TuiSharedCliOptions {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Args for TuiSharedCliOptions {
    fn augment_args(cmd: clap::Command) -> clap::Command {
        mark_tui_args(SharedCliOptions::augment_args(cmd))
    }

    fn augment_args_for_update(cmd: clap::Command) -> clap::Command {
        mark_tui_args(SharedCliOptions::augment_args_for_update(cmd))
    }
}

impl FromArgMatches for TuiSharedCliOptions {
    fn from_arg_matches(matches: &clap::ArgMatches) -> Result<Self, clap::Error> {
        SharedCliOptions::from_arg_matches(matches).map(Self)
    }

    fn update_from_arg_matches(&mut self, matches: &clap::ArgMatches) -> Result<(), clap::Error> {
        self.0.update_from_arg_matches(matches)
    }
}

fn mark_tui_args(cmd: clap::Command) -> clap::Command {
    cmd.mut_arg("dangerously_bypass_approvals_and_sandbox", |arg| {
        arg.conflicts_with("approval_policy")
    })
}
