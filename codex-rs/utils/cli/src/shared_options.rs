//! Shared command-line flags used by both interactive and non-interactive Codex entry points.

use crate::SandboxModeCliArg;
use clap::Args;
use codex_protocol::config_types::ProfileV2Name;
use std::path::PathBuf;

#[derive(Args, Clone, Debug, Default)]
pub struct SharedCliOptions {
    /// Optional image(s) to attach to the initial prompt.
    #[arg(
        long = "image",
        short = 'i',
        value_name = "FILE",
        value_delimiter = ',',
        num_args = 1..
    )]
    pub images: Vec<PathBuf>,

    /// Model to use and remember. Use `--model select` to open the startup chooser.
    #[arg(long, short = 'm')]
    pub model: Option<String>,

    /// Keep a newly launched model server running after this session exits.
    #[arg(long = "keep-model-server", default_value_t = false)]
    pub keep_model_server: bool,

    /// Select a local provider (llamacpp, vllm, lmstudio, or ollama).
    #[arg(long = "local-provider")]
    pub oss_provider: Option<String>,

    /// Layer $MILLIE_HOME/<name>.config.toml on top of the base user config.
    #[arg(long = "profile", short = 'p')]
    pub config_profile_v2: Option<ProfileV2Name>,

    /// Select the sandbox policy to use when executing model-generated shell
    /// commands.
    #[arg(long = "sandbox", short = 's')]
    pub sandbox_mode: Option<SandboxModeCliArg>,

    /// Skip all confirmation prompts and execute commands without sandboxing.
    /// EXTREMELY DANGEROUS. Intended solely for running in environments that are externally sandboxed.
    #[arg(
        long = "dangerously-bypass-approvals-and-sandbox",
        alias = "yolo",
        default_value_t = false
    )]
    pub dangerously_bypass_approvals_and_sandbox: bool,

    /// Run enabled hooks without requiring persisted hook trust for this invocation.
    /// DANGEROUS. Intended only for automation that already vets hook sources.
    #[arg(long = "dangerously-bypass-hook-trust", default_value_t = false)]
    pub bypass_hook_trust: bool,

    /// Force the local/plaintext compaction path and formula-derived auto-compact
    /// limits, even for providers that support remote/encrypted compaction. Also
    /// sets a default model context window of 64000 tokens unless one is already
    /// configured via config.toml/profile/-c overrides or --model-context-window.
    #[arg(long = "simple-context", default_value_t = false)]
    pub simple_context: bool,

    /// Override the model's context window size, in tokens. Takes precedence over
    /// config.toml/profile/-c overrides and over --simple-context's 64000 default.
    #[arg(long = "model-context-window", value_name = "TOKENS")]
    pub model_context_window: Option<i64>,

    /// Tell the agent to use the specified directory as its working root.
    #[clap(long = "cd", short = 'C', value_name = "DIR")]
    pub cwd: Option<PathBuf>,

    /// Additional directories that should be writable alongside the primary workspace.
    #[arg(long = "add-dir", value_name = "DIR", value_hint = clap::ValueHint::DirPath)]
    pub add_dir: Vec<PathBuf>,
}

impl SharedCliOptions {
    pub fn inherit_exec_root_options(&mut self, root: &Self) {
        let self_selected_sandbox_mode =
            self.sandbox_mode.is_some() || self.dangerously_bypass_approvals_and_sandbox;
        let Self {
            images,
            model,
            keep_model_server,
            oss_provider,
            config_profile_v2,
            sandbox_mode,
            dangerously_bypass_approvals_and_sandbox,
            bypass_hook_trust,
            simple_context,
            model_context_window,
            cwd,
            add_dir,
        } = self;
        let Self {
            images: root_images,
            model: root_model,
            keep_model_server: root_keep_model_server,
            oss_provider: root_oss_provider,
            config_profile_v2: root_config_profile_v2,
            sandbox_mode: root_sandbox_mode,
            dangerously_bypass_approvals_and_sandbox: root_dangerously_bypass_approvals_and_sandbox,
            bypass_hook_trust: root_bypass_hook_trust,
            simple_context: root_simple_context,
            model_context_window: root_model_context_window,
            cwd: root_cwd,
            add_dir: root_add_dir,
        } = root;

        *keep_model_server |= *root_keep_model_server;
        if model.is_none() {
            model.clone_from(root_model);
        }
        if model_context_window.is_none() {
            *model_context_window = *root_model_context_window;
        }
        if oss_provider.is_none() {
            oss_provider.clone_from(root_oss_provider);
        }
        if config_profile_v2.is_none() {
            config_profile_v2.clone_from(root_config_profile_v2);
        }
        if sandbox_mode.is_none() {
            *sandbox_mode = *root_sandbox_mode;
        }
        if !self_selected_sandbox_mode {
            *dangerously_bypass_approvals_and_sandbox =
                *root_dangerously_bypass_approvals_and_sandbox;
        }
        if !*bypass_hook_trust {
            *bypass_hook_trust = *root_bypass_hook_trust;
        }
        if !*simple_context {
            *simple_context = *root_simple_context;
        }
        if cwd.is_none() {
            cwd.clone_from(root_cwd);
        }
        if !root_images.is_empty() {
            let mut merged_images = root_images.clone();
            merged_images.append(images);
            *images = merged_images;
        }
        if !root_add_dir.is_empty() {
            let mut merged_add_dir = root_add_dir.clone();
            merged_add_dir.append(add_dir);
            *add_dir = merged_add_dir;
        }
    }

    pub fn apply_subcommand_overrides(&mut self, subcommand: Self) {
        let subcommand_selected_sandbox_mode = subcommand.sandbox_mode.is_some()
            || subcommand.dangerously_bypass_approvals_and_sandbox;
        let Self {
            images,
            model,
            keep_model_server,
            oss_provider,
            config_profile_v2,
            sandbox_mode,
            dangerously_bypass_approvals_and_sandbox,
            bypass_hook_trust,
            simple_context,
            model_context_window,
            cwd,
            add_dir,
        } = subcommand;

        self.keep_model_server |= keep_model_server;
        if let Some(model) = model {
            self.model = Some(model);
        }
        if let Some(oss_provider) = oss_provider {
            self.oss_provider = Some(oss_provider);
        }
        if let Some(config_profile_v2) = config_profile_v2 {
            self.config_profile_v2 = Some(config_profile_v2);
        }
        if subcommand_selected_sandbox_mode {
            self.sandbox_mode = sandbox_mode;
            self.dangerously_bypass_approvals_and_sandbox =
                dangerously_bypass_approvals_and_sandbox;
        }
        if bypass_hook_trust {
            self.bypass_hook_trust = true;
        }
        if simple_context {
            self.simple_context = true;
        }
        if let Some(model_context_window) = model_context_window {
            self.model_context_window = Some(model_context_window);
        }
        if let Some(cwd) = cwd {
            self.cwd = Some(cwd);
        }
        if !images.is_empty() {
            self.images = images;
        }
        if !add_dir.is_empty() {
            self.add_dir.extend(add_dir);
        }
    }
}
