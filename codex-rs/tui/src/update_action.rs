#[cfg(any(not(debug_assertions), test))]
use codex_install_context::InstallContext;

/// Update action the CLI should perform after the TUI exits.
///
/// Millie has no self-update channel: it is installed only by the Millie
/// bundle's installer, never by npm/bun/brew or upstream install scripts.
/// Running any upstream update command would replace Millie with a different
/// product, so no install method maps to a runnable update action.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpdateAction {
    /// Re-run the Millie bundle installer.
    RerunMillieInstaller,
}

impl UpdateAction {
    #[cfg(any(not(debug_assertions), test))]
    pub(crate) fn from_install_context(_context: &InstallContext) -> Option<Self> {
        // Never offer an automatic update: every install method upstream knew
        // about (npm/bun/brew/standalone script) belongs to the parent
        // project's release channel, not Millie's.
        None
    }

    /// Returns the list of command-line arguments for invoking the update.
    pub fn command_args(self) -> (&'static str, &'static [&'static str]) {
        match self {
            UpdateAction::RerunMillieInstaller => (
                "sh",
                &[
                    "-c",
                    "echo 'Update Millie by re-running install.sh from a newer Millie bundle.'",
                ],
            ),
        }
    }

    /// Returns string representation of the command-line arguments for invoking the update.
    pub fn command_str(self) -> String {
        let (command, args) = self.command_args();
        shlex::try_join(std::iter::once(command).chain(args.iter().copied()))
            .unwrap_or_else(|_| format!("{command} {}", args.join(" ")))
    }
}

#[cfg(not(debug_assertions))]
pub fn get_update_action() -> Option<UpdateAction> {
    UpdateAction::from_install_context(InstallContext::current())
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_install_context::InstallMethod;
    use codex_install_context::StandalonePlatform;
    use codex_utils_absolute_path::AbsolutePathBuf;
    use pretty_assertions::assert_eq;

    #[test]
    fn no_install_method_maps_to_an_update_action() {
        let native_release_dir =
            AbsolutePathBuf::from_absolute_path(std::env::temp_dir().join("native-release"))
                .expect("temp dir path should be absolute");

        let methods = [
            InstallMethod::Other,
            InstallMethod::Npm,
            InstallMethod::Bun,
            InstallMethod::Brew,
            InstallMethod::Standalone {
                platform: StandalonePlatform::Unix,
                release_dir: native_release_dir.clone(),
                resources_dir: Some(native_release_dir.join("codex-resources")),
            },
            InstallMethod::Standalone {
                platform: StandalonePlatform::Windows,
                release_dir: native_release_dir.clone(),
                resources_dir: Some(native_release_dir.join("codex-resources")),
            },
        ];
        for method in methods {
            assert_eq!(
                UpdateAction::from_install_context(&InstallContext {
                    method,
                    package_layout: None,
                }),
                None
            );
        }
    }

    #[test]
    fn update_command_never_invokes_an_upstream_installer() {
        let cmd = UpdateAction::RerunMillieInstaller.command_str();
        assert!(cmd.contains("Millie bundle"), "{cmd}");
        for forbidden in ["chatgpt.com", "@openai", "brew", "npm "] {
            assert!(!cmd.contains(forbidden), "{cmd}");
        }
    }
}
