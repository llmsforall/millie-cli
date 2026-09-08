//! Runtime resolution of the served system prompt.
//!
//! The prompt is a plain text file, never compiled into the binary, so it can
//! be changed without a rebuild. It is looked up, in order, at:
//!
//! 1. the path in the `MILLIE_SYSTEM_PROMPT` environment variable,
//! 2. `<MILLIE_HOME>/system_prompt.md`,
//! 3. `system_prompt.md` next to the running executable (release bundles),
//! 4. `prompts/system_prompt.md` in a source checkout (development only).
//!
//! If none exists that is a hard error: a session must never start on a
//! prompt nobody chose.

use std::path::Path;
use std::path::PathBuf;

/// Environment variable naming an explicit prompt file.
pub const SYSTEM_PROMPT_ENV: &str = "MILLIE_SYSTEM_PROMPT";
/// File name looked up under `MILLIE_HOME` and next to the executable.
pub const SYSTEM_PROMPT_FILE_NAME: &str = "system_prompt.md";
/// Placeholder in the prompt file replaced by the personality text.
pub const PERSONALITY_PLACEHOLDER: &str = "{{ personality }}";

/// Checkout-relative location, for `cargo run`/`cargo test` from the repo.
#[cfg(any(debug_assertions, test))]
const DEV_CHECKOUT_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../prompts/",
    "system_prompt.md"
);

/// The prompt text and where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedSystemPrompt {
    pub text: String,
    pub path: PathBuf,
}

/// Resolve the system prompt for the given `MILLIE_HOME`.
pub fn resolve_system_prompt(codex_home: &Path) -> std::io::Result<ResolvedSystemPrompt> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(explicit) = std::env::var_os(SYSTEM_PROMPT_ENV) {
        // An explicitly named file that is missing is an error, not a
        // reason to fall through to a prompt the user did not choose.
        let explicit = PathBuf::from(explicit);
        if !explicit.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!(
                    "{SYSTEM_PROMPT_ENV} is set to {} but no such file exists",
                    explicit.display()
                ),
            ));
        }
        candidates.push(explicit);
    }
    candidates.push(codex_home.join(SYSTEM_PROMPT_FILE_NAME));
    if let Some(exe_dir) = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(Path::to_path_buf))
    {
        candidates.push(exe_dir.join(SYSTEM_PROMPT_FILE_NAME));
    }
    #[cfg(any(debug_assertions, test))]
    candidates.push(PathBuf::from(DEV_CHECKOUT_PATH));

    for path in &candidates {
        match std::fs::read_to_string(path) {
            Ok(text) if !text.trim().is_empty() => {
                tracing::info!(path = %path.display(), "system prompt loaded");
                return Ok(ResolvedSystemPrompt {
                    text,
                    path: path.clone(),
                });
            }
            Ok(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("system prompt file {} is empty", path.display()),
                ));
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => continue,
            Err(err) => {
                return Err(std::io::Error::new(
                    err.kind(),
                    format!("failed to read system prompt {}: {err}", path.display()),
                ));
            }
        }
    }

    let looked_in = candidates
        .iter()
        .map(|p| format!("  {}", p.display()))
        .collect::<Vec<_>>()
        .join("\n");
    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!(
            "no system prompt found. Looked for {SYSTEM_PROMPT_FILE_NAME} at:\n{looked_in}\n\
             Set {SYSTEM_PROMPT_ENV}=<file> or place the file in one of those locations."
        ),
    ))
}

/// Resolve the prompt for tests and tooling that run from a source checkout.
/// Panics with the resolver's message if nothing is found.
pub fn resolve_system_prompt_for_tests(codex_home: &Path) -> ResolvedSystemPrompt {
    match resolve_system_prompt(codex_home) {
        Ok(prompt) => prompt,
        Err(err) => panic!("{err}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkout_prompt_exists_and_is_the_str_replace_world() {
        let text = std::fs::read_to_string(DEV_CHECKOUT_PATH).expect("prompts/system_prompt.md");
        assert!(
            text.contains("## File editing: create_file and str_replace"),
            "prompt must carry the str_replace editing doc"
        );
        assert!(
            !text.contains("apply_patch"),
            "prompt must not mention apply_patch: that tool is not declared"
        );
        assert_eq!(
            text.matches(PERSONALITY_PLACEHOLDER).count(),
            1,
            "prompt must contain the personality placeholder exactly once"
        );
    }

    #[test]
    fn millie_home_file_wins_over_checkout() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(SYSTEM_PROMPT_FILE_NAME), "custom prompt\n").unwrap();
        let resolved = resolve_system_prompt(dir.path()).expect("resolves");
        assert_eq!(resolved.text, "custom prompt\n");
        assert_eq!(resolved.path, dir.path().join(SYSTEM_PROMPT_FILE_NAME));
    }

    #[test]
    fn explicit_env_path_must_exist() {
        // Run serially with other env-dependent tests: set, check, unset.
        let dir = tempfile::tempdir().expect("tempdir");
        let missing = dir.path().join("nope.md");
        unsafe { std::env::set_var(SYSTEM_PROMPT_ENV, &missing) };
        let result = resolve_system_prompt(dir.path());
        unsafe { std::env::remove_var(SYSTEM_PROMPT_ENV) };
        let err = result.expect_err("missing explicit prompt file must be an error");
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn empty_file_is_an_error() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join(SYSTEM_PROMPT_FILE_NAME), "  \n").unwrap();
        let err = resolve_system_prompt(dir.path()).expect_err("empty prompt rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
    }
}
