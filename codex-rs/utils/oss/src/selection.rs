use std::io;
use std::io::IsTerminal;
use std::path::Path;

/// Resolve the last explicit choice before constructing session configuration.
/// Saving a choice does not approve its download or depend on server startup.
pub fn select_startup_model(
    home: &Path,
    config_path: &Path,
    explicit: Option<&str>,
    configured: Option<&str>,
) -> io::Result<String> {
    let manual_path =
        std::env::var("MILLIE_LLAMACPP_MODEL_PATH").is_ok_and(|path| !path.trim().is_empty());
    let chooser = explicit == Some("select")
        || (explicit.is_none()
            && configured.is_none()
            && !manual_path
            && io::stdin().is_terminal());
    let selected = if chooser {
        if !io::stdin().is_terminal() || !io::stderr().is_terminal() {
            return Err(io::Error::other(
                "The model chooser requires a terminal. Use --model MODEL instead.",
            ));
        }
        let recommendation = codex_models_manager::catalog::pick_local_model_slug(home)?;
        return super::choose_model_interactively(home, config_path, &recommendation);
    } else if let Some(model) = explicit.or(configured) {
        model.to_owned()
    } else {
        codex_models_manager::catalog::pick_local_model_slug(home)?
    };
    let models = codex_models_manager::catalog::resolve_model_catalog(home)?.models;
    if !models.iter().any(|m| m.slug == selected) {
        return Err(io::Error::other(format!(
            "Unknown model `{selected}`. Use `millie --model select` to choose. Available models: {}",
            models
                .iter()
                .map(|m| m.slug.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if explicit.is_some() {
        codex_core::config::edit::ConfigEditsBuilder::for_config_path(config_path)
            .set_model(Some(&selected), None)
            .apply_blocking()
            .map_err(|e| io::Error::other(format!("Could not remember model selection: {e}")))?;
    }
    Ok(selected)
}

pub(super) fn approve_download(model: &str, files: &[String]) -> io::Result<()> {
    eprintln!(
        "Missing files for {model}:\n  {}\nTo choose another model, run millie --model select.",
        files.join("\n  ")
    );
    if io::stdin().is_terminal() && io::stderr().is_terminal() {
        eprint!("Download the missing files for {model}? [y/N]: ");
        let mut answer = String::new();
        io::stdin().read_line(&mut answer)?;
        if matches!(answer.trim().to_ascii_lowercase().as_str(), "y" | "yes") {
            return Ok(());
        }
    }
    Err(io::Error::other(format!(
        "Download not approved for {model}. Retry with --download to approve. Your model selection is unchanged."
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_profile_remembers_choice_without_changing_base_or_other_profile() {
        let home =
            std::env::temp_dir().join(format!("millie-profile-choice-{}", std::process::id()));
        std::fs::create_dir_all(&home).unwrap();
        let base = home.join("config.toml");
        let profile = home.join("work.toml");
        let other = home.join("other.toml");
        for path in [&base, &profile, &other] {
            std::fs::write(path, "model = 'old'\n").unwrap();
        }
        let models = codex_models_manager::catalog::resolve_model_catalog(&home)
            .unwrap()
            .models;
        select_startup_model(&home, &profile, Some(&models[0].slug), None).unwrap();
        select_startup_model(&home, &profile, Some(&models[1].slug), None).unwrap();
        assert_eq!(std::fs::read_to_string(&base).unwrap(), "model = 'old'\n");
        assert_eq!(std::fs::read_to_string(&other).unwrap(), "model = 'old'\n");
        let saved = std::fs::read_to_string(&profile)
            .unwrap()
            .parse::<toml::Table>()
            .unwrap();
        assert_eq!(saved["model"].as_str(), Some(models[1].slug.as_str()));
        std::fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn last_valid_selection_is_saved_before_startup_and_invalid_names_do_not_replace_it() {
        let home = std::env::temp_dir().join(format!("millie-choice-test-{}", std::process::id()));
        std::fs::create_dir_all(&home).unwrap();
        let models = codex_models_manager::catalog::resolve_model_catalog(&home)
            .unwrap()
            .models;
        let first = &models[0].slug;
        let second = &models[1].slug;
        assert_eq!(
            select_startup_model(&home, &home.join("config.toml"), Some(first), None).unwrap(),
            *first
        );
        assert_eq!(
            select_startup_model(&home, &home.join("config.toml"), Some(second), Some(first))
                .unwrap(),
            *second
        );
        // No server or download was attempted, so persistence cannot depend on startup success.
        let saved = std::fs::read_to_string(home.join("config.toml")).unwrap();
        let parsed = saved.parse::<toml::Table>().unwrap();
        assert_eq!(
            parsed.get("model").and_then(|v| v.as_str()),
            Some(second.as_str())
        );
        assert_eq!(
            select_startup_model(&home, &home.join("config.toml"), None, Some(second)).unwrap(),
            *second
        );
        assert!(
            select_startup_model(
                &home,
                &home.join("config.toml"),
                Some("invalid-model-name"),
                Some(second)
            )
            .is_err()
        );
        assert_eq!(
            std::fs::read_to_string(home.join("config.toml")).unwrap(),
            saved
        );
        std::fs::remove_dir_all(home).unwrap();
    }
}
