pub(crate) use codex_skills::install_system_skills;
pub(crate) use codex_skills::system_cache_root_dir;

use codex_utils_absolute_path::AbsolutePathBuf;

/// Removes the bundled sample skills from disk. No longer called on the
/// disable path -- turning bundled skills off keeps the files and only stops
/// advertising them in the prompt -- but kept for explicit cleanup callers.
#[allow(dead_code)]
pub(crate) fn uninstall_system_skills(codex_home: &AbsolutePathBuf) {
    let _ = std::fs::remove_dir_all(system_cache_root_dir(codex_home));
}
