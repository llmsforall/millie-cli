//! Built-in pet asset cache ownership.
//!
//! Built-in pet artwork is not bundled with this build and is never
//! downloaded. A built-in pet is available only if a validated spritesheet is
//! already present in the versioned cache under MILLIE_HOME; otherwise it is
//! reported as unavailable. Custom pets (MILLIE_HOME/pets/<id>/pet.json) are
//! unaffected.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;

use super::catalog;

const PET_PACK_VERSION: &str = "v1";
const PET_PACK_DIR: &str = "cache/tui-pets";

pub(crate) fn builtin_spritesheet_path(codex_home: &Path, file: &str) -> PathBuf {
    pack_dir(codex_home).join("assets").join(file)
}

/// Returns Ok only if a validated built-in spritesheet is already cached.
/// Nothing is downloaded: callers should treat an error as "the asset is
/// unavailable".
pub(crate) fn ensure_builtin_pet(codex_home: &Path, pet: catalog::BuiltinPet) -> Result<()> {
    let destination = builtin_spritesheet_path(codex_home, pet.spritesheet_file);
    if validate_cached_spritesheet(&destination).is_ok() {
        return Ok(());
    }
    bail!(
        "built-in pet artwork is not available in this build (no spritesheet at {})",
        destination.display()
    )
}

fn pack_dir(codex_home: &Path) -> PathBuf {
    codex_home.join(PET_PACK_DIR).join(PET_PACK_VERSION)
}

fn validate_cached_spritesheet(path: &Path) -> Result<()> {
    let (width, height) =
        image::image_dimensions(path).with_context(|| format!("read {}", path.display()))?;
    if width != catalog::SPRITESHEET_WIDTH || height != catalog::SPRITESHEET_HEIGHT {
        bail!(
            "invalid pet spritesheet dimensions for {}: expected {}x{}, got {}x{}",
            path.display(),
            catalog::SPRITESHEET_WIDTH,
            catalog::SPRITESHEET_HEIGHT,
            width,
            height
        );
    }
    Ok(())
}

#[cfg(test)]
pub(crate) fn write_test_pack(codex_home: &Path) {
    let assets_dir = pack_dir(codex_home).join("assets");
    fs::create_dir_all(&assets_dir).unwrap();
    for pet in catalog::BUILTIN_PETS {
        let path = assets_dir.join(pet.spritesheet_file);
        catalog::write_test_spritesheet(&path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn write_test_pack_installs_all_builtins() {
        let dir = tempfile::tempdir().unwrap();

        write_test_pack(dir.path());

        for pet in catalog::BUILTIN_PETS {
            let path = builtin_spritesheet_path(dir.path(), pet.spritesheet_file);
            assert!(path.is_file());
            validate_cached_spritesheet(&path).unwrap();
        }
    }
}
