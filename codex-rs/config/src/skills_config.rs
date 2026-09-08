//! Skill-related configuration types shared across crates.

use codex_utils_absolute_path::AbsolutePathBuf;
use schemars::JsonSchema;
use serde::Deserialize;
use serde::Serialize;

/// Bundled (sample) skills are OFF by default: they index two meta-skills
/// -- guides for authoring skills and plugins -- which cost ~764 prompt
/// tokens on every request while having nothing to do with the user's work.
/// The sample files are still installed on disk; this only governs whether
/// they are advertised in the prompt. Turning them on restores the previous
/// prompt exactly.
const fn default_enabled() -> bool {
    false
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct SkillConfig {
    /// Path-based selector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<AbsolutePathBuf>,
    /// Name-based selector.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub enabled: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq, Eq, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct SkillsConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bundled: Option<BundledSkillsConfig>,

    /// Whether turns receive the automatic skills instructions block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_instructions: Option<bool>,

    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub config: Vec<SkillConfig>,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq, JsonSchema)]
#[schemars(deny_unknown_fields)]
pub struct BundledSkillsConfig {
    #[serde(default = "default_enabled")]
    pub enabled: bool,
}

impl Default for BundledSkillsConfig {
    fn default() -> Self {
        Self {
            enabled: default_enabled(),
        }
    }
}

impl TryFrom<toml::Value> for SkillsConfig {
    type Error = toml::de::Error;

    fn try_from(value: toml::Value) -> Result<Self, Self::Error> {
        SkillsConfig::deserialize(value)
    }
}
