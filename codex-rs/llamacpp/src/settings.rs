/// Interpret the vision setting identically for download approval and launch.
/// An absent or blank value leaves the choice automatic. The launcher's
/// existing true values are `1`, `true`, and `on`; other nonempty values disable it.
pub fn parse_vision_setting(value: Option<&str>) -> Option<bool> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| matches!(value, "1" | "true" | "on"))
}

#[cfg(test)]
mod tests {
    use super::parse_vision_setting;

    #[test]
    fn vision_encodings_match_launch_behavior() {
        for value in [None, Some(""), Some("  ")] {
            assert_eq!(parse_vision_setting(value), None);
        }
        for value in ["0", "false", "off", " 0 "] {
            assert_eq!(parse_vision_setting(Some(value)), Some(false));
        }
        for value in ["1", "true", "on", " true "] {
            assert_eq!(parse_vision_setting(Some(value)), Some(true));
        }
    }
}
