use codex_tools::JsonSchema;
use codex_tools::ResponsesApiTool;
use codex_tools::ToolSpec;
use std::collections::BTreeMap;

pub(crate) const CREATE_FILE_TOOL_NAME: &str = "create_file";
pub(crate) const STR_REPLACE_TOOL_NAME: &str = "str_replace";
pub(crate) const DELETE_FILE_TOOL_NAME: &str = "delete_file";

pub(crate) fn create_create_file_tool() -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: CREATE_FILE_TOOL_NAME.to_string(),
        description: "Create a new file with the given content. Fails if the file already exists."
            .to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            BTreeMap::from([
                ("path".to_string(), JsonSchema::string(None)),
                ("content".to_string(), JsonSchema::string(None)),
            ]),
            Some(vec!["path".to_string(), "content".to_string()]),
            None,
        ),
        output_schema: None,
    })
}

pub(crate) fn create_str_replace_tool() -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: STR_REPLACE_TOOL_NAME.to_string(),
        description: "Replace one exact occurrence of old_str in the file with new_str. old_str must match character-for-character and appear exactly once."
            .to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            BTreeMap::from([
                ("path".to_string(), JsonSchema::string(None)),
                ("old_str".to_string(), JsonSchema::string(None)),
                ("new_str".to_string(), JsonSchema::string(None)),
            ]),
            Some(vec![
                "path".to_string(),
                "old_str".to_string(),
                "new_str".to_string(),
            ]),
            None,
        ),
        output_schema: None,
    })
}

pub(crate) fn create_delete_file_tool() -> ToolSpec {
    ToolSpec::Function(ResponsesApiTool {
        name: DELETE_FILE_TOOL_NAME.to_string(),
        description: "Delete the file at path.".to_string(),
        strict: false,
        defer_loading: None,
        parameters: JsonSchema::object(
            BTreeMap::from([("path".to_string(), JsonSchema::string(None))]),
            Some(vec!["path".to_string()]),
            None,
        ),
        output_schema: None,
    })
}

#[cfg(test)]
#[path = "str_replace_spec_tests.rs"]
mod tests;
