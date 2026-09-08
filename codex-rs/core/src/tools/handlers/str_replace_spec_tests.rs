use super::*;
use pretty_assertions::assert_eq;
use serde_json::json;

#[test]
fn str_replace_tool_schemas_preserve_names_and_parameters() {
    let create_file = serde_json::to_value(create_create_file_tool()).expect("serialize");
    assert_eq!(create_file["name"], "create_file");
    assert_eq!(
        create_file["description"],
        "Create a new file with the given content. Fails if the file already exists."
    );
    assert_eq!(
        create_file["parameters"],
        json!({
            "type": "object",
            "properties": {
                "content": { "type": "string" },
                "path": { "type": "string" }
            },
            "required": ["path", "content"]
        })
    );

    let str_replace = serde_json::to_value(create_str_replace_tool()).expect("serialize");
    assert_eq!(str_replace["name"], "str_replace");
    assert_eq!(
        str_replace["description"],
        "Replace one exact occurrence of old_str in the file with new_str. old_str must match character-for-character and appear exactly once."
    );
    assert_eq!(
        str_replace["parameters"],
        json!({
            "type": "object",
            "properties": {
                "new_str": { "type": "string" },
                "old_str": { "type": "string" },
                "path": { "type": "string" }
            },
            "required": ["path", "old_str", "new_str"]
        })
    );

    let delete_file = serde_json::to_value(create_delete_file_tool()).expect("serialize");
    assert_eq!(delete_file["name"], "delete_file");
    assert_eq!(delete_file["description"], "Delete the file at path.");
    assert_eq!(
        delete_file["parameters"],
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" }
            },
            "required": ["path"]
        })
    );
}
