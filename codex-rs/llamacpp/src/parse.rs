//! Parses raw completion text produced by this model family back into
//! structured tool calls, translated directly from vLLM's
//! `qwen3coder_tool_parser.py` (the non-streaming `extract_tool_calls` /
//! `_get_function_calls` / `_parse_xml_function_call` path) -- the parser
//! actually proven working in production against this model, not a
//! reinvented one.

use serde_json::Value as JsonValue;

const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const FUNCTION_PREFIX: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_PREFIX: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    /// Present when the model emitted a call_id (the model's own JSON
    /// form round-trips these); None for the XML form.
    pub call_id: Option<String>,
    /// Raw string value per parameter, in the order they appeared. Type
    /// coercion against the tool's JSON schema (number/bool/array/etc.) is
    /// the caller's job -- matching the parser this was derived from, which
    /// coerces using the request's tool schema, not something parsing alone
    /// can know.
    pub arguments: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtractedToolCalls {
    pub tools_called: bool,
    pub tool_calls: Vec<ParsedToolCall>,
    /// Plain content preceding the first tool call (or the whole output, if
    /// no tool call was found).
    pub content: Option<String>,
}

/// Finds all `<tool_call>...</tool_call>` spans, tolerating an unclosed
/// trailing block (still-streaming-shaped output fed in whole), matching
/// `tool_call_regex`'s `<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$`.
fn find_tool_call_bodies(text: &str) -> Vec<&str> {
    let mut bodies = Vec::new();
    let mut rest = text;
    while let Some(start_rel) = rest.find(TOOL_CALL_START) {
        let after_start = &rest[start_rel + TOOL_CALL_START.len()..];
        match after_start.find(TOOL_CALL_END) {
            Some(end_rel) => {
                bodies.push(&after_start[..end_rel]);
                rest = &after_start[end_rel + TOOL_CALL_END.len()..];
            }
            None => {
                // Unclosed trailing tool_call: takes the rest of the text.
                bodies.push(after_start);
                break;
            }
        }
    }
    bodies
}

/// Within one `<tool_call>` body, finds the `<function=...>...</function>`
/// span, tolerating an unclosed trailing block. Matches
/// `tool_call_function_regex`.
fn find_function_body(tool_call_body: &str) -> Option<&str> {
    let start_rel = tool_call_body.find(FUNCTION_PREFIX)?;
    let after_start = &tool_call_body[start_rel + FUNCTION_PREFIX.len()..];
    Some(match after_start.find(FUNCTION_END) {
        Some(end_rel) => &after_start[..end_rel],
        None => after_start,
    })
}

/// Parses one `<function=NAME>...params...` body (already past the
/// `<function=` prefix and before `</function>`) into a name + ordered
/// parameter list. Matches `_parse_xml_function_call`.
/// Parses the JSON tool-call form `{"name": "...", "arguments": {...}}`.
/// String argument values are passed through raw (matching the XML path's
/// contract that type coercion happens against the tool schema later);
/// non-string values keep their JSON serialization.
fn parse_json_call(body: &str) -> Option<ParsedToolCall> {
    let trimmed = body.trim();
    if !trimmed.starts_with('{') {
        return None;
    }
    let v: serde_json::Value = serde_json::from_str(trimmed).ok()?;
    let name = v.get("name")?.as_str()?.to_string();
    if name.is_empty() {
        return None;
    }
    let call_id = v
        .get("call_id")
        .and_then(|c| c.as_str())
        .map(str::to_string);
    let mut arguments = Vec::new();
    let args_val = match v.get("arguments") {
        Some(serde_json::Value::String(inner)) => {
            serde_json::from_str::<serde_json::Value>(inner).ok()
        }
        Some(other) => Some(other.clone()),
        None => None,
    };
    if let Some(args) = args_val.as_ref().and_then(|a| a.as_object()) {
        for (k, val) in args {
            let s = match val {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            arguments.push((k.clone(), s));
        }
    }
    Some(ParsedToolCall {
        name,
        call_id,
        arguments,
    })
}

fn parse_function_call(function_body: &str) -> Option<ParsedToolCall> {
    let name_end = function_body.find('>')?;
    let name = function_body[..name_end].to_string();
    // A tool name never contains quotes, braces or whitespace. Seeing them
    // means the model mixed formats (observed: JSON-call fragments after
    // `<function=`, where the terminating '>' was a shell redirect inside the
    // arguments) -- minting a call from that garbage sends an unusable name
    // to the router and the model can retry it in a loop. Treat it as
    // not-a-call so the text surfaces to the model/user instead.
    if name.is_empty()
        || name
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '"' | '\'' | '{' | '}' | ':' | ','))
    {
        return None;
    }
    let params_text = &function_body[name_end + 1..];

    let mut arguments = Vec::new();
    let mut rest = params_text;
    while let Some(start_rel) = rest.find(PARAMETER_PREFIX) {
        let after_start = &rest[start_rel + PARAMETER_PREFIX.len()..];
        let Some(name_end) = after_start.find('>') else {
            break;
        };
        let param_name = after_start[..name_end].to_string();
        let value_region = &after_start[name_end + 1..];

        // Value ends at the first of: </parameter>, the next <parameter=,
        // or end of string. Matches tool_call_parameter_regex's lookahead
        // chain.
        let end_idx = [
            value_region.find(PARAMETER_END),
            value_region.find(PARAMETER_PREFIX),
        ]
        .into_iter()
        .flatten()
        .min()
        .unwrap_or(value_region.len());

        let mut value = value_region[..end_idx].to_string();
        if let Some(stripped) = value.strip_prefix('\n') {
            value = stripped.to_string();
        }
        if let Some(stripped) = value.strip_suffix('\n') {
            value = stripped.to_string();
        }
        arguments.push((param_name, value));

        let consumed = if value_region[end_idx..].starts_with(PARAMETER_END) {
            end_idx + PARAMETER_END.len()
        } else {
            end_idx
        };
        rest = &value_region[consumed..];
    }

    Some(ParsedToolCall {
        name,
        call_id: None,
        arguments,
    })
}

/// Extracts tool calls from one full (non-streaming) completion, matching
/// `Qwen3CoderToolParser.extract_tool_calls`.
pub fn extract_tool_calls(model_output: &str) -> ExtractedToolCalls {
    if !model_output.contains(FUNCTION_PREFIX) && !model_output.contains(TOOL_CALL_START) {
        return ExtractedToolCalls {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(model_output.to_string()),
        };
    }

    let tool_call_bodies = find_tool_call_bodies(model_output);
    // Back-off: no <tool_call> tags found even though <function= is
    // present, so treat the whole output as a single tool-call body.
    let raw_tool_calls: Vec<&str> = if tool_call_bodies.is_empty() {
        vec![model_output]
    } else {
        tool_call_bodies
    };

    let tool_calls: Vec<ParsedToolCall> = raw_tool_calls
        .iter()
        .filter_map(|body| match find_function_body(body) {
            Some(fb) => parse_function_call(fb),
            // Some sampling regimes emit the JSON call form inside the same
            // <tool_call> tags: {"name": ..., "arguments": {...}}. Accept it
            // rather than minting garbage from (or dropping) the call.
            None => parse_json_call(body),
        })
        .collect();

    if tool_calls.is_empty() {
        return ExtractedToolCalls {
            tools_called: false,
            tool_calls: Vec::new(),
            content: Some(model_output.to_string()),
        };
    }

    let tool_call_idx = model_output.find(TOOL_CALL_START);
    let function_idx = model_output.find(FUNCTION_PREFIX);
    let content_end = match (tool_call_idx, function_idx) {
        (Some(a), Some(b)) => a.min(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => model_output.len(),
    };
    let content = &model_output[..content_end];

    ExtractedToolCalls {
        tools_called: !tool_calls.is_empty(),
        tool_calls,
        content: if content.is_empty() {
            None
        } else {
            Some(content.to_string())
        },
    }
}

/// Coerces a raw string parameter value to the JSON type declared for it in
/// the tool's schema, matching `_convert_param_value`/`coerce_to_schema_type`.
/// Falls back to the raw string if the schema doesn't specify a type or the
/// value doesn't parse as that type -- never fails outright, since a
/// malformed coercion shouldn't drop an otherwise-valid tool call.
pub fn coerce_param_value(raw: &str, schema_type: Option<&str>) -> JsonValue {
    match schema_type {
        Some("integer") => {
            let t = raw.trim();
            // models frequently emit integers with a trailing .0 -- accept them
            t.parse::<i64>()
                .ok()
                .or_else(|| {
                    t.parse::<f64>()
                        .ok()
                        .filter(|f| f.fract() == 0.0)
                        .map(|f| f as i64)
                })
                .map(JsonValue::from)
                .unwrap_or_else(|| JsonValue::String(raw.to_string()))
        }
        Some("number") => {
            let t = raw.trim();
            // integral values are emitted as integers so u64-typed fields accept them
            match t.parse::<f64>() {
                Ok(f) if f.fract() == 0.0 && f.abs() < 9.0e15 => JsonValue::from(f as i64),
                Ok(f) => serde_json::Number::from_f64(f)
                    .map(JsonValue::Number)
                    .unwrap_or_else(|| JsonValue::String(raw.to_string())),
                Err(_) => JsonValue::String(raw.to_string()),
            }
        }
        Some("boolean") => match raw.trim() {
            "true" => JsonValue::Bool(true),
            "false" => JsonValue::Bool(false),
            _ => JsonValue::String(raw.to_string()),
        },
        Some("object") | Some("array") => {
            serde_json::from_str(raw).unwrap_or_else(|_| JsonValue::String(raw.to_string()))
        }
        _ => JsonValue::String(raw.to_string()),
    }
}

/// Builds the final JSON-object arguments blob (matching what
/// `ResponseItem::FunctionCall.arguments` expects: a JSON string) from a
/// parsed call plus a lookup of each parameter's declared schema type.
///
/// Serialized by hand, in the parameter order the model GENERATED them.
/// A `serde_json::Map` here would silently alphabetize the keys (BTreeMap),
/// and every later re-render of the call would then diverge byte-wise from
/// what the model produced -- a wire-format mismatch and a prompt-cache
/// divergence on every multi-parameter call. (A coerced object-typed VALUE
/// still normalizes its nested keys; top-level order is what the
/// `<parameter=>` serialization renders.)
pub fn arguments_to_json_string(
    call: &ParsedToolCall,
    param_type: impl Fn(&str) -> Option<String>,
) -> String {
    let mut out = String::from("{");
    for (i, (name, raw_value)) in call.arguments.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&serde_json::to_string(name).unwrap_or_else(|_| String::from("\"\"")));
        out.push(':');
        let ty = param_type(name);
        out.push_str(&coerce_param_value(raw_value, ty.as_deref()).to_string());
    }
    out.push('}');
    out
}

/// Top-level key/value pairs of a JSON object string, in DOCUMENT order.
/// serde streams map entries in encounter order; ordering is only lost when
/// they are stored into `serde_json::Map` (a BTreeMap). Used wherever a
/// stored `arguments` string is re-rendered as `<parameter=>` blocks, so the
/// render byte-matches the model's own generation order. `None` when the
/// string is not a JSON object.
pub fn ordered_object_pairs(json_str: &str) -> Option<Vec<(String, JsonValue)>> {
    struct PairsVisitor;
    impl<'de> serde::de::Visitor<'de> for PairsVisitor {
        type Value = Vec<(String, JsonValue)>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("a JSON object")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut access: A,
        ) -> Result<Self::Value, A::Error> {
            let mut out = Vec::new();
            while let Some((k, v)) = access.next_entry::<String, JsonValue>()? {
                out.push((k, v));
            }
            Ok(out)
        }
    }
    let mut de = serde_json::Deserializer::from_str(json_str);
    serde::de::Deserializer::deserialize_map(&mut de, PairsVisitor).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_function_prefix_is_plain_content() {
        let result = extract_tool_calls("just a normal answer, no tools here");
        assert!(!result.tools_called);
        assert_eq!(
            result.content.as_deref(),
            Some("just a normal answer, no tools here")
        );
    }

    #[test]
    fn parses_single_tool_call_with_content_before() {
        let output = "Let me check that.\n\n<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://example.com\n</parameter>\n</function>\n</tool_call>";
        let result = extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].name, "fetch_url");
        assert_eq!(
            result.tool_calls[0].arguments,
            vec![("url".to_string(), "https://example.com".to_string())]
        );
        assert_eq!(result.content.as_deref(), Some("Let me check that.\n\n"));
    }

    #[test]
    fn parses_multiple_parallel_tool_calls() {
        let output = "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>";
        let result = extract_tool_calls(output);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].name, "a");
        assert_eq!(result.tool_calls[1].name, "b");
    }

    #[test]
    fn multiline_parameter_value_is_preserved() {
        let output = "<tool_call>\n<function=write>\n<parameter=content>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
        let result = extract_tool_calls(output);
        assert_eq!(
            result.tool_calls[0].arguments,
            vec![("content".to_string(), "line one\nline two".to_string())]
        );
    }

    #[test]
    fn unclosed_trailing_tool_call_still_parses() {
        // Matches the streaming-shaped case: the tag never got a closing
        // </tool_call> because that's literally where generation stopped.
        let output = "<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://x.test\n</parameter>\n</function>\n</tool_call";
        let result = extract_tool_calls(output);
        assert!(result.tools_called);
        assert_eq!(result.tool_calls[0].name, "fetch_url");
    }

    #[test]
    fn coerces_types_per_schema() {
        assert_eq!(
            coerce_param_value("42", Some("integer")),
            JsonValue::from(42)
        );
        assert_eq!(
            coerce_param_value("true", Some("boolean")),
            JsonValue::Bool(true)
        );
        assert_eq!(
            coerce_param_value("not a bool", Some("boolean")),
            JsonValue::String("not a bool".to_string())
        );
        assert_eq!(
            coerce_param_value("hello", None),
            JsonValue::String("hello".to_string())
        );
    }
}

#[cfg(test)]
mod format_robustness_tests {
    use super::*;

    #[test]
    fn json_form_tool_call_parses() {
        let out = "I'll write the file.\n<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"cmd\": \"printf '14' > out.txt\"}}\n</tool_call>";
        let ex = extract_tool_calls(out);
        assert!(ex.tools_called);
        assert_eq!(ex.tool_calls.len(), 1);
        assert_eq!(ex.tool_calls[0].name, "shell");
        assert_eq!(ex.tool_calls[0].arguments[0].1, "printf '14' > out.txt");
    }

    #[test]
    fn hybrid_garbage_does_not_mint_a_call() {
        // Observed in the wild: JSON fragments after `<function=`, where the
        // first '>' is a shell redirect inside the arguments. Must not
        // produce a tool call named `exec_command","arguments":{"cmd":...`.
        let out = "<tool_call><function=exec_command\",\"arguments\":{\"cmd\":\"printf '14' > out.txt\"}}</function></tool_call>";
        let ex = extract_tool_calls(out);
        assert!(ex.tool_calls.iter().all(|c| !c.name.contains('"')));
    }

    #[test]
    fn xml_form_still_parses() {
        let out = "<tool_call>\n<function=shell>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>";
        let ex = extract_tool_calls(out);
        assert!(ex.tools_called);
        assert_eq!(ex.tool_calls[0].name, "shell");
    }
}

#[cfg(test)]
mod argument_order_tests {
    use super::*;

    #[test]
    fn arguments_string_preserves_generation_order() {
        // Keys deliberately in reverse-alphabetical order: a sorted map
        // would flip them and every re-render would diverge from the
        // model's own bytes.
        let call = ParsedToolCall {
            name: "create_file".to_string(),
            arguments: vec![
                ("path".to_string(), "a.txt".to_string()),
                ("content".to_string(), "hello".to_string()),
            ],
            call_id: None,
        };
        let s = arguments_to_json_string(&call, |_| Some("string".to_string()));
        assert_eq!(s, "{\"path\":\"a.txt\",\"content\":\"hello\"}");
    }

    #[test]
    fn ordered_pairs_preserve_document_order() {
        let pairs = ordered_object_pairs("{\"zeta\":\"1\",\"alpha\":\"2\"}").unwrap();
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, vec!["zeta", "alpha"]);
    }

    #[test]
    fn ordered_pairs_none_on_non_object() {
        assert!(ordered_object_pairs("[1,2]").is_none());
        assert!(ordered_object_pairs("not json").is_none());
    }
}
