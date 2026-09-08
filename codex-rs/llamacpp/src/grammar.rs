//! Builds a lazy GBNF grammar that constrains tool-call syntax to exactly the
//! XML form `parse.rs` accepts. The grammar is dormant during free text and
//! reasoning; it activates at the `<tool_call>` trigger and from that point
//! only well-formed calls against the declared tools are representable. This
//! mirrors the structural-tag enforcement vLLM applies via its tool parser --
//! without it, occasional malformed calls surface as parse failures instead of
//! being unrepresentable.

use serde_json::Value;

/// Sanitizes a tool/parameter name into a GBNF rule-name fragment.
fn rule_frag(name: &str) -> String {
    name.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

/// Escapes a literal for a GBNF double-quoted string.
fn lit(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Emits rules matching any text that does not contain `forbidden` as a
/// substring (the standard prefix-automaton expansion; GBNF has no negative
/// lookahead). Returns the entry rule name.
fn not_containing_rules(out: &mut String, base: &str, forbidden: &str) -> String {
    let chars: Vec<char> = forbidden.chars().collect();
    let n = chars.len();
    // state i = the last i chars matched a prefix of `forbidden`.
    // From state i on char c: extend to i+1 if c == chars[i]; otherwise fall
    // back to the longest prefix of `forbidden` that is a suffix of
    // prefix[0..i] + c (KMP failure), which for the terminator strings used
    // here ("</parameter" etc., no repeated prefixes beyond '<') is either
    // 1 (if c == chars[0]) or 0.
    let entry = format!("{base}-v");
    for i in 0..n {
        let name = if i == 0 {
            entry.clone()
        } else {
            format!("{base}-v{i}")
        };
        let next = |j: usize| {
            if j == 0 {
                entry.clone()
            } else {
                format!("{base}-v{j}")
            }
        };
        let mut alts: Vec<String> = Vec::new();
        // chars that keep us at/return us to a shallow state
        let c_i = chars[i];
        let c_0 = chars[0];
        if i + 1 < n {
            alts.push(format!("{} {}", lit(&c_i.to_string()), next(i + 1)));
        }
        // at i+1 == n we simply do not offer the extension: the full
        // forbidden string is unrepresentable from this rule.
        if c_i != c_0 {
            alts.push(format!("{} {}", lit(&c_0.to_string()), next(1)));
        }
        // any other char resets to state 0
        let mut class = String::from("[^");
        for c in [c_i, c_0] {
            match c {
                '\n' => class.push_str("\\n"),
                ']' => class.push_str("\\]"),
                '\\' => class.push_str("\\\\"),
                _ => class.push(c),
            }
        }
        class.push(']');
        alts.push(format!("{class} {entry}"));
        // every state is accepting (value may end anywhere)
        alts.push("\"\"".to_string());
        out.push_str(&format!("{name} ::= {}\n", alts.join(" | ")));
    }
    entry
}

/// Builds the full lazy grammar from flattened tool JSON (objects with
/// "name" and optional "parameters".properties). Returns None when there are
/// no usable tools (no grammar should be attached).

pub fn tool_call_grammar(tools: &[Value]) -> Option<String> {
    let mut names: Vec<(String, Vec<String>)> = Vec::new();
    for t in tools {
        let Some(name) = t.get("name").and_then(|v| v.as_str()) else {
            continue;
        };
        let params: Vec<String> = t
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object())
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        names.push((name.to_string(), params));
    }
    if names.is_empty() {
        return None;
    }
    let mut g = String::new();
    // One or more tool calls, then the turn may end. The trailing newline the
    // template uses between/after calls is optional so EOG stays reachable
    // right after a closing tag.
    g.push_str("root ::= tool-call (\"\\n\" tool-call)* \"\\n\"?\n");
    let func_alts: Vec<String> = names
        .iter()
        .map(|(n, _)| format!("func-{}", rule_frag(n)))
        .collect();
    g.push_str(&format!(
        "tool-call ::= \"<tool_call>\\n\" ({}) \"\\n</tool_call>\"\n",
        func_alts.join(" | ")
    ));

    let value_entry = not_containing_rules(&mut g, "val", "</parameter");

    for (name, params) in &names {
        let frag = rule_frag(name);
        if params.is_empty() {
            g.push_str(&format!(
                "func-{frag} ::= \"<function={name}>\" \"\\n\"? \"</function>\"\n"
            ));
            continue;
        }
        let param_alts: Vec<String> = params
            .iter()
            .map(|p| format!("param-{frag}-{}", rule_frag(p)))
            .collect();
        g.push_str(&format!(
            "func-{frag} ::= \"<function={name}>\\n\" ({})* \"</function>\"\n",
            param_alts.join(" | ")
        ));
        for p in params {
            g.push_str(&format!(
                "param-{frag}-{} ::= \"<parameter={p}>\\n\" {value_entry} \"\\n</parameter>\\n\"\n",
                rule_frag(p)
            ));
        }
    }
    Some(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn shell_tools() -> Vec<Value> {
        vec![json!({
            "name": "shell",
            "parameters": {"properties": {"cmd": {"type": "string"}, "timeout": {"type": "number"}}}
        })]
    }

    #[test]
    fn builds_rules_for_each_tool_and_param() {
        let g = tool_call_grammar(&shell_tools()).unwrap();
        assert!(g.contains("root ::="));
        assert!(g.contains("func-shell ::= \"<function=shell>"));
        assert!(g.contains("param-shell-cmd ::= \"<parameter=cmd>"));
        assert!(g.contains("param-shell-timeout"));
    }

    #[test]
    fn no_tools_no_grammar() {
        assert!(tool_call_grammar(&[]).is_none());
    }
}
