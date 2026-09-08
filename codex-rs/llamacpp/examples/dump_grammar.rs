fn main() {
    let tools = vec![serde_json::json!({
        "name": "exec_command",
        "parameters": {"properties": {"cmd": {"type":"string"}, "workdir": {"type":"string"}}}
    })];
    print!("{}", codex_llamacpp::tool_call_grammar(&tools).unwrap());
}
