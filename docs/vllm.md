# Connecting to an external vLLM server

Millie can connect to an independently managed, compatible vLLM server.
The server must already support the selected model and its tool-call format.
This release does not include a vLLM model-serving implementation or installer.

```toml
model_provider = "vllm"

[vllm]
base_url = "http://127.0.0.1:8300/v1"
served_model = "millie"
```

Set `served_model` to the model name exposed by your server. Millie does not
start or stop this server; it checks that the endpoint responds before startup.
Generation settings under `[llamacpp]` also apply to this backend; see
[configuration](./config.md#vllm-backend).
