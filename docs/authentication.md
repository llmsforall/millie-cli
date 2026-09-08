# Authentication

Millie does not need an account. The default setup runs the Millie model on
your own machine through the bundled llama.cpp server, and nothing is sent
anywhere.

If you want to point Millie at a different, OpenAI-compatible model server
instead, configure a `model_provider` in `~/.millie/config.toml`; see
[config.md](./config.md). Providers that require an API key read it from the
environment variable named in their configuration.
