# Millie

Keep the bundle's bin directory and its supporting files together. If installed
through the installer, run millie from a terminal in your project. After manual
extraction, you can run ./bin/millie from the extracted bundle directory, or add
its bin directory to PATH and run millie from your project.

Start: millie
Choose a model: millie --model select
Resume: millie resume
Memory help example: millie --model-context-window 16384 --no-vision
Check model updates: millie models check-updates
Install model updates: millie models update
Command reference: millie --help

Your last explicit model choice is remembered. Missing model files require
approval; add --download for noninteractive use. On a 16 GB Apple Silicon Mac,
9GB is recommended; 11GB fits but is tight. Memory estimates need headroom for
other applications. Close sessions using the old server before changing model,
vision or memory settings. The session that starts the managed server owns it;
exiting that session stops it unless --keep-model-server was requested.

Guides:
- Getting started: https://github.com/llmsforall/millie-cli/blob/main/docs/getting-started.md
- Memory troubleshooting: https://github.com/llmsforall/millie-cli/blob/main/docs/memory.md
- Configuration: https://github.com/llmsforall/millie-cli/blob/main/docs/config.md
- Model updates: https://github.com/llmsforall/millie-cli/blob/main/docs/config.md#model-updates

The links above follow the main branch. For documentation matching a specific
release, select its tag in the source repository and open the docs directory.
