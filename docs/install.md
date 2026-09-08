## Installing & building

### Install a release (recommended)

Use the [complete Mac/Linux walkthrough in the README](../README.md#install-and-start).
It gives every command for downloading, persistent PATH setup, checking the
installation, entering your project folder and starting Millie.

The installer downloads the platform bundle, verifies its checksum and installs
it under `~/.millie/app/`. It links the command into `~/.local/bin` and adds that
directory to future Bash/Zsh sessions through the shell's startup files. It
cannot change the parent terminal's environment, so the walkthrough includes an
explicit `export PATH` command for the terminal already open.

Models are separate downloads. First launch asks you to choose a model and
approve the missing files. Later launches reuse them. Windows source support and
a PowerShell installer are present, but no Windows prebuilt bundle is provided
for this release.

### System requirements

| Requirement                 | Details                                                                                                                                     |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Operating systems           | macOS 12+ (Apple silicon), Linux x86_64 (glibc 2.31+, e.g. Ubuntu 20.04+), Windows 10+ x86_64                                               |
| GPU                         | optional: any Vulkan-capable GPU (Linux/Windows) or Apple silicon (Metal); CPU-only works                                                   |
| Memory                      | Start with the model chooser; 16 GB Apple Silicon: 9GB recommended, 11GB tight. Other devices depend on model/profile and RAM/VRAM headroom |
| Git (optional, recommended) | 2.23+ for built-in PR helpers                                                                                                               |

See [getting started](getting-started.md) for model selection and downloads,
and [memory help](memory.md) for context/device settings. Available release
archives include a README with links to these guides.

### Build from source

Millie needs a `llama-server` binary from the Millie fork of llama.cpp
(https://github.com/llmsforall/llama.cpp) to serve the model; release
packages bundle one. For a source build, build that fork and point
`[llamacpp] server_bin` in `~/.millie/config.toml` at the resulting
`llama-server`. Use the [Millie runtime build instructions](https://github.com/llmsforall/llama.cpp/blob/main/docs/build-millie.md).
The default chat transport also requires that checkout's `millie-native.jinja`
beside the CLI binary, or at `MILLIE_LLAMACPP_CHAT_TEMPLATE`.

The system prompt lives in `prompts/system_prompt.md` and the model catalog in `codex-rs/models-manager/millie-models.json` (both ship next to the binary in a bundle); the prompt is read at
runtime. Debug builds can find both files in the source checkout. Release
builds require them beside the executable, under `MILLIE_HOME`, or at the
explicit `MILLIE_SYSTEM_PROMPT` and `MILLIE_MODELS` paths
(see [config.md](./config.md)).
Set `MILLIE_VERSION=<version>` in the environment when
building a release so the binary reports a clean version instead of
`<version>-dev+g<commit>`.

```bash
# Clone the repository and navigate to the root of the Cargo workspace.
git clone https://github.com/llmsforall/millie-cli.git
cd millie-cli/codex-rs

# Install the Rust toolchain, if necessary.
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup component add rustfmt
rustup component add clippy
# Install helper tools used by the workspace justfile:
cargo install --locked just
# Install nextest for the `just test` helper.
cargo install --locked cargo-nextest

# Build Millie.
cargo build

# Launch the TUI with a sample prompt.
cargo run --bin millie -- "explain this codebase to me"

# After making changes, use the root justfile helpers (they default to codex-rs):
just fmt
just fix -p <crate-you-touched>

# Run the relevant tests (project-specific is fastest), for example:
just test -p codex-tui
# `just test` runs the test suite via nextest:
just test
# Avoid `--all-features` for routine local runs because it increases build
# time and `target/` disk usage by compiling additional feature combinations.
```

### Assembling a release bundle

The two repositories together contain the CLI and runtime source. A platform
bundle contains both executables, the model catalog, system prompt, native chat
template and license files. See [release-builds.md](release-builds.md) for the
Mac build script, runtime-file layout and portable Linux build inputs.

### Building llama-server with CUDA on Linux

From a checkout of https://github.com/llmsforall/llama.cpp, with a CUDA
toolkit at `$CUDA_HOME` (13.x tested):

```bash
L="-L$CUDA_HOME/lib64 -Wl,-rpath-link,$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64"
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc -DCUDAToolkit_ROOT=$CUDA_HOME \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK \
  "-DCMAKE_EXE_LINKER_FLAGS=$L" "-DCMAKE_SHARED_LINKER_FLAGS=$L"
cmake --build build-cuda --target llama-server -j
```

Set `CMAKE_CUDA_ARCHITECTURES` to the compute capabilities you ship for (120
is Blackwell; use a list like `"80;86;89;90;120"` for a general package).
`-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK` is needed with recent host
compilers; the linker flags are needed when the toolkit is not in a system
library path. Build with `-DGGML_METAL=ON` on macOS instead.

## Tracing / verbose logging

Millie is written in Rust, so it honors the `RUST_LOG` environment variable to configure its logging behavior.

The TUI records diagnostics in bounded local stores by default. Set `log_dir` explicitly to enable a plaintext TUI log for a run:

```bash
millie -c log_dir=./.millie-log
tail -F ./.millie-log/millie-tui.log
```

The non-interactive mode (`millie exec`) defaults to `RUST_LOG=error`, but messages are printed inline, so there is no need to monitor a separate file.

See the Rust documentation on [`RUST_LOG`](https://docs.rs/env_logger/latest/env_logger/#enabling-logging) for more information on the configuration options.
