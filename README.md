# Millie CLI

Millie is a coding agent that runs entirely on your own computer. It drives a
local language model, reads and edits your code, runs commands, and iterates
until the job is done. There is no account to create and no API key to paste:
the model is the open-weight [Millie](https://huggingface.co/llmsforall) model,
served by a bundled copy of [llama.cpp](https://github.com/llmsforall/llama.cpp).
Nothing leaves your machine unless you configure it to.

## Install and start

Choose your platform: **[Mac](#macos-apple-silicon)** · **[Linux](#linux-x86_64)**.
Run the commands one at a time. They are for **Bash or Zsh**; these are the
usual shells on Linux and macOS respectively. No administrator password is needed
for the default installation.

### macOS (Apple Silicon)

Requires an Apple Silicon Mac (M1 or newer) with macOS 12 or newer. See the
[model and memory guidance](#models) before downloading model weights.

**1. Install Millie.** This downloads the application and its matching model
runtime; you do not need to clone the source or compile anything.

```sh
curl -fsSL https://raw.githubusercontent.com/llmsforall/millie-cli/main/scripts/install/install.sh | sh
```

**2. Activate it in this terminal.** Copy this line exactly:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

The installer saves PATH for future Bash/Zsh terminals, including after reboot.
The line above activates it in the terminal that was already open.

**3. Check the installation.**

```sh
millie --version
```

You should see a version such as `millie 0.1.0`.

**4. Go to the folder you want Millie to work on.** Replace the example path
with your folder's path. Keep the quotes, especially if the path has spaces.

```sh
cd "$HOME/Documents/My Project"
```

If `cd` reports “No such file or directory,” correct the folder path before
continuing. To check which folder you are in, run:

```sh
pwd
```

**5. Start Millie.**

```sh
millie
```

On Mac, you can also type `cd ` (including the space), drag your project folder
from Finder into Terminal, and press Return. Then run `millie`.

### Linux (x86_64)

Requires Linux x86_64 with glibc 2.31 or newer (for example, Ubuntu 20.04+).
GPU acceleration requires a working Vulkan driver; CPU-only operation is also
available. See the [memory guide](docs/memory.md) for model selection.

**1. Install Millie.** This downloads the application and its matching model
runtime; you do not need to clone the source or compile anything.

```sh
curl -fsSL https://raw.githubusercontent.com/llmsforall/millie-cli/main/scripts/install/install.sh | sh
```

**2. Activate it in this terminal.** Copy this line exactly:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

The installer saves PATH for future Bash/Zsh terminals, including after reboot.
The line above activates it in the terminal that was already open.

**3. Check the installation.**

```sh
millie --version
```

You should see a version such as `millie 0.1.0`.

**4. Go to the folder you want Millie to work on.** Replace the example path
with your folder's path. Keep the quotes, especially if the path has spaces.

```sh
cd "$HOME/projects/my-project"
```

If `cd` reports “No such file or directory,” correct the folder path before
continuing. To check which folder you are in, run:

```sh
pwd
```

**5. Start Millie.**

```sh
millie
```

### First launch

Millie shows a model chooser and recommends an option for your machine. Select
one and approve its download. **Model files are several GB**, separate from the
smaller application download; Millie shows the size before asking. Allow time
for the download and model loading. Later launches reuse the installed files.

When the input prompt is ready, try: **“Explain what this project does. Don't
change any files yet.”** Millie uses the folder you launched it from as its workspace.

For later sessions, open a terminal, use `cd` to enter your project folder,
and run `millie` again. You do not need to reinstall or redo PATH setup.
Use `millie resume` instead to return to a saved conversation.

### If you see “millie: command not found”

For the default install, run these two commands one at a time:

```sh
export PATH="$HOME/.local/bin:$PATH"
```

```sh
millie --version
```

If that still fails, check the installed command directly:

```sh
"$HOME/.local/bin/millie" --version
```

If that file is missing, rerun the install command above and check its output
for an error. If the direct command works but new terminals cannot find `millie`,
rerun the installer to repair PATH setup; it prints which shell files it updates
and any file it could not update. For a custom install directory, use the exact
activation command printed by the installer instead of the default above.

The installer adds a guarded PATH entry to `.zshrc` (under `ZDOTDIR` if set), or
to Bash's `.bashrc` and its active login profile. It preserves existing contents
and does not duplicate its entry when rerun. Other shells need their own PATH
setup; the installer reports when it cannot configure one automatically.

Prebuilt archives are also available on the
[Releases page](https://github.com/llmsforall/millie-cli/releases).
For more usage examples see [Getting started](docs/getting-started.md).
Developers can use [source-build instructions](docs/install.md#build-from-source)
or the [release build guide](docs/release-builds.md).

## Models

| Model                                | Approximate download including vision | Choosing a model                                   |
| ------------------------------------ | ------------------------------------- | -------------------------------------------------- |
| Millie 35B-A3B 7GB                   | 6.9 GB                                | Smallest download; useful when memory is limited   |
| Millie 35B-A3B 9GB (ternary experts) | 8.9 GB                                | Recommended for Apple Silicon Macs with 16 GB RAM  |
| Millie 35B-A3B 11GB                  | 10.9 GB                               | Needs more headroom; fits but tight on a 16 GB Mac |

Download size is not the amount of memory needed to run a model. Context,
image support, device selection and other applications all affect memory use.
Text-only profiles skip the vision tower download; it can be added later.
See [Make Millie fit in memory](./docs/memory.md) for practical steps.

Run `millie --model select` to open the chooser, even if models are already
cached. Your last explicit choice is remembered, including after a failed
startup. Ordinary launches use that choice. The initial recommendation does
not replace it. To choose directly, use `millie --model millie-35B-A3B-9GB`.
Model selection and permission to download are separate.

## Using Millie

- `millie` starts an interactive session in the current directory.
- `millie exec "fix the failing test"` runs one task without the UI, for
  scripts and CI.
- `millie resume` picks up a previous conversation.
- Inside a session, type `/` to see the available commands.

Millie asks before running commands or editing files outside the workspace;
see [docs/sandbox.md](./docs/sandbox.md) for the approval and sandbox model and
[docs/config.md](./docs/config.md) for every setting.

## Docs

- [Getting started](./docs/getting-started.md)
- [Make Millie fit in memory](./docs/memory.md)
- [Configuration](./docs/config.md) and [example config](./docs/example-config.md)
- [Non-interactive mode (`millie exec`)](./docs/exec.md)
- [Sandbox and approvals](./docs/sandbox.md)
- [Skills](./docs/skills.md) and [slash commands](./docs/slash_commands.md)
- [Installing and building from source](./docs/install.md)
- [Contributing](./docs/contributing.md)

## Credits and license

Millie CLI is a fork of [OpenAI Codex CLI](https://github.com/openai/codex),
licensed under the [Apache-2.0 License](LICENSE); see [NOTICE](NOTICE). The
runtime is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT)
that adds the quantization formats the Millie models use.
