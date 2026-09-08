# Getting started with Millie

Follow the complete [Mac or Linux install-and-start steps](../README.md#install-and-start)
first. They cover installation, persistent PATH setup, verification and choosing
your workspace folder.

Once installed, start each new session with these commands. Replace the example
folder with yours and keep the quotes if its path contains spaces:

```sh
cd "/path/to/your/project"
```

```sh
millie
```

If `millie` is not found, use the [copyable PATH recovery commands](../README.md#if-you-see-millie-command-not-found).

On an initial interactive launch, the chooser recommends a model for your
machine. Select one, then approve downloading any missing files. Later launches
reuse the installed files offline. The chooser estimates fit from machine
capacity; close memory-heavy applications if needed. See [memory help](memory.md).

Describe a task, such as “add a unit test for the parser and make it pass.”
Millie reads and edits files and runs commands subject to your configured
[sandbox and approval policy](sandbox.md). Type `/` for available commands;
`/status` shows session information and `/compact` shortens conversation history.
Compacting history does not shrink the server's allocated context buffer.

## Select or change a model

```sh
millie --model select
millie --model millie-35B-A3B-9GB
```

Both routes remember your last explicit choice. The chooser is available even
when model files are already downloaded. A failed startup keeps your choice;
free memory and retry. Millie does not silently substitute a smaller model.
With `--profile work`, the choice is saved to that active user configuration.

Switch models at startup: close existing sessions using the shared server,
then launch the other model. There is no in-session model switch. Different
models or opposite vision modes cannot share the same running server.

## Resume or run a single task

```sh
millie resume
millie resume --last
millie resume SESSION_ID
millie exec --model millie-35B-A3B-9GB --download "explain this project's tests"
```

Replace `SESSION_ID` with the ID printed on exit. `--download` approves missing
model files; omit it once the files are cached. In noninteractive use, a saved
model choice alone does not approve downloads. See [exec.md](exec.md).

## Add image support later

Close sessions using the server, then run:

```sh
millie --vision --download
```

This adds the matching vision tower and enables images for that launch. To
keep images enabled, set `vision = true` under `[llamacpp]` in
`~/.millie/config.toml`. Images require additional memory. If an existing model has no recorded remote revision and the tower is missing,
provide a matching local tower or explicitly use `millie models update` to install
a matched set. Millie will not silently replace the existing model.

## Check for model updates

```sh
millie models check-updates
millie models update
```

Updates are explicit and do not require reinstalling Millie. Close the
server-owning session before launching the updated model. See [model updates](config.md#model-updates).

## Multiple windows and server lifetime

The first session that starts the managed model server owns it. Other sessions
can attach to that same model and vision mode. Closing a borrowing session does
not stop the server; closing the owner does, even if another window remains.
If a window loses its server, restart it with `millie resume SESSION_ID` to
attempt loading the server again. Clearing a conversation does not release or
switch its server connection.

`millie --keep-model-server` explicitly leaves a newly started server running
after exit. Millie does not automatically terminate independently started
servers. See [changing server settings](memory.md#apply-changes-to-a-new-server).
