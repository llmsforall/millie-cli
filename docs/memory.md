# Make Millie fit in memory

Start with the chooser: `millie --model select`. On a 16 GB Apple Silicon Mac,
9GB is recommended; 11GB is a tight fit. Leave memory for the operating system,
your editor, browsers and builds. A profile uses machine capacity to estimate
fit; it does not reserve memory or guarantee that startup will succeed under
current load. Download size is not runtime memory usage.

## If startup runs out of memory

First close memory-heavy applications and any old model server you started.
Then try these changes one at a time, starting a new server each time:

1. **Choose a smaller model.** `millie --model select` lets you compare choices.
   This reduces weight memory but changes the model you use. The choice is
   remembered; failed startup never automatically selects a different model.
2. **Reduce context.** `millie --model-context-window 16384` requests a smaller
   context buffer. This leaves less room for conversation and tool results,
   so compaction may happen sooner. The profile can cap it further. If you
   previously set `[llamacpp] ctx_size`, remove that advanced override first.
3. **Turn off image input.** `millie --no-vision` avoids loading the vision
   tower. Text/code tools still work; image input is unavailable. Existing
   downloaded files stay on disk. You can add images later with
   `millie --vision --download` after closing existing sessions.
4. **Review device placement and profiles** below if you still need more room.

A combined example for a smaller context without image input:

```sh
millie --model millie-35B-A3B-9GB --model-context-window 16384 --no-vision
```

To retain those settings, merge these entries into `~/.millie/config.toml`
without creating a second `[llamacpp]` table:

```toml
model = "millie-35B-A3B-9GB"
model_context_window = 16384

[llamacpp]
vision = false
```

These are memory-saving examples, not settings everyone needs. The ordinary
launch uses the catalog's model-specific defaults and automatic serving profile.
Changing sampling or the system prompt is not a memory-tuning step.

## Mac unified memory

Apple Silicon shares one memory pool between the CPU and GPU. Moving work to
the CPU does not create another pool of RAM. The 9GB `metal-16gb` profile uses
65,536-token context and no vision by default; the 11GB variant also uses memory mapping
and CPU expert placement to make its tighter fit possible.

Prefer a smaller model or shorter context before changing placement. Advanced
users can try memory mapping and keeping more expert layers on the CPU:

```sh
millie -c llamacpp.mmap=true -c llamacpp.n_cpu_moe=16
```

This can reduce Metal's wired-memory pressure and allow mapped weights to be
paged. It does not guarantee lower total memory use or a comfortable fit.
Paging and CPU work can slow responses, especially under pressure. Remove the
overrides to return to the profile defaults.

## Linux: system RAM and GPU VRAM

Dedicated GPU memory and system RAM are separate limits. Choose the actual
GPU with `millie --llama-gpu 1` (replace `1` with your device index). Automatic
profile selection uses the selected device memory and system RAM; it does not
assume another, larger GPU is available to that launch.

For the 9GB and 11GB models, hybrid profiles put expert weights in system RAM:

```sh
millie --serving-profile hybrid
millie --serving-profile deep-hybrid
```

`hybrid` uses more CPU/RAM to reduce GPU weight memory; `deep-hybrid` also
uses a shorter context and places all experts on CPU. Both disable vision.
These profiles still need enough system RAM and can be slower. Profile names
are model-specific: the 7GB model does not have these hybrid profiles. Pinning
a profile is an explicit override, not proof that it fits your hardware.

For CPU-only use:

```sh
millie -c 'llamacpp.gpu=[]'
```

CPU-only serving needs sufficient system RAM and is generally slower. On CPU,
`n_cpu_moe` does not free a separate GPU memory pool.

## Advanced controls

| Setting under `[llamacpp]` | Effect and tradeoff |
| --- | --- |
| `profile` | Pins a model-specific serving profile; normally leave automatic |
| `kv_cache` | Profiles use `q8_0`; `f16` uses more cache memory. `q4_0` is accepted but not a validated general recommendation |
| `mmap` | Allows file-backed weight mapping; paging under pressure can be slow |
| `n_cpu_moe` | Places expert layers on CPU; shifts work and may relieve GPU pressure |
| `ctx_size` | Overrides total server context directly; prefer `model_context_window` for ordinary use |
| `parallel` | Server request slots; total context is divided among slots. More slots do not create more context or memory |

`--serving-profile` selects a model's memory/placement defaults. `--profile`
selects a named user configuration; they are different options. Explicit
settings in your config can override individual serving-profile defaults, so
check old overrides before assuming a new profile changed everything.

## Apply changes to a new server

Settings take effect when the server starts. They do not resize an already
running shared server. Close all sessions sharing it, including its owner,
then launch with the new settings. Conversation clearing and `/compact` do
not release the loaded model or shrink the allocated context buffer.

If you opted into `--keep-model-server`, or started llama-server yourself,
stop that server yourself before applying new launch settings. Millie leaves
independently launched servers alone. Other windows that lose their server
can be reopened with `millie resume SESSION_ID`.

At startup, inspect the reported profile and resolved settings. If it still
fails, retain the startup error and the model/device/settings you used;
`millie doctor` can help diagnose installation and runtime problems. Avoid
repeatedly forcing a larger configuration while the machine is under pressure.
