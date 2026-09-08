# Building release bundles

Millie releases combine two repositories:

| Repository              | Provides                                                        |
| ----------------------- | --------------------------------------------------------------- |
| `llmsforall/millie-cli` | CLI, system prompt, model catalog, installer and bundle tooling |
| `llmsforall/llama.cpp`  | Matching model runtime and native chat template                 |

Keep the checkouts side by side. For a release, select the corresponding tag in
both repositories and record both commit IDs in the release notes. A source
archive from one repository alone does not include the other repository.
Models are separate downloads from the locations in the catalog.

## macOS / Apple Silicon

Install Xcode Command Line Tools, CMake, Python 3, Perl, make and Rust (the
workspace pins its toolchain in `codex-rs/rust-toolchain.toml`). The dependency
helper uses curl to fetch a checksummed OpenSSL source crate. Cargo may download
its locked dependencies on the first build.

From the `millie-cli` checkout:

```sh
JOBS=8 VERSION=0.1.0 bash scripts/release/build-macos.sh
```

The script builds the sibling `../llama.cpp` checkout with Metal and embedded
shaders, builds static OpenSSL and the Rust CLI, runs CPU/Metal correctness and
local startup/install/update checks, strips debug records, checks for embedded
build paths, and assembles an unsigned archive. Mock model tests use local HTTP
servers and placeholder files; they do not download model weights. Run real-model
checks separately on the completed bundle.

Defaults:

| Setting                    | Default                      |
| -------------------------- | ---------------------------- |
| `RUNTIME_SOURCE`           | Sibling `llama.cpp` checkout |
| `CLI_SOURCE`               | This checkout                |
| `RUNTIME_BUILD`            | `.release-build/runtime`     |
| `CARGO_TARGET_DIR`         | `.release-build/cargo`       |
| `OUT`                      | `dist`                       |
| `PROFILE`                  | `fastrel`                    |
| `MACOSX_DEPLOYMENT_TARGET` | `12.0`                       |

Use absolute paths when overriding directory settings. `OUT` must not already
contain the same bundle or archive; use a fresh output directory for a rebuild.
Build directories and `dist/` are ignored by Git. The OpenSSL helper caches its
source under `.release-build/openssl`; `MILLIE_OPENSSL_ARCHIVE` can point to a
previously downloaded crate, which is checked against the same pinned digest.

Outputs for version 0.1.0:

- `dist/millie-0.1.0-macos-arm64/`
- `dist/millie-0.1.0-macos-arm64.tar.gz`
- `dist/SHA256SUMS`

This script does not Developer-ID sign or notarize. Apply release signing and
notarization to the completed executables before creating the public archive.
Recreate the archive and checksum from those signed files; exclude macOS
extended attributes and AppleDouble sidecars as the script does.

## Linux release build reference

The distributed Linux bundle targets x86_64 with glibc 2.31 or newer and uses
Vulkan. A normal source build against a newer host's libraries is useful for
local development but does not establish compatibility with that baseline.
See the runtime's [Linux/Vulkan build instructions](https://github.com/llmsforall/llama.cpp/blob/main/docs/build-millie.md#linux--vulkan)
for a local build.

The 0.1.0 release used these inputs:

| Component                             | Version or setting                                         |
| ------------------------------------- | ---------------------------------------------------------- |
| Rust                                  | 1.95.0                                                     |
| CLI profile                           | `fastrel`, locked Cargo dependencies                       |
| cargo-zigbuild                        | 0.23.3                                                     |
| Zig                                   | 0.16.0                                                     |
| CMake                                 | 3.31.10                                                    |
| Rust target                           | `x86_64-unknown-linux-gnu.2.31`                            |
| C/C++ target                          | `x86_64-linux-gnu.2.31`                                    |
| shaderc                               | v2026.3, commit `2fbab0561c3cc466e992a954e9e33e2cf8c94555` |
| Runtime Vulkan headers                | 1.4.312                                                    |
| Bundled Vulkan loader and its headers | Vulkan SDK 1.4.357.0                                       |
| CLI OpenSSL                           | 3.5.5 from `openssl-src` 300.5.5+3.5.5                     |

The [OpenSSL helper](../scripts/release/build-openssl.sh) supports Linux x86_64
through the supplied `zigcc231` wrapper. Add `scripts/release/linux` to PATH
and select a staging directory with `MILLIE_OPENSSL_BUILD`. It builds static
libraries with a neutral `/usr/local` prefix; DESTDIR holds the staged files.
For Cargo, set `OPENSSL_NO_VENDOR=1`, `OPENSSL_STATIC=1` and `OPENSSL_DIR` to
that staging directory's `install/usr/local` subdirectory. Set
`MILLIE_VERSION=0.1.0`, then build from `codex-rs` with:

```sh
cargo zigbuild --locked --profile fastrel --target x86_64-unknown-linux-gnu.2.31 -p codex-cli --bin millie -j 8
```

For the runtime, use the supplied `zigcc231` and `zigcxx231` as the C/C++
compilers. Configure CMake with `CMAKE_BUILD_TYPE=Release`,
`BUILD_SHARED_LIBS=OFF`, `GGML_NATIVE=OFF`, `GGML_VULKAN=ON`,
`LLAMA_BUILD_SERVER=ON`, `LLAMA_OPENSSL=OFF`, `LLAMA_BUILD_COMMIT=release` and
`LLAMA_BUILD_NUMBER=0`. Provide the runtime Vulkan headers, `glslc` and staged
loader explicitly through CMake's Vulkan package settings when multiple SDKs
are installed. Build the `llama-server` target.

The bundled loader is built from [Vulkan-Loader](https://github.com/KhronosGroup/Vulkan-Loader/tree/vulkan-sdk-1.4.357.0)
and matching [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers/tree/vulkan-sdk-1.4.357.0).
The [loader build notes and patches](../scripts/release/linux/README.md) describe
its configuration, including duplicate-driver detection. Stage a baseline-compatible
loader as `bin/libvulkan.so.1`. Set the runtime's library search path to `$ORIGIN`
so it loads that adjacent library. GPU vendor drivers remain installed by the user.

Build all native dependencies against a glibc-2.31-compatible development
sysroot/environment; Zig's target setting alone does not make newer prebuilt
host libraries portable. Use Rust `--remap-path-prefix` and C/C++
`-ffile-prefix-map` for source, dependency and output directories, mapping them
to neutral `/src` or `/build` paths. The Mac script shows the mapping approach.
Strip debug symbols, inspect dynamic dependencies and run `checks/check-build-paths.py`
on the assembled `bin/` directory. Test on another supported Linux machine.

These are the compiler/dependency settings and helpers for the Linux release;
a complete automated Linux environment/bootstrap script is not included.
They describe a matching build configuration, not a guarantee of byte-identical
output across compilers, host environments or timestamps.

## Required bundle files

Both platforms use this layout:

```text
millie-VERSION-PLATFORM/
  bin/
    millie
    llama-server
    millie-models.json
    system_prompt.md
    millie-native.jinja
  README.md
  LICENSE
  NOTICE
  LICENSE.llama.cpp
  LICENSE.OpenSSL
```

Copy the catalog from `codex-rs/models-manager/millie-models.json`, the prompt
from `prompts/system_prompt.md`, the template from the runtime checkout's root,
and the bundle README from `docs/RELEASE_README.md`. Preserve both repositories'
licenses/notices and the license of linked OpenSSL. Linux also includes
`bin/libvulkan.so.1`; preserve the loader's applicable license notices when
packaging it. Keep model weights, compiler caches and local test logs outside
the bundle.

The local checks in `scripts/release/checks/` can be run against a completed
bundle: `check-startup.py` takes the CLI executable; `check-sampling.py` also
takes the bundled catalog; the remaining startup/install/update checks take
the executable and CLI source directory. `check-build-paths.py` takes `bin/`.
The Mac build script supplies the full command sequence. A changed test wrapper
or documentation page does not require rebuilding otherwise unchanged binaries.

## GitHub release placement

Attach both platform archives and one combined `SHA256SUMS` to a release in
`llmsforall/millie-cli`. For version 0.1.0 the installer expects tag `v0.1.0`
and assets `millie-0.1.0-macos-arm64.tar.gz` and
`millie-0.1.0-linux-x86_64.tar.gz`. Tag the corresponding runtime source too;
a separate runtime binary release is optional. Generate the combined checksums
after final signing/packaging. Keep build workspaces and local validation logs out of source and release assets.
