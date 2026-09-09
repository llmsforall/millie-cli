#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd -P)
ROOT=$(cd "$SCRIPT_DIR/../.." && pwd -P)
if [ "${1:-}" = --help ]; then
    echo "Usage: JOBS=8 VERSION=0.1.1 bash scripts/release/build-macos.sh"
    echo "Uses a sibling llama.cpp checkout; override RUNTIME_SOURCE if needed."
    echo "Outputs: .release-build/ and dist/ (unsigned). See docs/release-builds.md."
    exit 0
fi
if [ "$#" -ne 0 ]; then echo "Unknown argument; use --help" >&2; exit 2; fi
VERSION=${VERSION:-0.1.1}
JOBS=${JOBS:-8}
PROFILE=${PROFILE:-fastrel}
MACOSX_DEPLOYMENT_TARGET=${MACOSX_DEPLOYMENT_TARGET:-12.0}
RUNTIME_SOURCE=${RUNTIME_SOURCE:-$ROOT/../llama.cpp}
CLI_SOURCE=${CLI_SOURCE:-$ROOT}
RUNTIME_BUILD=${RUNTIME_BUILD:-$ROOT/.release-build/runtime}
CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$ROOT/.release-build/cargo}
OUT=${OUT:-$ROOT/dist}
export CARGO_TARGET_DIR MACOSX_DEPLOYMENT_TARGET
test "$(uname -s)" = Darwin
test "$(uname -m)" = arm64
test -f "$RUNTIME_SOURCE/CMakeLists.txt" || { echo "Set RUNTIME_SOURCE to the Millie llama.cpp checkout" >&2; exit 1; }
RUNTIME_SOURCE=$(cd "$RUNTIME_SOURCE" && pwd -P)
CLI_SOURCE=$(cd "$CLI_SOURCE" && pwd -P)
NAME="millie-$VERSION-macos-arm64"
DEST="$OUT/$NAME"
if [ -e "$DEST" ] || [ -e "$OUT/$NAME.tar.gz" ]; then
    echo "Output already exists; choose a fresh OUT directory." >&2
    exit 1
fi
mkdir -p "$OUT"
# Separate outputs avoid reusing objects compiled before path remapping.
# Encoded Rust flags preserve source paths containing spaces.
export MILLIE_BUILD_SOURCE_ROOT="$ROOT" CLI_SOURCE RUNTIME_SOURCE
export CARGO_ENCODED_RUSTFLAGS="$(python3 - <<'FLAGS'
import os, shlex
flags = (os.environ["CARGO_ENCODED_RUSTFLAGS"].split("\x1f")
         if os.environ.get("CARGO_ENCODED_RUSTFLAGS")
         else shlex.split(os.environ.get("RUSTFLAGS", "")))
for key, target in [("HOME", "/build/user"), ("CARGO_HOME", "/build/cargo"),
                    ("RUSTUP_HOME", "/build/rustup"), ("MILLIE_BUILD_SOURCE_ROOT", "/src"),
                    ("CLI_SOURCE", "/src/millie-cli"), ("RUNTIME_SOURCE", "/src/llama.cpp"),
                    ("CARGO_TARGET_DIR", "/build/cargo-target")]:
    if os.environ.get(key):
        flags.append("--remap-path-prefix=" + os.path.realpath(os.environ[key]) + "=" + target)
print("\x1f".join(flags), end="")
FLAGS
)"
PREFIX_MAP_FLAGS="-ffile-prefix-map=\"$HOME=/build/user\" -ffile-prefix-map=\"$ROOT=/src\" -ffile-prefix-map=\"$RUNTIME_SOURCE=/src/llama.cpp\" -ffile-prefix-map=\"$CLI_SOURCE=/src/millie-cli\" -ffile-prefix-map=\"$RUNTIME_BUILD=/build/runtime\""
export CFLAGS="${CFLAGS:-} $PREFIX_MAP_FLAGS"
export CXXFLAGS="${CXXFLAGS:-} $PREFIX_MAP_FLAGS"

cmake -S "$RUNTIME_SOURCE" -B "$RUNTIME_BUILD" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_C_FLAGS="$PREFIX_MAP_FLAGS" -DCMAKE_CXX_FLAGS="$PREFIX_MAP_FLAGS" \
    -DCMAKE_OBJC_FLAGS="$PREFIX_MAP_FLAGS" -DCMAKE_OBJCXX_FLAGS="$PREFIX_MAP_FLAGS" \
    -DLLAMA_BUILD_COMMIT=release -DLLAMA_BUILD_NUMBER=0 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" \
    -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_OPENSSL=OFF
cmake --build "$RUNTIME_BUILD" --target llama-server test-ternary-dot test-custom-type-remap test-backend-ops -j "$JOBS"
"$RUNTIME_BUILD/bin/test-ternary-dot"
"$RUNTIME_BUILD/bin/test-custom-type-remap"
# MTL0 must appear as an executed backend; an unmatched filter can skip all tests.
"$RUNTIME_BUILD/bin/test-backend-ops" test -b MTL0 -p 'qt_ms32k4|q1_sym32k|q2_sym32k4|q4_sym16k' \
    2>&1 | tee "$OUT/metal-correctness.log"
grep -Eq '[1-9][0-9]*/[1-9][0-9]* tests passed' "$OUT/metal-correctness.log" || {
    echo "No executed Metal test results found; inspect the backend name." >&2
    exit 1
}
# Use the same pinned static crypto source, with neutral install metadata.
export MILLIE_OPENSSL_BUILD="$ROOT/.release-build/openssl"
JOBS="$JOBS" bash "$SCRIPT_DIR/build-openssl.sh"
export OPENSSL_NO_VENDOR=1 OPENSSL_STATIC=1
export OPENSSL_DIR="$MILLIE_OPENSSL_BUILD/install/usr/local"
(
    cd "$CLI_SOURCE/codex-rs"
    MILLIE_VERSION="$VERSION" cargo build --locked --profile "$PROFILE" -p codex-cli --bin millie -j "$JOBS"
)
mkdir -p "$DEST/bin"
cp "$RUNTIME_BUILD/bin/llama-server" "$DEST/bin/"
cp "$CARGO_TARGET_DIR/$PROFILE/millie" "$DEST/bin/"
# Remove debug records before testing and signing the final executables.
strip -S "$DEST/bin/millie" "$DEST/bin/llama-server"
cp "$CLI_SOURCE/codex-rs/models-manager/millie-models.json" "$DEST/bin/"
cp "$CLI_SOURCE/prompts/system_prompt.md" "$DEST/bin/"
cp "$RUNTIME_SOURCE/millie-native.jinja" "$DEST/bin/"
cp "$CLI_SOURCE/docs/RELEASE_README.md" "$DEST/README.md"
cp "$CLI_SOURCE/LICENSE" "$DEST/LICENSE"
cp "$CLI_SOURCE/NOTICE" "$DEST/NOTICE"
cp "$MILLIE_OPENSSL_BUILD/source/openssl-src-300.5.5+3.5.5/openssl/LICENSE.txt" "$DEST/LICENSE.OpenSSL"
cat >> "$DEST/NOTICE" <<'NOTICE'

OpenSSL is licensed under the Apache License, Version 2.0.
See LICENSE.OpenSSL for the license supplied with the pinned dependency.
NOTICE
cp "$RUNTIME_SOURCE/LICENSE" "$DEST/LICENSE.llama.cpp"
"$DEST/bin/millie" --version
python3 "$SCRIPT_DIR/checks/check-startup.py" "$DEST/bin/millie"
python3 "$SCRIPT_DIR/checks/check-sampling.py" "$DEST/bin/millie" "$DEST/bin/millie-models.json"
python3 "$SCRIPT_DIR/checks/check-local-startup.py" "$DEST/bin/millie" "$CLI_SOURCE"
python3 "$SCRIPT_DIR/checks/check-text-only-startup.py" "$DEST/bin/millie" "$CLI_SOURCE"
python3 "$SCRIPT_DIR/checks/check-two-clis.py" "$DEST/bin/millie" "$CLI_SOURCE"
python3 "$SCRIPT_DIR/checks/check-install.py" "$DEST/bin/millie" "$CLI_SOURCE"
python3 "$SCRIPT_DIR/checks/check-updates.py" "$DEST/bin/millie" "$CLI_SOURCE"
otool -L "$DEST/bin/millie"
otool -L "$DEST/bin/llama-server"
python3 "$SCRIPT_DIR/checks/check-build-paths.py" "$DEST/bin"
COPYFILE_DISABLE=1 tar --no-xattrs --no-acls -C "$OUT" -czf "$OUT/$NAME.tar.gz" "$NAME"
(cd "$OUT" && shasum -a 256 "$NAME.tar.gz" > SHA256SUMS)
echo "Bundle: $OUT/$NAME.tar.gz"
