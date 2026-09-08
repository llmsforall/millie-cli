#!/usr/bin/env bash
# Build the pinned static crypto dependency without local install paths in metadata.
set -euo pipefail
SUPPORT=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SUPPORT/../.." && pwd)
OUT=${MILLIE_OPENSSL_BUILD:-$ROOT/.release-build/openssl}
JOBS=${JOBS:-8}
ARCHIVE=${MILLIE_OPENSSL_ARCHIVE:-$OUT/downloads/openssl-src-300.5.5+3.5.5.crate}
if [ ! -f "$ARCHIVE" ]; then
    mkdir -p "$(dirname "$ARCHIVE")"
    curl --fail --location --retry 3 --output "$ARCHIVE.part" 'https://static.crates.io/crates/openssl-src/openssl-src-300.5.5+3.5.5.crate'
    mv "$ARCHIVE.part" "$ARCHIVE"
fi
python3 - "$ARCHIVE" <<'VERIFY'
import hashlib, sys
from pathlib import Path
expected = '3f1787d533e03597a7934fd0a765f0d28e94ecc5fb7789f8053b1e699a56f709'
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != expected:
    raise SystemExit('OpenSSL source checksum mismatch; remove the cached archive and retry')
VERIFY
mkdir -p "$OUT/source"
if [ ! -f "$OUT/source/openssl-src-300.5.5+3.5.5/openssl/Configure" ]; then
    tar -xzf "$ARCHIVE" -C "$OUT/source"
fi
cd "$OUT/source/openssl-src-300.5.5+3.5.5/openssl"
unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS CROSS_COMPILE
FLAGS=(-O2 -ffunction-sections -fdata-sections -fPIC -w)
case "$(uname -s):$(uname -m)" in
    Darwin:arm64)
        TARGET=darwin64-arm64-cc
        export CC=clang AR=ar RANLIB=ranlib
        FLAGS+=("-mmacosx-version-min=${MACOSX_DEPLOYMENT_TARGET:-12.0}")
        ;;
    Linux:x86_64)
        TARGET=linux-x86_64
        # zigcc231 should invoke: zig cc -target x86_64-linux-gnu.2.31 "$@"
        export CC=zigcc231 AR=ar RANLIB=ranlib
        FLAGS+=(-m64)
        ;;
    *) echo "Unsupported OpenSSL build platform" >&2; exit 1 ;;
esac
perl ./Configure --prefix=/usr/local --openssldir=/usr/local/ssl --libdir=lib \
    no-shared no-module no-tests no-comp no-zlib no-zlib-dynamic no-ssl3 \
    no-md2 no-rc5 no-weak-ssl-ciphers no-camellia no-idea no-seed \
    "$TARGET" "${FLAGS[@]}"
make -j "$JOBS" build_libs
make -j "$JOBS" install_dev DESTDIR="$OUT/install"
