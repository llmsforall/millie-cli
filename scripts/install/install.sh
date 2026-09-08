#!/bin/sh
# Millie installer: downloads the platform bundle from GitHub releases,
# verifies its checksum, unpacks it under ~/.millie/app/<version>/, and links
# the millie binary into ~/.local/bin. Sets persistent PATH for Bash and Zsh.
#
# Usage: install.sh [--release VERSION]
# Environment:
#   MILLIE_RELEASE      Version to install (default: latest).
#   MILLIE_INSTALL_DIR  Where the millie symlink goes (default: ~/.local/bin).
#   MILLIE_HOME         Millie home directory (default: ~/.millie).

set -eu

REPO="llmsforall/millie-cli"
RELEASE="${MILLIE_RELEASE:-latest}"
BIN_DIR="${MILLIE_INSTALL_DIR:-$HOME/.local/bin}"
MILLIE_HOME_DIR="${MILLIE_HOME:-$HOME/.millie}"
APP_ROOT="$MILLIE_HOME_DIR/app"

step() { printf '==> %s\n' "$1"; }
fail() { printf 'ERROR: %s\n' "$1" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    --release)
      [ "$#" -ge 2 ] || fail "--release requires a value"
      RELEASE="$2"; shift ;;
    -h|--help)
      sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) fail "unknown argument: $1" ;;
  esac
  shift
done

case "$BIN_DIR" in
  /*) ;;
  *) BIN_DIR="$PWD/$BIN_DIR" ;;
esac
case "$BIN_DIR" in
  *:*|*'
'*) fail "MILLIE_INSTALL_DIR must not contain a colon or newline" ;;
esac

os="$(uname -s)"
arch="$(uname -m)"
case "$os/$arch" in
  Linux/x86_64)  asset_base="linux-x86_64";  archive_ext="tar.gz" ;;
  Darwin/arm64)  asset_base="macos-arm64";   archive_ext="tar.gz" ;;
  *) fail "unsupported platform: $os/$arch (Windows: use install.ps1; other platforms: build from source, see docs/install.md)" ;;
esac

fetch() { # url -> stdout
  if command -v curl >/dev/null 2>&1; then curl -fsSL "$1"
  elif command -v wget >/dev/null 2>&1; then wget -q -O - "$1"
  else fail "curl or wget is required"; fi
}

if [ "$RELEASE" = "latest" ]; then
  step "Resolving latest release"
  RELEASE="$(fetch "https://api.github.com/repos/$REPO/releases/latest" | sed -n 's/.*"tag_name" *: *"v\{0,1\}\([^"]*\)".*/\1/p' | head -1)"
  [ -n "$RELEASE" ] || fail "could not resolve the latest release"
fi

name="millie-$RELEASE-$asset_base"
archive="$name.$archive_ext"
url="https://github.com/$REPO/releases/download/v$RELEASE/$archive"
sums_url="https://github.com/$REPO/releases/download/v$RELEASE/SHA256SUMS"

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT

step "Downloading $archive"
fetch "$url" > "$tmp_dir/$archive"

step "Verifying checksum"
fetch "$sums_url" > "$tmp_dir/SHA256SUMS" || fail "could not download SHA256SUMS"
expected="$(grep " $archive\$" "$tmp_dir/SHA256SUMS" | awk '{print $1}')"
[ -n "$expected" ] || fail "no checksum listed for $archive"
if command -v sha256sum >/dev/null 2>&1; then
  actual="$(sha256sum "$tmp_dir/$archive" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
  actual="$(shasum -a 256 "$tmp_dir/$archive" | awk '{print $1}')"
else
  fail "sha256sum or shasum is required"
fi
[ "$actual" = "$expected" ] || fail "checksum mismatch for $archive"

step "Installing to $APP_ROOT/$RELEASE"
mkdir -p "$APP_ROOT"
rm -rf "$APP_ROOT/$RELEASE"
tar -xzf "$tmp_dir/$archive" -C "$APP_ROOT"
mv "$APP_ROOT/$name" "$APP_ROOT/$RELEASE"

step "Linking $BIN_DIR/millie"
mkdir -p "$BIN_DIR"
ln -sf "$APP_ROOT/$RELEASE/bin/millie" "$BIN_DIR/millie"

# The installer is a child process: it cannot change its parent terminal's PATH.
# Persist the entry for future shells and print an activation command for this one.
shell_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}
quoted_bin=$(shell_quote "$BIN_DIR")
path_line=$(printf 'case ":$PATH:" in *:%s:*) ;; *) export PATH=%s:"$PATH" ;; esac' "$quoted_bin" "$quoted_bin")
path_setup_ok=true

add_path_to_file() {
  profile_file=$1
  if [ -f "$profile_file" ] && grep -Fqx "$path_line" "$profile_file"; then
    return
  fi
  if mkdir -p "$(dirname "$profile_file")" && printf '\n# Millie CLI: make the installed command available in new terminals.\n%s\n' "$path_line" >> "$profile_file"; then
    step "Configured PATH in $profile_file"
  else
    path_setup_ok=false
    printf 'Could not update %s. Add this line there for persistent PATH:\n%s\n' "$profile_file" "$path_line" >&2
  fi
}

user_shell=${SHELL:-}
case "${user_shell##*/}" in
  zsh)
    add_path_to_file "${ZDOTDIR:-$HOME}/.zshrc"
    ;;
  bash)
    add_path_to_file "$HOME/.bashrc"
    # Login Bash reads the first existing file in this list.
    if [ -f "$HOME/.bash_profile" ]; then
      login_profile="$HOME/.bash_profile"
    elif [ -f "$HOME/.bash_login" ]; then
      login_profile="$HOME/.bash_login"
    else
      login_profile="$HOME/.profile"
    fi
    add_path_to_file "$login_profile"
    ;;
  *)
    path_setup_ok=false
    printf 'Automatic persistent PATH setup supports Bash and Zsh.\n'
    printf 'For another shell, add %s to its PATH configuration.\n' "$BIN_DIR"
    ;;
esac

step "Installed Millie"
printf '\nIn this terminal (Bash/Zsh), run these commands, one at a time:\n\n'
printf 'export PATH=%s:"$PATH"\n' "$quoted_bin"
printf 'millie --version\n'
printf '\nThen replace the example path below with your project folder:\n\n'
printf 'cd "/path/to/your/project"\n'
printf 'millie\n'
if [ "$path_setup_ok" = true ]; then
  printf '\nPATH is saved for new terminals and after reboot; no need to rerun the export then.\n'
else
  printf '\nPersistent PATH setup was not completed. Follow the instructions above for your shell.\n'
fi
printf 'The first launch asks you to choose a model and approve its download.\n'
