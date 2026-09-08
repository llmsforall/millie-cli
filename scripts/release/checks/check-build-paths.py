#!/usr/bin/env python3
"""Reject bundled binaries containing user or build-workspace paths."""
import re
import sys
from pathlib import Path

MARKERS = re.compile(rb"/Users/|/home/|/workspace/|codex_workspaces|local_test")
MAGIC = (b"\x7fELF", b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf",
         b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xce", b"\xca\xfe\xba\xbe", b"MZ", b"!<arch>\n")

def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: check-build-paths.py BUNDLE_BIN_DIRECTORY")
    root = Path(sys.argv[1])
    if not root.is_dir():
        raise SystemExit("Bundle bin directory does not exist")
    failures = []
    checked = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if not data.startswith(MAGIC):
            continue
        checked += 1
        count = 0
        for match in MARKERS.finditer(data):
            before = data[max(0, match.start() - 32):match.start()]
            after = data[match.end():match.end() + 32]
            # Public HTTP routes and Emscripten's virtual home are not local paths.
            if match.group() == b"/workspace/" and before.endswith(
                    (b"/ps/plugins", b"/public/plugins")):
                continue
            if (match.group() == b"/home/" and before.endswith(b'HOME:"')
                    and after.startswith(b'web_user"')):
                continue
            count += 1
        if count:
            failures.append(f"{path.relative_to(root)}: {count} build-path markers")
    if not checked:
        raise SystemExit("No executable or library artifacts found to check")
    if failures:
        raise SystemExit("Build-path check failed; rebuild affected artifacts and dependencies:\n"
                         + "\n".join(failures))
    print(f"Build-path check passed for {checked} executable/library artifacts")

if __name__ == "__main__":
    main()
