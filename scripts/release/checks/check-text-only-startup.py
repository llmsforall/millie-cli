#!/usr/bin/env python3
"""Check download approval for cached text models without a vision tower.

Usage: python3 check-text-only-startup.py /path/to/millie /path/to/millie-cli
Uses a deliberately failing mock launcher, so no model is loaded or downloaded.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile


def check(binary, source, profile, flags, needs_approval):
    with tempfile.TemporaryDirectory(prefix='millie-text-only-') as tmp:
        home = Path(tmp)
        (home / 'millie-native.jinja').write_text('{{ messages }}')
        catalog_path = source / 'codex-rs/models-manager/millie-models.json'
        catalog = json.loads(catalog_path.read_text())
        slug = next(iter(catalog['downloads']))
        entry = catalog['downloads'][slug]
        assert entry.get('mmproj_file'), 'Fixture requires a catalog vision tower'
        (home / 'models').mkdir()
        (home / 'models' / entry['model_file']).touch()
        assert not (home / 'models' / entry['mmproj_file']).exists()
        entry['model_sha256'] = hashlib.sha256(b'').hexdigest()
        (home / 'millie-models.json').write_text(json.dumps(catalog))
        shutil.copy2(source / 'prompts/system_prompt.md', home / 'system_prompt.md')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        (home / 'config.toml').write_text(f'[llamacpp]\nport = {port}\n')
        launched = home / 'launched'
        launcher = home / 'mock-server'
        launcher.write_text('#!/bin/sh\nprintf launched > ' + shlex.quote(str(launched)) + '\nexit 12\n')
        launcher.chmod(0o755)
        env = {k: v for k, v in os.environ.items() if not k.startswith('MILLIE_')}
        env.update(MILLIE_HOME=str(home), MILLIE_LLAMACPP_PROFILE=profile,
                   MILLIE_LLAMACPP_SERVER_BIN=str(launcher), MILLIE_LLAMACPP_GPU='none',
                   MILLIE_LLAMACPP_STARTUP_TIMEOUT_SECS='3', OMP_NUM_THREADS='1')
        result = subprocess.run([str(binary), 'exec', '--skip-git-repo-check',
                                 '--model', slug, *flags, 'Reply with ok.'],
                                cwd=home, env=env, stdin=subprocess.DEVNULL,
                                text=True, capture_output=True, timeout=20)
        assert result.returncode != 0, 'The mock intentionally fails model loading'
        if needs_approval:
            assert 'no recorded remote revision' in result.stderr, result.stderr
            assert not launched.exists(), 'Vision-enabled startup bypassed required approval'
        else:
            assert 'Download not approved' not in result.stderr, result.stderr
            assert 'llama-server exited during startup' in result.stderr, result.stderr
            assert launched.exists(), 'Text-only startup did not reach the launcher'
        print('PASS:', profile, flags, 'matching revision required' if needs_approval else 'no tower approval requested')


if __name__ == '__main__':
    binary, source = (Path(arg).resolve() for arg in sys.argv[1:3])
    # Explicit CLI false must override a vision-enabled profile.
    check(binary, source, 'cpu-full', ['--no-vision'], False)
    # A text-only profile exports vision=0 without any CLI vision override.
    check(binary, source, 'cpu-compact', [], False)
    # Positive control: explicitly enabled vision still requires the missing file.
    check(binary, source, 'cpu-compact', ['--vision'], True)
