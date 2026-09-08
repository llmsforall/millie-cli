#!/usr/bin/env python3
"""Installer integration checks with fake downloads and isolated shell homes."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

INSTALLER = Path(__file__).with_name('install.sh')

def check(shell, operating_system, profile_name=None, custom_directory=False):
    with tempfile.TemporaryDirectory(prefix='millie-installer-') as tmp:
        root = Path(tmp)
        home = root / "home space 'quote' $(touch BAD)"
        home.mkdir()
        mocks = root / 'commands'
        mocks.mkdir()
        platform = 'macos-arm64' if operating_system == 'Darwin' else 'linux-x86_64'
        name = 'millie-0.1.0-' + platform
        archive = root / (name + '.tar.gz')
        with tarfile.open(archive, 'w:gz') as output:
            data = ('#!' + sys.executable + '\n' + '''import json,os,sys
from pathlib import Path
if sys.argv[1:] == ['--version']:
    print('millie 0.1.0')
else:
    print(json.dumps({'args':sys.argv[1:], 'cwd':os.getcwd(),
        'stdin':sys.stdin.read(), 'marker':os.environ['USER_MARKER'],
        'resource':(Path(sys.argv[0]).parent/'resource.txt').read_text()}))
    sys.exit(17)
''').encode()
            info = tarfile.TarInfo(name + '/bin/millie')
            info.mode = 0o755
            info.size = len(data)
            output.addfile(info, io.BytesIO(data))
            resource = b'bundled resource'
            info = tarfile.TarInfo(name + '/bin/resource.txt')
            info.size = len(resource)
            output.addfile(info, io.BytesIO(resource))
        sums = root / 'SHA256SUMS'
        sums.write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name + '\n')
        curl = mocks / 'curl'
        curl.write_text('#!' + sys.executable + '\n' + '''import os,sys
from pathlib import Path
url=sys.argv[-1]
if url.endswith('/releases/latest'): print('{"tag_name":"v0.1.0"}')
else: sys.stdout.buffer.write((Path(os.environ['FIXTURE'])/url.rsplit('/',1)[-1]).read_bytes())
''')
        curl.chmod(0o755)
        uname = mocks / 'uname'
        uname.write_text('#!/bin/sh\ncase "$1" in -s) printf "%s\\n" "$TEST_OS";; -m) printf "%s\\n" "$TEST_ARCH";; *) exit 1;; esac\n')
        uname.chmod(0o755)
        env = {k: v for k, v in os.environ.items() if not k.startswith('MILLIE_') and k not in ('ZDOTDIR', 'BASH_ENV', 'ENV')}
        env.update(HOME=str(home), SHELL=shell, PATH=str(mocks) + ':/usr/bin:/bin', FIXTURE=str(root), TEST_OS=operating_system, TEST_ARCH='arm64' if operating_system == 'Darwin' else 'x86_64')
        bindir = home / '.local/bin'
        if custom_directory:
            bindir = home / "custom 'bin' $literal"
            env['MILLIE_INSTALL_DIR'] = str(bindir)
        if Path(shell).name == 'zsh':
            zdir = home / 'shell settings'
            zdir.mkdir()
            env['ZDOTDIR'] = str(zdir)
            profiles = [zdir / '.zshrc']
        else:
            profiles = [home / '.bashrc', home / (profile_name or '.profile')]
        for p in profiles:
            p.write_text('export USER_MARKER=preserved\n')
        def install():
            return subprocess.run(['sh', str(INSTALLER)], env=env, cwd=root, capture_output=True, text=True, timeout=20)
        # Upgrade from the previous symlink installer without overwriting its target.
        bindir.mkdir(parents=True)
        old_binary = root / 'previous-binary'
        old_binary.write_bytes(b'previous signed executable')
        (bindir / 'millie').symlink_to(old_binary)
        result = install()
        assert result.returncode == 0, result.stderr
        assert not (bindir / 'millie').is_symlink()
        assert old_binary.read_bytes() == b'previous signed executable'
        assert (home / '.millie/app/0.1.0/bin/millie').read_bytes() == data
        args = ['--probe', 'space argument', "quote'argument", '$(touch BAD)', '']
        probe = subprocess.run([str(bindir / 'millie'), *args], env=dict(env, USER_MARKER='forwarded'), cwd=root,
                               input='forwarded stdin', capture_output=True, text=True, timeout=10)
        assert probe.returncode == 17, probe.stderr
        assert json.loads(probe.stdout) == {'args': args, 'cwd': str(root.resolve()),
            'stdin': 'forwarded stdin', 'marker': 'forwarded', 'resource': 'bundled resource'}
        first = {p: p.read_bytes() for p in profiles}
        assert all(b'USER_MARKER=preserved' in v for v in first.values())
        assert install().returncode == 0
        assert first == {p: p.read_bytes() for p in profiles}, 'Reinstall duplicated PATH configuration'
        # Activation printed by the installer must work immediately in a clean shell.
        activation = next(line for line in result.stdout.splitlines() if line.startswith('export PATH='))
        command = activation + '; test "$USER_MARKER" = preserved; millie --version'
        active = subprocess.run([shell, '-c', command], env=dict(env, USER_MARKER='preserved'), cwd=root, capture_output=True, text=True)
        assert active.returncode == 0 and 'millie 0.1.0' in active.stdout, (active.stdout, active.stderr)
        # A new interactive shell must find Millie without rerunning activation.
        for mode in (['-ic', '-lic'] if Path(shell).name == 'bash' else ['-ic']):
            new_shell = subprocess.run([shell, mode, 'test "$USER_MARKER" = preserved && millie --version'], env=env, cwd=root, capture_output=True, text=True, timeout=10)
            assert new_shell.returncode == 0 and 'millie 0.1.0' in new_shell.stdout, (mode, new_shell.stdout, new_shell.stderr)
        assert not (root / 'BAD').exists(), 'Shell path quoting allowed command substitution'
        # Failed checksum verification must not modify shell configuration or the install.
        sums.write_text('0' * 64 + '  ' + archive.name + '\n')
        failed = install()
        assert failed.returncode != 0 and 'checksum mismatch' in failed.stderr
        assert first == {p: p.read_bytes() for p in profiles}
        print('PASS:', operating_system, Path(shell).name, profile_name or 'default profile', 'custom directory' if custom_directory else 'default directory')

if __name__ == '__main__':
    bash = shutil.which('bash')
    assert bash, 'Bash is required for these checks'
    for profile in (None, '.bash_profile', '.bash_login'):
        check(bash, 'Linux', profile, custom_directory=profile == '.bash_login')
    zsh = shutil.which('zsh')
    if zsh:
        check(zsh, 'Darwin', custom_directory=True)
    else:
        print('SKIP: Zsh not installed; run the macOS shell check on a Mac')
