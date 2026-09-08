#!/usr/bin/env python3
"""Exercise packaged server supervision using a mock server, without models/GPUs.

Usage: python3 check-startup.py /path/to/millie
"""
import argparse
import http.server
import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time


def fake_server():
    if '--list-devices' in sys.argv:
        return
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    parser.add_argument('--port', type=int)
    parser.add_argument('--mmproj')
    args, _ = parser.parse_known_args(sys.argv[2:])
    if os.environ.get('TEST_FAIL') == '1':
        sys.exit(12)
    start = time.monotonic()
    delay = float(os.environ.get('TEST_LOAD_DELAY', '0.3'))
    with open(os.environ['TEST_LAUNCHES'], 'a') as f:
        f.write(str(os.getpid()) + '\n')

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(503 if self.path == '/health' and time.monotonic() - start < delay else 200)
            body = json.dumps({'model_path': args.model, 'modalities': {'vision': bool(args.mmproj)}, 'default_generation_settings': {'n_ctx': 32768}}).encode()
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    http.server.ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


def fake_owner(binary):
    child = subprocess.Popen([binary, '__millie_model_server'], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
    print(child.stdout.readline().strip(), flush=True)
    time.sleep(3600)


def read_ready(process):
    readable, _, _ = select.select([process.stdout], [], [], 20)
    assert readable, 'Supervisor did not report readiness/error within 20 seconds'
    line = process.stdout.readline()
    assert line, 'Supervisor exited without a response'
    return json.loads(line)


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(.1)
        return sock.connect_ex(('127.0.0.1', port)) == 0


def wait_stopped(port):
    deadline = time.monotonic() + 10
    while listening(port) and time.monotonic() < deadline:
        time.sleep(.05)
    assert not listening(port), 'Owned server remained after owner exit'


def main(binary):
    with tempfile.TemporaryDirectory(prefix='millie-startup-') as tmp:
        root = Path(tmp)
        (root / 'millie-native.jinja').write_text('{{ messages }}')
        for name in ('one.gguf', 'two.gguf'):
            (root / name).touch()
        launcher = root / 'mock-server'
        # The script is only run on Linux/macOS, matching the release builds.
        import shlex
        launcher.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' ' + shlex.quote(str(Path(__file__).resolve())) + ' --fake-server "$@"\n')
        launcher.chmod(0o755)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        launches = root / 'launches'
        env = {k: v for k, v in os.environ.items() if not k.startswith('MILLIE_')}
        env.update(MILLIE_HOME=str(root), MILLIE_LLAMACPP_MODEL_PATH=str(root / 'one.gguf'),
                   MILLIE_LLAMACPP_SERVER_BIN=str(launcher), MILLIE_LLAMACPP_PORT=str(port),
                   MILLIE_LLAMACPP_GPU='none', MILLIE_LLAMACPP_VISION='false',
                   MILLIE_LLAMACPP_STARTUP_TIMEOUT_SECS='3', TEST_LAUNCHES=str(launches))
        processes = []
        logs = (root / 'supervisor.log').open('w+')

        def launch(**overrides):
            p = subprocess.Popen([binary, '__millie_model_server'], env=dict(env, **overrides),
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=logs, text=True)
            processes.append(p)
            return p

        try:
            (root/'temp-a').mkdir(); (root/'temp-b').mkdir()
            first, second = launch(TMPDIR=str(root/'temp-a')), launch(TMPDIR=str(root/'temp-b'))
            responses = [read_ready(first), read_ready(second)]
            assert sorted(r.get('owned') for r in responses) == [False, True], responses
            owner = first if responses[0]['owned'] else second
            borrower = second if owner is first else first
            borrower.stdin.close()
            borrower.wait(timeout=5)
            assert listening(port)
            assert len(launches.read_text().splitlines()) == 1
            print('PASS: simultaneous cold starts launch once; closing borrower leaves owner running')

            wrong = launch(MILLIE_LLAMACPP_MODEL_PATH=str(root / 'two.gguf'))
            assert 'different model' in read_ready(wrong)['error']
            wrong.wait(timeout=5)
            assert listening(port)
            print('PASS: different model rejected without disrupting server')

            wrong_vision = launch(MILLIE_LLAMACPP_VISION='true', MILLIE_LLAMACPP_MMPROJ_PATH=str(root/'two.gguf'))
            assert 'different vision mode' in read_ready(wrong_vision)['error']
            wrong_vision.wait(timeout=5)
            assert listening(port)
            print('PASS: requested vision refuses text-only server without disruption')

            owner.stdin.close()  # Equivalent EOF when the owning CLI exits or is killed.
            owner.wait(timeout=10)
            wait_stopped(port)
            print('PASS: owner pipe closure stops and reaps server')

            owner = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--fake-owner', binary],
                                     env=env, stdout=subprocess.PIPE, stderr=logs, text=True)
            processes.append(owner)
            assert read_ready(owner)['owned']
            owner.kill()
            owner.wait(timeout=5)
            wait_stopped(port)
            print('PASS: SIGKILL of owning client releases model server')

            before = len(launches.read_text().splitlines())
            loading_owner = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--fake-owner', binary],
                                             env=dict(env, TEST_LOAD_DELAY='60'), stdout=subprocess.PIPE, stderr=logs, text=True)
            processes.append(loading_owner)
            deadline = time.monotonic() + 5
            while len(launches.read_text().splitlines()) == before and time.monotonic() < deadline:
                time.sleep(.05)
            assert len(launches.read_text().splitlines()) > before
            loading_owner.kill()
            loading_owner.wait(timeout=5)
            wait_stopped(port)
            print('PASS: owner crash during loading cleans up child')

            for _ in range(3):
                owner = launch()
                assert read_ready(owner)['owned']
                owner.stdin.close()
                contender = launch()
                response = read_ready(contender)
                assert 'error' not in response, response
                owner.wait(timeout=10)
                contender.stdin.close()
                contender.wait(timeout=10)
                wait_stopped(port)
            print('PASS: attachment/shutdown races leave no duplicate or orphan server')

            failing = launch(TEST_FAIL='1')
            assert 'error' in read_ready(failing)
            failing.wait(timeout=5)
            wait_stopped(port)
            timeout = launch(TEST_LOAD_DELAY='60', MILLIE_LLAMACPP_STARTUP_TIMEOUT_SECS='1')
            assert 'error' in read_ready(timeout)
            timeout.wait(timeout=5)
            wait_stopped(port)
            print('PASS: failed and timed-out launches leave no server')

            persistent = launch(MILLIE_LLAMACPP_KEEP_ALIVE='1')
            assert read_ready(persistent)['owned']
            persistent.stdin.close()
            time.sleep(.2)
            assert listening(port)
            pid = int(launches.read_text().splitlines()[-1])
            os.kill(pid, signal.SIGTERM)
            persistent.wait(timeout=10)
            wait_stopped(port)
            print('PASS: persistence requires explicit opt-in')

            external = subprocess.Popen([str(launcher), '--model', str(root/'one.gguf'), '--port', str(port)], env=env, stderr=logs)
            processes.append(external)
            deadline = time.monotonic() + 5
            while not listening(port) and time.monotonic() < deadline:
                time.sleep(.05)
            time.sleep(.4)  # Allow the independently launched mock's loading phase to finish.
            attached = launch()
            assert read_ready(attached) == {'owned': False}
            attached.stdin.close()
            attached.wait(timeout=5)
            assert external.poll() is None and listening(port)
            external.terminate()
            external.wait(timeout=5)
            print('PASS: independently launched server remains independently owned')

            # Exercise ordinary CLI startup as well as the supervisor protocol.
            catalog_path = (Path(sys.argv[2]) if len(sys.argv)>2 else Path(__file__).resolve().parent / 'millie-cli') / 'codex-rs/models-manager/millie-models.json'
            catalog = json.loads(catalog_path.read_text())
            (root / 'millie-models.json').write_text(json.dumps(catalog))
            choices = list(catalog['downloads'])[:2]

            def start_cli(*flags):
                return subprocess.run([binary, 'exec', '--skip-git-repo-check', *flags, 'Reply with ok.'],
                                      cwd=root, env=dict(env, TEST_FAIL='1'), stdin=subprocess.DEVNULL,
                                      capture_output=True, text=True, timeout=30)

            for choice in choices:
                failed = start_cli('--model', choice)
                assert failed.returncode and 'llama-server exited during startup' in failed.stderr, failed.stderr
                assert f'model = "{choice}"' in (root / 'config.toml').read_text(), failed.stderr
            remembered = (root / 'config.toml').read_text()
            ordinary = start_cli()
            assert ordinary.returncode and 'llama-server exited during startup' in ordinary.stderr, ordinary.stderr
            assert (root / 'config.toml').read_text() == remembered
            invalid = start_cli('--model', 'definitely-not-a-valid-model')
            assert invalid.returncode and 'Unknown model' in invalid.stderr, invalid.stderr
            assert (root / 'config.toml').read_text() == remembered
            removed = start_cli('--oss')
            assert removed.returncode and "unexpected argument '--oss'" in removed.stderr, removed.stderr
            print('PASS: ordinary startup remembers last explicit choice through failure; invalid names and --oss rejected')
        except Exception:
            logs.flush()
            logs.seek(0)
            print(logs.read(), file=sys.stderr)
            raise
        finally:
            for p in processes:
                if p.stdin and not p.stdin.closed:
                    p.stdin.close()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    p.wait(timeout=5)
            if launches.exists():
                for pid in launches.read_text().splitlines():
                    # All are this harness's short-lived mock children.
                    command = subprocess.run(['ps', '-p', pid, '-o', 'command='], capture_output=True, text=True).stdout
                    if str(root) in command and '--fake-server' in command:
                        try:
                            os.kill(int(pid), signal.SIGTERM)
                        except ProcessLookupError:
                            pass
            logs.close()


if __name__ == '__main__':
    if sys.argv[1] == '--fake-server':
        fake_server()
    elif sys.argv[1] == '--fake-owner':
        fake_owner(sys.argv[2])
    else:
        main(str(Path(sys.argv[1]).resolve()))
