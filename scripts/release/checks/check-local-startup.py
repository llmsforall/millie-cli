#!/usr/bin/env python3
"""Regression checks using cached placeholders and a local mock; no model downloads."""
import http.server, json, os, select, shlex, shutil, socket, subprocess, sys, tempfile, threading, time
from pathlib import Path
binary, source = (Path(p).resolve() for p in sys.argv[1:3])
with tempfile.TemporaryDirectory(prefix='millie-startup-') as tmp:
    home = Path(tmp)
    catalog = json.loads((source/'codex-rs/models-manager/millie-models.json').read_text())
    (home/'millie-models.json').write_text(json.dumps(catalog))
    shutil.copy2(source/'prompts/system_prompt.md', home/'system_prompt.md')
    first, second = list(catalog['downloads'])[:2]
    existing = home/'existing.gguf'; existing.touch()
    marker = home/'launched'
    launcher = home/'server'
    launcher.write_text('#!/bin/sh\nif [ "$1" = "--list-devices" ]; then exit 0; fi\nprintf launched > '+shlex.quote(str(marker))+'\nexit 12\n')
    launcher.chmod(0o755)
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def do_GET(self):
            body=json.dumps({'model_path':str(existing),'modalities': {'vision': False}, 'default_generation_settings':{'n_ctx':32768}}).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    env={k:v for k,v in os.environ.items() if not k.startswith('MILLIE_')}
    env.update({key:'http://127.0.0.1:9' for key in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']})
    env.update(NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
    env.update(MILLIE_HOME=str(home),MILLIE_LLAMACPP_PORT=str(server.server_port),MILLIE_LLAMACPP_SERVER_BIN=str(launcher),MILLIE_LLAMACPP_GPU='none',MILLIE_LLAMACPP_VISION='0',MILLIE_LLAMACPP_STARTUP_TIMEOUT_SECS='2')
    def run(*flags, extra=None):
        return subprocess.run([str(binary),'exec','--skip-git-repo-check',*flags,'Reply with ok.'],env=dict(env,**(extra or {})),cwd=home,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=20)
    try:
        for flags in [[],['--download']]:
            result=run('--model',first,*flags)
            assert result.returncode and 'different model is already running' in result.stderr,result.stderr
            assert 'Download not approved' not in result.stderr,result.stderr
            assert not (home/'models').exists(),'Download began before conflict rejection'
            assert not marker.exists()
            assert first in (home/'config.toml').read_text()
        print('PASS: conflicting model refused before approval/download; explicit selection remembered')
        result=run('--model',first,extra={'MILLIE_OSS_BASE_URL':'http://127.0.0.1:1/v1'})
        assert result.returncode and 'conflicting API URL' in result.stderr,result.stderr
        print('PASS: conflicting inherited API URL refused')
        result=run('--local-provider','vllm','-c','vllm.temperature=1')
        assert result.returncode and 'unknown field' in result.stderr and 'temperature' in result.stderr,result.stderr
        print('PASS: misplaced vLLM sampling setting rejected without strict-config')
    finally:
        server.shutdown();server.server_close();thread.join()
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env.update(MILLIE_LLAMACPP_PORT=str(port),MILLIE_LLAMACPP_MODEL_PATH=str(existing))
    # A hard link keeps this check cheap while giving current_exe a directory with no template.
    original_binary=binary
    with tempfile.TemporaryDirectory(prefix='template-check-',dir=binary.parent) as isolated:
        binary=Path(isolated)/original_binary.name
        os.link(original_binary,binary)
        result=run('--model',first)
        assert result.returncode and 'Required millie-native.jinja was not found' in result.stderr,result.stderr
        assert not marker.exists()
    binary=original_binary
    print('PASS: missing packaged default chat template fails before server launch')
    # Deliberately nonexistent explicit template works even if packaged binary has a sibling template.
    bad_template={'MILLIE_LLAMACPP_CHAT_TEMPLATE':str(home/'missing.jinja')}
    result=run('--model',first,extra=bad_template)
    assert result.returncode and 'Required Millie chat template is missing' in result.stderr,result.stderr
    assert not marker.exists()
    print('PASS: invalid explicit chat template fails before server launch')
    result=run('--model',first,extra=dict(bad_template,MILLIE_LLAMACPP_TRANSPORT='raw'))
    assert result.returncode and 'llama-server exited during startup' in result.stderr,result.stderr
    assert marker.exists()
    print('PASS: explicit raw transport can start without a chat template')
    marker.unlink()
    (home/'millie-native.jinja').write_text('{{ messages }}')
    (home/'config.toml').write_text(f'model = "{first}"\n')
    (home/'work.config.toml').write_text(f'model = "{first}"\n')
    (home/'other.config.toml').write_text(f'model = "{first}"\n')
    base=(home/'config.toml').read_text();other=(home/'other.config.toml').read_text()
    result=run('--profile','work','--model',second)
    assert result.returncode and 'llama-server exited during startup' in result.stderr,result.stderr
    assert second in (home/'work.config.toml').read_text()
    assert (home/'config.toml').read_text()==base
    assert (home/'other.config.toml').read_text()==other
    remembered=(home/'work.config.toml').read_text()
    result=run('--profile','work')
    assert result.returncode and 'llama-server exited during startup' in result.stderr,result.stderr
    assert (home/'work.config.toml').read_text()==remembered
    print('PASS: named profile retains last explicit choice after failure; base and other profile unchanged')

    (home/'work.config.toml').write_text(f'model = "{first}"\n')
    result=run('--local-provider','vllm','--profile','work','--model',second,'-c',f'vllm.base_url="http://127.0.0.1:{port}/v1"')
    assert result.returncode, 'No remote server should be running'
    assert second in (home/'work.config.toml').read_text()
    assert (home/'config.toml').read_text()==base
    print('PASS: vLLM explicit model choice persists in active profile even after failed startup')

    # Exercise the chooser with real terminal handles, even when the model path is cached.
    master, slave = os.openpty()
    process = subprocess.Popen([str(binary),'exec','--skip-git-repo-check','--profile','work','--model','select','Reply with ok.'],env=env,cwd=home,stdin=slave,stdout=slave,stderr=slave)
    os.close(slave)
    output=b'';answered=False;deadline=time.monotonic()+20
    entries=[m for m in catalog['models'] if m['slug'] in catalog['downloads']]
    choice=next(i+1 for i,m in enumerate(entries) if m['slug']==first)
    try:
        while time.monotonic()<deadline:
            readable,_,_=select.select([master],[],[],.1)
            if readable:
                try:data=os.read(master,65536)
                except OSError:break
                if not data:break
                output+=data
                if b'Use which model?' in output and not answered:
                    os.write(master,f'{choice}\n'.encode());answered=True
            if process.poll() is not None and not readable:break
        process.wait(timeout=3)
        assert answered,output.decode(errors='replace')
        assert b'llama-server exited during startup' in output,output.decode(errors='replace')
        assert first in (home/'work.config.toml').read_text()
        assert (home/'config.toml').read_text()==base
        assert (home/'other.config.toml').read_text()==other
        print('PASS: terminal chooser updates active profile even with cached explicit GGUF path')
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=3)
        os.close(master)
