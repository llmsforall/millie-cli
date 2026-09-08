#!/usr/bin/env python3
"""Check real CLI ownership using a blocked mock generation, without inference."""
import argparse, http.server, json, os, shlex, shutil, socket, subprocess, sys, tempfile, time
from pathlib import Path

if sys.argv[1] == '--server':
    parser=argparse.ArgumentParser();parser.add_argument('--model');parser.add_argument('--port',type=int)
    args,_=parser.parse_known_args(sys.argv[2:])
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def send(self,value):
            body=json.dumps(value).encode();self.send_response(200);self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
        def do_GET(self):
            self.send({'model_path':args.model,'modalities': {'vision': False}, 'default_generation_settings':{'n_ctx':32768}})
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
            if body.get('stream'):
                with open(os.environ['CLI_REQUEST_MARKER'],'a') as f:f.write('request\n')
                time.sleep(60)
            if self.path=='/apply-template':self.send({'prompt':'test'})
            elif self.path=='/tokenize':self.send({'tokens':[1,2]})
            else:self.send({'success':True,'content':'','stop':True})
    http.server.ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()
    sys.exit()

binary=Path(sys.argv[1]).resolve(); source=Path(sys.argv[2]).resolve()
with tempfile.TemporaryDirectory(prefix='millie-two-clis-') as tmp:
    root=Path(tmp);model=root/'model.gguf';model.touch();marker=root/'requests'
    launcher=root/'server';launcher.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(Path(__file__).resolve()))+' --server "$@"\n');launcher.chmod(0o755)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env={k:v for k,v in os.environ.items() if not k.startswith('MILLIE_')}
    env.update(MILLIE_LLAMACPP_MODEL_PATH=str(model),MILLIE_LLAMACPP_SERVER_BIN=str(launcher),MILLIE_LLAMACPP_PORT=str(port),MILLIE_LLAMACPP_GPU='none',MILLIE_LLAMACPP_VISION='false',CLI_REQUEST_MARKER=str(marker))
    processes=[];logs=[]
    def alive():
        with socket.socket() as sock:sock.settimeout(.1);return sock.connect_ex(('127.0.0.1',port))==0
    def start(name):
        home=root/name;home.mkdir();(home/'millie-native.jinja').write_text('{{ messages }}');shutil.copy2(source/'codex-rs/models-manager/millie-models.json',home/'millie-models.json');shutil.copy2(source/'prompts/system_prompt.md',home/'system_prompt.md')
        slug=next(iter(json.loads((home/'millie-models.json').read_text())['downloads']))
        (home/'config.toml').write_text(f'[llamacpp]\nvision = false\ntemperature = 1.0\ntop_p = 0.95\ntop_k = 64\nmin_p = 0.0\n')
        log=(root/(name+'.log')).open('w+');logs.append(log)
        p=subprocess.Popen([str(binary),'exec','--skip-git-repo-check','--model',slug,'--sandbox','read-only','Reply with ok.'],env=dict(env,MILLIE_HOME=str(home),TMPDIR=str(home)),cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log)
        processes.append(p);return p
    def await_requests(count):
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            if marker.exists() and len(marker.read_text().splitlines())>=count:return
            assert all(p.poll() is None for p in processes), 'CLI exited before request'
            time.sleep(.1)
        raise AssertionError('CLI did not reach mock generation')
    try:
        owner=start('owner');await_requests(1)
        borrower=start('borrower');await_requests(2)
        borrower.kill();borrower.wait(timeout=5);time.sleep(.3);assert alive()
        owner.kill();owner.wait(timeout=5)
        deadline=time.monotonic()+10
        while alive() and time.monotonic()<deadline:time.sleep(.1)
        assert not alive(),'Actual owner CLI exit did not release server'
        print('PASS: two ordinary CLI processes with separate homes and TMPDIR values attach; borrower SIGKILL keeps server; owner SIGKILL releases it')
    except Exception:
        for log in logs:log.flush();log.seek(0);print(log.read(),file=sys.stderr)
        raise
    finally:
        for p in processes:
            if p.poll() is None:p.kill();p.wait(timeout=5)
        for log in logs:log.close()
