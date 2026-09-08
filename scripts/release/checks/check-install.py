#!/usr/bin/env python3
"""Tiny localhost install fixtures. No real weights or inference."""
import pty, select, shutil, argparse, hashlib, http.server, json, os, shlex, socket, subprocess, sys, tempfile, threading, time
from pathlib import Path

def serve():
    if '--list-devices' in sys.argv:return
    parser=argparse.ArgumentParser();parser.add_argument('--model');parser.add_argument('--mmproj');parser.add_argument('--port',type=int)
    args,_=parser.parse_known_args(sys.argv[2:])
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def send(self,obj,sse=False):
            b=obj.encode() if sse else json.dumps(obj).encode();self.send_response(200);self.send_header('Content-Length',str(len(b)));self.send_header('Content-Type','text/event-stream' if sse else 'application/json');self.end_headers();self.wfile.write(b)
        def do_GET(self):self.send({'model_path':args.model,'modalities':{'vision':bool(args.mmproj)},'default_generation_settings':{'n_ctx':32768}})
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
            if body.get('stream'):
                self.send('data: '+json.dumps({'choices':[{'delta':{'content':'ok'},'finish_reason':'stop'}]})+'\n\ndata: [DONE]\n\n',True)
            elif self.path=='/tokenize':self.send({'tokens':[1,2]})
            elif self.path=='/apply-template':self.send({'prompt':'fixture'})
            elif 'slots/' in self.path:self.send({'filename':'fixture','n_saved':1,'n_written':1})
            else:self.send({'content':'','stop':True})
    http.server.ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if '--server' in sys.argv:
    serve();sys.exit(0)
binary,source=(Path(p).resolve() for p in sys.argv[1:3])
with tempfile.TemporaryDirectory(prefix='millie-initial-check-') as tmp:
    home=Path(tmp).resolve();catalog=json.loads((source/'codex-rs/models-manager/millie-models.json').read_text());slug=next(iter(catalog['downloads']));entry=catalog['downloads'][slug]
    entry['repo']='fixture/model';entry['model_file']='model.gguf';entry['mmproj_file']='vision.gguf';entry['model_sha256']='0'*64;entry['mmproj_sha256']='0'*64
    (home/'millie-models.json').write_text(json.dumps(catalog));(home/'system_prompt.md').write_bytes((source/'prompts/system_prompt.md').read_bytes());(home/'millie-native.jinja').write_text('{{ messages }}')
    revision=['a'*40];calls=[];payloads={'model.gguf':b'fresh model','vision.gguf':b'matching tower'}
    class HF(http.server.BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_HEAD(self):
            calls.append(('HEAD',self.path));parts=self.path.split('/');rev=revision[0] if parts[-2]=='main' else parts[-2];data=payloads[parts[-1]]
            self.send_response(302);self.send_header('x-repo-commit',rev);self.send_header('x-linked-etag',hashlib.sha256(data).hexdigest());self.send_header('x-linked-size',str(len(data)));self.end_headers()
        def do_GET(self):
            calls.append(('GET',self.path));assert self.path.split('/')[-2]=='a'*40,self.path
            data=payloads[self.path.split('/')[-1]];self.send_response(200);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    hf=http.server.ThreadingHTTPServer(('127.0.0.1',0),HF);thread=threading.Thread(target=hf.serve_forever,daemon=True);thread.start()
    launcher=home/'server';launcher.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(Path(__file__).resolve()))+' --server "$@"\n');launcher.chmod(0o755)
    env={k:v for k,v in os.environ.items() if not k.startswith('MILLIE_')};env.update(MILLIE_HOME=str(home),HF_ENDPOINT=f'http://127.0.0.1:{hf.server_port}',MILLIE_LLAMACPP_SERVER_BIN=str(launcher),MILLIE_LLAMACPP_GPU='none',MILLIE_LLAMACPP_STARTUP_TIMEOUT_SECS='4',OMP_NUM_THREADS='1')
    def run(*flags):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        result=subprocess.run([str(binary),'exec','--skip-git-repo-check','--model',slug,'-c',f'llamacpp.port={port}',*flags,'Reply ok'],cwd=home,env=env,stdin=subprocess.DEVNULL,text=True,capture_output=True,timeout=40)
        time.sleep(.3)
        return result
    def decline_interactively(*flags):
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        master,slave=pty.openpty()
        proc=subprocess.Popen([str(binary),'exec','--skip-git-repo-check','--model',slug,'-c',f'llamacpp.port={port}',*flags,'Reply ok'],cwd=home,env=env,stdin=slave,stdout=slave,stderr=slave)
        os.close(slave);output=b'';deadline=time.monotonic()+15;answered=False
        try:
            while time.monotonic()<deadline:
                ready,_,_=select.select([master],[],[],.1)
                if ready:
                    try:chunk=os.read(master,65536)
                    except OSError:break
                    if not chunk:break
                    output+=chunk
                    if b'[y/N]' in output and not answered:os.write(master,b'n\n');answered=True
                if proc.poll() is not None:break
            proc.wait(timeout=3)
            assert answered and proc.returncode!=0,output.decode(errors='replace')
            return output.decode(errors='replace')
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            os.close(master)
    try:
        denied=run('--no-vision');assert denied.returncode and 'Download not approved' in denied.stderr,denied.stderr;assert all(m=='HEAD' for m,p in calls),calls
        prompt=decline_interactively('--no-vision');assert 'model.gguf (0.00 GB)' in prompt and 'millie --model select' in prompt,prompt
        assert not any(m=='GET' for m,p in calls),calls
        result=run('--no-vision','--download');assert result.returncode==0,result.stderr
        assert [p for m,p in calls if m=='GET']==['/fixture/model/resolve/'+'a'*40+'/model.gguf'],calls
        manifests=list((home/'models/updates').glob('*/active.json'));assert len(manifests)==1
        before=manifests[0].read_bytes();saved=json.loads(before);assert len(saved['files'])==2
        print('PASS: initial download uses current HF checksum despite stale catalog; text-only skips tower')
        revision[0]='b'*40;head_count=sum(m=='HEAD' for m,p in calls)
        call_start=len(calls)
        prompt=decline_interactively('--vision');listing=prompt.split('Missing files for',1)[1].split('To choose another',1)[0]
        assert 'vision.gguf (0.00 GB)' in listing and 'model.gguf' not in listing,listing
        assert all('/resolve/'+'a'*40+'/' in p for m,p in calls[call_start:] if m=='HEAD'),calls
        head_count=sum(m=='HEAD' for m,p in calls)
        print('PASS: interactive prompts list only missing files and sizes; pinned tower metadata follows installed revision')
        result=run('--vision','--download');assert result.returncode==0,result.stderr
        assert sum(m=='HEAD' for m,p in calls)==head_count,calls
        assert [p for m,p in calls if m=='GET']==['/fixture/model/resolve/'+'a'*40+'/'+f for f in ['model.gguf','vision.gguf']],calls
        assert manifests[0].read_bytes()==before
        print('PASS: adding vision after HF main changes fetches matching pinned tower only')
        count=len(calls);result=run('--vision');assert result.returncode==0,result.stderr;assert len(calls)==count,calls
        print('PASS: installed model and vision restart without metadata checks or transfers')
        result=run('--vision','-c','llamacpp.profile="misspelled"');assert result.returncode and 'Unknown serving profile' in result.stderr,result.stderr;assert len(calls)==count
        print('PASS: explicit invalid serving profile refuses without download or startup')
        # Existing test-build files need no checksum match, copy, or network access.
        shutil.rmtree(home/'models')
        models=home/'models';models.mkdir()
        model=models/'model.gguf';tower=models/'vision.gguf'
        model.write_bytes(b'older installed model');tower.write_bytes(b'older installed tower')
        original=(model.stat().st_ino,model.stat().st_size,model.stat().st_mtime_ns)
        count=len(calls)
        result=run('--vision');assert result.returncode==0,result.stderr
        assert len(calls)==count,calls
        records=list(models.glob('updates/*/local.json'));assert len(records)==1
        record=json.loads(records[0].read_text())
        assert record==[{'file':'model.gguf','path':str(model)},{'file':'vision.gguf','path':str(tower)}],record
        assert not list(models.glob('huggingface/*/model.gguf'))
        assert original==(model.stat().st_ino,model.stat().st_size,model.stat().st_mtime_ns)
        result=run('--vision');assert result.returncode==0,result.stderr;assert len(calls)==count
        print('PASS: different-checksum existing model/tower reused at original paths, offline, across launches')
        tower.unlink()
        result=run('--vision','--download');assert result.returncode and 'no recorded remote revision' in result.stderr,result.stderr
        assert len(calls)==count
        result=run('--no-vision');assert result.returncode==0,result.stderr;assert len(calls)==count
        tower.write_bytes(b'older installed tower')
        print('PASS: missing unversioned tower cannot trigger implicit model replacement or revision mixing')
        saved_min=entry['sampling'].pop('min_p');(home/'millie-models.json').write_text(json.dumps(catalog))
        result=run('--vision');assert result.returncode and 'Missing or invalid sampling setting `min_p`' in result.stderr,result.stderr
        result=run('--vision','-c','llamacpp.min_p=0.0');assert result.returncode==0,result.stderr
        entry['sampling']['min_p']=saved_min;(home/'millie-models.json').write_text(json.dumps(catalog))
        result=run('--vision','-c','llamacpp.temperature=-1.0');assert result.returncode and 'Missing or invalid sampling setting `temperature`' in result.stderr,result.stderr
        assert len(calls)==count
        print('PASS: managed sampling must be complete/valid; explicit min_p zero fills a missing catalog setting')
        revision[0]='a'*40
        result=subprocess.run([str(binary),'models','update',slug],cwd=home,env=env,stdin=subprocess.DEVNULL,text=True,capture_output=True,timeout=40)
        assert result.returncode==0,result.stderr
        assert model.read_bytes()==b'older installed model' and tower.read_bytes()==b'older installed tower'
        count=len(calls);result=run('--vision');assert result.returncode==0,result.stderr;assert len(calls)==count
        print('PASS: explicit update activates verified revision while preserving original installed files')
    finally:hf.shutdown();hf.server_close();thread.join()
