#!/usr/bin/env python3
"""CLI update/config regression checks: tiny local HTTP fixtures only, no inference."""
import hashlib,http.server,json,os,subprocess,sys,tempfile,threading
from pathlib import Path
binary=Path(sys.argv[1]).resolve(); source=Path(sys.argv[2]).resolve()
with tempfile.TemporaryDirectory(prefix='millie-updates-') as tmp:
 home=Path(tmp); (home/'models').mkdir()
 catalog=json.loads((source/'codex-rs/models-manager/millie-models.json').read_text());slug=next(iter(catalog['downloads']));entry=catalog['downloads'][slug];repo=entry['repo'];file=entry['model_file'];tower=entry.get('mmproj_file')
 (home/'millie-models.json').write_text(json.dumps(catalog));(home/'system_prompt.md').write_bytes((source/'prompts/system_prompt.md').read_bytes())
 (home/'models'/file).write_bytes(b'old model')
 if tower:(home/'models'/tower).write_bytes(b'old vision')
 data={file:b'new model',tower:b'new vision'} if tower else {file:b'new model'}
 requests=[];fail=[False];inference=[];commit='a'*40
 key=hashlib.sha256((repo+'\0'+file).encode()).hexdigest();update_dir=home/'models/updates'/key
 class Handler(http.server.BaseHTTPRequestHandler):
  def log_message(self,*args):pass
  def send(self,body,status=200,content_type='application/json'):
   if not isinstance(body,bytes):body=json.dumps(body).encode()
   self.send_response(status);self.send_header('Content-Length',str(len(body)));self.send_header('Content-Type',content_type);self.end_headers();self.wfile.write(body)
  def do_HEAD(self):
   requests.append(('HEAD',self.path));name=self.path.split('/')[-1];assert name in data,self.path
   self.send_response(302);self.send_header('x-repo-commit',commit);self.send_header('x-linked-etag',hashlib.sha256(data[name]).hexdigest());self.send_header('x-linked-size',str(len(data[name])));self.end_headers()
  def do_GET(self):
   requests.append(('GET',self.path))
   if '/resolve/' in self.path:
    assert '/'+commit+'/' in self.path,self.path
    if fail[0]:self.send(b'error',503)
    else:self.send(data[self.path.split('/')[-1]])
   else:
    active=json.loads((update_dir/'active.json').read_text());path=update_dir/'files'/active['files'][0]['sha256']
    self.send({'model_path':str(path),'modalities': {'vision': False}, 'default_generation_settings':{'n_ctx':32768}})
  def do_POST(self):
   body=json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))))
   if self.path=='/tokenize':self.send({'tokens':[1,2]})
   elif self.path=='/v1/chat/completions':
    inference.append(body);self.send(('data: '+json.dumps({'choices':[{'delta':{'content':'ok'},'finish_reason':'stop'}]})+'\n\ndata: [DONE]\n\n').encode(),content_type='text/event-stream')
   else:self.send({'prompt':'test','content':'','stop':True})
 server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
 (home/'config.toml').write_text(f'model = "{slug}"\n[llamacpp]\nvision = false\ngpu = []\nport = {server.server_port}\n')
 env={k:v for k,v in os.environ.items() if not k.startswith('MILLIE_')};env.update(MILLIE_HOME=str(home),HF_ENDPOINT=f'http://127.0.0.1:{server.server_port}',MILLIE_LLAMACPP_GPU='none',NO_PROXY='127.0.0.1,localhost',no_proxy='127.0.0.1,localhost')
 for k in ['HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy']:env[k]='http://127.0.0.1:9'
 def run(*args):return subprocess.run([str(binary),*args],env=env,cwd=home,stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=45)
 def ok(*args):
  result=run(*args);assert result.returncode==0,(args,result.stdout,result.stderr);return result
 try:
  ok('models','check-updates');assert all(method=='HEAD' for method,_ in requests);assert not update_dir.exists()
  print('PASS: check-updates performs only metadata requests and no cache writes')
  ok('models','update');oldmanifest=(update_dir/'active.json').read_bytes();assert (home/'models'/file).read_bytes()==b'old model'
  for name in data:assert (update_dir/'files'/hashlib.sha256(data[name]).hexdigest()).read_bytes()==data[name]
  print('PASS: update verifies complete file set and preserves previous model')
  requests.clear();ok('models','update');assert all(method=='HEAD' for method,_ in requests)
  print('PASS: repeated update performs no weight transfer')
  ok('exec','--skip-git-repo-check','--sandbox','read-only','Reply with ok.');assert inference
  expected=entry['sampling']
  for request in inference:
   for k in ['temperature','top_p','top_k','min_p']:assert abs(request[k]-expected[k])<1e-6,(k,request[k],expected[k])
  print('PASS: ordinary startup selects updated file and preserves catalog sampling including min_p=0')
  data[file]=b'newer model';fail[0]=True
  result=run('models','update');assert result.returncode and (update_dir/'active.json').read_bytes()==oldmanifest
  print('PASS: failed update leaves active revision unchanged')
  result=run('exec','--local-provider','vllm','--model','typo-model','--skip-git-repo-check','x');assert result.returncode and 'Unknown model' in result.stderr,result.stderr
  result=run('-c','llamacpp.temperture=1.0','models','check-updates');assert result.returncode and 'temperture' in result.stderr,result.stderr
  print('PASS: invalid vLLM model and misspelled sampling override fail clearly')
  (home/'work.config.toml').write_text(f'model = "{slug}"\n[llamacpp]\nvision = false\n')
  ok('--profile','work','models','check-updates')
  print('PASS: model update commands respect named profiles')
 finally:server.shutdown();server.server_close();thread.join()
