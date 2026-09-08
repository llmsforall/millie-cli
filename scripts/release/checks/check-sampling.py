#!/usr/bin/env python3
"""Check a built CLI's outgoing sampling values without loading any model.

Usage: python3 check-sampling.py /path/to/millie /path/to/millie-models.json
All config and placeholder model files are confined to a temporary directory.
"""
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading


def run_case(binary, catalog, expected, config_values=None, flag_values=None, transport="chat"):
    requests = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, value, content_type="application/json"):
            data = value.encode() if isinstance(value, str) else json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send({"status": "ok", "model_path": str(home / "models" / catalog["downloads"][slug]["model_file"]), "modalities": {"vision": False}, "default_generation_settings": {"n_ctx": 32768}})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path == "/tokenize":
                self.send({"tokens": [1, 2, 3]})
            elif self.path == "/apply-template":
                self.send({"prompt": "<|im_start|>user\ntest<|im_end|>\n<|im_start|>assistant\n<think>"})
            elif self.path == "/completion" and body.get("stream"):
                requests.append(body)
                event = {"content": "</think>ok", "stop": True, "stopped_eos": True,
                         "tokens_predicted": 4, "tokens_evaluated": 20}
                self.send("data: " + json.dumps(event) + "\n\n", "text/event-stream")
            elif self.path == "/v1/chat/completions" and body.get("stream"):
                requests.append(body)
                events = [
                    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24}},
                ]
                self.send("".join("data: " + json.dumps(e) + "\n\n" for e in events) + "data: [DONE]\n\n", "text/event-stream")
            else:
                self.send({"content": "", "stop": True})

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix="millie-sampling-") as tmp:
            root = Path(tmp)
            home = root / "home"
            (home / "models").mkdir(parents=True)
            slug = next(iter(catalog["downloads"]))
            for entry in catalog["downloads"].values():
                entry["model_sha256"] = hashlib.sha256(b"").hexdigest()
                entry["mmproj_sha256"] = hashlib.sha256(b"").hexdigest()
                (home / "models" / entry["model_file"]).touch()
                if entry.get("mmproj_file"):
                    (home / "models" / entry["mmproj_file"]).touch()
            (home / "millie-models.json").write_text(json.dumps(catalog))
            config = [f'model = "{slug}"', '[llamacpp]', f'port = {server.server_port}', 'vision = false']
            for key, value in (config_values or {}).items():
                config.append(f"{key} = {value}")
            if transport == "vllm":
                config += ['[vllm]', f'base_url = "http://127.0.0.1:{server.server_port}/v1/"']
            (home / "config.toml").write_text("\n".join(config) + "\n")
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith("MILLIE_") and k not in ("CODEX_HOME",)}
            env.update(MILLIE_HOME=str(home), MILLIE_LLAMACPP_GPU="none", GGML_VK_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
            if transport == "raw":
                env["MILLIE_LLAMACPP_TRANSPORT"] = "raw"
            args = [str(binary), "exec", "--local-provider", "vllm" if transport == "vllm" else "llamacpp",
                    "--model", slug, "--skip-git-repo-check", "--sandbox", "read-only"]
            for key, value in (flag_values or {}).items():
                args += ["-c", f"llamacpp.{key}={value}"]
            args += ["Reply with ok."]
            result = subprocess.run(args, cwd=root, env=env, text=True,
                                    stdin=subprocess.DEVNULL, capture_output=True, timeout=45)
            if result.returncode or not requests:
                raise AssertionError(f"CLI failed ({result.returncode}) or no request captured:\n{result.stderr}\n{result.stdout}")
            for request in requests:
                for key, value in expected.items():
                    actual = request.get(key)
                    if actual is None or abs(actual - value) > 1e-6:
                        raise AssertionError(f"{key}: expected {value}, got {actual}")
            print("PASS:", transport, expected)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def main():
    binary = Path(sys.argv[1]).resolve()
    for transport in ("chat", "raw", "vllm"):
        catalog = json.loads(Path(sys.argv[2]).read_text())
        keys = ("temperature", "top_p", "top_k", "min_p")
        defaults = next(iter(catalog["downloads"].values()))["sampling"]
        run_case(binary, catalog, {k: defaults[k] for k in keys}, transport=transport)
        changed = {"temperature": 0.73, "top_p": 0.83, "top_k": 23, "min_p": 0.012}
        for entry in catalog["downloads"].values():
            entry["model_sha256"] = hashlib.sha256(b"").hexdigest()
            entry["mmproj_sha256"] = hashlib.sha256(b"").hexdigest()
            entry["sampling"].update(changed)
        run_case(binary, catalog, changed, transport=transport)
        configured = {"temperature": 1.0, "top_p": 0.95, "top_k": 64, "min_p": 0.0}
        run_case(binary, catalog, configured, config_values=configured, transport=transport)
        flags = dict(configured, temperature=0.91)
        run_case(binary, catalog, flags, config_values=configured, flag_values={"temperature": 0.91}, transport=transport)
    print("PASS: bundled defaults, runtime catalog edit, config override, CLI override on both transports")


if __name__ == "__main__":
    main()
