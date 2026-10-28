#!/usr/bin/env python3
"""Measure llama-server prompt/decode timings and load RSS for the test models.

Runs one server at a time on port 18086. No speculative decoding.
"""
import json, os, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path

ROOT = Path(os.environ["LAB_ROOT"]).resolve()
LAB = ROOT / "inference-ab-001"
LLAMA = LAB / "native/build/bin/llama-server"
FLINT_LLAMA = ROOT / "flint-test-001/llama.cpp/build/bin/llama-server"
FLINT_MODEL = ROOT / "flint-test-001/model/flint-preview-UD-Q2_K_XL.gguf"
FLINT_TPL = ROOT / "flint-test-001/flint-chat-template.jinja"
BASE = ROOT / "measure-001"

FROG_CMD = os.environ.get(
    "FROG_CMD", f"cd {LAB} && PORT=18086 CTX=131072 SPEC=none exec sh serve.sh")

CONFIGS = [
    ("frog", "sh", ["-c", FROG_CMD]),
    ("qwen36", str(LLAMA), ["-m", str(BASE/"qwen36/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"), "-c", "131072", "-ngl", "999", "--host", "127.0.0.1", "--port", "18086", "--threads", "5", "--jinja"]),
    ("sage", str(LLAMA), ["-m", str(BASE/"sage/Sage-A1.5.Q3_K_L.gguf"), "-c", "131072", "-ngl", "999", "--host", "127.0.0.1", "--port", "18086", "--threads", "5", "--jinja"]),
    ("lfm", str(LLAMA), ["-m", str(BASE/"lfm/LFM2.5-24B-A2B-UD-Q4_K_XL.gguf"), "-c", "131072", "-ngl", "999", "--host", "127.0.0.1", "--port", "18086", "--threads", "5", "--jinja"]),
    ("flint", str(FLINT_LLAMA), ["-m", str(FLINT_MODEL), "-c", "131072", "-ngl", "999", "--host", "127.0.0.1", "--port", "18086", "--threads", "5", "--jinja", "--chat-template-file", str(FLINT_TPL)]),
]
PREFILL = [512, 4096, 16384, 32768, 65536, 128000]
DECODE = [0, 4096, 32768]

def port_open(port=18086):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def wait_port(timeout=600):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with socket.create_connection(("127.0.0.1", 18086), 1):
                return True
        except OSError:
            time.sleep(1)
    return False

def wait_ready(timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen("http://127.0.0.1:18086/v1/models", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def rss(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return int(out.split()[0]) * 1024 if out else 0
    except Exception:
        return 0

def complete(n_tokens):
    """Exact-128-token decode on the raw completion endpoint, matching measure_decode."""
    payload = {"prompt": "word " * n_tokens, "n_predict": 128, "ignore_eos": True,
               "temperature": 0, "stream": False, "cache_prompt": False}
    req = urllib.request.Request("http://127.0.0.1:18086/completion",
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read())
    return body, round(time.monotonic() - t, 3)


def request(model, n_tokens, max_tokens, cache_prompt=False):
    text = ("word " * n_tokens) + "\nReply with exactly OK."
    payload = {"model": model, "messages": [{"role": "user", "content": text}],
               "temperature": 0, "max_tokens": max_tokens, "stream": False,
               "cache_prompt": cache_prompt,
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:18086/v1/chat/completions",
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read())
    return body, round(time.monotonic() - t, 3)

def main():
    results = {}
    for name, exe, args in CONFIGS:
        if name != "frog" and ("/measure-001/" in exe or "measure-001" in " ".join(args)) and not Path(args[1]).exists():
            print("SKIP missing", name, args[1], flush=True); continue
        if "--chat-template-file" in args and not Path(args[args.index("--chat-template-file") + 1]).exists():
            print("SKIP missing template", name, flush=True); continue
        if port_open():
            results[name] = {"error": "port 18086 already in use"}
            print("SKIP port busy", name, flush=True)
            continue
        print("START", name, flush=True)
        t0 = time.monotonic()
        p = subprocess.Popen([exe] + args, cwd=str(LAB), stdout=open(f"/tmp/measure-{name}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if not wait_port() or not wait_ready():
                results[name] = {"error": "server timeout"}; print("TIMEOUT", name, flush=True); continue
            load_s = round(time.monotonic() - t0, 2)
            try:
                models = json.loads(urllib.request.urlopen("http://127.0.0.1:18086/v1/models", timeout=30).read())
                model = models["data"][0]["id"]
            except Exception:
                model = name
            request(model, 32, 1)  # warmup
            rows = {"load_s": load_s, "rss_after_load": rss(p.pid), "prefill": [], "decode": []}
            for n in PREFILL:
                try:
                    body, wall = request(model, n, 1)
                except Exception as exc:
                    rows["prefill"].append({"target": n, "error": f"{type(exc).__name__}: {exc}"})
                    print(name, "pp", n, rows["prefill"][-1], flush=True)
                    continue
                tim = body.get("timings") or {}
                rows["prefill"].append({"target": n, "prompt_n": tim.get("prompt_n"), "prompt_ms": tim.get("prompt_ms"),
                                        "prefill_tps": tim.get("prompt_per_second"), "wall_s": wall,
                                        "rss": rss(p.pid), "error": body.get("error")})
                print(name, "pp", n, rows["prefill"][-1], flush=True)
            for n in DECODE:
                try:
                    body, wall = complete(n)
                except Exception as exc:
                    rows["decode"].append({"depth": n, "error": f"{type(exc).__name__}: {exc}"})
                    print(name, "tg", n, rows["decode"][-1], flush=True)
                    continue
                tim = body.get("timings") or {}
                rows["decode"].append({"depth": n, "prompt_n": tim.get("prompt_n"), "prompt_ms": tim.get("prompt_ms"),
                                       "predicted_n": tim.get("predicted_n"), "predicted_ms": tim.get("predicted_ms"),
                                       "decode_tps": tim.get("predicted_per_second"), "wall_s": wall,
                                       "rss": rss(p.pid), "error": body.get("error")})
                print(name, "tg", n, rows["decode"][-1], flush=True)
            results[name] = rows
        except Exception as exc:
            results[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print("ERR", name, results[name], flush=True)
        finally:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait(timeout=10)
            except Exception:
                try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception: pass
            time.sleep(2)
    print("RESULT", json.dumps(results, indent=2), flush=True)
    Path("/tmp/measure-result.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
