#!/usr/bin/env python3
"""Decode-only llama-server timings with ignore_eos, exact 128 generated tokens."""
import json, os, signal, socket, subprocess, time, urllib.request
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
DEPTHS = [0, 4096, 32768]

def port_open(port=18086):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def wait_ready(timeout=900):
    end = time.time() + timeout
    while time.time() < end:
        try:
            with urllib.request.urlopen("http://127.0.0.1:18086/v1/models", timeout=3) as r:
                if r.status == 200: return True
        except Exception: pass
        time.sleep(2)
    return False

def rss(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return int(out.split()[0]) * 1024 if out else 0
    except Exception: return 0

def complete(depth):
    prompt = "word " * depth
    payload = {"prompt": prompt, "n_predict": 128, "ignore_eos": True,
               "temperature": 0, "stream": False, "cache_prompt": False}
    req = urllib.request.Request("http://127.0.0.1:18086/completion",
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read())
    return body, round(time.monotonic() - t, 3)

def main():
    out = {}
    for name, exe, args in CONFIGS:
        if port_open():
            out[name] = {"error": "port 18086 already in use"}
            print("SKIP port busy", name, flush=True)
            continue
        if exe != "sh" and not Path(args[1]).exists():
            out[name] = {"error": "missing %s" % args[1]}
            print("SKIP missing", name, args[1], flush=True)
            continue
        if "--chat-template-file" in args and not Path(args[args.index("--chat-template-file") + 1]).exists():
            out[name] = {"error": "missing chat template"}
            print("SKIP missing template", name, flush=True)
            continue
        print("START", name, flush=True)
        p = subprocess.Popen([exe] + args, cwd=str(LAB), stdout=open(f"/tmp/decode-{name}.log", "w"), stderr=subprocess.STDOUT, start_new_session=True)
        try:
            if not wait_ready():
                out[name] = {"error": "timeout"}; print("TIMEOUT", name, flush=True); continue
            complete(8)  # warmup
            rows = {"load_rss": rss(p.pid), "rows": []}
            for d in DEPTHS:
                try:
                    body, wall = complete(d)
                    tim = body.get("timings") or {}
                    rows["rows"].append({"depth": d, "prompt_n": tim.get("prompt_n"), "prompt_ms": tim.get("prompt_ms"),
                                         "predicted_n": tim.get("predicted_n"), "predicted_ms": tim.get("predicted_ms"),
                                         "decode_tps": tim.get("predicted_per_second"), "prefill_s": (tim.get("prompt_ms") or 0)/1000,
                                         "wall_s": wall, "rss": rss(p.pid), "error": body.get("error")})
                except Exception as exc:
                    rows["rows"].append({"depth": d, "error": f"{type(exc).__name__}: {exc}"})
                print(name, rows["rows"][-1], flush=True)
            out[name] = rows
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}; print("ERR", name, out[name], flush=True)
        finally:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM); p.wait(timeout=10)
            except Exception:
                try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception: pass
            time.sleep(2)
    print("RESULT", json.dumps(out, indent=2), flush=True)
    Path("/tmp/decode-result.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
