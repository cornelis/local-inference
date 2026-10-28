#!/usr/bin/env python3
"""Qwen3.6 serving levers: MTP depth, flash attention, ubatch, KV q8, IQ3, ngram+MTP."""
import json, os, signal, socket, subprocess, time, urllib.request
from pathlib import Path

ROOT = Path(os.environ["LAB_ROOT"]).resolve()
BASE = ROOT / "qwen-levers-001"
LLAMA = ROOT / "inference-ab-001/native/build/bin/llama-server"
Q3 = BASE / "q3/Qwen3.6-35B-A3B-UD-Q3_K_M.gguf"
IQ3 = BASE / "iq3/Qwen3.6-35B-A3B-UD-IQ3_S.gguf"

def args(model, extra):
    return [str(LLAMA), "-m", str(model), "-c", "131072", "-ngl", "999",
            "--host", "127.0.0.1", "--port", "18086", "--threads", "5", "--jinja", "-np", "1"] + extra

MTP2 = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
CONFIGS = [("no-mtp", args(Q3, []))]
for n in (1, 2, 3):
    CONFIGS.append((f"mtp-n{n}", args(Q3, ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n)])))
CONFIGS += [
    ("mtp-n2-fa", args(Q3, MTP2 + ["-fa", "on"])),
    ("mtp-n2-ub1024", args(Q3, MTP2 + ["-ub", "1024"])),
    ("mtp-n2-ub2048", args(Q3, MTP2 + ["-ub", "2048"])),
    ("mtp-n2-ub4096", args(Q3, MTP2 + ["-ub", "4096"])),
    ("mtp-n2-kvq8", args(Q3, MTP2 + ["-ctk", "q8_0", "-ctv", "q8_0"])),
    ("mtp-iq3", args(IQ3, ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"])),
    ("mtp-n2-ngram", args(Q3, ["--spec-type", "draft-mtp,ngram-simple", "--spec-draft-n-max", "2",
                                "--spec-ngram-simple-size-n", "12", "--spec-ngram-simple-size-m", "48"])),
]

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
    payload = {"prompt": "word " * depth, "n_predict": 128, "ignore_eos": True,
               "temperature": 0, "stream": False, "cache_prompt": False}
    req = urllib.request.Request("http://127.0.0.1:18086/completion", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    t = time.monotonic()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        body = json.loads(resp.read())
    return body, round(time.monotonic() - t, 3)

def main():
    out = {}
    for name, cmd in CONFIGS:
        if port_open():
            out[name] = {"error": "port 18086 already in use"}
            print("SKIP port busy", name, flush=True)
            continue
        if not Path(cmd[2]).exists():
            out[name] = {"error": "missing %s" % cmd[2]}
            print("SKIP missing", name, cmd[2], flush=True)
            continue
        print("START", name, flush=True)
        p = subprocess.Popen(cmd, cwd=str(ROOT / "inference-ab-001"),
                             stdout=open(f"/tmp/qlever-{name}.log", "w"), stderr=subprocess.STDOUT,
                             start_new_session=True)
        try:
            if not wait_ready():
                out[name] = {"error": "timeout"}; print("TIMEOUT", name, flush=True); continue
            # warmup
            complete(8)
            row = {"load_rss": rss(p.pid), "rows": []}
            for d in (4096, 32768):
                body, wall = complete(d)
                tim = body.get("timings") or {}
                row["rows"].append({"depth": d, "prompt_n": tim.get("prompt_n"), "prompt_ms": tim.get("prompt_ms"),
                                    "prefill_tps": tim.get("prompt_per_second"),
                                    "predicted_n": tim.get("predicted_n"), "predicted_ms": tim.get("predicted_ms"),
                                    "decode_tps": tim.get("predicted_per_second"),
                                    "draft_n": tim.get("draft_n"), "draft_n_accepted": tim.get("draft_n_accepted"),
                                    "wall_s": wall, "rss": rss(p.pid), "error": body.get("error")})
                print(name, row["rows"][-1], flush=True)
            out[name] = row
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
    Path("/tmp/qwen-levers-result.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
