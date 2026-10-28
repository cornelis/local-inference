#!/usr/bin/env python3
"""Speed matrix on one host: identical request protocol per arm, one server at a time.

Arms:
  lfm3-qad   LFM3-1.7B QAD-Q4_0   (stock llama.cpp v0.8.0, Metal)
  lfm3-q4    LFM3-1.7B Q4_K_M     (stock llama.cpp v0.8.0, Metal)
  lfm3-q8    LFM3-1.7B Q8_0       (stock llama.cpp v0.8.0, Metal)
  frog       FrogNano-4B-2609 Q4_K_M (production flags, n-gram off)

Protocol (llama.cpp-native /completion, no chat template, no speculation):
  * prefill: N-token filler prompt, n_predict=1, cache_prompt=false -> prompt_per_second
  * decode : N-token filler prompt, n_predict=128, ignore_eos=true, cache_prompt=false -> predicted_per_second
  * chat   : 8k-token agent-shaped prompt via /v1/chat/completions, streaming -> TTFT, then decode t/s

Every arm is launched on the same port, measured, and torn down, so only one model is resident.
This competes with production for GPU memory/time: run it when the serving path can be idle.

Env:
  LAB_ROOT   directory holding inference-ab-001/ (Frog launcher) and lfm3/
  LFM3_ROOT  LFM3 models and build (default $LAB_ROOT/lfm3)
  FROG_CMD   overrides the Frog launch command (default: sh serve.sh in LAB)

    LAB_ROOT=/path/to/lab python3 measure.py --arm lfm3-qad --ctx 32768
    LAB_ROOT=/path/to/lab python3 measure.py --arm frog --ctx 32768
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

LAB_ROOT = Path(os.environ["LAB_ROOT"]).resolve()
ROOT = Path(os.environ.get("LFM3_ROOT", LAB_ROOT / "lfm3"))
LAB = LAB_ROOT / "inference-ab-001"
FROG_CMD = os.environ.get("FROG_CMD")
MODELS = {
    "lfm3-qad": ROOT / "models/LFM3-1.7B-QAD-Q4_0.gguf",
    "lfm3-q4": ROOT / "models/LFM3-1.7B-Q4_K_M.gguf",
    "lfm3-q8": ROOT / "models/LFM3-1.7B-Q8_0.gguf",
}
DECODE_DEPTHS = [0, 4096, 16384, 32768]
PREFILL_SIZES = [1024, 4096, 16384, 32768]


def rss(pid: int) -> int:
    try:
        return int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).split()[0]) * 1024
    except Exception:
        return 0


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def wait_ready(port: int, proc: subprocess.Popen, start: float, timeout: float = 300) -> float:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError("server exited with %s" % proc.returncode)
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2) as r:
                if r.status == 200:
                    return time.monotonic() - start
        except Exception:
            pass
        time.sleep(0.25)
    raise RuntimeError("server did not become ready")


def post(port: int, path: str, payload: dict, timeout: int = 1800):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (port, path),
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def complete(port, depth, n_predict, cache_prompt=False):
    payload = {"prompt": "word " * depth, "n_predict": n_predict, "temperature": 0,
               "stream": False, "cache_prompt": cache_prompt}
    if n_predict > 1:
        payload["ignore_eos"] = True
    body = post(port, "/completion", payload)
    t = body.get("timings") or {}
    return {"depth": depth, "prompt_n": t.get("prompt_n"), "prompt_ms": t.get("prompt_ms"),
            "prefill_tps": round(t.get("prompt_per_second") or 0, 1),
            "predicted_n": t.get("predicted_n"), "decode_tps": round(t.get("predicted_per_second") or 0, 1),
            "prefill_s": round((t.get("prompt_ms") or 0) / 1000, 3)}


def stream_chat(port, prompt, max_tokens=128):
    """Streaming chat: first-token latency and steady-state decode rate on the real API shape."""
    payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
               "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 json.dumps(payload).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = None
    n = 0
    usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            usage = obj.get("usage") or usage
            delta = (obj.get("choices") or [{}])[0].get("delta") or {}
            text = delta.get("content") or delta.get("reasoning_content") or ""
            if text:
                if first is None:
                    first = time.monotonic() - t0
                n += 1
    total = time.monotonic() - t0
    gen = (usage or {}).get("completion_tokens") or n
    return {"prompt_n": (usage or {}).get("prompt_tokens"), "gen_n": gen,
            "ttft_s": round(first or 0, 3), "wall_s": round(total, 2),
            "decode_tps": round((gen - 1) / (total - first), 1) if first and total > first and gen > 1 else None}


def launch(arm: str, port: int, ctx: int, ubatch: int | None, threads: int, log: Path,
           flags: str | None = None):
    if arm == "frog":
        env = dict(os.environ, PORT=str(port), CTX=str(ctx), SPEC="none", THREADS=str(threads),
                   EXTRA_FLAGS=flags or "")
        cmd = ["sh", "-c", FROG_CMD] if FROG_CMD else ["sh", "serve.sh"]
        cwd = str(LAB)
    else:
        cmd = [str(ROOT / "build/bin/llama-server"), "-m", str(MODELS[arm]), "-c", str(ctx),
               "-ngl", "999", "-fa", "on", "--host", "127.0.0.1", "--port", str(port),
               "-t", str(threads), "--jinja"]
        if ubatch:
            cmd += ["-b", str(max(2048, ubatch)), "-ub", str(ubatch)]
        if flags:
            cmd += shlex.split(flags)
        env = dict(os.environ)
        cwd = str(ROOT)
    if not port_free(port):
        raise RuntimeError("port %d already in use" % port)
    fh = open(log, "w")
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, fh


def stop(proc, fh):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
    fh.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(MODELS) + ["frog"])
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--ctx", type=int, default=40960)
    ap.add_argument("--ubatch", type=int, default=None)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--flags", default=None, help="extra llama-server flags (spec decoding etc.)")
    ap.add_argument("--sizes", default=None, help="comma list, overrides the prefill depths")
    ap.add_argument("--depths", default=None, help="comma list, overrides the decode depths")
    ap.add_argument("--skip-decode", action="store_true")
    ap.add_argument("--skip-prefill", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = {"arm": a.arm, "ctx": a.ctx, "ubatch": a.ubatch, "threads": a.threads,
           "flags": a.flags,
           "llama": str(ROOT / "build/bin/llama-server") if a.arm != "frog" else "frog-fork",
           "model": str(MODELS.get(a.arm, "bartowski/frognano-4b-2609-gguf:Q4_K_M")),
           "rows": []}
    log = Path("/tmp/lfm3-speed-%s.log" % a.arm)
    proc, fh = launch(a.arm, a.port, a.ctx, a.ubatch, a.threads, log, a.flags)
    started = time.monotonic()
    try:
        out["load_s"] = round(wait_ready(a.port, proc, started), 2)
        out["rss_load"] = rss(proc.pid)
        complete(a.port, 8, 8)  # warmup: compile Metal kernels, fill caches
        out["rss_warm"] = rss(proc.pid)
        def probe(kind, depth, n_predict):
            try:
                r = complete(a.port, depth, n_predict)
            except Exception as exc:
                r = {"kind": kind, "depth": depth, "error": "%s: %s" % (type(exc).__name__, exc)}
            r["kind"] = kind
            out["rows"].append(r)
            if "error" in r:
                print("%-7s %-8s %6d  ERROR %s" % (kind, a.arm, depth, r["error"]), flush=True)
            else:
                print("%-7s %-8s %6d  %7.1f tok/s" % (kind, a.arm, r["prompt_n"] or 0,
                                                      r["decode_tps"] if kind == "decode" else r["prefill_tps"]),
                      flush=True)
            return r

        # the filler adds one space per word; keep the prompt plus the reply inside the context
        budget = a.ctx - 256
        sizes = [int(x) for x in a.sizes.split(",")] if a.sizes else PREFILL_SIZES
        depths = [int(x) for x in a.depths.split(",")] if a.depths else DECODE_DEPTHS
        if not a.skip_prefill:
            for s in sizes:
                if s <= budget:
                    probe("prefill", s, 1)
        if not a.skip_decode:
            for d in depths:
                if d <= budget:
                    probe("decode", d, 128)
        text = " ".join(["def merge(intervals):"] * 2000)
        r = stream_chat(a.port, "Refactor this code and explain the change.\n\n" + text)
        out["chat"] = r
        print("chat    %-8s %6s  ttft %.2fs  %s tok/s" % (a.arm, r["prompt_n"], r["ttft_s"], r["decode_tps"]),
              flush=True)
        out["rss_end"] = rss(proc.pid)
    finally:
        stop(proc, fh)

    dest = Path(a.out) if a.out else Path("/tmp/lfm3-speed-%s.json" % a.arm)
    dest.write_text(json.dumps(out, indent=1))
    print("WROTE", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
