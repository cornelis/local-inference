"""OpenAI-compatible front door for a local llama.cpp server.

Policy: a reasoning model sometimes spends the whole output budget thinking and
returns no answer (finish_reason "length", empty content). When that happens and
the reasoning exceeds a character budget, the proxy retries once upstream with
the truncated reasoning injected as context and thinking disabled, so the answer
cannot be crowded out. Everything else is byte-for-byte passthrough.

Env:
  FROG_REASON_BUDGET  reasoning chars before the forced-answer retry fires
                      (default 16000, ~4k tokens; 0 disables the policy)
  FROG_UPSTREAM_BASE  upstream llama-server (default http://host.docker.internal:18085)
  UPSTREAM_MODEL      model id sent upstream and accepted from clients
                      (default hf.co/bartowski/FrogNano-4B-2609-GGUF:Q4_K_M)
  FROG_MAX_TOKENS     per-request output cap enforced by the proxy (default 8192)
  BIND / PORT         listen address (default 127.0.0.1:8080; run-proxy.sh
                      sets BIND=0.0.0.0 inside the container and publishes
                      127.0.0.1:18084)
"""
import json, os, time, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = os.environ.get('UPSTREAM_MODEL', 'hf.co/bartowski/FrogNano-4B-2609-GGUF:Q4_K_M')
UP_BASE = os.environ.get('FROG_UPSTREAM_BASE', 'http://host.docker.internal:18085')
BUDGET = int(os.environ.get('FROG_REASON_BUDGET', '16000'))
ALLOWED = {'model', 'messages', 'tools', 'tool_choice', 'max_tokens', 'max_completion_tokens',
           'temperature', 'top_p', 'top_k', 'min_p', 'logprobs', 'top_logprobs', 'logit_bias', 'n',
           'stop', 'grammar', 'echo', 'prompt', 'chat_template_kwargs', 'seed', 'stream',
           'stream_options', 'reasoning_effort', 'parallel_tool_calls'}


def upstream(payload, timeout=900):
    req = urllib.request.Request(UP_BASE + '/v1/chat/completions',
                                 json.dumps(payload).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parse_reasoning_delta(d):
    try:
        obj = json.loads(d)
    except Exception:
        return None, 0, 0, None
    ch = (obj.get('choices') or [{}])[0]
    delta = ch.get('delta') or {}
    return (delta.get('reasoning_content') or delta.get('reasoning') or '',
            len(delta.get('content') or ''), len(delta.get('reasoning_content')
                                                or delta.get('reasoning') or ''),
            ch.get('finish_reason'))


def budget_exceeded(reason_chars, content_chars, finish):
    """True when the reply is the measured failure: long reasoning, no answer, never closed."""
    return bool(BUDGET and reason_chars > BUDGET and content_chars == 0
                and finish in (None, 'length'))


def forced_answer_retry(payload, reason_text):
    """The retry shape: the reasoning already happened and was truncated, so force the answer out.
    Returns a fresh non-streaming response body, or None if the retry is impossible/failed."""
    retry = dict(payload)
    retry['stream'] = False
    retry['chat_template_kwargs'] = {'enable_thinking': False}
    retry['messages'] = list(payload.get('messages') or []) + [
        {'role': 'user', 'content': 'Reasoning from the previous step (it ran out of room '
         'before concluding):\n' + reason_text[-8000:] + '\n\nNow write the final answer.'}]
    try:
        out = upstream(retry)
        if out.get('choices'):
            out['model'] = payload.get('model') or MODEL
            out['g3_retry'] = True
            return out
    except Exception as e:
        print(json.dumps({'g3_retry_error': str(e)[:300]}), flush=True)
    return None


def sse_for(body_answer, model):
    parts = [body_answer[i:i + 200] for i in range(0, len(body_answer), 200)] or ['']
    lines = ['data: ' + json.dumps({'model': model,
                                    'choices': [{'delta': {'role': 'assistant', 'content': part},
                                                 'index': 0}]}) + '\n\n' for part in parts]
    lines.append('data: ' + json.dumps({'model': model,
               'choices': [{'delta': {}, 'finish_reason': 'stop', 'index': 0}]}) + '\n\n')
    lines.append('data: [DONE]\n\n')
    return lines


class Handler(BaseHTTPRequestHandler):
    def error(self, code, message):
        data = json.dumps({'error': message}).encode()
        self.send_response(code)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.error(200 if self.path == '/health' else 403,
                   'ready' if self.path == '/health' else 'denied')

    def do_POST(self):
        if self.path not in ('/v1/chat/completions', '/v1/completions'):
            return self.error(403, 'endpoint denied')
        try:
            n = int(self.headers.get('Content-Length', '0'))
            if not 0 < n <= 2 * 1024 * 1024:
                return self.error(413, 'request size denied')
            p = json.loads(self.rfile.read(n))
            extra = set(p) - ALLOWED
            if extra:
                return self.error(403, 'unsupported fields: ' + str(sorted(extra)))
            if p.get('model') not in (None, 'frog', MODEL):
                return self.error(403, 'model denied')
            p['model'] = MODEL
            CAP = int(os.environ.get('FROG_MAX_TOKENS', '8192'))
            p['max_tokens'] = min(int(p.pop('max_completion_tokens', p.get('max_tokens', CAP))), CAP)
            p.pop('reasoning_effort', None)
            p['chat_template_kwargs'] = {'enable_thinking':
                p.get('chat_template_kwargs', {}).get('enable_thinking', True)}
            start = time.monotonic()

            if not p.get('stream'):
                # Non-streaming: full control. Parse, decide, maybe retry, answer.
                try:
                    req = urllib.request.Request(UP_BASE + self.path, json.dumps(p).encode(),
                                                 {'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=900) as r:
                        body = json.loads(r.read().decode())
                except urllib.error.HTTPError as e:
                    return self.error(e.code, e.read().decode()[:2000])
                msg = (body.get('choices') or [{}])[0].get('message') or {}
                reason = msg.get('reasoning_content') or msg.get('reasoning') or ''
                finish = (body.get('choices') or [{}])[0].get('finish_reason')
                policy = 'passthrough'
                if budget_exceeded(len(reason), len(msg.get('content') or ''), finish):
                    retry = forced_answer_retry(p, reason)
                    if retry is not None:
                        body, policy = retry, 'g3-retry'
                payload = json.dumps(body).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(payload)
                print(json.dumps({'seconds': round(time.monotonic() - start, 2),
                                  'thinking': p['chat_template_kwargs']['enable_thinking'],
                                  'max_tokens': p['max_tokens'], 'policy': policy}), flush=True)
                return

            # Streaming: buffer until decidable (think-end seen, finish, or budget breach), then
            # either forward the accumulated SSE or replace it with the forced-answer retry as SSE.
            try:
                req = urllib.request.Request(UP_BASE + self.path, json.dumps(p).encode(),
                                             {'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=900) as response:
                    ctype = response.headers.get('Content-Type', 'text/event-stream')
                    # Keep EVERY line verbatim: SSE events are separated by blank lines, and dropping
                    # them makes a client concatenate two events into invalid JSON (this cost one
                    # deployment: omp reported "JSON Parse error" on healthy passthrough traffic).
                    chunks, reason_parts, reason_chars, content_chars, finished = [], [], 0, 0, None
                    for raw in response:
                        line = raw.decode('utf-8', 'replace')
                        chunks.append(line)
                        if not line.startswith('data:'):
                            continue
                        d = line[5:].strip()
                        if d == '[DONE]':
                            continue
                        t, cch, rch, fr = parse_reasoning_delta(d)
                        if t:
                            reason_parts.append(t)
                        reason_chars += rch
                        content_chars += cch
                        if fr:
                            finished = fr
            except urllib.error.HTTPError as e:
                return self.error(e.code, e.read().decode()[:2000])

            policy, sse = 'passthrough', None
            if budget_exceeded(reason_chars, content_chars, finished):
                retry = forced_answer_retry(p, ''.join(reason_parts))
                if retry is not None:
                    msg = (retry.get('choices') or [{}])[0].get('message') or {}
                    sse = sse_for(msg.get('content') or '', p['model'])
                    policy = 'g3-retry'
            if sse is None:
                sse = chunks          # byte-verbatim upstream passthrough
            self.send_response(200)
            self.send_header('Content-Type', ctype)
            self.send_header('Connection', 'close')
            self.end_headers()
            try:
                for line in sse:
                    self.wfile.write(line.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass                  # the client hung up; not our problem, and not a traceback
            print(json.dumps({'seconds': round(time.monotonic() - start, 2),
                              'thinking': p['chat_template_kwargs']['enable_thinking'],
                              'max_tokens': p['max_tokens'], 'policy': policy,
                              'reason_chars': reason_chars, 'content_chars': content_chars}), flush=True)
        except urllib.error.HTTPError as e:
            self.error(e.code, e.read().decode()[:2000])
        except Exception as e:
            self.error(502, str(e))


if __name__ == '__main__':
    ThreadingHTTPServer((os.environ.get('BIND', '127.0.0.1'),
                         int(os.environ.get('PORT', '8080'))), Handler).serve_forever()
