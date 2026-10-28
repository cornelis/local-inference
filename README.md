# Fast and efficient local inference on an M1 Max

Working toward local agents that are both fast and efficient: fewer seconds per
turn, fewer tokens per answer, and more correct answers, on hardware one person
owns.

Six models on one laptop, measured on two axes: how fast they run (decode,
prefill, task wall) and what they cost (memory, context, tokens per answer).
Production serves a 4B model; the rest are the comparison set.

The two axes do not pick the same winner. LFM2.5-24B-A2B is the fastest engine
(84 tok/s, 10.8 GB) and loses the agent wall; LFM3-1.7B is the cheapest to keep
resident (159 tok/s, 1.6–2.5 GB) and trails on hard code; Qwen3.6-35B runs next
to production at 128k (IQ3_S, 18.6 GB) and wins the agent task at 71 s on 1,559
tokens.

## Bench overview

One row per model: decode at 4k, prefill at 4k and 128k, memory, and the agent
wall with its best speculation setting. `n/a` means not measured.

| model | role | quant / file | ctx | decode 4k | prefill 4k / 128k | RSS GB | agent wall s | output tokens | quality |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| **FrogNano-4B** | production | Q4_K_M, 2.8 GB | 128k | 59.7 | 837 / 338 | 9.15 | 116 / 134 | 4,088 | hidden 3/3 |
| **Qwen3.6-35B** | side | IQ3_S, 15.35 GB | 128k | 52.9* | 856 / 313* | 19.47* | 83 / 71 | 1,559 | hidden 3/3 |
| **LFM3-1.7B** | experiment | QAD-Q4_0, 1.05 GB | 40k | 159 | 2496 / n/a | 1.6–2.5 | n/a | n/a | easy 12/12, hard 5/18 |
| **LFM2.5-24B-A2B** | tested | Q4_K_XL, 14.9 GB | 131k | 83.8 | 1229 / 517 | 10.79 | 155 / 268 | 6,848 | hidden 3/3 |
| **Sage-A1.5** | tested | Q3_K_L, 18.7 GB | 131k | 48.6 | 817 / 303 | 15.55 | 193 / 178 | 4,518 | hidden 3/3 |
| **Flint** | tested | Q2_K_XL, 13.6 GB | 131k | 60.5 | 969 / 252 | 12.80 | 177 / 203 | 6,961 | hidden 3/3 |

Agent wall is `no speculation / with the model's speculation`, lower is better.
Frog's production choice wins on its own turn, not on this task. Tokens are
output tokens in the speculation run. Quality is the 3-test hidden suite (n=1)
unless a model folder says otherwise. `*` Qwen's matrix numbers are Q3_K_M
(16.6 GB); the side endpoint serves IQ3_S, which decodes 63.7 / 48.4 tok/s at
4k / 32k. LFM3 is measured at 40k; at 128k it prefills 994 tok/s on a 120k
prompt (see [lfm3/](lfm3/)).

## What each model has

| model | full matrix | 128k prefill | agent wall | code-bank quality | speculation |
|---|---|---|---|---|---|
| Frog | yes | yes | yes | no | n-gram |
| Qwen | yes (Q3) | yes | yes | no | MTP |
| LFM3 | 40k | spot check | no | easy + hard | n-gram |
| LFM2.5 | yes | yes | yes | no | n-gram |
| Sage | yes | yes | yes | no | n-gram, MTP |
| Flint | yes | yes | yes | no | n-gram |

Only LFM3 has code-bank quality. The agent wall exists for five models; LFM3 has
no omp run. All six have throughput and memory numbers; LFM3's are at 40k, the
others at 131072.

## Model folders

Frog, Qwen3.6 and LFM3 have their own folders with notes, numbers and runners;
the rest appear only in the comparison tables. Shared benchmarks and the front
door are at the root.

| folder | model | quant / file | role | runners |
|---|---|---|---|---|
| [frog](frog/) | FrogNano-4B-2609 | Q4_K_M, 2.8 GB | production | `launch-frog.sh` |
| [qwen36](qwen36/) | Qwen3.6-35B-A3B | IQ3_S, 15.35 GB | side endpoint | `launch-qwen-mtp.sh`, `measure_levers.py` |
| [lfm3](lfm3/) | LFM3-1.7B | QAD-Q4_0, 1.05 GB | experiment | `measure.py` |

## The machine

| | |
|---|---|
| host | `MacBookPro18,2` |
| chip | Apple M1 Max, 10 cores (8 performance + 2 efficiency) |
| memory | 32 GB unified, ~400 GB/s |
| engine | `llama-server`, Metal; the lab's `frog-mtp-profile-001` build (b9879-metal) for Frog and MTP, upstream v0.8.0 (b11960) for LFM3; 5 threads |
| Metal buffer limit | ~20.1 GB (the allocation ceiling, not RSS) |

Decode is not simply bandwidth-bound. Single-stream Q4_K_M decode costs about
16 ms/token, and only ~7 ms of that is reading the 2.79 GB of weights at
~400 GB/s; the rest is per-token dispatch and synchronisation, which batching
amortises (four streams give 1.27× aggregate, not 4×). LFM3's Q8_0 build reaches
~230 GB/s of the available bandwidth, while its Q4_0 build reaches 167 GB/s
because dequantisation sets the pace there. The arithmetic is `weights in GB ×
decode tok/s`: LFM3 Q4_0 is 1.05 × 159 ≈ 167 GB/s. Memory capacity decides
context length.

## How to read the numbers

- **ppN / tgN**: prefill with an N-token prompt, and decode at depth N, in tok/s,
  read from the server's own timings. Depth is the number of tokens already in
  context.
- **load**: seconds from start to ready. **RSS**: resident GB after load,
  including CPU-side buffers, so it can exceed the Metal limit.
- **`-b` / `-ub`**: batch and ubatch, prompt tokens processed per step and per
  prefill chunk.
- **MTP**: a small draft head proposes tokens and the target model verifies them.
  **n-gram**: drafts proposed from repeated patterns in the context.
- **Agent wall**: wall-clock seconds for a whole 3-turn repo task through omp, my
  coding agent, which talks OpenAI-compatible HTTP to the server. It includes
  harness overhead, not just model time.

Prefill is a chat request with a one-token output; decode is a `/completion`
request with `ignore_eos` and exactly 128 generated tokens. One server at a time
on port 18086. Quant names are GGUF formats: Q4_K_M and Q3_K_M are 4- and 3-bit
k-quants, IQ3_S is an i-quant, QAD is Liquid's distillation-trained 4-bit build.

## Full prefill and decode matrix

No speculation, 131072 server context. The Frog row is the served Q4_K_M; the
Qwen row is Q3_K_M, not the IQ3_S side endpoint.

| model | load s | RSS GB | pp512 | pp4k | pp16k | pp32k | pp64k | pp128k | tg4k | tg32k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **FrogNano-4B** | 3.0 | 9.15 | 782 | 837 | 738 | 631 | 487 | 338 | 59.7 | 48.3 |
| **Qwen3.6-35B-A3B** | 9.1 | 19.47 | 784 | 856 | 762 | 637 | 473 | 313 | 52.9 | 42.5 |
| **Sage-A1.5** | 13.1 | 15.55 | 697 | 817 | 721 | 607 | 454 | 303 | 48.6 | 39.5 |
| **LFM2.5-24B-A2B** | 11.1 | 10.79 | 1010 | 1229 | 1143 | 974 | 748 | **517** | **83.8** | **70.4** |
| **Flint Q2_K_XL** | 11.1 | 12.80 | 908 | 969 | 771 | 596 | 407 | 252 | 60.5 | 44.8 |

LFM2.5 (24B total, 2B active) is the fastest engine at every length, including
128k prefill, and does it on 10.8 GB. Frog is slower but has the smallest
footprint of the five. Flint's 40B total weight and MoE routing show up as the
worst 128k prefill of the group.

LFM3 is measured separately at 40960 context, where it decodes 159/145/133 tok/s
and prefills 2495/2496/2167/1828; see [lfm3/](lfm3/).

## Agent tasks

The same three-turn repo task through omp against each server, with a 3-test
suite outside the repository deciding the result. One task, one sample per model;
bold marks the faster wall. Every model passed, so the table measures cost, not
capability.

| model | no speculation | with speculation | output tokens | tools |
|---|---:|---:|---:|---:|
| [**Qwen3.6**](qwen36/) | 83 s | **71 s** (MTP depth 2) | 1,559 | 13 |
| [**Frog 4B**](frog/) | **116 s** | 134 s (n-gram) | 4,088 | 24 |
| **LFM2.5-24B-A2B** | **155 s** | 268 s (n-gram) | 6,848 | 22 |
| **Flint Q2** | **177 s** | 203 s (n-gram) | 6,961 | 19 |
| **Sage-A1.5** | 193 s | **178 s** (n-gram) | 4,518 | 22 |

Tokens and tool calls are from the speculation runs. Qwen wins the wall by
writing less: 1,559 output tokens against Frog's 4,088. Only Qwen and Sage gain
from speculation on this task, and Sage's MTP variant regressed to 312 s. Frog
still runs n-gram in production because its long reasoning turn gains ~6%. On a
one-shot `calc.py` repair: Qwen 43 s, Sage 56 s, LFM2.5 78 s; on a different
128k repair, Flint took 66 s against Frog's 42 s.

## The serving path

This folder ships the measurements, the launch lines and the front door. A
request on this machine passes through:

| piece | role | in this folder |
|---|---|---|
| launchd agent `com.frog.llamaserver` | starts the server at boot, keeps it up | no (host-specific plist) |
| `serve.sh` | production launcher: resolves the model from the Docker Model Runner store, adds mmproj and the spec knobs | no; `frog/launch-frog.sh` is the portable equivalent |
| `llama-server` | the lab's `frog-mtp-profile-001` build (b9879-metal); MTP needs it | no (build from source) |
| `frog-omp-default-proxy` | front door on 127.0.0.1:18084: model allowlist, output cap, thinking default, reasoning budget with one forced-answer retry ([details](#cheaper-thinking-budget-then-answer)) | **yes**: `proxy.py` + `run-proxy.sh` |
| omp provider | points the coding harness at 18084 | no (harness config) |

The proxy's streaming path buffers upstream output until the retry decision is
clear, then forwards it or replaces it with the retried answer.

## Cheaper thinking: budget, then answer

A thinking model can spend the whole output budget reasoning and return nothing.
On a 4B model with an 8k cap that is not a rare tail: in the 18-task bank, six
tasks hit it, and all six came back with `finish_reason: length` and no code. A
bigger cap is the wrong fix: at 16k, thinking-on recovered 2 of the 6 for
87,531 tokens, while the wider design that uses the same bounded-then-answer
step recovered 4 for 49,067. Across the bank that is ~1.9× the tokens for 11
points less.

The proxy makes the empty reply impossible:

- Requests run with thinking on and a hard output cap of 8,192 tokens
  (`FROG_MAX_TOKENS`).
- If the reply has more than `FROG_REASON_BUDGET` characters of reasoning
  (16,000, about 4k tokens), no content, and `finish_reason` is `length` or
  missing, the proxy retries once upstream: the last 8,000 characters of
  reasoning are injected as context, thinking is forced off, and the retry gets
  its own budget.
- Everything else is byte-for-byte passthrough. `FROG_REASON_BUDGET=0` disables
  the policy.

Measured:

| arm | result |
|---|---|
| flat 8k, thinking on (production shape) | 0/8 on the failure set |
| budget then answer, harness level | 4/8 converted; 12/16 vs 8/16 on the bank |
| **through the live proxy** | **5/8**, retry fired on 4 |
| thinking off, single call | 2/8 converted |

In the standalone test the retry caught a 29,726-character runaway with zero
content; live, `json_pointer` returned 998 characters with `g3_retry: true`.

The cost per solved task, same bank and same two arms:

| arm | solved | tokens total | tokens per solve | wall per solve |
|---|---:|---:|---:|---:|
| production (thinking on, 8k, one sample) | 12/18 | 115,197 | 9,600 | 212 s |
| design (best-of-4 plus escalation) | 16/18 | 80,319 | 5,020 | 94 s |

The design solves four more tasks for 30% fewer tokens and 41% less wall in
total; per solved task that is roughly half the tokens and less than half the
time. The proxy implements one step of this shape, the retry, not the best-of-4.

What it does and does not save. The retry is one extra call, and it only fires on
requests that would otherwise come back empty; the first call still runs its full
budget. So this is not a token discount on every request; it converts a
zero-answer into an answer and avoids paying for a bigger cap everywhere. The
same budget-then-answer shape measured 16/18 at 80,319 tokens against
production's 12/18 at 115,197. On ordinary multi-turn agent work the gate rarely
fires: those turns deliberate only ~300–1,100 tokens, and the runaway reasoning
lives in the one-shot/puzzle shape.

Why it works: the bottleneck is emission, not reasoning. Thinking-on suppresses
code output on some tasks even when the budget is not exhausted, so forcing the
answer out (or turning thinking off) changes the outcome. That is why the policy
is one retry with thinking off rather than a bigger thinking cap.

## Shared scripts

| file | what it does |
|---|---|
| `measure_matrix.py` | prefill, load and decode for five models; writes `/tmp/measure-result.json` |
| `measure_decode.py` | exact-128 decode at 0/4k/32k; writes `/tmp/decode-result.json` |
| `proxy.py` | front door: allowlist, output cap, thinking default, one forced-answer retry |
| `run-proxy.sh` | seeds the Docker volume and starts the proxy on 127.0.0.1:18084 |

The measurement scripts take `LAB_ROOT` and expect this layout:

```text
$LAB_ROOT/
  inference-ab-001/native/build/bin/llama-server   # engine build
  inference-ab-001/serve.sh                        # Frog launcher (PORT/CTX/SPEC)
  measure-001/{qwen36,sage,lfm}/*.gguf
  qwen-levers-001/{q3,iq3}/*.gguf
  flint-test-001/llama.cpp/build/bin/llama-server
  flint-test-001/model/*.gguf
  flint-test-001/flint-chat-template.jinja
  lfm3/build/bin/llama-server                     # stock v0.8.0 for LFM3
  lfm3/models/*.gguf
```

`FROG_CMD` overrides the Frog launcher. Each script starts its own server on
18086, one at a time, skips a model whose file is missing, and records an error
instead of running if the port is already busy. The Qwen MTP arms need the MTP
GGUF (the draft head is in the file); Flint needs a build with the `flint` arch.

Builds: the lab's `b9879-metal` fork produced every Frog and MTP number, upstream
v0.8.0 (b11960) is required for LFM3, and Flint needs Unsloth's `model-arch/flint`
branch.

Proxy environment: `FROG_REASON_BUDGET` (16000 chars), `FROG_UPSTREAM_BASE`
(`http://host.docker.internal:18085`), `UPSTREAM_MODEL`, `FROG_MAX_TOKENS`
(8192), `BIND`/`PORT` (`0.0.0.0:8080`).

Not reproducible from this folder: the prefix-cache, slot, batched-decode,
ubatch and alternate-server numbers were one-off measurements, the LFM3 quality
scores came from a separate quality runner, and the agent table came from omp
sessions, not from a script. The raw JSON behind every table was written to
`/tmp` on the measurement host and is not committed. The models and the Frog
`serve.sh` are also not included.

## Reproduce one number

The scripts run the sweeps; these are the same requests by hand, with a server
on 18086.

```sh
# prefill: prompt tokens/second, one token out
curl -s localhost:18086/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"word word"}],"max_tokens":1,"temperature":0}' \
  | jq '.timings | {prompt_n, prompt_per_second}'

# decode: exactly 128 tokens
curl -s localhost:18086/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"word word","n_predict":128,"ignore_eos":true,"temperature":0,"cache_prompt":false}' \
  | jq '.timings | {predicted_n, predicted_per_second}'

# front door: passthrough or budget retry?
curl -s localhost:18084/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"frog","messages":[{"role":"user","content":"write a function"}]}' \
  | jq '{retry: (.g3_retry // false), content: .choices[0].message.content}'
```

`jq` is only for readability; the numbers are in the JSON `timings` object. The
proxy logs one JSON line per request with `policy`, `reason_chars` and
`content_chars`; `docker logs -f frog-omp-default-proxy` is the live view.

## Limitations

- One host, single runs, no repeated measurement of variance.
- The cross-model table uses one quant per model; quality is only measured for
  LFM3 (see [lfm3/](lfm3/)).
- The agent table is one task, one sample per model. The generated workspaces
  behind this family were later found to contain pristine sibling copies of each
  defect, so a mechanical copy can pass: read the walls as cost, not capability.
- 128k is configured wherever it is claimed, but most agent prompts are ~16k, so
  these are not 128k KV measurements.
- Memory numbers are RSS, not GPU buffer bytes; the Metal ceiling is the limit
  that actually fails.
- The thinking-budget gate was validated on an 8-task failure set plus a live
  smoke test, not a broad benchmark. It adds one retry call on the requests where
  it fires, and the first call still spends its full budget.

## Models used

| model | source | local file |
|---|---|---|
| FrogNano-4B-2609 | <https://huggingface.co/bartowski/frognano-4b-2609-gguf> (base <https://huggingface.co/microsoft/FrogNano-4B-2609>) | `model.gguf`, Q4_K_M, 2.8 GB |
| Qwen3.6-35B-A3B | <https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF> | `Qwen3.6-35B-A3B-UD-IQ3_S.gguf`, 15.35 GB; MTP head included, `-np > 1` unsupported |
| Sage-A1.5 | <https://huggingface.co/mradermacher/Sage-A1.5-GGUF> | `Sage-A1.5.Q3_K_L.gguf`, 18.7 GB |
| LFM2.5-24B-A2B | <https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF> | `LFM2.5-24B-A2B-UD-Q4_K_XL.gguf`, 14.9 GB |
| Flint | <https://huggingface.co/unsloth/flint-GGUF> | `flint-preview-UD-Q2_K_XL.gguf`, 13.6 GB |
| LFM3-1.7B | <https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF> | `LFM3-1.7B-QAD-Q4_0.gguf`, 1.05 GB |

## Change log

- **2026-10-28**: LFM3-1.7B evaluated and kept as an experiment: 2.7× the decode
  and 3× the prefill of Frog at a fraction of the memory, but 5/18 on the hard
  bank against Frog's 7/18 at the same temperature. Production untouched.
- **2026-10-23**: Qwen side moved to `-ub 2048` after a 1024/2048/4096 A/B.
  Q3_K_M cannot share the box with Frog at 32k, so the side endpoint is IQ3_S at
  128k with 18.55 GB resident.
- **2026-10-22**: full five-model matrix plus the Qwen lever sweep. Kept the
  plain llama.cpp path on the one-shot `calc.py` repair: 23 s for production
  Q4_K_M + n-gram, against 37 s with `--ctx-checkpoints 32` and 68 s for the MLX
  4-bit server. MTP depth 2 confirmed as the Qwen multiturn setting; depth 3
  wins raw decode and loses the agent wall.
- **2026-10-17**: the existing proxy in front of Frog gains the reasoning
  budget: 16k characters, then one retry with thinking off. The model and its
  launch flags do not move.
- **2026-10-13**: Frog production set to 128k context with `-b 2048 -ub 1024`
  and n-gram speculation. The n-gram gain was measured separately at 32k:
  60.0 → 63.4–63.7 tok/s on the production-shaped turn, byte-identical output.

## License

The scripts, the proxy and the text are MIT (see `LICENSE`). No model weights are
redistributed; each model keeps its own license from the source in the table above.
