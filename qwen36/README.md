# Qwen3.6-35B-A3B (side endpoint)

[← index](../README.md)

| | |
|---|---|
| model | Qwen3.6-35B-A3B (35B total, 3B active) |
| quant / file | IQ3_S, 15.35 GB on the side endpoint; the matrix used Q3_K_M, 16.6 GB |
| context | 128k |
| flags | MTP draft depth 2, `-ub 2048`, f16 KV, prefix reuse |
| port | 18086 |
| resident | 18.55 GB at 128k |

## Launch

```sh
./launch-qwen-mtp.sh /path/to/Qwen3.6-35B-A3B-UD-IQ3_S.gguf [port]
```

Needs a llama.cpp build with `--spec-type draft-mtp`; the draft head is in the
MTP GGUF. `-np > 1` and `--mmproj` are not supported with MTP.

Why IQ3_S: Q3_K_M next to Frog is a Metal OOM even at 32k. IQ3_S is the only
quant tested that coexists with production at 128k.

## Speed

Matrix row (Q3_K_M, no speculation, 131072 context): load 9.1 s, RSS 19.47 GB,
pp 784 / 856 / 762 / 637 / 473 / 313, tg 52.9 / 42.5 at 4k / 32k.

MTP, exact-128 decode, 131k context:

| config | decode 4k | decode 32k | prefill 4k | prefill 32k | RSS GB |
|---|---:|---:|---:|---:|---:|
| no MTP | 52.9 | 42.5 | 855 | 637 | 19.47 |
| depth 1 | 66.3 | 49.5 | 884 | 611 | 20.27 |
| **depth 2 (daily)** | **71.3** | **54.1** | 882 | 609 | 20.54 |
| depth 3 | **75.6** | **57.7** | 879 | 608 | 20.61 |

Depth 3 is +43% / +36% over no-MTP on raw decode, but on the agent task (run with
`-ub 1024`) it came out at 72 s against 71 s for depth 2: it wrote more, 1,687
against 1,559 tokens, and the extra output ate the saving. Depth 2 stays.

## Ubatch

From a separate A/B on uncached 30k and 8k prompts:

| ubatch | 30k prefill | 8k prefill |
|---|---:|---:|
| 1024 | 617.0 | 820 |
| **2048** | **625.4** | **834** |
| 4096 | 609.0 | 803 |

## Prefix cache

Exact repeated prefix at 32k: first prefill 31,716 tokens in 50.9 s; the exact
repeat reuses 30,688 tokens and costs 1,028 new tokens in **2.34 s**; appending
costs 3.54 s. A shifted prefix matches nothing and returns to full time. Keep the
system prompt and tool definitions byte-stable between turns.

## Slot save/restore

A 13,201-token conversation state saves in 55 ms (336.5 MB) and restores in
40 ms; continuing with the restored cache takes 46 ms instead of 17.06 s of cold
prefill. The constraint is storage and retention, not compute.

## Batched decode

With 512-token prompts, four slots aggregate 70.0 tok/s against 55.25 for one:
1.27×, not 4×, because the streams share the same memory system and per-token
overhead. MTP and `-np > 1` do not work together.

## Rejected

- **KV q8** saves 1.25 GB and collapses long-context decode: 49.7 vs 71.3 at 4k,
  **17.0 vs 54.1** at 32k.
- **Flash attention** on this path: 71.2 vs 71.3 at 4k, 54.1 vs 54.1 at 32k.
- **MTP plus n-gram**: no decode gain.
- **IQ3_S as a speed lever**: 15.35 GB against Q3_K_M's 16.6 and 18.55 GB
  resident at 128k, but 63.7 / 48.4 tok/s instead of 71.3 / 54.1. It is the
  memory lever that makes coexistence possible.

## Agent wall

One sample, hidden suite 3/3: 83 s without speculation, **71 s** with MTP depth 2,
1,559 output tokens and 13 tool calls.

## Runner

```sh
LAB_ROOT=/path/to/lab python3 measure_levers.py
```

Arms: no-MTP baseline, MTP depth 1–3, flash attention, ubatch 1024/2048/4096,
KV q8, IQ3, MTP+n-gram. Writes `/tmp/qwen-levers-result.json`.
