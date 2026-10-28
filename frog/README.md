# FrogNano-4B-2609 (production)

[← index](../README.md)

| | |
|---|---|
| model | FrogNano-4B-2609 (Qwen3.5-4B base) |
| quant / file | Q4_K_M, 2.8 GB |
| context | 128k, one shared context over 4 slots |
| flags | `-b 2048 -ub 1024`, n-gram speculation size-n 12 / size-m 48 |
| port | 18085, behind the proxy on 18084 |
| resident | 9.15 GB |

## Launch

```sh
./launch-frog.sh /path/to/model.gguf [mmproj.gguf] [port]
```

The n-gram flags are upstream llama.cpp; MTP is not used here. Dropping the
mmproj (text-only) also enables llama.cpp's KV-shifting prefix reuse, which it
disables for multimodal models.

In production this server sits behind the proxy, which adds the thinking budget:
a request that burns 16k characters of reasoning without emitting an answer is
retried once with thinking off. See the main README, "Cheaper thinking".

## Speed

From the shared matrix, no speculation, 131072 context:

| load s | RSS GB | pp512 | pp4k | pp16k | pp32k | pp64k | pp128k | tg4k | tg32k |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.0 | 9.15 | 782 | 837 | 738 | 631 | 487 | 338 | 59.7 | 48.3 |

At 128k the prefill alone is about 6.5 minutes, which is why exact-prefix reuse
matters more than raw prefill speed.

Context is the other cost. The hybrid model's KV cache is about 32 KB per token,
so 128k is roughly 4.2 GB and a 262k setting would be roughly 8.4 GB. That is
why 128k is the served size: it matches what the harness advertises and leaves
the rest of the 32 GB to weights and compute buffers.

## Ubatch

`-ub 1024` is the optimum. At 128k, 2048 and 4096 start cleanly but prefill 3–7%
slower; only at 262k do they OOM the Metal command buffer.

## n-gram speculation

- Production-shaped turn (thinking on, 8k budget, 32k context): 60.0 tok/s with
  no speculation, 63.4 with `size_m 16`, 63.5 with `size_m 48`. Both matcher
  settings are ~+6%, and the output is byte-identical.
- The wall did not get shorter (100.6 s vs 95.3 s): the n-gram arm wrote 14% more
  reasoning, so the gain is decode rate, not time.
- Verbatim echo: 62 → 268 tok/s with `size_m 48`; m48 adds +39% on echo-heavy
  prompts while staying neutral on the reasoning turn.
- On the three-turn repo task n-gram costs 18 s (116 → 134 s), so the setting is
  workload-dependent, not universal.

## Agent wall

One sample, hidden suite 3/3: 116 s without speculation, 134 s with n-gram,
4,088 output tokens and 24 tool calls in the speculation run.

## Verdict

Production. It has the smallest footprint of the six and leaves room for the
proxy and containers; n-gram pays on its long reasoning turn, not on the repo
task.
