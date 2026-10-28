# LFM3-1.7B (experiment)

[← index](../README.md)

| | |
|---|---|
| model | LFM3-1.7B |
| quant / file | QAD-Q4_0, 1.05 GB |
| context | 40960 measured; 128k also fits |
| build | llama.cpp v0.8.0 (b11960), the first tag with `arch lfm3` |
| port | 18086 when running |

## Speed

Same `/completion` protocol as the shared matrix, 40960 context:

| model | decode 4k | decode 16k | decode 32k | prefill 1k | prefill 4k | prefill 16k | prefill 32k | RSS GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **LFM3-1.7B QAD-Q4_0** | 159 | 145 | 133 | 2495 | 2496 | 2167 | 1828 | 1.61–2.49 |
| FrogNano-4B Q4_K_M | 60 | 54 | 48 | 843 | 843 | 740 | 631 | 5.71–8.23 |

At 40k it loads in 0.5 s, against 2.6 s for Frog. At 128k context it still
prefills 994 tok/s on a 120k prompt, decodes 116 tok/s at 65k depth, and holds
3.75 GB.

## Quants

QAD (quantisation-aware distillation) is both the fastest and the best of the
4-bit checkpoints: 159 vs 142 tok/s for plain Q4_K_M, 2496 vs 2193 prefill, and
12/12 vs 10/12 on the easy code bank. Q8_0 costs 21% decode for the same 10/12.
Q4_0 runs at ~167 GB/s effective, so dequantisation, not bandwidth, sets its pace.

## n-gram speculation

The gain scales with how much of the output repeats, and the drafts are cheaper
than on a 35B model:

- verbatim echo 170.6 → **578.3** tok/s, code echo 166.9 → **516.7**;
- novel prose 180.6 → 178.7, neutral;
- generated code byte-identical at temperature 0 on 12/12 easy tasks;
- on the hard bank, 25% fewer tokens and 27% less wall time.

## Quality

Easy bank 12/12 in 14 s against Frog's 11/12 in 39 s, tied at this resolution.
Hard bank 5/18 at temp 0.6 and 6/18 at 0.1 against Frog's 7/18 at 0.6. Thinking
is a pure loss at these caps: 12/12 but 16× the tokens on easy, and 4/18 on hard
with 13 of 18 truncated.

## Runner

```sh
LAB_ROOT=/path/to/lab python3 measure.py --arm lfm3-qad --ctx 32768
LAB_ROOT=/path/to/lab python3 measure.py --arm frog --ctx 32768
```

Arms `lfm3-qad`, `lfm3-q4`, `lfm3-q8`, `frog`; `LFM3_ROOT` overrides the model
and build directory (default `$LAB_ROOT/lfm3`), `FROG_CMD` overrides the Frog
launcher. Writes `/tmp/lfm3-speed-<arm>.json`.

## Verdict

Much faster and much smaller than Frog, and byte-identical speculation at greedy.
Hard-bank quality is behind Frog, so it stays an experiment rather than
production.
