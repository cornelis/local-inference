#!/usr/bin/env bash
set -euo pipefail

# Needs an MTP GGUF (the draft head is in the file). Production side endpoint:
# Qwen3.6-35B-A3B-UD-IQ3_S.gguf.
MODEL="${1:?usage: launch-qwen-mtp.sh /path/to/Qwen3.6-35B-A3B-UD-IQ3_S.gguf [port]}"
PORT="${2:-18086}"

exec llama-server \
  -m "$MODEL" \
  -c 131072 \
  -ngl 999 \
  -np 1 \
  --jinja \
  --host 127.0.0.1 \
  --port "$PORT" \
  --threads 5 \
  --spec-type draft-mtp \
  --spec-draft-n-max 2 \
  -ub 2048 \
  --cache-reuse 256
