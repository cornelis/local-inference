#!/usr/bin/env sh
# Serve FrogNano-4B the way production runs it: 128k context, ubatch 1024,
# n-gram speculation at size-n 12 / size-m 48.
#
#   ./launch-frog.sh /path/to/frognano-4b-2609-Q4_K_M.gguf
#   ./launch-frog.sh /path/to/model.gguf /path/to/model.mmproj 18086
#
# The second argument is optional (omit it for text-only, which also enables
# llama.cpp's KV-shifting prefix reuse). Needs a Metal llama.cpp build; the
# n-gram flags are upstream, MTP is not used here.
set -eu
MODEL=${1:?usage: launch-frog.sh <model.gguf> [mmproj.gguf] [port]}
MMPROJ=${2:-}
PORT=${3:-18085}

if [ -n "$MMPROJ" ]; then MMPROJ_ARG="--mmproj $MMPROJ"; else MMPROJ_ARG=""; fi

# $MMPROJ_ARG is intentionally unquoted so the empty case disappears.
exec llama-server \
  --model "$MODEL" \
  $MMPROJ_ARG \
  --host 127.0.0.1 --port "$PORT" \
  --ctx-size 131072 \
  --n-gpu-layers 999 \
  --no-mmap \
  --threads 5 \
  --batch-size 2048 --ubatch-size 1024 \
  --spec-type ngram-simple --spec-draft-n-max 16 \
  --spec-ngram-simple-size-n 12 --spec-ngram-simple-size-m 48
