#!/usr/bin/env sh
# Run the front-door proxy in Docker: seeds the script volume from this folder
# and starts the container on 127.0.0.1:18084. Upstream defaults to
# host.docker.internal:18085 (one llama-server per host).
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
VOL=frog-default-proxy-scripts

docker volume create "$VOL" >/dev/null
docker run --rm --entrypoint sh -v "$VOL":/scripts -v "$HERE":/src:ro \
  python:3.12-alpine -c 'cp /src/proxy.py /scripts/proxy.py'
docker rm -f frog-omp-default-proxy >/dev/null 2>&1 || true

set -- docker run -d --name frog-omp-default-proxy --restart unless-stopped \
  -p 127.0.0.1:18084:8080 -v "$VOL":/scripts
if [ -n "${FROG_REASON_BUDGET:-}" ]; then set -- "$@" -e "FROG_REASON_BUDGET=$FROG_REASON_BUDGET"; fi
if [ -n "${UPSTREAM_MODEL:-}" ]; then set -- "$@" -e "UPSTREAM_MODEL=$UPSTREAM_MODEL"; fi
set -- "$@" python:3.12-alpine python -u /scripts/proxy.py
exec "$@"
