#!/usr/bin/env bash
# Starts the exploration static-file server (server.py) — plain HTML/JS
# pages under pages/, no build step. Port defaults to 8100 (matching this
# project's own established convention); pass a different one as $1, e.g.
# `./start.sh 8101`.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PORT="${1:-8100}"

source .venv/bin/activate
exec uvicorn server:app --port "$PORT"
