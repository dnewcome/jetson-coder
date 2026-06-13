#!/usr/bin/env bash
# run-server.sh — launch the persistent llama-server for Pi to talk to.
# Serves Gemma 4 26B-A4B with MTP speculative decoding + vision, OpenAI-compatible on :8080.
# Matches Kyle's flags; CTX/FA overridable via env.
set -u

LLAMA=${LLAMA:-/mnt/nvme/zen/llm/llama.cpp/build/bin}
GEMMA_DIR=${GEMMA_DIR:-/mnt/nvme/zen/llm/models/gemma-4-26B-A4B-it-GGUF}
CTX=${CTX:-32768}
FA=${FA:-on}
SPEC_N=${SPEC_N:-3}

exec "$LLAMA/llama-server" \
  -m  "$GEMMA_DIR/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf" \
  --model-draft "$GEMMA_DIR/MTP/gemma-4-26B-A4B-it-Q8_0-MTP.gguf" \
  --mmproj "$GEMMA_DIR/mmproj-F16.gguf" \
  --spec-type draft-mtp --spec-draft-n-max "$SPEC_N" \
  --alias gemma4-local --jinja \
  --no-mmap -ngl 999 -fa "$FA" -c "$CTX" --parallel 1 \
  --host 0.0.0.0 --port 8080
