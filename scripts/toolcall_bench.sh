#!/usr/bin/env bash
# toolcall_bench.sh — run the tool-calling probe (toolcall_test.py) against all 3 models.
# Tool-calling correctness is hardware-independent (same GGUF + --jinja → same tokens at temp 0),
# so this runs once on the workstation; the verdict transfers to the Jetson with the same GGUFs.
#
# Each server launches with --jinja (required for the chat template's tool-call grammar).
# BENCH_NO_THINK=1 disables the reasoning channel (matches our coding-rubric methodology).
#   BENCH_NO_THINK=1 ./toolcall_bench.sh
set -u
B=$HOME/llm/llama.cpp/build/bin/llama-server
MD=$HOME/llm/models
PORT=8099; CTX=${CTX:-8192}; SRVLOG=/tmp/tc_srv.log
HERE="$(cd "$(dirname "$0")" && pwd)"
export BASE="http://127.0.0.1:$PORT/v1"
export BENCH_NO_THINK=${BENCH_NO_THINK:-1}
export MAXTOK=${MAXTOK:-512}

SP=""
stop(){ [ -n "$SP" ] && kill "$SP" 2>/dev/null; pkill -x llama-server 2>/dev/null; sleep 2; SP=""; }
trap stop EXIT

SUMMARY=""
run(){ local name="$1"; shift
  stop
  echo "### launching $name"
  "$B" "$@" --jinja --no-mmap -fa on -c "$CTX" --parallel 1 \
       --host 127.0.0.1 --port "$PORT" >"$SRVLOG" 2>&1 &
  SP=$!
  local ready=0
  for i in $(seq 1 90); do
    grep -q "server is listening" "$SRVLOG" 2>/dev/null && { ready=1; break; }
    kill -0 "$SP" 2>/dev/null || break; sleep 2
  done
  if [ "$ready" -ne 1 ]; then echo "$name | LOAD_FAILED"; tail -6 "$SRVLOG"; SUMMARY+="
$name | LOAD_FAILED"; return; fi
  local out; out=$(NAME="$name" python3 "$HERE/toolcall_test.py")
  echo "$out"
  local line; line=$(echo "$out" | grep '^RESULT' | tail -1)
  SUMMARY+="
$name | $(echo "$line" | cut -f3)/$(echo "$line" | cut -f4) | $(echo "$line" | cut -f5)%"
}

echo "=== tool-calling reliability (CTX=$CTX, --jinja, BENCH_NO_THINK=$BENCH_NO_THINK) ==="
run "gemma4-26b-a4b"   -m "$MD/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf" -ngl 999 --n-cpu-moe 16
run "qwen36-35b-a3b"   -m "$MD/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"     -ngl 999 --n-cpu-moe 30
run "qwen36-27b-dense" -m "$MD/Qwen3.6-27B-UD-Q4_K_XL.gguf"        -ngl 30
echo; echo "===== SUMMARY tool-calling (model | passed/total | pct) ====="
echo "$SUMMARY"
