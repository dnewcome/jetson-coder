#!/usr/bin/env bash
# bench.sh — Reproduce Kyle Howells' local-coding-agent benchmark on Jetson AGX Xavier.
# Ref: https://ikyle.me/blog/2026/how-to-setup-a-local-coding-agent-on-macos
#
# Measures prompt tok/s and generation tok/s for each config by launching
# llama-server, polling /health, sending Kyle's exact bench prompt to the native
# /completion endpoint, and parsing server-reported timings. Averages over REPS.
#
# Override any of these via env, e.g.:  CTX=65536 REPS=5 ./bench.sh
set -u

LLAMA=${LLAMA:-/mnt/nvme/zen/llm/llama.cpp/build/bin}
MODELS=${MODELS:-/mnt/nvme/zen/llm/models}
GEMMA_DIR=$MODELS/gemma-4-26B-A4B-it-GGUF
GEMMA=$GEMMA_DIR/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
GEMMA_MTP=$GEMMA_DIR/MTP/gemma-4-26B-A4B-it-Q8_0-MTP.gguf
GEMMA_MMPROJ=$GEMMA_DIR/mmproj-BF16.gguf
QWEN=$MODELS/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

HOST=127.0.0.1
PORT=${PORT:-8089}
CTX=${CTX:-16384}          # Kyle used 65536 on 64GB M1; gen tok/s is ctx-independent, keep it fitting
NGL=${NGL:-999}
FA=${FA:-on}               # flash-attn; flip to off if Volta/sm_72 errors
REPS=${REPS:-3}
MAXTOK=${MAXTOK:-128}      # Kyle's run generated ~128 tokens
SPEC_N=${SPEC_N:-3}        # --spec-draft-n-max (Kyle's optimal was 3)
SRVLOG=/tmp/llama_srv.log

PROMPT='Write a compact Python function that parses a unified diff and returns changed file paths. Then explain two edge cases.'

SRV_PID=""
cleanup() { [ -n "$SRV_PID" ] && kill "$SRV_PID" 2>/dev/null; wait "$SRV_PID" 2>/dev/null; SRV_PID=""; }
trap cleanup EXIT INT TERM

start_server() {  # args: extra llama-server flags
  cleanup
  "$LLAMA/llama-server" -m "$1" "${@:2}" \
      --no-mmap -ngl "$NGL" -fa "$FA" -c "$CTX" --parallel 1 --no-warmup \
      --host "$HOST" --port "$PORT" >"$SRVLOG" 2>&1 &
  SRV_PID=$!
}

wait_ready() {  # returns 1 on death/timeout
  for _ in $(seq 1 180); do
    if curl -s "http://$HOST:$PORT/health" 2>/dev/null | grep -q '"status":"ok"'; then return 0; fi
    kill -0 "$SRV_PID" 2>/dev/null || { echo "  !! server exited early; tail of log:"; tail -15 "$SRVLOG" | sed 's/^/     /'; return 1; }
    sleep 2
  done
  echo "  !! server did not become ready in time"; tail -15 "$SRVLOG" | sed 's/^/     /'; return 1
}

# POST Kyle's prompt (wrapped in Gemma turn format) to /completion,
# print "prompt_per_second predicted_per_second"
bench_once() {
  curl -s "http://$HOST:$PORT/completion" -H 'Content-Type: application/json' \
    -d "$(PROMPT="$PROMPT" MAXTOK="$MAXTOK" python3 - <<'PY'
import json, os
p = os.environ["PROMPT"]
wrapped = f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
print(json.dumps({"prompt": wrapped, "n_predict": int(os.environ["MAXTOK"]),
                  "temperature": 0, "cache_prompt": False}))
PY
)" | python3 -c '
import sys, json
try:
    t = json.load(sys.stdin).get("timings", {})
    print(f"{t.get(\"prompt_per_second\",0):.1f} {t.get(\"predicted_per_second\",0):.1f}")
except Exception:
    print("0 0")
'
}

run_config() {  # name, then llama-server args
  local name="$1"; shift
  printf '\n### %s\n' "$name"
  start_server "$@"
  if ! wait_ready; then echo "$name | FAILED | FAILED"; return; fi
  # one warmup, then REPS measured
  bench_once >/dev/null
  local pp_sum=0 tg_sum=0 n=0
  for r in $(seq 1 "$REPS"); do
    read -r pp tg < <(bench_once)
    printf '  rep %d: prompt %5s tok/s   gen %5s tok/s\n' "$r" "$pp" "$tg"
    pp_sum=$(python3 -c "print($pp_sum+$pp)"); tg_sum=$(python3 -c "print($tg_sum+$tg)"); n=$((n+1))
  done
  # capture spec acceptance if present
  local acc; acc=$(grep -aoE 'n_accepted *= *[0-9]+|accept[^,]*' "$SRVLOG" | tail -1)
  RESULTS+=("$name|$(python3 -c "print(f'{$pp_sum/$n:.1f}')")|$(python3 -c "print(f'{$tg_sum/$n:.1f}')")|$acc")
  cleanup
}

echo "=== Jetson Xavier llama.cpp bench (CTX=$CTX FA=$FA REPS=$REPS MAXTOK=$MAXTOK) ==="
echo "=== $(date) ==="
"$LLAMA/llama-server" --version 2>&1 | head -2

declare -a RESULTS=()

# A. Gemma 4 Q4 alone (baseline)            -> Kyle: 298.0 prompt / 58.2 gen
[ -f "$GEMMA" ] && run_config "Gemma4 Q4_K_XL (baseline)" "$GEMMA" \
  || echo "skip: $GEMMA missing"

# B. + Q8 MTP draft (speculative)           -> Kyle: 295.6 prompt / 72.2 gen (1.24x)
[ -f "$GEMMA" ] && [ -f "$GEMMA_MTP" ] && run_config "Gemma4 + MTP (n=$SPEC_N)" "$GEMMA" \
  --model-draft "$GEMMA_MTP" --spec-type draft-mtp --spec-draft-n-max "$SPEC_N" \
  || echo "skip: MTP draft missing"

# C. + mmproj (multimodal loaded)           -> Kyle: 297.4 prompt / 72.2 gen (no text slowdown)
[ -f "$GEMMA" ] && [ -f "$GEMMA_MTP" ] && [ -f "$GEMMA_MMPROJ" ] && \
  run_config "Gemma4 + MTP + mmproj" "$GEMMA" \
  --model-draft "$GEMMA_MTP" --spec-type draft-mtp --spec-draft-n-max "$SPEC_N" \
  --mmproj "$GEMMA_MMPROJ" \
  || echo "skip: mmproj missing"

# D. Qwen3.6-35B-A3B (Q4_K_M present)        -> Kyle: 55 gen (his was Q4_K_XL)
[ -f "$QWEN" ] && run_config "Qwen3.6-35B-A3B Q4_K_M" "$QWEN" \
  || echo "skip: $QWEN missing"

echo
echo "================= SUMMARY (Xavier) ================="
printf '%-34s %12s %10s\n' "config" "prompt t/s" "gen t/s"
printf '%-34s %12s %10s\n' "----------------------------------" "----------" "--------"
for row in "${RESULTS[@]}"; do
  IFS='|' read -r n pp tg acc <<<"$row"
  printf '%-34s %12s %10s   %s\n' "$n" "$pp" "$tg" "$acc"
done
echo
echo "Kyle (M1 Max 64GB) targets:  Gemma base 298/58.2 | +MTP 295.6/72.2 | +mmproj 297.4/72.2 | Qwen 55 gen"
