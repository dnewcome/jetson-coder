#!/usr/bin/env bash
# bench2.sh — reliable Xavier benchmark, modeled on the proven diag3 flow.
# Readiness = grep server log for "listening" (curl-health polling was flaky).
set -u
BIN=/mnt/nvme/zen/llm/llama.cpp/build/bin/llama-server
GD=/mnt/nvme/zen/llm/models/gemma-4-26B-A4B-it-GGUF
GEMMA=$GD/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
GMTP=$GD/MTP/gemma-4-26B-A4B-it-Q8_0-MTP.gguf
QWEN=/mnt/nvme/zen/llm/models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
PORT=8099; CTX=${CTX:-16384}; REPS=${REPS:-3}; MAXTOK=${MAXTOK:-128}
SRVLOG=/tmp/bench_srv.log
export PROMPT='Write a compact Python function that parses a unified diff and returns changed file paths. Then explain two edge cases.'

SP=""
stop(){ [ -n "$SP" ] && kill "$SP" 2>/dev/null; pkill -x llama-server 2>/dev/null; sleep 2; SP=""; }
trap stop EXIT

measure(){ # -> "promptPS genPS"
  MAXTOK="$MAXTOK" python3 - > /tmp/req.json <<'PY'
import json,os
p=os.environ["PROMPT"]
w=f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
print(json.dumps({"prompt":w,"n_predict":int(os.environ["MAXTOK"]),"temperature":0,"cache_prompt":False}))
PY
  curl -s -m 120 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' --data-binary @/tmp/req.json \
   | python3 -c 'import sys,json
t=json.load(sys.stdin).get("timings",{})
print(round(t.get("prompt_per_second",0),1), round(t.get("predicted_per_second",0),1))'
}

run(){ local name="$1"; shift
  stop; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  echo "### $name"
  nohup "$BIN" "$@" --no-mmap -ngl 999 -fa on -c "$CTX" --parallel 1 \
        --host 127.0.0.1 --port "$PORT" >"$SRVLOG" 2>&1 &
  SP=$!
  local ready=0 t0=$SECONDS
  for i in $(seq 1 90); do
    grep -q "server is listening" "$SRVLOG" 2>/dev/null && { ready=1; break; }
    kill -0 "$SP" 2>/dev/null || break; sleep 3
  done
  if [ "$ready" -ne 1 ]; then echo "$name | LOAD_FAILED"; tail -6 "$SRVLOG"; RESULTS+="
$name | LOAD_FAILED"; return; fi
  echo "  (loaded in ~$((SECONDS-t0))s)"
  measure >/dev/null    # warmup
  local ps=0 gs=0
  for r in $(seq 1 "$REPS"); do
    read -r pp tg < <(measure); echo "  rep$r: prompt $pp / gen $tg tok/s"
    ps=$(python3 -c "print($ps+$pp)"); gs=$(python3 -c "print($gs+$tg)")
  done
  local PA GA; PA=$(python3 -c "print(f'{$ps/$REPS:.1f}')"); GA=$(python3 -c "print(f'{$gs/$REPS:.1f}')")
  echo "  AVG: prompt $PA / gen $GA tok/s"
  RESULTS+="
$name | $PA | $GA"
}

echo "=== Xavier bench2 (CTX=$CTX REPS=$REPS MAXTOK=$MAXTOK, fa on, no-mmap, MAXN) ==="
RESULTS=""
run "Gemma4 26B-A4B Q4_K_XL baseline"  -m "$GEMMA"
run "Gemma4 + MTP draft (n=3)"         -m "$GEMMA" --model-draft "$GMTP" --spec-type draft-mtp --spec-draft-n-max 3
run "Qwen3.6-35B-A3B Q4_K_M"           -m "$QWEN"
echo; echo "================ SUMMARY (config | prompt t/s | gen t/s) ================"
echo "$RESULTS"
echo
echo "Kyle M1 Max: Gemma base 58.2 gen | +MTP 72.2 gen | Qwen 55 gen"
