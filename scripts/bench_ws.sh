#!/usr/bin/env bash
# bench_ws.sh — workstation benchmark (RTX 4070 partial-offload OR CPU-only).
# Same models, prompt, and methodology as the Jetson's bench2.sh.
#   MODE=gpu ./bench_ws.sh     # 4070: attention on GPU, MoE experts on CPU (--n-cpu-moe)
#   MODE=cpu ./bench_ws.sh     # i9-14900K, -ngl 0
# Uses --no-mmap (same as Jetson). NCMOE tunable for the GPU expert/VRAM split.
set -u
BIN=${BIN:-$HOME/llm/llama.cpp/build/bin/llama-server}
MD=${MD:-$HOME/llm/models}
GEMMA=$MD/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
GMTP=$MD/gemma-4-26B-A4B-it-Q8_0-MTP.gguf
QWEN=$MD/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
MODE=${MODE:-gpu}
PORT=8099; CTX=${CTX:-16384}; REPS=${REPS:-3}; MAXTOK=${MAXTOK:-128}
NCMOE=${NCMOE:-99}            # GPU mode: how many layers' experts to keep on CPU (99 = all)
THREADS=${THREADS:-8}         # CPU mode: P-core count on the 14900K
SRVLOG=/tmp/ws_srv.log
export PROMPT='Write a compact Python function that parses a unified diff and returns changed file paths. Then explain two edge cases.'

if [ "$MODE" = cpu ]; then
  DEV=(-ngl 0 -t "$THREADS"); FA=on; TAG="CPU i9-14900K (${THREADS}t)"
else
  DEV=(-ngl 999 --n-cpu-moe "$NCMOE"); FA=on; TAG="RTX4070 (ncmoe=$NCMOE)"
fi

SP=""
stop(){ [ -n "$SP" ] && kill "$SP" 2>/dev/null; pkill -x llama-server 2>/dev/null; sleep 2; SP=""; }
trap stop EXIT

measure(){
  MAXTOK="$MAXTOK" python3 - > /tmp/ws_req.json <<'PY'
import json,os
p=os.environ["PROMPT"]
w=f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
print(json.dumps({"prompt":w,"n_predict":int(os.environ["MAXTOK"]),"temperature":0,"cache_prompt":False}))
PY
  curl -s -m 600 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' --data-binary @/tmp/ws_req.json \
   | python3 -c 'import sys,json
t=json.load(sys.stdin).get("timings",{})
print(round(t.get("prompt_per_second",0),1), round(t.get("predicted_per_second",0),1))'
}

run(){ local name="$1"; shift
  stop
  echo "### $name"
  "$BIN" "$@" "${DEV[@]}" --no-mmap -fa "$FA" -c "$CTX" --parallel 1 \
        --host 127.0.0.1 --port "$PORT" >"$SRVLOG" 2>&1 &
  SP=$!
  local ready=0 t0=$SECONDS
  for i in $(seq 1 200); do
    grep -q "server is listening" "$SRVLOG" 2>/dev/null && { ready=1; break; }
    kill -0 "$SP" 2>/dev/null || break; sleep 3
  done
  if [ "$ready" -ne 1 ]; then echo "$name | LOAD_FAILED"; tail -6 "$SRVLOG"; RESULTS+="
$name | LOAD_FAILED"; return; fi
  echo "  (loaded in ~$((SECONDS-t0))s)"
  measure >/dev/null
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

echo "=== Workstation bench [$TAG] (CTX=$CTX REPS=$REPS MAXTOK=$MAXTOK, --no-mmap) ==="
RESULTS=""
run "Gemma4 26B-A4B Q4_K_XL baseline" -m "$GEMMA"
run "Gemma4 + MTP draft (n=3)"        -m "$GEMMA" --model-draft "$GMTP" --spec-type draft-mtp --spec-draft-n-max 3
run "Qwen3.6-35B-A3B Q4_K_M"          -m "$QWEN"
echo; echo "============ SUMMARY [$TAG] (config | prompt t/s | gen t/s) ============"
echo "$RESULTS"
