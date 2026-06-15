#!/usr/bin/env bash
# gpu_bench.sh — RTX 4070 partial-offload bench. Per-model --n-cpu-moe to fill the 12GB card.
# Attention/non-expert weights on GPU, enough MoE experts spilled to CPU to fit. --no-mmap.
set -u
B=$HOME/llm/llama.cpp/build/bin/llama-server
MD=$HOME/llm/models
GEMMA=$MD/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
GMTP=$MD/gemma-4-26B-A4B-it-Q8_0-MTP.gguf
QWEN=$MD/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
PORT=8099; CTX=4096; REPS=3; MAXTOK=128; SRVLOG=/tmp/gpu_srv.log
export PROMPT='Write a compact Python function that parses a unified diff and returns changed file paths. Then explain two edge cases.'

SP=""
stop(){ [ -n "$SP" ] && kill "$SP" 2>/dev/null; pkill -x llama-server 2>/dev/null; sleep 2; SP=""; }
trap stop EXIT

measure(){
  MAXTOK="$MAXTOK" python3 - > /tmp/gpu_req.json <<'PY'
import json,os
p=os.environ["PROMPT"]
w=f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
print(json.dumps({"prompt":w,"n_predict":int(os.environ["MAXTOK"]),"temperature":0,"cache_prompt":False}))
PY
  curl -s -m 600 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' --data-binary @/tmp/gpu_req.json \
   | python3 -c 'import sys,json
t=json.load(sys.stdin).get("timings",{})
print(round(t.get("prompt_per_second",0),1), round(t.get("predicted_per_second",0),1))'
}

run(){ local name="$1"; shift
  stop
  echo "### $name"
  "$B" "$@" --no-mmap -ngl 999 -fa on -c "$CTX" --parallel 1 --host 127.0.0.1 --port "$PORT" >"$SRVLOG" 2>&1 &
  SP=$!
  local ready=0 t0=$SECONDS
  for i in $(seq 1 60); do
    grep -q "server is listening" "$SRVLOG" 2>/dev/null && { ready=1; break; }
    kill -0 "$SP" 2>/dev/null || break; sleep 2
  done
  if [ "$ready" -ne 1 ]; then echo "$name | LOAD_FAILED"; tail -5 "$SRVLOG"; RESULTS+="
$name | LOAD_FAILED"; return; fi
  local vram; vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  echo "  (loaded ~$((SECONDS-t0))s, VRAM ${vram}MiB)"
  measure >/dev/null
  local ps=0 gs=0
  for r in $(seq 1 "$REPS"); do read -r pp tg < <(measure); echo "  rep$r: prompt $pp / gen $tg"; ps=$(python3 -c "print($ps+$pp)"); gs=$(python3 -c "print($gs+$tg)"); done
  local PA GA; PA=$(python3 -c "print(f'{$ps/$REPS:.1f}')"); GA=$(python3 -c "print(f'{$gs/$REPS:.1f}')")
  echo "  AVG: prompt $PA / gen $GA   (VRAM ${vram}MiB)"
  RESULTS+="
$name | $PA | $GA | ${vram}MiB"
}

echo "=== RTX 4070 SUPER partial-offload bench (CTX=$CTX, --no-mmap, fa on) ==="
RESULTS=""
run "Gemma4 26B-A4B Q4_K_XL"      -m "$GEMMA" --n-cpu-moe 16
run "Gemma4 + MTP draft (n=3)"    -m "$GEMMA" --n-cpu-moe 18 --model-draft "$GMTP" --spec-type draft-mtp --spec-draft-n-max 3
run "Qwen3.6-35B-A3B Q4_K_M"      -m "$QWEN"  --n-cpu-moe 30
echo; echo "===== SUMMARY [RTX 4070] (config | prompt t/s | gen t/s | VRAM) ====="
echo "$RESULTS"
