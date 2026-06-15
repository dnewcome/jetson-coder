#!/usr/bin/env bash
# dense_bench.sh — Qwen3.6-27B DENSE on the workstation. Dense → no MoE experts, so offload is
# plain -ngl N (layers). MODE=gpu tunes -ngl to fill 12GB; MODE=cpu uses -ngl 0.
#   NGL=40 MODE=gpu ./dense_bench.sh   |   MODE=cpu ./dense_bench.sh
set -u
B=$HOME/llm/llama.cpp/build/bin/llama-server
M=$HOME/llm/models/Qwen3.6-27B-UD-Q4_K_XL.gguf
MODE=${MODE:-gpu}; PORT=8099; CTX=${CTX:-4096}; REPS=${REPS:-3}; MAXTOK=${MAXTOK:-128}
THREADS=${THREADS:-8}; SRVLOG=/tmp/dense_srv.log
export PROMPT='Write a compact Python function that parses a unified diff and returns changed file paths. Then explain two edge cases.'
if [ "$MODE" = cpu ]; then DEV=(-ngl 0 -t "$THREADS"); TAG="CPU i9 (${THREADS}t)"; else DEV=(-ngl "${NGL:-40}" -t "$THREADS"); TAG="4070 (-ngl ${NGL:-40})"; fi

SP=""
stop(){ [ -n "$SP" ] && kill "$SP" 2>/dev/null; pkill -x llama-server 2>/dev/null; sleep 2; SP=""; }
trap stop EXIT
measure(){
  MAXTOK="$MAXTOK" python3 - > /tmp/dense_req.json <<'PY'
import json,os
p=os.environ["PROMPT"]; w=f"<start_of_turn>user\n{p}<end_of_turn>\n<start_of_turn>model\n"
print(json.dumps({"prompt":w,"n_predict":int(os.environ["MAXTOK"]),"temperature":0,"cache_prompt":False}))
PY
  curl -s -m 600 "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' --data-binary @/tmp/dense_req.json \
   | python3 -c 'import sys,json
t=json.load(sys.stdin).get("timings",{})
print(round(t.get("prompt_per_second",0),1), round(t.get("predicted_per_second",0),1))'
}
stop
echo "### Qwen3.6-27B dense  [$TAG]  CTX=$CTX"
"$B" -m "$M" "${DEV[@]}" --no-mmap -fa on -c "$CTX" --parallel 1 --host 127.0.0.1 --port "$PORT" >"$SRVLOG" 2>&1 &
SP=$!
ok=0; for i in $(seq 1 90); do grep -q "server is listening" "$SRVLOG" 2>/dev/null && { ok=1; break; }; kill -0 "$SP" 2>/dev/null || break; sleep 2; done
[ "$ok" -ne 1 ] && { echo "LOAD_FAILED"; tail -6 "$SRVLOG"; exit 1; }
vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
echo "  loaded (VRAM ${vram}MiB)"
measure >/dev/null
ps=0; gs=0
for r in $(seq 1 "$REPS"); do read -r pp tg < <(measure); echo "  rep$r: prompt $pp / gen $tg"; ps=$(python3 -c "print($ps+$pp)"); gs=$(python3 -c "print($gs+$tg)"); done
echo "  AVG: prompt $(python3 -c "print(f'{$ps/$REPS:.1f}')") / gen $(python3 -c "print(f'{$gs/$REPS:.1f}')")  [$TAG, VRAM ${vram}MiB]"
