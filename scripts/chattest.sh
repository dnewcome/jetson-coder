#!/usr/bin/env bash
# Test the chat endpoint with --jinja (use GGUF's embedded template). Pass extra args to compare.
set -u
M=/mnt/nvme/zen/llm/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
BIN=/mnt/nvme/zen/llm/llama.cpp/build/bin/llama-server
PORT=8099
pkill -f llama-server 2>/dev/null; sleep 3
echo "### server args: $*"
nohup "$BIN" -m "$M" --no-mmap -ngl 999 -fa on -c 8192 --parallel 1 "$@" \
  --host 127.0.0.1 --port "$PORT" >/tmp/chat.log 2>&1 &
SP=$!
for i in $(seq 1 60); do
  grep -q "server is listening" /tmp/chat.log 2>/dev/null && { echo "listening"; break; }
  kill -0 "$SP" 2>/dev/null || { echo "DIED"; tail -8 /tmp/chat.log; exit 1; }
  sleep 3
done
echo "=== /v1/chat/completions ==="
curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Write a one-line Python function to add two numbers."}],"max_tokens":256,"temperature":0}' \
  > /tmp/chat.json 2>/tmp/chat.err
python3 - <<'PY'
import json
try:
    d=json.load(open("/tmp/chat.json"))
    m=d.get("choices",[{}])[0].get("message",{})
    print("content      =", repr(m.get("content"))[:300])
    print("reasoning    =", repr(m.get("reasoning_content"))[:200])
    print("finish       =", d.get("choices",[{}])[0].get("finish_reason"))
    print("gen tok/s    =", round(d.get("timings",{}).get("predicted_per_second",0),1))
except Exception as e:
    print("ERR", e, open("/tmp/chat.json").read()[:300])
PY
kill "$SP" 2>/dev/null
echo "=== DONE ==="
