#!/usr/bin/env bash
# Isolate output-quality: one server load, three prompt paths. Usage: diag3.sh <fa on|off>
set -u
FA="${1:-off}"
M=/mnt/nvme/zen/llm/models/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf
BIN=/mnt/nvme/zen/llm/llama.cpp/build/bin/llama-server
PORT=8099
pkill -f llama-server 2>/dev/null; sleep 3
echo "### FA=$FA"
nohup "$BIN" -m "$M" --no-mmap -ngl 999 -fa "$FA" -c 4096 --parallel 1 \
  --host 127.0.0.1 --port "$PORT" >/tmp/diag.log 2>&1 &
SP=$!
for i in $(seq 1 60); do
  grep -q "server is listening" /tmp/diag.log 2>/dev/null && { echo "listening ~$((i*3))s"; break; }
  kill -0 "$SP" 2>/dev/null || { echo "SERVER DIED"; tail -6 /tmp/diag.log; exit 1; }
  sleep 3
done

probe() { # $1 endpoint  $2 json
  curl -s -m 90 "http://127.0.0.1:$PORT/$1" -H 'Content-Type: application/json' --data-binary "$2"
}

echo "--- (1) /completion RAW prompt ---"
probe completion '{"prompt":"Write a haiku about GPUs.","n_predict":48,"temperature":0}' \
 | python3 -c 'import sys,json;d=json.load(sys.stdin);print("content=",repr(d.get("content"))[:160])'

echo "--- (2) /completion explicit Gemma turn format ---"
probe completion '{"prompt":"<start_of_turn>user\nWrite a haiku about GPUs.<end_of_turn>\n<start_of_turn>model\n","n_predict":48,"temperature":0}' \
 | python3 -c 'import sys,json;d=json.load(sys.stdin);print("content=",repr(d.get("content"))[:160])'

echo "--- (3) /v1/chat/completions ---"
probe v1/chat/completions '{"messages":[{"role":"user","content":"Write a haiku about GPUs."}],"max_tokens":48,"temperature":0}' \
 | python3 -c 'import sys,json;d=json.load(sys.stdin);
c=d.get("choices",[{}])[0].get("message",{}).get("content");print("content=",repr(c)[:160])'

kill "$SP" 2>/dev/null
echo "=== DONE FA=$FA ==="
