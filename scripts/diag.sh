#!/usr/bin/env bash
# Coherence + speed probe. Usage: diag.sh <fa on|off>
set -u
FA="${1:-on}"
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
  kill -0 "$SP" 2>/dev/null || { echo "SERVER DIED"; tail -5 /tmp/diag.log; exit 1; }
  sleep 3
done

REQ='{"messages":[{"role":"user","content":"Write a haiku about GPUs."}],"max_tokens":48,"temperature":0,"cache_prompt":false}'
# warmup
curl -s -m 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "$REQ" >/dev/null 2>&1
# measured
curl -s -m 90 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "$REQ" >/tmp/chat.json 2>/tmp/chat.err
python3 - <<'PY'
import json
try:
    d=json.load(open("/tmp/chat.json"))
    t=d.get("timings",{})
    msg=d["choices"][0]["message"]["content"]
    print("prompt_t/s =", round(t.get("prompt_per_second",0),1), " gen_t/s =", round(t.get("predicted_per_second",0),1))
    print("CONTENT:", repr(msg)[:200])
except Exception as e:
    print("PARSE ERROR:", e); print(open("/tmp/chat.json").read()[:300])
PY
kill "$SP" 2>/dev/null
echo "=== DONE FA=$FA ==="
