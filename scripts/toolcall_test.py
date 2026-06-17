#!/usr/bin/env python3
"""toolcall_test.py — deterministic tool-calling reliability probe for a running llama-server.

Tool-calling correctness is a property of (model + quant + chat template + sampling), NOT the
hardware. The same GGUF at temp 0 emits the same tokens on the Jetson and the 4070, so this only
needs to run once per model (we run it on the workstation where the models live).

Talks to the OpenAI-compatible /v1/chat/completions endpoint with a `tools` array and tool_choice
"auto". Scores three things the HN thread flagged as MoE weak spots:
  1. CALLS WHEN IT SHOULD   — emits a tool_call for prompts that require a tool.
  2. PICKS THE RIGHT TOOL    — correct function name + required args present with valid JSON + right values.
  3. ABSTAINS WHEN IT SHOULDN'T CALL — no tool_call for a chit-chat prompt (over-eager calling = fail).

Usage: BASE=http://127.0.0.1:8099/v1 MODEL=ignored python3 toolcall_test.py
Exit 0 always; prints a per-case table and a final "name | passed/total | pct".
"""
import json, os, sys, re
import requests

BASE = os.environ.get("BASE", "http://127.0.0.1:8099/v1")
NAME = os.environ.get("NAME", "model")
NO_THINK = bool(os.environ.get("BENCH_NO_THINK"))
MAXTOK = int(os.environ.get("MAXTOK", "512"))

# --- tool catalogue (offered on every request so the model must SELECT) ---
TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": "City name, e.g. Tokyo"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["location"]}}},
    {"type": "function", "function": {
        "name": "book_flight",
        "description": "Book a one-way flight between two airports on a date.",
        "parameters": {"type": "object", "properties": {
            "origin": {"type": "string", "description": "3-letter IATA origin code"},
            "destination": {"type": "string", "description": "3-letter IATA destination code"},
            "date": {"type": "string", "description": "YYYY-MM-DD"}},
            "required": ["origin", "destination", "date"]}}},
    {"type": "function", "function": {
        "name": "run_sql",
        "description": "Run a read-only SQL query against the analytics database.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "A SELECT statement"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "create_file",
        "description": "Create a text file at a path with given contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "contents": {"type": "string"}},
            "required": ["path", "contents"]}}},
]

# Each case: prompt, expected tool ("" => expect NO call), and a validator(args)->(ok,msg).
def v_weather(a):
    loc = str(a.get("location", "")).lower()
    return ("tokyo" in loc, f"location={a.get('location')!r}")

def v_flight(a):
    o = str(a.get("origin", "")).upper(); d = str(a.get("destination", "")).upper()
    date = str(a.get("date", ""))
    ok = ("SFO" in o) and ("JFK" in d) and bool(re.match(r"2026-0?7-0?1", date))
    return (ok, f"origin={o!r} dest={d!r} date={date!r}")

def v_sql(a):
    q = str(a.get("query", "")).lower()
    return (("select" in q and "users" in q), f"query={a.get('query')!r}")

def v_file(a):
    p = str(a.get("path", "")); c = str(a.get("contents", ""))
    return (p.endswith("hello.py") and "print" in c.lower(), f"path={p!r} contents~{c[:40]!r}")

def v_none(a):
    return (False, "should not have been called")

CASES = [
    ("What's the current weather in Tokyo? Use celsius.", "get_weather", v_weather),
    ("Book me a one-way flight from San Francisco (SFO) to New York JFK on July 1st, 2026.", "book_flight", v_flight),
    ("How many rows are in the users table? Query the analytics DB.", "run_sql", v_sql),
    ("Create a file hello.py that prints Hello, world.", "create_file", v_file),
    ("Hi there! Briefly, what are you good at? Just chat, do not call any tool.", "", v_none),
]

def chat(messages):
    body = {"model": NAME, "messages": messages, "tools": TOOLS, "tool_choice": "auto",
            "temperature": 0, "max_tokens": MAXTOK}
    if NO_THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    r = requests.post(f"{BASE}/chat/completions", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]

def first_toolcall(msg):
    tcs = msg.get("tool_calls") or []
    if not tcs:
        return None
    fn = tcs[0].get("function", {})
    name = fn.get("name", "")
    raw = fn.get("arguments", "")
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except Exception:
        args = {"__parse_error__": raw}
    return name, args

def main():
    passed = 0
    print(f"\n=== tool-calling: {NAME} ({BASE}) ===")
    for prompt, expect, validate in CASES:
        try:
            msg = chat([{"role": "user", "content": prompt}])
        except Exception as e:
            print(f"  FAIL  [{expect or 'no-call'}]  request error: {e}")
            continue
        tc = first_toolcall(msg)
        if expect == "":
            ok = tc is None
            detail = "no tool_call" if ok else f"called {tc[0]}({tc[1]})"
        elif tc is None:
            ok = False
            detail = f"NO tool_call (content={ (msg.get('content') or '')[:50]!r})"
        elif tc[0] != expect:
            ok = False
            detail = f"wrong tool: {tc[0]} (wanted {expect})"
        elif "__parse_error__" in tc[1]:
            ok = False
            detail = f"invalid JSON args: {tc[1]['__parse_error__']!r:.60}"
        else:
            ok, why = validate(tc[1])
            detail = ("args ok " if ok else "bad args ") + why
        passed += int(ok)
        print(f"  {'PASS' if ok else 'FAIL'}  [{expect or 'no-call'}]  {detail}")
    pct = 100.0 * passed / len(CASES)
    print(f"  ---\n  {NAME} | {passed}/{len(CASES)} | {pct:.0f}%")
    # machine-readable line for the runner to grep
    print(f"RESULT\t{NAME}\t{passed}\t{len(CASES)}\t{pct:.0f}")

if __name__ == "__main__":
    main()
