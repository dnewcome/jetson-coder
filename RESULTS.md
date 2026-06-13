# Benchmark Results — Xavier vs Kyle's M1 Max

Bench prompt (identical to Kyle's):
> "Write a compact Python function that parses a unified diff and returns changed file paths. Then explain two edge cases." (~128 tokens generated)

## Kyle Howells — Apple M1 Max, 64 GB, Metal (reference targets)
| Config | Prompt tok/s | Gen tok/s | Note |
|---|---|---|---|
| Gemma 4 26B-A4B Q4_K_XL | 298.0 | 58.2 | baseline |
| + Q8 MTP draft (n=3) | 295.6 | **72.2** | 1.24× speedup |
| + mmproj (multimodal) | 297.4 | 72.2 | no text slowdown |
| Qwen3.6 35B-A3B Q4_K_XL | — | 55 | "much better" coder, slower |

## This box — Jetson AGX Xavier, 32 GB, CUDA 11.4 / sm_72
Measured by `scripts/bench2.sh` — MAXN + jetson_clocks, `-fa on`, `--no-mmap`, CTX=16384,
3 reps each (variance <2%). Prompt = Kyle's exact coding prompt, wrapped in Gemma turn format.

| Config | Prompt tok/s | Gen tok/s | vs Kyle (gen) |
|---|---|---|---|
| Gemma 4 26B-A4B Q4_K_XL | 68.0 | **16.6** | 0.29× (Kyle 58.2) |
| + Q8 MTP draft (n=3) | 67.8 | **20.7** | 0.29× (Kyle 72.2) |
| Qwen3.6 35B-A3B **Q4_K_M** | 60.1 | **17.1** | 0.31× (Kyle 55) |
| Gemma 4 + MTP + mmproj-F16 | 67.9 | **20.5** | vision loaded = **no text slowdown** (Kyle 72.2) |

**Vision confirmed working**: read "SECRET CODE / 4729" from a test image correctly (13.4 s, 25 tok/s)
after rebuilding with `GGML_CUDA_NO_VMM=ON` (the CUDA-VMM pool aborts on Tegra during clip encode).

**Pi coding agent confirmed**: generation + agentic write/bash tool loop, via `~/.pi/agent/models.json`
→ the Jetson server (`--provider local-llama --model gemma4-local`).

### Headline findings
- **MTP works on Volta: 16.6 → 20.7 gen tok/s = 1.25× speedup**, essentially matching Kyle's **1.24×**.
  Contrary to the [RTX 3090 + Qwen3.6 net-negative result](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090),
  MTP speculative decoding is a clear net win here.
- **~3.4× slower than the M1 Max overall**, which tracks the **memory-bandwidth ratio**
  (~137 GB/s vs ~400 GB/s ≈ 2.9–3.5×). Generation is bandwidth-bound → the gap is physics, not config.
- **Prefill gap is wider (~4.4×, 68 vs 298)** — prompt processing is compute-bound and Volta is weak there.

## Run log / observations
- `-fa on` is fine on sm_72 (earlier "gibberish" was an **unformatted prompt**, not FlashAttention).
- Model load needs **`--no-mmap`** + a page-cache drop, else the 16 GB weight buffer can't be allocated.
- `curl` had to be installed; harness readiness uses **grep server log for "listening"** (curl-health polling was flaky).
- llama-server's **built-in chat template for this Gemma 4 GGUF returns empty** (it's a *thinking* model with
  `<|channel>thought` tags) — `/completion` with explicit `<start_of_turn>` formatting works; chat template needs
  fixing before Pi can use the chat endpoint.
- Context: verified at **both CTX=16384 and CTX=65536** — Kyle's full 64k context **fits in 32 GB**
  (with `--no-mmap` + cache drop) and gen tok/s is identical (16.6 / 20.6 / 17.1), confirming
  generation is context-independent. Setup now matches Kyle's exactly except hardware.
