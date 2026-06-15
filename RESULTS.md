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

## Cross-platform comparison (same models, prompt, methodology)

Generation tok/s. RTX 4070 = partial offload (`--n-cpu-moe`, attention on GPU / experts on CPU,
CTX=4096); CPU = i9-14900K, 8 P-cores, `-ngl 0`, CTX=2048; both `--no-mmap`. M1 Max from Kyle's post.

| Config | M1 Max (Kyle) | RTX 4070 SUPER 12GB | i9-14900K CPU | Jetson Xavier |
|---|---|---|---|---|
| Gemma 4 26B-A4B Q4_K_XL | 58.2 | 46.4 | 15.1 | 16.6 |
| Gemma 4 + MTP (n=3) | 72.2 | 57.7 | 15.6 | 20.7 |
| Qwen3.6-35B-A3B Q4 | 55 | 55.3 | 12.9 | 17.1 |
| **MTP speedup** | 1.24× | 1.24× | **1.03×** | 1.25× |
| prompt t/s (Gemma+MTP) | 295.6 | 166 | 73 | 67.8 |

Notes:
- **4070 SUPER ≈ M1 Max** for these MoE models despite only 12GB VRAM (partial offload): Qwen ties (55.3 vs 55).
  Works because the models are MoE (~3–4B active) — the CPU-resident experts aren't a heavy tax.
- **MTP speculative decoding is a GPU-only win** (~1.24× on M1 Max / 4070 / Jetson) but **net-zero on CPU**
  (1.03×): draft-verification overhead cancels the gain when there's no GPU to absorb it.
- **CPU-only (i9-14900K) is usable** (~13–16 tok/s) and roughly matches the Jetson Xavier GPU.
- 4070 per-model VRAM fit: Gemma `--n-cpu-moe 16` (~10.7GB), Qwen `--n-cpu-moe 30` (~8.2GB; experts larger, lower values OOM).
- Build gotcha: **CUDA 12.4 needs GCC ≤13** — set `-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13` (system GCC 15 ICEs on FlashAttention).
  Cap `cmake --build -j4` — unlimited `-j` OOMs during nvcc/FA compile.

## Projected configs (ESTIMATES — not measured)

Extrapolated from the measured 4070 SUPER + i9/DDR5 numbers. Generation is bandwidth-bound; the
split is GPU-bandwidth (offloaded layers) + CPU-RAM-bandwidth (CPU-resident MoE experts).

### RTX 3060 12 GB + 64 GB DDR4 + 8-core i7
Same 12 GB VRAM as the 4070 → same offload tuning (`--n-cpu-moe 16` Gemma / `30` Qwen). 3060 is
~0.71× the 4070's bandwidth (360 vs 504 GB/s) and DDR4 (~50 GB/s) is ~0.55× the DDR5 box's CPU bandwidth,
so the CPU-resident experts are slower than in the measured runs.

| Config | 4070+DDR5 (measured) | 3060 12GB + DDR4 (est.) |
|---|---|---|
| Gemma 4 + MTP | 57.7 | ~36–44 |
| Gemma 4 baseline | 46.4 | ~30–36 |
| Qwen3.6-35B | 55.3 | ~28–36 |
| CPU-only Gemma | 15.1 | ~8–10 |
| CPU-only Qwen | 12.9 | ~7–8 |

- **Ranking flips vs DDR5:** on DDR4, Qwen drops more than Gemma (12 GB holds fewer of Qwen's larger
  experts → more on slow DDR4), so **Gemma + MTP is the daily driver** (~36–44 tok/s, very usable).
- **64 GB DDR4 is a capacity win, not bandwidth:** removes RAM limits (CPU Qwen w/o swap, both models
  loaded, big context, room for larger models) but doesn't speed the CPU path.
- **CPU-only ~halves** vs the DDR5 box (~7–10 tok/s) — fallback, not the main path.
- Build: 3060 is Ampere → `-DCMAKE_CUDA_ARCHITECTURES=86`; same g++-13 / `-j4` caveats.
- 3060 **8 GB** variant: more experts forced onto DDR4 → roughly ~25–32 (Gemma+MTP), Qwen tighter.

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
