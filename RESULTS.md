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

## Coding *quality* (benchmarks — speed ≠ ability)

All numbers above are **speed**. Coding *ability* (community/blog benchmarks, April 2026 — recent models,
varying harnesses, treat as directional):

| Model | SWE-bench Verified | Coding rank | Notes |
|---|---|---|---|
| Qwen 3.6-27B **dense** | ~77% (≈ +4 vs MoE) | 🥇 best | gap widens on hard agentic/multi-step (~11pt Terminal-Bench) |
| Qwen 3.6-35B-A3B **MoE** | 73.4% | 🥈 close | activates only ~3B/token (9 of 256 experts) |
| Gemma 4 26B-A4B | 52.0% | 🥉 well behind | ~21pt behind Qwen; wins on vision + size only |

Sources: [Qwen 35B-A3B vs Gemma 4](https://pub.towardsai.net/i-tested-alibaba-qwen3-6-35b-a3b-30cc4658a382),
[27B dense vs 35B-A3B](https://www.aimadetools.com/blog/qwen-3-6-27b-vs-35b-a3b/),
[Qwen3.6-27B official](https://qwen.ai/blog?id=qwen3.6-27b).

**Quality × speed × 12GB hardware → daily driver = Qwen 3.6-35B-A3B MoE** (73% quality at 55 tok/s):
nearly as fast as Gemma on the 4070 but a *far* better coder (73 vs 52). Gemma 4 only if you need vision/max speed.
The dense 27B is the best coder but only ~4pt better than the MoE on SWE-bench (bigger on hard agentic work) —
so the 24GB upgrade is justified by *quality* only if your work is the hard multi-step kind; otherwise the MoE suffices.

## Local coding benchmark — MEASURED (wartron/bench_llamacpp_jetson rubric)

Ran the [bench_llamacpp_jetson](https://github.com/wartron/bench_llamacpp_jetson) 8-task coding
suite on our three models on the **RTX 4070** (deterministic keyword rubric, passed/total markers).

| Model | Rubric score | avg tok/s |
|---|---|---|
| Qwen 3.6-27B dense | **43/45 (95.6%)** | 5.0 |
| Qwen 3.6-35B-A3B MoE | 42/45 (93.3%) | 49.8 |
| Gemma 4 26B-A4B | 42/45 (93.3%) | 46.9 |

Per-task: all three differ on only **1–2 markers total** (full marks on lru_cache, rust iterator,
regex parser, edit-distance, refactor). They're near-ceiling and effectively **tied**, dense by a hair.

**Two findings:**
- **This rubric saturates.** Easy tasks + substring matching can't resolve real quality gaps — note this
  contradicts the SWE-bench spread (Qwen 73 vs Gemma 52 above). Routine code: all three are excellent and
  indistinguishable; the differences only show up on hard, execution-verified benchmarks.
- **Thinking models need a big token budget** (methodology trap): a first run at `max_tokens=3072` with
  thinking ON gave *inverted, bogus* scores — the models burned the whole budget on hidden
  `reasoning_content` and emitted no answer (dense hit the cap 7/8, left 6/8 empty → falsely "worst").
  Fix: disable thinking (`chat_template_kwargs.enable_thinking=false`) for the rubric. With thinking ON the
  dense model would need ~8k tokens/task → **~3.5 h** at 5 tok/s — impractical to measure on this hardware.

## Dense vs MoE on limited VRAM (measured) — the decisive result

Qwen 3.6-27B **dense** (Q4_K_XL, 17.6GB) — the HN "quality" pick — same prompt/methodology:

| Platform | gen t/s | vs Qwen 35B-A3B MoE |
|---|---|---|
| RTX 4070 12GB (-ngl 30, partial) | **4.9** | 55.3 → **11× slower** |
| i9-14900K CPU-only | **2.8** | 12.9 |

- **Why:** a dense model reads *all* 27B params every token, so the ~8GB stranded on CPU throttles
  everything. MoE reads only ~3B active params/token → the CPU-resident part is small → partial offload
  barely hurts. **Partial offload rescues MoE; it kills dense.**
- **Consequence:** the dense quality model is **unusable on 12GB (~5 tok/s)**. It needs **full VRAM fit**.
  On a 24GB card (e.g. RTX 3090) the 17.6GB model fits entirely → expect ~30–50 tok/s (HN: single 3090
  ~70–80 for 27B dense). This is the **strongest case for the 24GB upgrade** — it's the only path to the
  quality model at usable speed. On 12GB, stick with the MoE models.

## Target benchmarks — HN community ("local model for daily coding" thread)

Reported gen tok/s for the **same models** (Gemma 4 26B-A4B / Qwen 3.6-35B-A3B, Q4-class), from
[HN 48542100](https://news.ycombinator.com/item?id=48542100). Use these as targets. Our **measured**
boxes are marked ◀; the single biggest determinant is **whether the model fits entirely in VRAM**.

| Tier | gen t/s | Hardware | VRAM fit? | ~Cost |
|---|---|---|---|---|
| **S** | ~120–160 | Dual RTX 3090 (48GB), RTX Pro 6000 Blackwell | full | $1.4k–7k+ |
| **A** | ~50–80 | Single RTX 3090/4090 (24GB), Strix Halo 128GB, M1/M4/M5 Max | full or near | $0.7k–4k |
| **A** | **55** ◀ | **RTX 4070 SUPER 12GB (ours)** | partial (`--n-cpu-moe`) | (have it) |
| **A** | **55–72** ◀ | **M1 Max 64GB (Kyle, ref)** | full | — |
| **B** | ~20–30 | M4 Pro 48GB, low-bandwidth unified | full/slow | — |
| **B** | **17–21** ◀ | **Jetson AGX Xavier (ours)** | full, 137 GB/s | — |
| **C** | ~7–17 | CPU-only, old Xeons, Optane | n/a | — |
| **C** | **13–15** ◀ | **i9-14900K CPU DDR5 (ours)** | n/a | (have it) |

Community consensus worth weighing alongside speed:
- **VRAM fit is the cliff:** full-fit 24GB+ cards hit ~120–150; our 12GB card is capped at ~55 purely by partial offload.
- **Qwen 3.6-27B *dense* = best coding quality** (~3× slower than A3B MoE, but "the sweet spot"); A3B MoE is the speed pick.
- Memory bandwidth (not compute) governs generation — confirms our cross-platform results.
- `llama.cpp` over Ollama; `preserve_thinking`/cache-reuse to avoid reprocessing on thinking models.

### Optimal setup for *this* hardware (i9-14900K + 4070 SUPER + Jetsons)
- **Highest-leverage upgrade: a single 24GB GPU (used RTX 3090 ~$700–900).** It makes the 16–22GB models
  fit **entirely in VRAM** → no CPU experts → realistically **~2.5–3× our current 55 → ~120–150 tok/s** (Tier S),
  *and* unlocks **Qwen 3.6-27B dense** (the quality pick) at usable speed. Keep the i9-14900K.
- **Bump RAM 30 → 64GB+** (cheap): removes the swap/context limits we hit running CPU-only Qwen.
- **The 4070 SUPER is already solid** (ties the M1 Max) — if not upgrading, run **Gemma 4 + MTP** as the daily
  driver (~58 tok/s) and reserve CPU-only as fallback.
- **Jetsons:** not the speed play (17–21 tok/s); best as always-on low-power inference or the 2-board
  experiment — not your main coding driver.

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
