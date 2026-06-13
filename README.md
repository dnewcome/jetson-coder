# jetson-coder

Replicating Kyle Howells' [local coding agent setup](https://ikyle.me/blog/2026/how-to-setup-a-local-coding-agent-on-macos)
on a **Jetson AGX Xavier (32 GB)** and benchmarking against his M1 Max numbers.

## Target stack (from the blog)
- **Inference:** llama.cpp with **MTP speculative decoding** (`--spec-type draft-mtp`)
- **Primary model:** `gemma-4-26B-A4B-it-UD-Q4_K_XL` + `Q8_0-MTP` draft + `mmproj-BF16` (vision)
- **Alt model:** `Qwen3.6-35B-A3B` (Q4)
- **Agent:** [Pi](https://github.com/earendil-works/pi) → local `llama-server` (OpenAI-compatible)

## Hardware reality check
| | Kyle's box | This box |
|---|---|---|
| Device | Apple M1 Max | Jetson AGX Xavier |
| RAM | 64 GB unified | 32 GB unified |
| Accel | Metal | CUDA 11.4, **sm_72 (Volta)** |
| Mem BW | ~400 GB/s | ~137 GB/s |
| OS | macOS 15.7 | Ubuntu 20.04 / L4T R35.6.4 (JetPack 5.1.4) |

Lower memory bandwidth + older GPU arch ⇒ **generation tok/s will be a fraction of Kyle's**;
this repo measures *how close* and whether MTP still gives a net win on Volta.

## The Jetson (`dan-jetson-1`, 192.168.68.62)
- SSH alias `jetson1` (see `~/.ssh/config` on the workstation); user `george`.
- Models + llama.cpp live on the 1 TB NVMe at `/mnt/nvme/zen/llm/` (eMMC only has ~7 GB free).
- llama.cpp built with `GGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=72` (needs cmake ≥3.18; pip cmake used).

## Files
- `scripts/bench2.sh` — reliable Xavier benchmark (grep-log readiness); prompt/gen tok/s per config.
- `scripts/run-server.sh` — persistent llama-server: Gemma 4 + MTP + F16 vision + `--jinja`, on `:8080`.
- `scripts/build_novmm.sh` — rebuild llama.cpp with `GGML_CUDA_NO_VMM=ON` (required for vision; see below).
- `scripts/maxperf.sh` — MAXN + jetson_clocks.
- `jetson/gemma-server.service` — systemd unit (auto-start on boot, auto-restart).
- `pi/models.json` — Pi agent config → `http://192.168.68.62:8080/v1` (install to `~/.pi/agent/models.json`).
- `RESULTS.md` — side-by-side comparison vs Kyle.

## Pi coding agent (runs on the workstation → Jetson)
```bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent   # Node.js
cp pi/models.json ~/.pi/agent/models.json
pi --provider local-llama --model gemma4-local --thinking low      # interactive
pi --provider local-llama --model gemma4-local -p "..."            # one-shot
```
Confirmed: generation, the agentic tool loop (write/bash/edit), and vision all work.

## Xavier/Tegra gotchas (hard-won)
- **`GGML_CUDA_NO_VMM=ON` is required.** CUDA Virtual Memory Management is broken on Tegra; the VMM
  pool aborts (`ggml_cuda_pool_vmm::alloc`) — text generation survives but **vision/clip encode crashes**
  the server. Rebuild with VMM disabled (uses the legacy `cudaMalloc` pool).
- **`--no-mmap` + drop page cache** before loading, or the ~16 GB weight buffer won't allocate.
- **`--jinja`** is required for Gemma 4 (a *thinking* model) or the chat endpoint returns empty.
- **Volta has no bf16** → use `mmproj-F16`, not `mmproj-BF16`.
- **`-fa on` is fine** (earlier "gibberish" was an unformatted prompt, not FlashAttention).
- **`pkill -x llama-server`, not `pkill -f`** — `-f` matches any shell whose command line contains the
  string "llama-server" (including your own SSH command), killing the wrong process.
- **Marginal PSU → brownout** under load (clean power-off, no panic). Use a strong supply.

## Usage
```bash
# on the workstation
scp scripts/bench.sh jetson1:/mnt/nvme/zen/llm/
ssh jetson1 'bash /mnt/nvme/zen/llm/bench.sh | tee /mnt/nvme/zen/llm/bench-results.txt'
# try Kyle's exact context size once memory allows:
ssh jetson1 'CTX=65536 bash /mnt/nvme/zen/llm/bench.sh'
```
