#!/usr/bin/env bash
# Rebuild llama.cpp with CUDA VMM disabled — required for multimodal (vision) on Tegra/Xavier,
# where ggml_cuda_pool_vmm::alloc aborts. Uses the legacy cudaMalloc pool instead.
cd /mnt/nvme/zen/llm/llama.cpp || exit 1
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export CUDACXX=/usr/local/cuda/bin/nvcc
echo "=== RECONFIGURE (GGML_CUDA_NO_VMM=ON) $(date '+%F %T') ==="
cmake -B build -DGGML_CUDA_NO_VMM=ON -DGGML_CCACHE=ON || { echo "=== CONFIGURE_FAILED ==="; exit 2; }
echo "=== BUILD $(date '+%F %T') ==="
cmake --build build -j6 || { echo "=== BUILD_FAILED ==="; exit 3; }
echo "=== BUILD_OK $(date '+%F %T') ==="
./build/bin/llama-server --version 2>&1 | head -2
