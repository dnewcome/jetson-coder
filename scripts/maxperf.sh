#!/usr/bin/env bash
# Put the Jetson AGX Xavier into maximum-performance mode for benchmarking.
set -u
echo "=== nvpmodel: set MAXN (mode 0 = highest) ==="
sudo nvpmodel -m 0
sudo nvpmodel -q 2>/dev/null | grep -i "power mode"
echo "=== jetson_clocks: lock CPU/GPU/EMC to max ==="
sudo jetson_clocks
echo "=== current clocks ==="
sudo jetson_clocks --show 2>/dev/null | grep -iE "GPU|EMC|MaxFreq|Online" | head -20
echo "=== DONE ==="
