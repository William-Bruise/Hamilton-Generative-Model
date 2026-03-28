#!/usr/bin/env bash
set -euo pipefail

python src/generate_unconditional.py \
  --ckpt checkpoints/div2k_hamiltonian.pt \
  --out-dir outputs/color_uncond \
  --mode rgb \
  --num-samples 32 \
  --height 256 \
  --width-out 256
