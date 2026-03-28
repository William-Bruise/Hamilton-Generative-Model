#!/usr/bin/env bash
set -euo pipefail

python src/generate_unconditional.py \
  --ckpt checkpoints/icvl_hamiltonian.pt \
  --out-dir outputs/hyper_uncond \
  --mode hyper \
  --num-samples 32 \
  --height 64 \
  --width-out 64
