#!/usr/bin/env bash
set -euo pipefail

python src/generate_hyper_anyshape.py \
  --ckpt checkpoints/hyspecnet3d_hamiltonian.pt \
  --out-dir outputs/hyper_anyshape_512x512x31 \
  --num-samples 8 \
  --channels 31 \
  --height 512 \
  --width-out 512
