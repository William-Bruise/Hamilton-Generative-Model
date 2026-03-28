#!/usr/bin/env bash
set -euo pipefail

python src/generate_hyper_anyshape.py \
  --ckpt checkpoints/hyspecnet3d_hamiltonian.pt \
  --out-dir outputs/hyper_anyshape_256x256x191 \
  --num-samples 8 \
  --channels 191 \
  --height 256 \
  --width-out 256
