#!/usr/bin/env bash
set -euo pipefail

python src/train_universal_hamiltonian.py \
  --dataset-type div2k \
  --data-root ./data \
  --resize 256x256 \
  --batch-size 8 \
  --epochs 20 \
  --out checkpoints/div2k_hamiltonian.pt
