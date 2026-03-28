#!/usr/bin/env bash
set -euo pipefail

python src/train_hyspecnet3d_hamiltonian.py \
  --data-root ./data \
  --auto-download \
  --num-shards 1 \
  --train-h 128 \
  --train-w 128 \
  --train-c 224 \
  --batch-size 2 \
  --epochs 20 \
  --out checkpoints/hyspecnet3d_hamiltonian.pt
