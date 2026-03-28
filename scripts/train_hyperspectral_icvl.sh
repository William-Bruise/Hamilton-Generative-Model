#!/usr/bin/env bash
set -euo pipefail

python src/train_universal_hamiltonian.py \
  --dataset-type icvl_31 \
  --data-root ./data \
  --hyper-patch 64 \
  --hyper-stride 32 \
  --icvl-max-files 120 \
  --batch-size 8 \
  --epochs 20 \
  --out checkpoints/icvl_hamiltonian.pt
