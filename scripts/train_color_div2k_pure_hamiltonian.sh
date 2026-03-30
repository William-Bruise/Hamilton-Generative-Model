#!/usr/bin/env bash
set -euo pipefail

RUN_ID="$(date +%Y%m%d_%H%M%S)_div2k_pure"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${RUN_ID}.train.log"
ERR_LOG="$LOG_DIR/${RUN_ID}.error.log"

python src/train_universal_pure_hamiltonian.py \
  --dataset-type div2k \
  --data-root ./data \
  --resize 256x256 \
  --preprocess "${PREPROCESS:-resize}" \
  --batch-size 8 \
  --epochs 2000 \
  --out checkpoints/div2k_pure_hamiltonian.pt \
  > >(tee -a "$TRAIN_LOG") \
  2> >(tee -a "$ERR_LOG" >&2)
