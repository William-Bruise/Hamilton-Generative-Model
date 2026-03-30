#!/usr/bin/env bash
set -euo pipefail

RUN_ID="$(date +%Y%m%d_%H%M%S)_hyspecnet3d"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${RUN_ID}.train.log"
ERR_LOG="$LOG_DIR/${RUN_ID}.error.log"

{
  echo "[INFO] RUN_ID=$RUN_ID"
  echo "[INFO] TRAIN_LOG=$TRAIN_LOG"
  echo "[INFO] ERR_LOG=$ERR_LOG"
} | tee -a "$TRAIN_LOG"

python src/train_hyspecnet3d_hamiltonian.py \
  --data-root ./data \
  --auto-download \
  --num-shards 1 \
  --train-h 128 \
  --train-w 128 \
  --train-c 224 \
  --batch-size 2 \
  --epochs 20 \
  --out checkpoints/hyspecnet3d_hamiltonian.pt \
  > >(tee -a "$TRAIN_LOG") \
  2> >(tee -a "$ERR_LOG" >&2)
