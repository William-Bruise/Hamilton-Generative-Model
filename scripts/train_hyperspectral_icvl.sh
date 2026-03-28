#!/usr/bin/env bash
set -euo pipefail

RUN_ID="$(date +%Y%m%d_%H%M%S)_icvl"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
TRAIN_LOG="$LOG_DIR/${RUN_ID}.train.log"
ERR_LOG="$LOG_DIR/${RUN_ID}.error.log"

{
  echo "[INFO] RUN_ID=$RUN_ID"
  echo "[INFO] TRAIN_LOG=$TRAIN_LOG"
  echo "[INFO] ERR_LOG=$ERR_LOG"
} | tee -a "$TRAIN_LOG"

python src/train_universal_hamiltonian.py \
  --dataset-type icvl_31 \
  --data-root ./data \
  --hyper-patch 64 \
  --hyper-stride 32 \
  --icvl-max-files 120 \
  --batch-size 8 \
  --epochs 20 \
  --out checkpoints/icvl_hamiltonian.pt \
  > >(tee -a "$TRAIN_LOG") \
  2> >(tee -a "$ERR_LOG" >&2)
