#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/generate_hyper_anyshape.sh 191x256x256 outputs/hyper_191x256x256 8
#   bash scripts/generate_hyper_anyshape.sh 31x512x512 outputs/hyper_31x512x512 4

NOISE_SHAPE="${1:-191x256x256}"
OUT_DIR="${2:-outputs/hyper_anyshape}"
NUM_SAMPLES="${3:-8}"
CKPT="${CKPT:-checkpoints/hyspecnet3d_hamiltonian.pt}"

python src/generate_hyper_anyshape.py \
  --ckpt "$CKPT" \
  --out-dir "$OUT_DIR" \
  --num-samples "$NUM_SAMPLES" \
  --noise-shape "$NOISE_SHAPE"
