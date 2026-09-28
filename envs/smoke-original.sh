#!/usr/bin/env bash
# One-node smoke test of the ORIGINAL code: bash envs/smoke-original.sh <config> <gpu_list>
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/math_phi2_efficient.json}"
GPUS="${2:?gpu list required, e.g. 2}"
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
TAG="$(basename "$CONFIG" .json)_gpu${GPUS//,/-}_np${NPROC}_$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
export WANDB_MODE="${WANDB_MODE:-disabled}" PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="$GPUS" .venv/bin/torchrun --nproc_per_node "$NPROC" --nnodes 1 \
  --rdzv_backend c10d --rdzv-endpoint=localhost:$((RANDOM % 20000 + 30000)) \
  -m colm.train.train "$CONFIG" 2>&1 | tee "logs/smoke_${TAG}.log"
