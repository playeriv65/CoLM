#!/usr/bin/env bash
# CoLM on MathInstruct with the batched last-layer MeZO estimate (SubsetTrainerEfficient).
#   scripts/run_math_efficient.sh [gpu_ids]        (or COLM_GPUS=...)
exec "$(dirname "${BASH_SOURCE[0]}")/run.sh" configs/math_phi2_efficient.json "${1:-${COLM_GPUS:-}}"
