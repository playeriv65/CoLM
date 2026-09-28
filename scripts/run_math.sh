#!/usr/bin/env bash
# CoLM on MathInstruct, one example per micro-batch (SubsetTrainer).
#   scripts/run_math.sh [gpu_ids]        (or COLM_GPUS=...)
exec "$(dirname "${BASH_SOURCE[0]}")/run.sh" configs/math_phi2.json "${1:-${COLM_GPUS:-}}"
