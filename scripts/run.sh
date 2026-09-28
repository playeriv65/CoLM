#!/usr/bin/env bash
# Launch CoLM training with torchrun inside the project's uv environment.
#
#   scripts/run.sh <config.json> [gpu_ids] [extra HfArgumentParser flags...]
#
# gpu_ids defaults to $COLM_GPUS; there is no built-in default because the GPUs are
# shared and must be reserved first. nproc_per_node is derived from the list.
# Output goes to logs/<config>-<gpus>-<timestamp>.log (python runs unbuffered).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config=${1:?"Usage: $0 <config.json> [gpu_ids] [extra args...]"}
gpus=${2:-${COLM_GPUS:-}}
if [[ -z "${gpus}" ]]; then
    echo "No GPUs given: pass them as the second argument or set COLM_GPUS (e.g. COLM_GPUS=2)." >&2
    exit 2
fi
shift $(( $# >= 2 ? 2 : 1 ))

IFS=',' read -r -a gpu_list <<< "${gpus}"
nproc=${#gpu_list[@]}
port=$(( 20000 + RANDOM % 40000 ))

log_dir="${COLM_LOG_DIR:-${repo_root}/logs}"
mkdir -p "${log_dir}"
log_file="${log_dir}/$(basename "${config}" .json)-gpu${gpus//,/_}-np${nproc}-$(date +%Y%m%d-%H%M%S).log"

cd "${repo_root}"
echo "config=${config} gpus=${gpus} nproc_per_node=${nproc} port=${port} log=${log_file}"
CUDA_VISIBLE_DEVICES="${gpus}" PYTHONUNBUFFERED=1 uv run --frozen torchrun \
    --nproc_per_node "${nproc}" \
    --nnodes 1 \
    --rdzv-id "${port}" \
    --rdzv-backend c10d \
    --rdzv-endpoint "localhost:${port}" \
    -m colm.train.train "${config}" "$@" 2>&1 | tee "${log_file}"
