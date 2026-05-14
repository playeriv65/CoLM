#!/bin/bash
CONFIG=${1:?"Usage: $0 <config.json> [gpu_ids]"}
GPU=${2:-0,1,2,3}

CUDA_VISIBLE_DEVICES=$GPU torchrun \
    --nproc_per_node 4 \
    --nnodes 1 \
    --rdzv-id=$((RANDOM % 90000 + 10000)) \
    --rdzv_backend c10d \
    --rdzv-endpoint=localhost:$((RANDOM % 90000 + 10000)) \
    -m colm.train.train "$CONFIG"
