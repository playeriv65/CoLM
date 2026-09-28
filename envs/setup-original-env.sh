#!/usr/bin/env bash
# Build the original-version CoLM venv (.venv) with uv. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
FREEZE_DATE="${FREEZE_DATE:-2024-08-31}"      # earliest date satisfying all pins (plotly 5.24.0)
TORCH_DATE="${TORCH_DATE:-2025-08-31}"        # torch 2.8.0 stack
TORCH_STACK=(torch torchvision torchaudio triton sympy mpmath
  nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cupti-cu12
  nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
  nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-cusparselt-cu12
  nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 nvidia-cufile-cu12)
EXTRA=()
for p in "${TORCH_STACK[@]}"; do EXTRA+=(--exclude-newer-package "$p=$TORCH_DATE"); done
[ -d .venv ] || uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python --exclude-newer "$FREEZE_DATE" "${EXTRA[@]}" \
  -r envs/original-requirements.txt
# submodlib: original README clones it into the repo root and installs editable;
# the code imports `submodlib.submodlib`, which relies on that layout.
[ -d submodlib ] || git clone https://github.com/decile-team/submodlib.git submodlib
uv pip install --python .venv/bin/python --exclude-newer "$FREEZE_DATE" "${EXTRA[@]}" -e ./submodlib
uv pip install --python .venv/bin/python --exclude-newer "$FREEZE_DATE" --no-deps -e .
[ -e data ] || ln -s /mnt/data2/zelin4593/datasets/colm data
.venv/bin/python -c "import torch, transformers, peft, accelerate; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'peft', peft.__version__, 'accelerate', accelerate.__version__)"
