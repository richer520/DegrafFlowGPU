#!/bin/bash
set -e

export DEGRAF_PROJECT_ROOT="${DEGRAF_PROJECT_ROOT:-/root/autodl-tmp/projects/DegrafFlowGPU}"
export INTERPONET_MODEL_DIR="${INTERPONET_MODEL_DIR:-/root/autodl-tmp/models/interponet}"

if [ -n "${INTERPONET_CONDA_ENV}" ]; then
  # Optional: use dedicated TF1 environment
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "${INTERPONET_CONDA_ENV}"
fi

cd "${DEGRAF_PROJECT_ROOT}/external/InterpoNet"
INTERPONET_MODEL_DIR="${INTERPONET_MODEL_DIR}" python3 interponet_batch_tcp_server.py
