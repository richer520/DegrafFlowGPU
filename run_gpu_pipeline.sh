#!/bin/bash
set -e

export DEGRAF_PROJECT_ROOT="${DEGRAF_PROJECT_ROOT:-/root/autodl-tmp/projects/DegrafFlowGPU}"
if [ ! -d "${DEGRAF_PROJECT_ROOT}" ]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  export DEGRAF_PROJECT_ROOT="${SCRIPT_DIR}"
fi
export DEGRAF_DATA_PATH="${DEGRAF_DATA_PATH:-/root/autodl-tmp/data/kitti/data_scene_flow}"
export DEGRAF_CALIB_PATH="${DEGRAF_CALIB_PATH:-/root/autodl-tmp/data/kitti/data_scene_flow_calib}"
export DEGRAF_INFERENCE_MODE="${DEGRAF_INFERENCE_MODE:-cpp}"

echo "[INFO] DEGRAF_PROJECT_ROOT=${DEGRAF_PROJECT_ROOT}"
echo "[INFO] DEGRAF_DATA_PATH=${DEGRAF_DATA_PATH}"
echo "[INFO] DEGRAF_CALIB_PATH=${DEGRAF_CALIB_PATH}"
echo "[INFO] DEGRAF_INFERENCE_MODE=${DEGRAF_INFERENCE_MODE} (single-process C++ pipeline)"

mkdir -p "${DEGRAF_PROJECT_ROOT}/data/outputs"
if command -v nvcc >/dev/null 2>&1; then
  BUILD_DIR="${DEGRAF_PROJECT_ROOT}/build_gpu"
  mkdir -p "${BUILD_DIR}"
  cd "${BUILD_DIR}"
  echo "[INFO] CUDA toolkit detected, building with USE_CUDA=ON"
  cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
else
  BUILD_DIR="${DEGRAF_PROJECT_ROOT}/build_cpu"
  mkdir -p "${BUILD_DIR}"
  cd "${BUILD_DIR}"
  echo "[WARN] nvcc not found, falling back to USE_CUDA=OFF"
  cmake .. -DUSE_CUDA=OFF
fi
make -j"$(nproc)"
echo "[INFO] degraf_flow args: $*"
./degraf_flow "$@"
