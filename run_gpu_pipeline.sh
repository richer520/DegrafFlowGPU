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
export DEGRAF_ALLOW_CPU_FALLBACK="${DEGRAF_ALLOW_CPU_FALLBACK:-0}"
export DEGRAF_RAFT_BACKEND="${DEGRAF_RAFT_BACKEND:-trt}"
export DEGRAF_ALLOW_LK_FALLBACK="${DEGRAF_ALLOW_LK_FALLBACK:-0}"
export DEGRAF_RAFT_ENGINE_PATH="${DEGRAF_RAFT_ENGINE_PATH:-${DEGRAF_PROJECT_ROOT}/external/RAFT/models/raft_kitti_fp16.engine}"

echo "[INFO] DEGRAF_PROJECT_ROOT=${DEGRAF_PROJECT_ROOT}"
echo "[INFO] DEGRAF_DATA_PATH=${DEGRAF_DATA_PATH}"
echo "[INFO] DEGRAF_CALIB_PATH=${DEGRAF_CALIB_PATH}"
echo "[INFO] DEGRAF_INFERENCE_MODE=${DEGRAF_INFERENCE_MODE} (single-process C++ pipeline)"
echo "[INFO] DEGRAF_ALLOW_CPU_FALLBACK=${DEGRAF_ALLOW_CPU_FALLBACK}"
echo "[INFO] DEGRAF_RAFT_BACKEND=${DEGRAF_RAFT_BACKEND}"
echo "[INFO] DEGRAF_ALLOW_LK_FALLBACK=${DEGRAF_ALLOW_LK_FALLBACK}"
echo "[INFO] DEGRAF_RAFT_ENGINE_PATH=${DEGRAF_RAFT_ENGINE_PATH}"

mkdir -p "${DEGRAF_PROJECT_ROOT}/data/outputs"
if command -v nvcc >/dev/null 2>&1; then
  BUILD_DIR="${DEGRAF_PROJECT_ROOT}/build_gpu"
  mkdir -p "${BUILD_DIR}"
  cd "${BUILD_DIR}"
  CMAKE_EXTRA_ARGS="${DEGRAF_CMAKE_EXTRA_ARGS:-}"
  echo "[INFO] CUDA toolkit detected, building with USE_CUDA=ON"
  if [ -n "${CMAKE_EXTRA_ARGS}" ]; then
    echo "[INFO] Additional CMake args: ${CMAKE_EXTRA_ARGS}"
  fi
  cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 ${CMAKE_EXTRA_ARGS}
else
  if [ "${DEGRAF_ALLOW_CPU_FALLBACK}" = "1" ]; then
    BUILD_DIR="${DEGRAF_PROJECT_ROOT}/build_cpu"
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    CMAKE_EXTRA_ARGS="${DEGRAF_CMAKE_EXTRA_ARGS:-}"
    echo "[WARN] nvcc not found, falling back to USE_CUDA=OFF (DEGRAF_ALLOW_CPU_FALLBACK=1)"
    if [ -n "${CMAKE_EXTRA_ARGS}" ]; then
      echo "[INFO] Additional CMake args: ${CMAKE_EXTRA_ARGS}"
    fi
    cmake .. -DUSE_CUDA=OFF ${CMAKE_EXTRA_ARGS}
  else
    echo "[ERROR] nvcc not found. Refusing CPU fallback because DEGRAF_ALLOW_CPU_FALLBACK=0."
    echo "[ERROR] Install CUDA toolkit or explicitly set DEGRAF_ALLOW_CPU_FALLBACK=1."
    exit 1
  fi
fi
make -j"$(nproc)"
echo "[INFO] degraf_flow args: $*"
./degraf_flow "$@"
