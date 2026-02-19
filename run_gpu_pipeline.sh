#!/bin/bash
set -e

export DEGRAF_PROJECT_ROOT="${DEGRAF_PROJECT_ROOT:-/root/autodl-tmp/projects/DegrafFlowGPU}"
export DEGRAF_DATA_PATH="${DEGRAF_DATA_PATH:-/root/autodl-tmp/data/kitti/data_scene_flow}"
export DEGRAF_CALIB_PATH="${DEGRAF_CALIB_PATH:-/root/autodl-tmp/data/kitti/data_scene_flow_calib}"
export RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-/root/autodl-tmp/models/raft/raft-kitti.pth}"
export INTERPONET_MODEL_DIR="${INTERPONET_MODEL_DIR:-/root/autodl-tmp/models/interponet}"
export INTERPONET_EDGE_MODEL_YML="${INTERPONET_EDGE_MODEL_YML:-${INTERPONET_MODEL_DIR}/model.yml}"

echo "[INFO] DEGRAF_PROJECT_ROOT=${DEGRAF_PROJECT_ROOT}"
echo "[INFO] DEGRAF_DATA_PATH=${DEGRAF_DATA_PATH}"
echo "[INFO] DEGRAF_CALIB_PATH=${DEGRAF_CALIB_PATH}"
echo "[INFO] RAFT_MODEL_PATH=${RAFT_MODEL_PATH}"
echo "[INFO] INTERPONET_MODEL_DIR=${INTERPONET_MODEL_DIR}"
echo "[INFO] INTERPONET_EDGE_MODEL_YML=${INTERPONET_EDGE_MODEL_YML}"

if [ ! -f "${RAFT_MODEL_PATH}" ]; then
  echo "[ERROR] RAFT model not found: ${RAFT_MODEL_PATH}"
  exit 1
fi

if [ ! -f "${INTERPONET_MODEL_DIR}/best_model_kitti2015.ckpt.meta" ]; then
  echo "[WARN] InterpoNet ckpt meta not found. Check INTERPONET_MODEL_DIR."
fi

if [ ! -f "${INTERPONET_EDGE_MODEL_YML}" ]; then
  echo "[ERROR] InterpoNet edge model file not found: ${INTERPONET_EDGE_MODEL_YML}"
  exit 1
fi

mkdir -p "${DEGRAF_PROJECT_ROOT}/build_gpu"
mkdir -p "${DEGRAF_PROJECT_ROOT}/data/outputs"

cd "${DEGRAF_PROJECT_ROOT}/build_gpu"
cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
make -j"$(nproc)"
./degraf_flow
