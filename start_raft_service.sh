#!/bin/bash
set -e

export DEGRAF_PROJECT_ROOT="${DEGRAF_PROJECT_ROOT:-/root/autodl-tmp/projects/DegrafFlowGPU}"
export RAFT_MODEL_PATH="${RAFT_MODEL_PATH:-/root/autodl-tmp/models/raft/raft-kitti.pth}"
export RAFT_TCP_PORT="${RAFT_TCP_PORT:-9998}"

cd "${DEGRAF_PROJECT_ROOT}/external/RAFT"
python3 raft_batch_tcp_server.py --port "${RAFT_TCP_PORT}" --model "${RAFT_MODEL_PATH}"
