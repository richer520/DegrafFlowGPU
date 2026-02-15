#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${RAFT_MODEL_PATH:-${SCRIPT_DIR}/checkpoints/raft-kitti.pth}"
PORT="${RAFT_TCP_PORT:-9998}"

echo "Starting RAFT TCP Server on port ${PORT}"
echo "Model checkpoint: ${MODEL_PATH}"
python3 "${SCRIPT_DIR}/raft_batch_tcp_server.py" --port "${PORT}" --model "${MODEL_PATH}"
