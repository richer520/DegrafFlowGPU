#!/bin/bash

echo "🔁 Stopping and removing old container (if exists)..."
docker rm -f raft_tcp_server &> /dev/null

echo "🚀 Starting RAFT TCP Server on port 9998..."
docker run -d \
  --gpus all \
  -v $(pwd):/app \
  -p 9998:9998 \
  --name raft_tcp_server \
  raft_flow \
#   python3 /app/tcp_server_raft.py

echo "✅ RAFT TCP Server started. Listening on port 9998."
