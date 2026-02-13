#!/bin/bash
echo "🚀 Starting RAFT TCP Server on port 9998..."
docker run -it --rm   --gpus all   -v $(pwd):/app   -p 9998:9998   raft_flow bash
echo "RAFT TCP Server started. Listening on port 9998."
