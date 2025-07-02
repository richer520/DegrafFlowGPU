#!/bin/bash

# 容器镜像名
IMAGE_NAME=degraf_flow_cuda

# 检查镜像是否已构建
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
  echo "⚠️ 镜像 $IMAGE_NAME 不存在，请先运行: docker build -f Dockerfile.cuda -t $IMAGE_NAME ."
  exit 1
fi

# 启动带 GPU 的容器
docker run --gpus all \
    -v $(pwd):/app \
    -w /app \
    -it $IMAGE_NAME \
    bash
