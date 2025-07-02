#!/bin/bash

set -e  # 遇到错误直接退出
PROJECT_NAME="DeGraFFlow"
BUILD_DIR="build"
TARGET_EXECUTABLE="gpu_main"  # 或改成 degraf_flow

echo "🛠  [1/4] Preparing build directory..."
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

echo "🧱  [2/4] Running CMake configuration..."
cmake .. -DENABLE_CUDA=ON

echo "🚀  [3/4] Building project..."
start_time=$(date +%s)
make -j$(nproc)
end_time=$(date +%s)
echo "✅  Build finished in $((end_time - start_time)) seconds."

echo "🏃  [4/4] Running executable: ${TARGET_EXECUTABLE}"
if [[ -f "./${TARGET_EXECUTABLE}" ]]; then
    echo "--------------------------------------"
    ./${TARGET_EXECUTABLE}
    echo "--------------------------------------"
else
    echo "❌  Error: ${TARGET_EXECUTABLE} not found in build/"
    exit 1
fi
