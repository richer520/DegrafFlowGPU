cmake_minimum_required(VERSION 3.12)
project(DeGraFFlow LANGUAGES CXX CUDA)

# 设置编译标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# 是否启用 CUDA 加速模块（默认开启）
option(ENABLE_CUDA "Enable CUDA support" ON)

# 查找 OpenCV
find_package(OpenCV REQUIRED)

# 包含头文件路径
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/external/RLOFLib/include
    ${PROJECT_SOURCE_DIR}/gpu/cuda
)

# 链接 RLOF 库目录（你已提供.so 文件）
link_directories(${PROJECT_SOURCE_DIR}/external/RLOFLib/lib)

# 收集源码
file(GLOB SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/*.cpp
)

file(GLOB GPU_SRC_FILES
    ${PROJECT_SOURCE_DIR}/gpu/*.cpp
    ${PROJECT_SOURCE_DIR}/gpu/cuda/*.cu
)

# -------------------- 可执行程序：主评估程序（CPU / GPU均可用） --------------------
add_executable(degraf_flow ${SRC_FILES})

# 链接 OpenCV 和 RLOF
target_link_libraries(degraf_flow
    ${OpenCV_LIBS}
    ${PROJECT_SOURCE_DIR}/external/RLOFLib/lib/libRLOF_64.so
)

# -------------------- 可选：调试 CUDA 的单独测试程序 --------------------
if(ENABLE_CUDA)
    add_executable(gpu_main ${PROJECT_SOURCE_DIR}/gpu/gpu_main.cpp ${GPU_SRC_FILES})
    target_link_libraries(gpu_main
        ${OpenCV_LIBS}
        ${PROJECT_SOURCE_DIR}/external/RLOFLib/lib/libRLOF_64.so
    )
endif()
