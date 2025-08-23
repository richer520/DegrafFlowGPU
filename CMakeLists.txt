cmake_minimum_required(VERSION 3.10)

# 项目名称及语言支持
project(degraf_flow LANGUAGES CXX CUDA)

# CUDA 架构设置（61 = Pascal 架构，适用于 GTX 1080 Ti）
set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)

# 添加 OpenCV 路径（推荐在终端用 -DOpenCV_DIR 或 export）
if(NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR "$ENV{OpenCV_DIR}")
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
# 输出 OpenCV 版本和包含路径
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# 包含路径
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/gpu/cuda
    /usr/local/cuda/include   # 明确添加 CUDA 头文件路径，防止 cuda_runtime.h 报错
    /usr/include/jsoncpp
)

# 添加源文件（C++ 和 CUDA）
file(GLOB_RECURSE SRC_FILES
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_SOURCE_DIR}/gpu/cuda/*.cu"
)

# 创建可执行文件
add_executable(degraf_flow ${SRC_FILES})


# 添加 CUDA 编译支持
set_target_properties(degraf_flow PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# 链接 OpenCV 库
target_link_libraries(degraf_flow ${OpenCV_LIBS} jsoncpp rt pthread)


# ---------------------------------------
# ✅ 构建 GT PNG → .flo 批量转换工具程序
# ---------------------------------------
add_executable(convert_kitti_png_to_flo tools/convert_kitti_png_to_flo.cpp)
target_link_libraries(convert_kitti_png_to_flo ${OpenCV_LIBS})


