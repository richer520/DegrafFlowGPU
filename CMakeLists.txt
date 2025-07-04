cmake_minimum_required(VERSION 3.12)
project(DeGraFFlow LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

option(ENABLE_CUDA "Enable CUDA support" ON)

# Set OpenCV path
set(OpenCV_DIR "/home/homeblue01/9/qkwv79/opencv_4.9_clean_cuda/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# Include paths
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/gpu/cuda
    ${OpenCV_INCLUDE_DIRS}
)

# Source files
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB GPU_CU_FILES ${PROJECT_SOURCE_DIR}/gpu/cuda/*.cu)

# Executable
add_executable(degraf_flow ${SRC_FILES})

# Add CUDA files to target
target_sources(degraf_flow PRIVATE ${GPU_CU_FILES})

# CUDA support for host-device code
set_target_properties(degraf_flow PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 14
    CUDA_STANDARD 14
)

# Link libraries
target_link_libraries(degraf_flow ${OpenCV_LIBS})
