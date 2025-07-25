cmake_minimum_required(VERSION 3.12)
project(DeGraFFlow LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

option(ENABLE_CUDA "Enable CUDA support" ON)

# ========================= OpenCV =========================
# Set OpenCV path (自定义 OpenCV 路径）
set(OpenCV_DIR "/home/homeblue01/9/qkwv79/opencv_4.9_clean_cuda/install/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")

# ========================= PCL =========================
# Set PCL path based on current Conda environment
# set(PCL_DIR "$ENV{CONDA_PREFIX}/lib/cmake/pcl")
# find_package(PCL REQUIRED COMPONENTS common io visualization)
# add_definitions(${PCL_DEFINITIONS})

# message(STATUS "PCL version: ${PCL_VERSION}")
# message(STATUS "PCL include dirs: ${PCL_INCLUDE_DIRS}")

# ========================= Include Paths =========================
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/gpu/cuda
    ${OpenCV_INCLUDE_DIRS}
    # ${PCL_INCLUDE_DIRS}      # ✅ 新增：PCL 路径
)

# ========================= Source Files =========================
file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB GPU_CU_FILES ${PROJECT_SOURCE_DIR}/gpu/cuda/*.cu)

add_executable(degraf_flow ${SRC_FILES})
target_sources(degraf_flow PRIVATE ${GPU_CU_FILES})

# CUDA support for host-device code
set_target_properties(degraf_flow PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 14
    CUDA_STANDARD 14
)

# ========================= Fast Guided Filter Sub-project =========================
# add_subdirectory(external/GuidedFilterCUDA)

# ========================= Linking Libraries =========================
target_link_libraries(degraf_flow
    ${OpenCV_LIBS}
    # fast_guided_filter
    # ${PCL_LIBRARIES}         # ✅ 新增：PCL 库
)

# target_include_directories(degraf_flow PRIVATE external/GuidedFilterCUDA)
