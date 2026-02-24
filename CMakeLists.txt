cmake_minimum_required(VERSION 3.12)

option(ENABLE_CUDA "Enable CUDA support" OFF)

if(ENABLE_CUDA)
    project(DeGraFFlow LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_STANDARD 14)
else()
    project(DeGraFFlow LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 14)

if(DEFINED ENV{OpenCV_DIR})
    set(OpenCV_DIR $ENV{OpenCV_DIR})
endif()
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")


include_directories(
    ${PROJECT_SOURCE_DIR}/include
    # ${PROJECT_SOURCE_DIR}/external/RLOFLib/include
    ${PROJECT_SOURCE_DIR}/gpu/cuda
    ${OpenCV_INCLUDE_DIRS}
)

# link_directories(${PROJECT_SOURCE_DIR}/external/RLOFLib/lib)

file(GLOB SRC_FILES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB GPU_SRC_FILES ${PROJECT_SOURCE_DIR}/gpu/*.cpp ${PROJECT_SOURCE_DIR}/gpu/cuda/*.cu)

add_executable(degraf_flow ${SRC_FILES})
set_property(TARGET degraf_flow PROPERTY CXX_STANDARD 14)

target_link_libraries(degraf_flow
    ${OpenCV_LIBS}
    # ${PROJECT_SOURCE_DIR}/external/RLOFLib/build/libRLOF.so
)

if(ENABLE_CUDA)
    add_executable(gpu_main ${PROJECT_SOURCE_DIR}/gpu/gpu_main.cpp ${GPU_SRC_FILES})
    set_property(TARGET gpu_main PROPERTY CXX_STANDARD 14)
    set_property(TARGET gpu_main PROPERTY CUDA_STANDARD 14)
    # target_link_libraries(gpu_main ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/external/RLOFLib/lib/libRLOF_64.so)
endif()
