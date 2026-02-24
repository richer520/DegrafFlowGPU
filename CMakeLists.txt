cmake_minimum_required(VERSION 3.10)

option(USE_CUDA "Build with CUDA support" OFF)
option(ENABLE_INPROCESS_VARIATIONAL "Link SrcVariational C library in-process" ON)

if(USE_CUDA)
    project(degraf_flow LANGUAGES C CXX CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 61)
    set(CMAKE_CUDA_STANDARD 14)
else()
    project(degraf_flow LANGUAGES C CXX)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT DEFINED OpenCV_DIR)
    set(OpenCV_DIR "$ENV{OpenCV_DIR}")
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")


include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
)

if(USE_CUDA)
    include_directories(
        ${CMAKE_SOURCE_DIR}/gpu/cuda
        /usr/local/cuda/include
    )
endif()

file(GLOB_RECURSE SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")
if(USE_CUDA)
    file(GLOB_RECURSE CUDA_SRC_FILES "${CMAKE_SOURCE_DIR}/gpu/cuda/*.cu")
    list(APPEND SRC_FILES ${CUDA_SRC_FILES})
endif()

if(ENABLE_INPROCESS_VARIATIONAL AND UNIX AND NOT APPLE)
    set(VARIATIONAL_SRC_DIR "${CMAKE_SOURCE_DIR}/external/InterpoNet/SrcVariational")
    if(EXISTS "${VARIATIONAL_SRC_DIR}/variational.c")
        list(APPEND SRC_FILES
            "${VARIATIONAL_SRC_DIR}/image.c"
            "${VARIATIONAL_SRC_DIR}/solver.c"
            "${VARIATIONAL_SRC_DIR}/variational_aux.c"
            "${VARIATIONAL_SRC_DIR}/variational.c"
        )
        include_directories("${VARIATIONAL_SRC_DIR}")
        add_compile_definitions(DEGRAF_HAVE_INPROCESS_VARIATIONAL=1)
    endif()
endif()

add_executable(degraf_flow ${SRC_FILES})

if(USE_CUDA)
    target_compile_definitions(degraf_flow PRIVATE USE_CUDA=1)
else()
    target_compile_definitions(degraf_flow PRIVATE USE_CUDA=0)
endif()

# Add CUDA compilation support
if(USE_CUDA)
    set_target_properties(degraf_flow PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
endif()

# Link OpenCV library
target_link_libraries(degraf_flow ${OpenCV_LIBS} pthread)
if(UNIX AND NOT APPLE)
    target_link_libraries(degraf_flow rt)
endif()



# add_executable(convert_kitti_png_to_flo tools/convert_kitti_png_to_flo.cpp)
# target_link_libraries(convert_kitti_png_to_flo ${OpenCV_LIBS})


