cmake_minimum_required(VERSION 3.10)
project(SegmentationComponents LANGUAGES CXX CUDA)

if(POLICY CMP0104)
    cmake_policy(SET CMP0104 NEW)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# Save all shared libraries to one place.
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libs)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(OpenCV 4.9.0 REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(Threads REQUIRED)

find_path(REALSENSE2_INCLUDE_DIR librealsense2/rs.hpp)
find_library(REALSENSE2_LIBRARY realsense2)
if(NOT REALSENSE2_INCLUDE_DIR OR NOT REALSENSE2_LIBRARY)
    message(FATAL_ERROR "librealsense2 headers or library not found")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Build local logger static library under src/logger.
add_subdirectory(src/logger)

file(GLOB_RECURSE COMMON_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/common/*.cpp
)
file(GLOB_RECURSE TRT_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/tensorrt/*.cpp
    ${PROJECT_SOURCE_DIR}/src/tensorrt/*.cu
)
file(GLOB_RECURSE TASK_SOURCES CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/src/tasks/*.cpp
    ${PROJECT_SOURCE_DIR}/src/tasks/*.cu
)

add_library(${PROJECT_NAME} SHARED ${COMMON_SOURCES} ${TRT_SOURCES} ${TASK_SOURCES})

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${PROJECT_SOURCE_DIR}/src
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ${REALSENSE2_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        logger
    PRIVATE
        Threads::Threads
        stdc++fs
        ${OpenCV_LIBS}
        ${TensorRT_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${REALSENSE2_LIBRARY}
        cuda
)

# Build app executables under src/apps (optional, controlled in src/apps/CMakeLists.txt).
add_subdirectory(src/apps)