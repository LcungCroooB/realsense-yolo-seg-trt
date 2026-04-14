cmake_minimum_required(VERSION 3.10)
project(logger LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

file(GLOB LOGGER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

list(REMOVE_ITEM LOGGER_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/demo.cpp)
add_library(${PROJECT_NAME} STATIC ${LOGGER_SRCS})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(Threads REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Threads::Threads)

option(LOGGER_BUILD_DEMO "Build logger demo executable" OFF)
if (LOGGER_BUILD_DEMO)
	add_executable(demo ${CMAKE_CURRENT_SOURCE_DIR}/demo.cpp)
	target_link_libraries(demo PRIVATE logger)
endif()