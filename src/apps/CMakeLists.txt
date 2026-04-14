cmake_minimum_required(VERSION 3.10)

find_package(yaml-cpp REQUIRED)

set(APP_TARGET yolo_seg_trt_app)

add_executable(${APP_TARGET}
	main.cpp
	yolo_compile.cpp
	yolo_compile_config.cpp
	bench_config.cpp
	benchmark.cpp
	d435_seg.cpp
	d435_seg_config.cpp
	platform_info.cpp
)

target_include_directories(${APP_TARGET}
	PRIVATE
		${CMAKE_SOURCE_DIR}/src
		${YAML_CPP_INCLUDE_DIR}
)

target_link_libraries(${APP_TARGET}
	PRIVATE
		SegmentationComponents
		logger
		${OpenCV_LIBS}
		yaml-cpp
)

set_target_properties(${APP_TARGET} PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

