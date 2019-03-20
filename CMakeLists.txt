# CMakeLists.txt for the cuda-api-wrappers project
#
# Notes:
#
# CUDA is very picky about which compiler you can use.
# CUDA 7.x supports gcc up to version 4.9.x
# CUDA 8.x supports gcc up to version 5.x
# CUDA 9.x supports gcc up to version 6.x
#
# (and newer versions of clang seems to support CUDA compilation as well,
# although the API wrappers haven't been tested with those yet)
#
# One way to force the use of a different compiler with CMake is to run it as follows:
#
# cmake -D CMAKE_C_COMPILER=/path/to/your/cc -D CMAKE_CXX_COMPILER=/path/to/your/c++ your_project_dir
#
# and recall also that libraries must also have been built with a compatible compiler
#

# We need version 3.8 for native CUDA support in CMake
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# -----------------------------------
# Project name, version & build type
# -----------------------------------

if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
	set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(WIN32 AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

PROJECT(cuda-api-wrappers
	LANGUAGES CUDA CXX)

#	# No versioning for now
#
#	set(PROJECT_MAJOR_VERSION 0)
#	set(PROJECT_MINOR_VERSION 1)
#	set(PROJECT_PATCH_VERSION 0)
#	set(PROJECT_VERSION ${PROJECT_MAJOR_VERSION}.${PROJECT_MINOR_VERSION}.${PROJECT_PATCH_VERSION})


# ----------------------------
# General C++ build settings
# ----------------------------

include(CompileWithWarnings)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories( "src/" )

# -------------
# CUDA
# -------------

#find_package(CUDA 7.0 REQUIRED) # Why do I to do this damn it ?!
#include_directories( "${CUDA_TOOLKIT_INCLUDE}" )
include_directories( ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )
include(HandleCUDAComputeCapability)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall" )
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

set(CUDA_SEPARABLE_COMPILATION ON) # Does this work with native CUDA support?
set(CUDA_PROPAGATE_HOST_FLAGS OFF) # Does this work with native CUDA support?

# -----------------------
# Main target(s)
# -----------------------

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")

add_library(
	cuda-api-wrappers
	src/cuda/api/device_properties.cpp
	src/cuda/api/profiling.cpp
)

target_include_directories(
	${PROJECT_NAME}
		PUBLIC
		"$<INSTALL_INTERFACE:include>"
		"$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
)

if(WIN32)
	# Windows users report that CMake has trouble 
	set(CUDA_LIBRARIES "cudadevrt.lib;cudart.lib")
	target_link_libraries(${PROJECT_NAME} PUBLIC ${CUDA_LIBRARIES})
endif()

# -----------------------
# Examples / Tests
# -----------------------

link_libraries(${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES} cuda-api-wrappers)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "examples/bin")
add_executable(vectorAdd EXCLUDE_FROM_ALL examples/modified_cuda_samples/vectorAdd/vectorAdd.cu)
add_executable(inlinePTX EXCLUDE_FROM_ALL examples/modified_cuda_samples/inlinePTX/inlinePTX.cu)
add_executable(simpleStreams EXCLUDE_FROM_ALL examples/modified_cuda_samples/simpleStreams/simpleStreams.cu)
add_executable(simpleIPC EXCLUDE_FROM_ALL examples/modified_cuda_samples/simpleIPC/simpleIPC.cu)
#----
add_custom_target(modified_cuda_samples)
add_dependencies(modified_cuda_samples vectorAdd inlinePTX simpleStreams simpleIPC)

add_executable(version_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/version_management.cpp)
add_executable(error_handling EXCLUDE_FROM_ALL examples/by_runtime_api_module/error_handling.cu)
add_executable(device_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/device_management.cpp)
add_executable(execution_control EXCLUDE_FROM_ALL examples/by_runtime_api_module/execution_control.cu)
target_compile_options(execution_control PRIVATE -rdc true)
add_executable(stream_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/stream_management.cu)
add_executable(event_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/event_management.cu)
add_executable(unified_addressing EXCLUDE_FROM_ALL examples/by_runtime_api_module/unified_addressing.cpp)
add_executable(io_compute_overlap_with_streams EXCLUDE_FROM_ALL examples/other/io_compute_overlap_with_streams.cu)
#----
add_custom_target(examples_by_runtime_api_module)
add_dependencies(examples_by_runtime_api_module
	version_management
	error_handling device_management
	execution_control
	stream_management
	event_management
	unified_addressing
	io_compute_overlap_with_streams
)

add_custom_target(examples)
add_dependencies(examples examples_by_runtime_api_module modified_cuda_samples)

add_custom_target(docs
	COMMAND doxygen doxygen.cfg
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

add_dependencies(examples examples_by_runtime_api_module modified_cuda_samples)

# -------------

install(
	TARGETS cuda-api-wrappers
	EXPORT cuda-api-wrappers
	RUNTIME DESTINATION bin
	ARCHIVE DESTINATION lib
	LIBRARY DESTINATION lib
	INCLUDES DESTINATION include
)

install(
#	EXPORT cuda-api-wrappers
	DIRECTORY src/cuda
	DESTINATION include
	FILES_MATCHING REGEX "\\.(h|hpp|cuh)"
)

install(
	EXPORT ${PROJECT_NAME}
	DESTINATION "share/cmake/${PROJECT_NAME}"
	NAMESPACE "${PROJECT_NAME}::"
	# In this CMake config file no dependencies are considered. But since we
	# do not use any `find_package` call here this approach is sufficient.
	FILE ${PROJECT_NAME}Config.cmake
)
