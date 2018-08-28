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

option(EXPORT_BUILD_DIR "Export build directory using CMake (enables external use without install)" OFF)
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)

# -----------------------------------
# Project name, version & build type
# -----------------------------------


if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

PROJECT(
	cuda-api-wrappers
	LANGUAGES CUDA CXX
)

#	# No versioning for now
#
#	set(PROJECT_MAJOR_VERSION 0)
#	set(PROJECT_MINOR_VERSION 1)
#	set(PROJECT_PATCH_VERSION 0)
#	set(PROJECT_VERSION ${PROJECT_MAJOR_VERSION}.${PROJECT_MINOR_VERSION}.${PROJECT_PATCH_VERSION})


# ----------------------------
# General C++ build settings
# ----------------------------

# Here we set a PRIVATE compile option for all targets in this project
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic" )

set(CMAKE_CXX_EXTENSIONS OFF)

# -------------
# CUDA
# -------------
include(HandleCUDAComputeCapability)

# Here we set a PRIVATE compile option for all targets in this project
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall" )

set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS ON)

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

# CUDA_STANDARD can not be set as `compile_feature` yet. Thus we
# set the c++ standard, since that is used in the header part of
# this library, to force users of this library to compile automatically
# with c++11 compatible standards.
set_target_properties(cuda-api-wrappers PROPERTIES CUDA_STANDARD 11)
target_compile_features(cuda-api-wrappers PUBLIC cxx_std_11)

target_include_directories(
	cuda-api-wrappers
	PUBLIC
		"$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
		"$<INSTALL_INTERFACE:include>"
)

target_link_libraries(cuda-api-wrappers cudart)

# To enable `target_link_library(foo cuda-api-wrappers::cuda-api-wrappers)`
# in other CMake projects, which include this project with `add_subdirectory`
add_library(cuda-api-wrappers::cuda-api-wrappers ALIAS cuda-api-wrappers)

# -----------------------
# Examples / Tests
# -----------------------

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "examples/bin")

add_executable(vectorAdd EXCLUDE_FROM_ALL examples/modified_cuda_samples/vectorAdd/vectorAdd.cu)
target_link_libraries(vectorAdd cuda-api-wrappers)

add_executable(inlinePTX EXCLUDE_FROM_ALL examples/modified_cuda_samples/inlinePTX/inlinePTX.cu)
target_link_libraries(inlinePTX cuda-api-wrappers)

add_executable(simpleStreams EXCLUDE_FROM_ALL examples/modified_cuda_samples/simpleStreams/simpleStreams.cu)
target_link_libraries(simpleStreams cuda-api-wrappers)

add_executable(simpleIPC EXCLUDE_FROM_ALL examples/modified_cuda_samples/simpleIPC/simpleIPC.cu)
target_link_libraries(simpleIPC cuda-api-wrappers)

#----
add_custom_target(modified_cuda_samples)
add_dependencies(
	modified_cuda_samples
		vectorAdd
		inlinePTX
		simpleStreams
		simpleIPC
)

#----
add_executable(version_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/version_management.cpp)
target_link_libraries(version_management cuda-api-wrappers)

add_executable(error_handling EXCLUDE_FROM_ALL examples/by_runtime_api_module/error_handling.cu)
target_link_libraries(error_handling cuda-api-wrappers)

add_executable(device_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/device_management.cpp)
target_link_libraries(device_management cuda-api-wrappers)

add_executable(execution_control EXCLUDE_FROM_ALL examples/by_runtime_api_module/execution_control.cu)
target_link_libraries(execution_control cuda-api-wrappers)
target_compile_options(execution_control PRIVATE -rdc true)

add_executable(stream_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/stream_management.cu)
target_link_libraries(stream_management cuda-api-wrappers)

add_executable(event_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/event_management.cu)
target_link_libraries(event_management cuda-api-wrappers)

add_executable(io_compute_overlap_with_streams EXCLUDE_FROM_ALL examples/other/io_compute_overlap_with_streams.cu)
target_link_libraries(io_compute_overlap_with_streams cuda-api-wrappers)

add_executable(ipc EXCLUDE_FROM_ALL examples/by_runtime_api_module/ipc.cpp)
target_link_libraries(ipc cuda-api-wrappers)

#----
add_custom_target(examples_by_runtime_api_module)
add_dependencies(
	examples_by_runtime_api_module
		version_management
		error_handling
		device_management
		execution_control
		stream_management
		event_management
		io_compute_overlap_with_streams
		ipc
)

#----
add_custom_target(examples)
add_dependencies(examples examples_by_runtime_api_module modified_cuda_samples)

#----
add_custom_target(docs
	COMMAND doxygen doxygen.cfg
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

#-----
install(
	TARGETS cuda-api-wrappers
	EXPORT cuda-api-wrappers
	ARCHIVE # For CMake, a static library is an ARCHIVE, a dynamic library is a RUNTIME
	DESTINATION lib
	INCLUDES DESTINATION include
)

install(
	DIRECTORY src/cuda
	DESTINATION include
	FILES_MATCHING REGEX "\\.(h|hpp|cuh)"
)

#set(RELATIVE_CONFIG_INSTALL_DIR "share/cmake/cuda-api-wrappers-${PROJECT_VERSION}")
# since there is no versioning yet:
set(RELATIVE_CONFIG_INSTALL_DIR "share/cmake/cuda-api-wrappers")

install(
	EXPORT cuda-api-wrappers
	DESTINATION "${RELATIVE_CONFIG_INSTALL_DIR}"
	NAMESPACE "cuda-api-wrappers::"
	FILE cuda-api-wrappersTargets.cmake
)
configure_file("cmake/cuda-api-wrappersConfig.cmake.in"
	"${PROJECT_BINARY_DIR}/cuda-api-wrappersConfig.cmake" @ONLY)

install(
	FILES "${PROJECT_BINARY_DIR}/cuda-api-wrappersConfig.cmake"
	DESTINATION "${RELATIVE_CONFIG_INSTALL_DIR}"
)

if(EXPORT_BUILD_DIR AND NOT CMAKE_EXPORT_NO_PACKAGE_REGISTRY)
	message("-- Exporting cuda-api-wrappers build directory to local CMake package registry.")

	export(
		EXPORT cuda-api-wrappers
		NAMESPACE "cuda-api-wrappers::"
		FILE "${PROJECT_BINARY_DIR}/cuda-api-wrappersTargets.cmake"
	)
	# Export the project build directory as a package into the local CMake package registry.
	export(PACKAGE cuda-api-wrappers)
	# Configure a *Config.cmake file for the export of the build directory from
	# the template, reflecting the current build options.
	set(SETUP_PACKAGE_CONFIG_FOR_INSTALLATION FALSE)

endif()
