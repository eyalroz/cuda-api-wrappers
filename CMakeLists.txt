# CMakeLists.txt for the cuda-api-wrappers project
#
# Notes:
#
# CUDA is very picky about which compiler you can use.
# CUDA 7.x supports gcc up to version 4.9.x (and not clang);
# CUDA 8.0 supports gcc up to version 5.4.x (and there seems
# to be some work on getting clang to support CUDA, but it
# looks like it's not there yet)
#
# Also, you will need your libraries compiled to be compatible
# with whatever CUDA-supported compiler you use.
#
# to use a different compiler with CMake, run it as follows:
#
# cmake -D CMAKE_C_COMPILER=/path/to/your/cc -D CMAKE_CXX_COMPILER=/path/to/your/c++ your_project_dir
#
#
cmake_minimum_required(VERSION 3.1)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

# -----------------------------------
# Project name, version & build type
# -----------------------------------


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release" )
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # This is the top-level project.
	PROJECT(cuda-api-wrappers)

#	# No versioning for now
#
#	set(PROJECT_MAJOR_VERSION 0)
#	set(PROJECT_MINOR_VERSION 1)
#	set(PROJECT_PATCH_VERSION 0)
#	set(PROJECT_VERSION ${PROJECT_MAJOR_VERSION}.${PROJECT_MINOR_VERSION}.${PROJECT_PATCH_VERSION})
endif()


# ----------------------------
# General C/C++ build settings
# ----------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wpedantic" )
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#SET(CMAKE_BUILD_TYPE distribution)

include_directories( "src/" )

# -------------
# CUDA
# -------------

find_package(CUDA 7.0 REQUIRED)
if (${CUDA_VERSION} GREATER 9.0)
    message(FATAL_ERROR "Unsupported CUDA version")
endif()
include(HandleCUDAComputeCapability)

# There is no CUDA equivalent of CMAKE_CXX_STANDARD* macros... so we're stuck with the following line:
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --std=c++11)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Xcompiler -Wall)
set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} -g --keep --generate-line-info --source-in-ptx)
set(CUDA_NVCC_FLAGS_RELEASE ${CUDA_NVCC_FLAGS_RELEASE} -O3 -DNDEBUG)

set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

# This should really have been automatic...
get_filename_component(CUDA_LIBRARY_DIR ${CUDA_CUDART_LIBRARY} DIRECTORY)
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} "-L${CUDA_LIBRARY_DIR}" )

# -----------------------
# Miscellaneous targets
# -----------------------

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)	
    # This is the top-level project.
	set_source_files_properties( tags PROPERTIES GENERATED true )
	add_custom_target(tags
	    COMMAND ctags --langmap=c++:+.cu.cuh.hpp  -R ./
	    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

	add_custom_target(cclean COMMAND rm -r CMakeCache.txt CMakeFiles/ cmake_install.cmake)
endif()

# -----------------------
# Main target(s)
# -----------------------

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
	set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
	set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")
endif()

cuda_add_library(
	cuda-api-wrappers
	src/cuda/api/device_properties.cpp
	src/cuda/api/profiling.cpp
)

# -----------------------
# Examples / Tests
# -----------------------

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

	link_libraries(cuda-api-wrappers)
 
	set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "examples/bin")
	cuda_add_executable(vectorAdd EXCLUDE_FROM_ALL examples/modified_cuda_samples/vectorAdd/vectorAdd.cu)
	cuda_add_executable(inlinePTX EXCLUDE_FROM_ALL examples/modified_cuda_samples/inlinePTX/inlinePTX.cu)
	cuda_add_executable(simpleStreams EXCLUDE_FROM_ALL examples/modified_cuda_samples/simpleStreams/simpleStreams.cu)
	cuda_add_executable(simpleIPC EXCLUDE_FROM_ALL examples/modified_cuda_samples/simpleIPC/simpleIPC.cu)
	#----
	add_custom_target(modified_cuda_samples)
	add_dependencies(modified_cuda_samples vectorAdd inlinePTX simpleStreams simpleIPC)

	add_executable(version_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/version_management.cpp)
	target_link_libraries(version_management  ${CUDA_LIBRARIES})
	cuda_add_executable(error_handling EXCLUDE_FROM_ALL examples/by_runtime_api_module/error_handling.cu)
	add_executable(device_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/device_management.cpp)
	cuda_add_executable(execution_control EXCLUDE_FROM_ALL examples/by_runtime_api_module/execution_control.cu)
	cuda_add_executable(stream_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/stream_management.cu)
	cuda_add_executable(event_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/event_management.cu)
	#----
	add_custom_target(examples_by_runtime_api_module)
	add_dependencies(
		examples_by_runtime_api_module
		version_management
		error_handling device_management
		execution_control
		stream_management
		event_management)

	add_custom_target(examples)
	add_dependencies(examples examples_by_runtime_api_module modified_cuda_samples)

	add_custom_target(docs 
		COMMAND doxygen doxygen.cfg
		WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

	add_dependencies(examples examples_by_runtime_api_module modified_cuda_samples)
endif()

# -------------

# We don't do installation for now
# set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")

