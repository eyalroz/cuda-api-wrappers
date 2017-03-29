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
cmake_minimum_required(VERSION 2.8)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

# -----------------------------------
# Project name, version & build type
# -----------------------------------

include (ReleaseOrDebug)

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

set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -g" )
include_directories( "src/" )

# -------------
# CUDA
# -------------

find_package(CUDA QUIET REQUIRED)

include(HandleCUDAComputeCapability)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --std=c++11 --keep  --generate-line-info --source-in-ptx -Xcompiler -Wall -Xcompiler -D__CORRECT_ISO_CPP11_MATH_H_PROTO )
# Note: Adding "-G" to the next line might give you debugging info, but it switches off all optimizers, so you're not really debugging what you're running; and it also increases the register pressure, and the local memory pressure tremendously, changing the behavior even further; finally, it _may_ cause linking trouble with CUDA 7.5 and Fedora 23 (or equivalent build environments)
set(CUDA_NVCC_FLAGS_DEBUG -g -lineinfo -DDEBUG)
set(CUDA_NVCC_FLAGS_RELEASE -O3 -DNDEBUG)

set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)

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

# -----------------------------------
# Be verbose about what we've found
# -----------------------------------

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)	
	message("--------------------------------------------------------------")
	message("Paths Used"                                                    )
	message("--------------------------------------------------------------")
	message("C compiler (for non-CUDA code):    " "${CMAKE_C_COMPILER}"     )
	message("C++ compiler (for non-CUDA code):  " "${CMAKE_CXX_COMPILER}"   )
	message("CUDA C++ forward-to host compiler: " "${CUDA_HOST_COMPILER}"   )
	message("CUB header-only library root:      " "${CUB_INCLUDE_DIR}"      )
	message("CUDA include directories:          " "${CUDA_INCLUDE_DIRS}"    )
	message("--------------------------------------------------------------")
	message("Package and Library versions"                                  )
	message("--------------------------------------------------------------")
	message("nVIDIA CUDA toolkit:               " "${CUDA_VERSION_STRING}"  )
	message("--------------------------------------------------------------")
endif()

# -----------------------
# Main target(s)
# -----------------------

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib/")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib/")

cuda_add_library(
	cuda-api-wrappers
	src/cuda/api/device_properties.cpp
	src/cuda/api/profiling.cpp
)

# -----------------------
# Examples
# -----------------------

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "examples/bin")
cuda_add_executable(vectorAdd EXCLUDE_FROM_ALL examples/modified_cuda_samples/vectorAdd/vectorAdd.cu)
target_link_libraries(vectorAdd cuda-api-wrappers)
cuda_add_executable(inlinePTX EXCLUDE_FROM_ALL examples/modified_cuda_samples/inlinePTX/inlinePTX.cu)
target_link_libraries(inlinePTX cuda-api-wrappers)
cuda_add_executable(simpleStreams EXCLUDE_FROM_ALL examples/modified_cuda_samples/simpleStreams/simpleStreams.cu)
target_link_libraries(simpleStreams cuda-api-wrappers)
cuda_add_executable(simpleIPC EXCLUDE_FROM_ALL examples/modified_cuda_samples/simpleIPC/simpleIPC.cu)
target_link_libraries(simpleIPC cuda-api-wrappers)
#----
add_custom_target(modified_cuda_samples)
add_dependencies(modified_cuda_samples vectorAdd inlinePTX simpleStreams simpleIPC)

add_executable(version_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/version_management.cpp)
target_link_libraries(version_management  ${CUDA_LIBRARIES})
cuda_add_executable(error_handling EXCLUDE_FROM_ALL examples/by_runtime_api_module/error_handling.cu)
target_link_libraries(error_handling cuda-api-wrappers)
add_executable(device_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/device_management.cpp)
target_link_libraries(device_management cuda-api-wrappers)
cuda_add_executable(execution_control EXCLUDE_FROM_ALL examples/by_runtime_api_module/execution_control.cu)
target_link_libraries(execution_control cuda-api-wrappers)
cuda_add_executable(stream_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/stream_management.cu)
target_link_libraries(stream_management cuda-api-wrappers)
cuda_add_executable(event_management EXCLUDE_FROM_ALL examples/by_runtime_api_module/event_management.cu)
target_link_libraries(event_management cuda-api-wrappers)
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

# -------------

# We don't do installation for now
# set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")

