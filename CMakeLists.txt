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


# ----------------------
# Project & build type settings
# ----------------------

PROJECT(kernel_tester)

set(PROJECT_MAJOR_VERSION 0)
set(PROJECT_MINOR_VERSION 1)
set(PROJECT_PATCH_VERSION 0)
set(PROJECT_VERSION ${PROJECT_MAJOR_VERSION}.${PROJECT_MINOR_VERSION}.${PROJECT_PATCH_VERSION})

if("$ENV{RELEASE}" STREQUAL ""  AND  "$ENV{DEBUG}" STREQUAL "" )
	message("Neither DEBUG nor RELEASE environment variables set, defaulting to DEBUG.")
	set(CMAKE_BUILD_TYPE "DEBUG")
endif( "$ENV{RELEASE}" STREQUAL ""  AND  "$ENV{DEBUG}" STREQUAL "" )

if (NOT "$ENV{DEBUG}" STREQUAL "")
	set(CMAKE_BUILD_TYPE "DEBUG")
endif (NOT "$ENV{DEBUG}" STREQUAL "")

if (NOT "$ENV{RELEASE}" STREQUAL "")
	set(CMAKE_BUILD_TYPE "RELEASE")
endif (NOT "$ENV{RELEASE}" STREQUAL "")

if ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
	message("\
  ____           _                               _ 
 |  _ \\    ___  | |   ___    __ _   ___    ___  | |
 | |_) |  / _ \\ | |  / _ \\  / _` | / __|  / _ \\ | |
 |  _ <  |  __/ | | |  __/ | (_| | \\__ \\ |  __/ |_|
 |_| \\_\\  \\___| |_|  \\___|  \\__,_| |___/  \\___| (_)

")
else ("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")
if("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
	message("\
  ____           _                     
 |  _ \\    ___  | |__    _   _    __ _ 
 | | | |  / _ \\ | '_ \\  | | | |  / _` |
 | |_| | |  __/ | |_) | | |_| | | (_| |
 |____/   \\___| |_.__/   \\__,_|  \\__, |
                                 |___/ 
")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "DEBUG")
endif("${CMAKE_BUILD_TYPE}" STREQUAL "RELEASE")

# ----------------------------
# General C/C++ build settings
# ----------------------------

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -std=c++11 -g" )
include_directories( "src/" )

# -------------
# CUDA
# -------------

find_package(CUDA QUIET REQUIRED)

if("$ENV{CUDA_SM}" STREQUAL "")
	message(WARNING  "CUDA_SM environment variable not set. Defaulting to \"30\" (i.e. a compute capability of 3.0)")
	set(CUDA_TARGET_COMPUTE_CAPABILITY "30")
else("$ENV{CUDA_SM}" STREQUAL "")
	set(CUDA_TARGET_COMPUTE_CAPABILITY $ENV{CUDA_SM})
	execute_process(COMMAND sh "-c" "echo ${CUDA_TARGET_COMPUTE_CAPABILITY} | sed 's/^\\([0-9]\\)\\([0-9]\\)/\\1.\\2/;' | xargs echo -n" OUTPUT_VARIABLE FORMATTED_COMPUTE_CAPABILITY) 
	message("Building for Compute Capability ${FORMATTED_COMPUTE_CAPABILITY}.")
endif("$ENV{CUDA_SM}" STREQUAL "")

set(CUDA_GENCODE "arch=compute_${CUDA_TARGET_COMPUTE_CAPABILITY},code=compute_${CUDA_TARGET_COMPUTE_CAPABILITY}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode ${CUDA_GENCODE} "-Xcompiler" "-D__CORRECT_ISO_CPP11_MATH_H_PROTO")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --keep  --generate-line-info --source-in-ptx)
# We would like to add "-G" to the next line, but for some reason that causes segmentation faults in the driver when linking with libcub (on FC 23 with CUDA 7.5)
set(CUDA_NVCC_FLAGS_DEBUG -g)
set(CUDA_NVCC_FLAGS_RELEASE -O3 --use_fast_math)
#
# have a look at http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/
# for more potential options

set(CUDA_SEPARABLE_COMPILATION OFF)

# -----------------------
# Miscellaneous targets
# -----------------------

# Add "tags" target and make project depending on this target.
set_source_files_properties( tags PROPERTIES GENERATED true )
add_custom_target(tags 
	COMMAND ctags --langmap=c++:+.cu.cuh.hpp -R ./ 
	WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_custom_target(cclean COMMAND rm -r CMakeCache.txt CMakeFiles/)

# -----------------------------------
# Be verbose about what we've found
# -----------------------------------

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
cuda_add_executable(inlinePTX EXCLUDE_FROM_ALL examples/modified_cuda_samples/inlinePTX/inlinePTX.cu)
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
#----
add_custom_target(examples_by_runtime_api_module)
add_dependencies(examples_by_runtime_api_module version_management error_handling)

add_custom_target(examples)
add_dependencies(examples examples_by_runtime_api_module modified_cuda_samples)

# -------------

# We don't do installation for now
# set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")

