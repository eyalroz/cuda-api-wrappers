cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")
enable_language(CXX)
enable_language(CUDA)

if(WIN32 AND "${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	# Avoid a build failure when the BUILD_SHARED_LIBS variable is set to true.
	set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()

include(FindCUDA) 

cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS_TMP Auto)
set(CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS_TMP} CACHE STRING "CUDA -gencode parameters")
string(REPLACE ";" " " CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS_STR}")

include(CompileWithWarnings)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall" )
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

link_libraries(runtime-api)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
add_executable(vectorAdd modified_cuda_samples/vectorAdd/vectorAdd.cu)
add_executable(vectorAddMapped modified_cuda_samples/vectorAddMapped/vectorAddMapped.cu)
add_executable(vectorAddManaged modified_cuda_samples/vectorAddManaged/vectorAddManaged.cu)
add_executable(inlinePTX modified_cuda_samples/inlinePTX/inlinePTX.cu)
add_executable(simpleStreams modified_cuda_samples/simpleStreams/simpleStreams.cu)
add_executable(simpleIPC modified_cuda_samples/simpleIPC/simpleIPC.cu)
add_executable(p2pBandwidthLatencyTest modified_cuda_samples/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest.cu)
add_executable(asyncAPI modified_cuda_samples/asyncAPI/asyncAPI.cu)
#----
add_custom_target(modified_cuda_samples)
add_dependencies(modified_cuda_samples vectorAdd inlinePTX simpleStreams simpleIPC)

add_executable(version_management by_runtime_api_module/version_management.cpp)
add_executable(error_handling by_runtime_api_module/error_handling.cu)
add_executable(device_management by_runtime_api_module/device_management.cpp)
add_executable(execution_control by_runtime_api_module/execution_control.cu)

add_executable(stream_management by_runtime_api_module/stream_management.cu)
add_executable(event_management by_runtime_api_module/event_management.cu)
add_executable(unified_addressing by_runtime_api_module/unified_addressing.cpp)
add_executable(io_compute_overlap_with_streams other/io_compute_overlap_with_streams.cu)
add_executable(inclusion_in_two_translation_units other/inclusion_in_two_translation_units/main.cpp other/inclusion_in_two_translation_units/second_tu.cpp )

if(NOT "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")
    # A weird NVCC-only linking issue
    set_target_properties(execution_control PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    # __nv_tex_surf_handler not yet implemented in Clang (as of 11.0)
    # https://reviews.llvm.org/D76365
    add_executable(array_management other/array_management.cu)
endif()
