enable_language(CXX)
enable_language(CUDA)

if(NOT DEFINED CUDA_ARCH_FLAGS_STR)
	include(FindCUDA/select_compute_arch)
	# After having trouble going through CUDA_ARCHITECTURES, we're falling back on the old-style method of directly appending to CMAKE_CUDA_FLAGS.
	if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
		cmake_policy(SET CMP0104 OLD)
	endif()
	if((NOT DEFINED CUDA_ARCH_FLAGS) OR ("${CUDA_ARCH_FLAGS}" STREQUAL ""))
		cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS_1 Auto)
		set(CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS} CACHE STRING "CUDA -gencode parameters")
		string(REPLACE ";" " " CUDA_ARCH_FLAGS_STR_ "${CUDA_ARCH_FLAGS}")
		set(CUDA_ARCH_FLAGS_STR ${CUDA_ARCH_FLAGS_STR_} CACHE INTERNAL "CUDA Architecture flags")
	else()
		set(CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS}" CACHE INTERNAL "CUDA Architecture flags")
	endif()
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS_STR}")

include(CompileWithWarnings)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

link_libraries(runtime-and-driver)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
add_executable(vectorAdd modified_cuda_samples/vectorAdd/vectorAdd.cu)
add_executable(vectorAddMapped modified_cuda_samples/vectorAddMapped/vectorAddMapped.cu)
add_executable(vectorAddManaged modified_cuda_samples/vectorAddManaged/vectorAddManaged.cu)
add_executable(vectorAdd_nvrtc modified_cuda_samples/vectorAdd_nvrtc/vectorAdd_nvrtc.cpp)
target_link_libraries(vectorAdd_nvrtc nvrtc)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.3")
	# Needs direct cubin access for NVRTC programs
	add_executable(clock_nvrtc modified_cuda_samples/clock_nvrtc/clock.cpp)
	target_link_libraries(clock_nvrtc nvrtc)
endif()
add_executable(simpleDrvRuntimePTX modified_cuda_samples/simpleDrvRuntimePTX/simpleDrvRuntimePTX.cpp)
add_executable(inlinePTX modified_cuda_samples/inlinePTX/inlinePTX.cu)
add_executable(simpleStreams modified_cuda_samples/simpleStreams/simpleStreams.cu)
add_executable(simpleIPC modified_cuda_samples/simpleIPC/simpleIPC.cu)
if ((NOT DEFINED CMAKE_CUDA_HOST_COMPILER) OR (CMAKE_CUDA_HOST_COMPILER STREQUAL ""))
	set(CCBIN_ARGUMENT)
else()
	set(CCBIN_ARGUMENT "-ccbin=${CMAKE_CUDA_HOST_COMPILER}")
endif()
IF((NOT ${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
   AND (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.2"))
	# compile the kernel into a fatbin : modified_cuda_samples/memMapIPCDrv/memMapIpc_kernel.cu
	add_executable(memMapIPCDrv
		modified_cuda_samples/memMapIPCDrv/memMapIPC.cpp 
		modified_cuda_samples/memMapIPCDrv/child.cpp 
		modified_cuda_samples/memMapIPCDrv/parent.cpp
		modified_cuda_samples/memMapIPCDrv/helper_multiprocess.cpp
	)
	target_link_libraries(memMapIPCDrv c rt)
	add_custom_command(
			OUTPUT memMapIPC_kernel.fatbin
			COMMAND ${CMAKE_CUDA_COMPILER} -fatbin
				${CCBIN_ARGUMENT}
				--generate-code arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}
				-o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/memMapIPC_kernel.fatbin
				${CMAKE_CURRENT_SOURCE_DIR}/modified_cuda_samples/memMapIPCDrv/memMapIPC_kernel.cu
			MAIN_DEPENDENCY modified_cuda_samples/memMapIPCDrv/memMapIPC_kernel.cu
	)
	add_custom_target(do_build_memMapIPCDrv_kernel ALL DEPENDS memMapIPC_kernel.fatbin)
	add_dependencies(memMapIPCDrv do_build_memMapIPCDrv_kernel)
	add_executable(vectorAddMMAP modified_cuda_samples/vectorAddMMAP/vectorAddMMAP.cpp)
	add_custom_command(
			OUTPUT vectorAddMMAP_kernel.fatbin
			COMMAND ${CMAKE_CUDA_COMPILER} -fatbin
				${CCBIN_ARGUMENT}
				--generate-code arch=compute_${CMAKE_CUDA_ARCHITECTURES},code=sm_${CMAKE_CUDA_ARCHITECTURES}
				-o ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/vectorAddMMAP_kernel.fatbin
				${CMAKE_CURRENT_SOURCE_DIR}/modified_cuda_samples/vectorAddMMAP/vectorAdd_kernel.cu
			MAIN_DEPENDENCY modified_cuda_samples/vectorAddMMAP/vectorAdd_kernel.cu
	)
	add_custom_target(do_build_vectorAddMMAP_kernel ALL DEPENDS vectorAddMMAP_kernel.fatbin)
	add_dependencies(vectorAddMMAP do_build_vectorAddMMAP_kernel)
endif()
add_executable(p2pBandwidthLatencyTest modified_cuda_samples/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest.cu)
add_executable(asyncAPI modified_cuda_samples/asyncAPI/asyncAPI.cu)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
	# This one needs cooperative_groups/reduce.h
	add_executable(binaryPartitionCG modified_cuda_samples/binaryPartitionCG/binaryPartitionCG.cu)
endif()
#----
add_custom_target(modified_cuda_samples)
add_dependencies(modified_cuda_samples vectorAdd inlinePTX simpleStreams simpleIPC)

add_executable(version_management by_runtime_api_module/version_management.cpp)
add_executable(error_handling by_runtime_api_module/error_handling.cu)
add_executable(device_management by_runtime_api_module/device_management.cpp)
add_executable(context_management by_driver_api_module/context_management.cpp)
add_executable(execution_control by_runtime_api_module/execution_control.cu)
set_property(TARGET execution_control  PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(stream_management by_runtime_api_module/stream_management.cu)
add_executable(event_management by_runtime_api_module/event_management.cu)
add_executable(unified_addressing by_runtime_api_module/unified_addressing.cpp)
add_executable(io_compute_overlap_with_streams other/io_compute_overlap_with_streams.cu)
add_executable(vectorAdd_profiled other/vectorAdd_profiled.cu)
target_link_libraries(vectorAdd_profiled nvtx)
add_executable(manipulate_current_device other/manipulate_current_device.cu)
add_executable(inclusion_in_two_translation_units other/inclusion_in_two_translation_units/main.cpp other/inclusion_in_two_translation_units/second_tu.cpp )
add_executable(jitify other/jitify/jitify.cpp)
set(jitify_headers_target_dir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/example_headers)
file(MAKE_DIRECTORY ${jitify_headers_target_dir})
execute_process(COMMAND ${CMAKE_COMMAND} -E make_directory ${jitify_headers_target_dir})
file(GLOB jitify_headers ${CMAKE_CURRENT_SOURCE_DIR}/other/jitify/*.cuh)
foreach(hdr ${jitify_headers})
	add_custom_command(
		TARGET jitify
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy ${hdr} ${jitify_headers_target_dir})
endforeach()
target_link_libraries(jitify nvrtc stdc++fs)

if(NOT "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")

	# __nv_tex_surf_handler not yet implemented in Clang (as of 11.0)
	# https://reviews.llvm.org/D76365
	add_executable(array_management other/array_management.cu)
endif()
