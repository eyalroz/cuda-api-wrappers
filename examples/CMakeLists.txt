list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")
include(CheckIncludeFiles)

option(CAW_WITH_COOPERATIVE_GROUPS "Enable cooperative-groups-related code in example programs" ON)
option(CAW_EXAMPLES_STATIC_DEPS "Have the example programs depend on the static versions of raw CUDA targets" OFF)

enable_language(CXX)
enable_language(CUDA)

if(NOT DEFINED CUDA_ARCH_FLAGS_STR)
	include(FindCUDA/select_compute_arch)
	# After having trouble going through CUDA_ARCHITECTURES, we're falling back on the old-style method of directly appending to CMAKE_CUDA_FLAGS.
	if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
		cmake_policy(SET CMP0104 OLD)
	endif()
	if((NOT DEFINED CUDA_ARCH_FLAGS) OR ("${CUDA_ARCH_FLAGS}" STREQUAL ""))
		cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS_ Auto)
		set(CUDA_ARCH_FLAGS ${CUDA_ARCH_FLAGS_} CACHE STRING "CUDA -gencode parameters")
		string(REPLACE ";" " " CUDA_ARCH_FLAGS_STR_ "${CUDA_ARCH_FLAGS}")
		set(CUDA_ARCH_FLAGS_STR ${CUDA_ARCH_FLAGS_STR_} CACHE INTERNAL "CUDA Architecture flags")
	else()
		string(REPLACE ";" " " CUDA_ARCH_FLAGS_STR_ "${CUDA_ARCH_FLAGS}")
		set(CUDA_ARCH_FLAGS_STR "${CUDA_ARCH_FLAGS_STR_}" CACHE INTERNAL "CUDA Architecture flags")
	endif()
endif()
# NVCC is unfriendly if you specify multiple architectures, only some of which are supported; and
# many versions of CMake will indeed tend to include older architectures in the auto-detection 
# fallback, so - we must do a bit of purging.
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
	string(REGEX REPLACE " ?-gencode arch=compute_(([1-6][0-9])|(7[0-2])),code=sm_(([1-6][0-9])|(7[0-2]))" "" CUDA_ARCH_FLAGS_STR_NO_UNSUPPORTED "${CUDA_ARCH_FLAGS_STR}")
else()
	string(CONCAT CUDA_ARCH_FLAGS_STR_NO_UNSUPPORTED "${CUDA_ARCH_FLAGS_STR}")
endif()
string(APPEND CMAKE_CUDA_FLAGS " ${CUDA_ARCH_FLAGS_STR_NO_UNSUPPORTED} -Wno-deprecated-gpu-targets")
separate_arguments(CMAKE_CUDA_FLAGS_AS_LIST NATIVE_COMMAND ${CMAKE_CUDA_FLAGS})

include(CompileWithWarnings)

#if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-gpu-targets")
#endif()

if(MSVC)
	# MSVC is very problematic! ... lets go easy on it

	set(CAW_WITH_COOPERATIVE_GROUPS OFF)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:__cplusplus")
	# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /NODEFAULTLIB:nvptxcompiler_static.lib")
	set(CMAKE_CXX_STANDARD 17)
	add_compile_definitions("$<$<COMPILE_LANGUAGE:CXX,CUDA>:CUDA_NO_HALF>")
	add_compile_definitions("$<$<COMPILE_LANGUAGE:C,CXX>:_CRT_SECURE_NO_DEPRECATE>")
	add_compile_definitions("$<$<COMPILE_LANGUAGE:C,CXX>:_CRT_SECURE_NO_WARNINGS>")
	set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
	set(CMAKE_CXX_STANDARD 11)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

if(CAW_WITH_COOPERATIVE_GROUPS AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "9.0")
	add_compile_definitions(USE_COOPERATIVE_GROUPS _CG_LIMIT_INCLUDED_DEPENDENCIES)
	message(STATUS "Using cooperative groups in example programs")
endif()

if (CAW_EXAMPLES_STATIC_DEPS)
	set(static_suffix "_static")
else()
	set(static_suffix "")
endif()
link_libraries(cuda-api-wrappers::runtime-and-driver${static_suffix})
set(rtc_lib_tgt cuda-api-wrappers::rtc${static_suffix})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
add_executable(vectorAdd modified_cuda_samples/vectorAdd/vectorAdd.cu)
if (TARGET CUDA::cublas)
	add_executable(matrixMulCUBLAS modified_cuda_samples/matrixMulCUBLAS/matrixMulCUBLAS.cpp)
	target_link_libraries(matrixMulCUBLAS CUDA::cublas${static_suffix})
endif()
add_executable(vectorAdd_unique_regions modified_cuda_samples/vectorAdd_unique_regions/vectorAdd_unique_regions.cu)
add_executable(vectorAddMapped modified_cuda_samples/vectorAddMapped/vectorAddMapped.cu)
add_executable(vectorAddManaged modified_cuda_samples/vectorAddManaged/vectorAddManaged.cu)
add_executable(vectorAdd_nvrtc modified_cuda_samples/vectorAdd_nvrtc/vectorAdd_nvrtc.cpp)
target_link_libraries(vectorAdd_nvrtc ${rtc_lib_tgt})
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.1")
	add_executable(vectorAdd_ptx modified_cuda_samples/vectorAdd_ptx/vectorAdd_ptx.cpp)
	target_link_libraries(vectorAdd_ptx ${rtc_lib_tgt})
endif()
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.2")
	add_executable(streamOrderedAllocation modified_cuda_samples/streamOrderedAllocation/streamOrderedAllocation.cu)
	add_executable(streamOrderedAllocationP2P modified_cuda_samples/streamOrderedAllocationP2P/streamOrderedAllocationP2P.cu)
	if (UNIX)
		set(streamOrderedAllocationIPC_include_files unistd.h sched.h sys/mman.h sys/wait.h linux/version.h)
		CHECK_INCLUDE_FILES("${simpleIPC_include_files}" have_streamOrderedAllocationIPC_include_files LANGUAGE CXX)
		if (have_streamOrderedAllocationIPC_include_files)
			add_executable(streamOrderedAllocationIPC
					modified_cuda_samples/streamOrderedAllocationIPC/streamOrderedAllocationIPC.cu
					modified_cuda_samples/streamOrderedAllocationIPC/helper_multiprocess.cpp)
		endif()
	endif()
endif()
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.3")
	# Needs direct cubin access for NVRTC programs
	add_executable(clock_nvrtc modified_cuda_samples/clock_nvrtc/clock.cpp)
	target_link_libraries(clock_nvrtc ${rtc_lib_tgt})
endif()
add_executable(simpleDrvRuntimePTX modified_cuda_samples/simpleDrvRuntimePTX/simpleDrvRuntimePTX.cpp)
add_executable(inlinePTX modified_cuda_samples/inlinePTX/inlinePTX.cu)
add_executable(simpleStreams modified_cuda_samples/simpleStreams/simpleStreams.cu)
if(UNIX AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.6" OR CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.2")
	add_executable(binaryPartitionCG modified_cuda_samples/binaryPartitionCG/binaryPartitionCG.cu)
endif()
if(UNIX)
	set(simpleIPC_include_files unistd.h sched.h sys/mman.h sys/wait.h linux/version.h)
	CHECK_INCLUDE_FILES("${simpleIPC_include_files}" have_simpleIPC_include_files LANGUAGE CXX)
	if(have_simpleIPC_include_files)
		add_executable(simpleIPC modified_cuda_samples/simpleIPC/simpleIPC.cu)
	endif()
endif()
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
	if (UNIX)
		target_link_libraries(memMapIPCDrv rt)
	endif()
	add_custom_command(
			OUTPUT memMapIPC_kernel.fatbin
			COMMAND ${CMAKE_CUDA_COMPILER} -fatbin
				${CCBIN_ARGUMENT}
				${CMAKE_CUDA_FLAGS_AS_LIST}
				-o ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/memMapIPC_kernel.fatbin
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
				${CMAKE_CUDA_FLAGS_AS_LIST}
				-o ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/vectorAddMMAP_kernel.fatbin
				${CMAKE_CURRENT_SOURCE_DIR}/modified_cuda_samples/vectorAddMMAP/vectorAdd_kernel.cu
			MAIN_DEPENDENCY modified_cuda_samples/vectorAddMMAP/vectorAdd_kernel.cu
	)
	add_custom_target(do_build_vectorAddMMAP_kernel ALL DEPENDS vectorAddMMAP_kernel.fatbin)
	add_dependencies(vectorAddMMAP do_build_vectorAddMMAP_kernel)
endif()

add_executable(simpleTextureDrv modified_cuda_samples/simpleTextureDrv/simpleTextureDrv.cpp)
add_custom_command(
		OUTPUT simpleTexture_kernel.fatbin
		COMMAND ${CMAKE_CUDA_COMPILER} -fatbin
		${CCBIN_ARGUMENT}
		${CMAKE_CUDA_FLAGS_AS_LIST}
		-o ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/simpleTexture_kernel.fatbin
		${CMAKE_CURRENT_SOURCE_DIR}/modified_cuda_samples/simpleTextureDrv/simpleTexture_kernel.cu
		MAIN_DEPENDENCY modified_cuda_samples/simpleTextureDrv/simpleTexture_kernel.cu
)
add_custom_target(do_build_simpleTexture_kernel ALL DEPENDS simpleTexture_kernel.fatbin)
add_dependencies(simpleTextureDrv do_build_simpleTexture_kernel)
foreach(img "ref_rotated" "teapot512")
	add_custom_command(
			TARGET simpleTextureDrv POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy
			${CMAKE_CURRENT_SOURCE_DIR}/modified_cuda_samples/simpleTextureDrv/data/${img}.pgm
			${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/data/${img}.pgm)
endforeach()

add_dependencies(simpleTextureDrv do_build_simpleTexture_kernel)

add_executable(p2pBandwidthLatencyTest modified_cuda_samples/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest.cu)
add_executable(asyncAPI modified_cuda_samples/asyncAPI/asyncAPI.cu)
if(USE_COOPERATIVE_GROUPS AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.0")
	# This one needs cooperative_groups/reduce.h
	add_executable(binaryPartitionCG modified_cuda_samples/binaryPartitionCG/binaryPartitionCG.cu)
endif()
add_executable(bandwidthtest modified_cuda_samples/bandwidthtest/bandwidthtest.cpp)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "10.0")
	if((NOT MSVC)
		OR (CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.2")
		OR (CUDAToolkit_VERSION VERSION_LESS "11.7")
	)
		add_executable(simpleCudaGraphs modified_cuda_samples/simpleCudaGraphs/simpleCudaGraphs.cu)
		# Note: Between versions 11.7 and 12.1.1, CUDA's own cooperative groups header
		# tries to include the CUDA faux standard library - which it shouldn't
		add_executable(jacobiCudaGraphs
			modified_cuda_samples/jacobiCudaGraphs/main.cpp
			modified_cuda_samples/jacobiCudaGraphs/jacobi.cu
		)
	endif()
endif()
#----

add_executable(version_management by_api_module/version_management.cpp)
add_executable(error_handling by_api_module/error_handling.cu)
add_executable(device_management by_api_module/device_management.cpp)
add_executable(context_management by_api_module/context_management.cpp)
if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.2")
	add_executable(memory_pools by_api_module/memory_pools.cpp)
endif()
add_executable(execution_control by_api_module/execution_control.cu)
set_property(TARGET execution_control  PROPERTY CUDA_SEPARABLE_COMPILATION ON)

add_executable(stream_management by_api_module/stream_management.cu)
add_executable(event_management by_api_module/event_management.cu)
add_executable(unified_addressing by_api_module/unified_addressing.cpp)
add_executable(kernel_argument other/kernel_argument_validity.cu)
add_executable(io_compute_overlap_with_streams other/io_compute_overlap_with_streams.cu)
if( WIN32 AND CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
	add_executable(vectorAdd_profiled other/vectorAdd_profiled.cu)
	target_link_libraries(vectorAdd_profiled cuda-api-wrappers::nvtx${static_suffix})
endif()
add_executable(manipulate_current_device other/manipulate_current_device.cu)
add_executable(inclusion_in_two_translation_units other/inclusion_in_two_translation_units/main.cpp other/inclusion_in_two_translation_units/second_tu.cpp)
target_link_libraries(inclusion_in_two_translation_units ${rtc_lib_tgt} cuda-api-wrappers::nvtx${static_suffix})
if( NOT MSVC )
	foreach(std_version "14" "17" "20" "23")
		if("cxx_std_${std_version}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
			if (("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU") AND (${CMAKE_CXX_COMPILER_VERSION} LESS_EQUAL 8))
				continue()
			endif()
			set(tgt "cpp_${std_version}")
			add_executable(${tgt} other/new_cpp_standard/main.cpp)
			target_link_libraries(${tgt} ${rtc_lib_tgt} cuda-api-wrappers::nvtx${static_suffix} cuda-api-wrappers::fatbin${static_suffix})
			if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
				target_link_libraries(${tgt} stdc++fs)
			endif()
			set_target_properties(${tgt} PROPERTIES
				CXX_STANDARD ${std_version}
				CXX_STANDARD_REQUIRED ON
				CXX_EXTENSIONS OFF)
		endif()
	endforeach()
endif()
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
target_link_libraries(jitify ${rtc_lib_tgt})
if(NOT MSVC)
	target_link_libraries(jitify stdc++fs)
endif()
add_executable(module_management by_api_module/module_management.cpp)
target_link_libraries(module_management ${rtc_lib_tgt})


if(NOT "${CMAKE_CUDA_COMPILER_ID}" STREQUAL "Clang")

	# __nv_tex_surf_handler not yet implemented in Clang (as of 11.0)
	# https://reviews.llvm.org/D76365
	add_executable(array_management other/array_management.cu)
endif()
