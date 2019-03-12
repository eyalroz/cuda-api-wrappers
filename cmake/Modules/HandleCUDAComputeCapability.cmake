# This module determines which compute capability / SM version
# we should be compiling our CUDA code for, and adds the appropriate
# switch to the NVCC compiler flags - so that you don't have to worry
# about it.
#
# TODO: Be willing to take CUDA_CC, CUDA_TARGET_COMPUTE_CAPABILITY, 
# CUDA_TARGET_COMPUTE or CUDA_TARGET_COMPUTE_CAP and maybe even 
# those without the CUDA_ prefix

if (NOT CUDA_TARGET_COMPUTE_CAPABILITY)
	if (CMAKE_CUDA_COMPILER_LOADED)
		set(CUDAFILE ${CMAKE_CURRENT_BINARY_DIR}/check_cuda.cu)
		file(WRITE ${CUDAFILE}
"#include <stdio.h>\n\
#include <cuda_runtime_api.h>\n\
int main()\n\
{\n\
	cudaDeviceProp prop;\n\
	cudaError_t status;\n\
	int device_count;\n\
	status = cudaGetDeviceCount(&device_count);\n\
	int device_index = 0;\n\
	if (status != cudaSuccess) {\n\
		fprintf(stderr,\"cudaGetDeviceCount() failed: %s\\n\", cudaGetErrorString(status));\n\
		return -1;\n\
	}\n\
	if (device_index >= device_count) {\n\
		fprintf(stderr, \"Specified device index %d exceeds the maximum (the device count on this system is %d)\\n\", device_index, device_count);\n\
		return -1;\n\
	}\n\
	status = cudaGetDeviceProperties(&prop, device_index);\n\
	if (status != cudaSuccess) {\n\
		fprintf(stderr,\"cudaGetDeviceProperties() for device ${device_index} failed: %s\\n\", cudaGetErrorString(status));\n\
		return -1;\n\
	}\n\
	printf(\"%d%d %d.%d\", prop.major, prop.minor, prop.major, prop.minor);\n\
	return 0;\n\
}")
		try_run(run_result compile_result ${CMAKE_CURRENT_BINARY_DIR} ${CUDAFILE}
			RUN_OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY_
			COMPILE_OUTPUT_VARIABLE compile_output)
		if(NOT "${compile_result}")
			message(SEND_ERROR "Detecting CUDA_TARGET_COMPUTE_CAPABILITY failed: Compilation failure")
			message(SEND_ERROR "${compile_output}")
		elseif(NOT ("${run_result}" EQUAL "0"))
			message(SEND_ERROR "Detecting CUDA_TARGET_COMPUTE_CAPABILITY failed: Run failure")
			message(SEND_ERROR "${CUDA_TARGET_COMPUTE_CAPABILITY_}")
		endif()
		separate_arguments(CUDA_TARGET_COMPUTE_CAPABILITY_ UNIX_COMMAND "${CUDA_TARGET_COMPUTE_CAPABILITY_}")
		list(GET CUDA_TARGET_COMPUTE_CAPABILITY_ 1 FORMATTED_COMPUTE_CAPABILITY)
		list(GET CUDA_TARGET_COMPUTE_CAPABILITY_ 0 CUDA_TARGET_COMPUTE_CAPABILITY_)
	endif()
endif()

set(CUDA_TARGET_COMPUTE_CAPABILITY "${CUDA_TARGET_COMPUTE_CAPABILITY_}" CACHE STRING "CUDA compute capability of the (first) \
	CUDA device on the system, in XY format (like the X.Y format but no dot); see table of features and capabilities by \
	capability X.Y value at https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications")

message(STATUS "CUDA device-side code will assume compute capability ${FORMATTED_COMPUTE_CAPABILITY}")

set(CUDA_GENCODE "arch=compute_${CUDA_TARGET_COMPUTE_CAPABILITY},code=compute_${CUDA_TARGET_COMPUTE_CAPABILITY}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode ${CUDA_GENCODE} )

