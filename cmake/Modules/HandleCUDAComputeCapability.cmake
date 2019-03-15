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
		set(CUDAFILE ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/query_gpu_compute_capability.cu)
		file(WRITE ${CUDAFILE}
"#include <string>\n\
#include <iostream>\n\
#include <vector>\n\
#include <algorithm>\n\
#include <sstream>\n\
#include <cuda_runtime_api.h>\n\
int main()\n\
{\n\
	cudaDeviceProp prop;\n\
	cudaError_t status;\n\
	int device_count;\n\
	status = cudaGetDeviceCount(&device_count);\n\
	std::vector<std::string> sm_values;\n\
	if (status != cudaSuccess) {\n\
		fprintf(stderr,\"cudaGetDeviceCount() failed: %s\\n\", cudaGetErrorString(status));\n\
		return -1;\n\
	}\n\
	for(int device_index = 0; device_index < device_count; device_index++){\n\
		status = cudaGetDeviceProperties(&prop, device_index);\n\
		if (status != cudaSuccess) {\n\
			fprintf(stderr,\"cudaGetDeviceProperties() for device ${device_index} failed: %s\\n\", cudaGetErrorString(status));\n\
			return -1;\n\
		}\n\
		std::ostringstream ss;\n\
		ss << prop.major << \".\" << prop.minor;\n\
		sm_values.push_back(ss.str());\n\
	}\n\
	std::vector<std::string>::iterator end = std::unique(sm_values.begin(), sm_values.end()); ;\n\
	sm_values.resize(std::distance(sm_values.begin(), end));\n\
	for(std::vector<std::string>::iterator it = sm_values.begin(); it != sm_values.end(); it++) {\n\
		std::cout << *it;\n\
		if((it+1) != sm_values.end()) std::cout << \";\";\n\
	}\n\
	return 0;\n\
}")
		try_run(RUN_RESULT COMPILE_RESULT ${CMAKE_CURRENT_BINARY_DIR} ${CUDAFILE}
			RUN_OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY_
			COMPILE_OUTPUT_VARIABLE compile_output)
		if(NOT "${COMPILE_RESULT}")
			message(SEND_ERROR "Detecting CUDA_TARGET_COMPUTE_CAPABILITY failed: Compilation failure")
			message(SEND_ERROR "${compile_output}")
		elseif(NOT ("${RUN_RESULT}" EQUAL "0"))
			message(SEND_ERROR "Detecting CUDA_TARGET_COMPUTE_CAPABILITY failed: Run failure")
			message(SEND_ERROR "${CUDA_TARGET_COMPUTE_CAPABILITY_}")
		endif()
	endif()
endif()

set(CUDA_TARGET_COMPUTE_CAPABILITY "${CUDA_TARGET_COMPUTE_CAPABILITY_}" CACHE STRING "List of CUDA compute capabilities of the targeted \
	CUDA devices in X.Y format - see table of features and capabilities by \
	capability X.Y value at https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications")

message(STATUS "CUDA device-side code will generated for the compute capabilitie(s) ${CUDA_TARGET_COMPUTE_CAPABILITY}")

set(FORMATTED_COMPUTE_CAPABILITY "")
foreach(COMPUTE_CAPABILITY ${CUDA_TARGET_COMPUTE_CAPABILITY})
	string(REPLACE "." "" COMPUTE_CAPABILITY "${COMPUTE_CAPABILITY}")
	string(APPEND FORMATTED_COMPUTE_CAPABILITY " -gencode arch=compute_${COMPUTE_CAPABILITY},code=compute_${COMPUTE_CAPABILITY}")
endforeach(COMPUTE_CAPABILITY)

message(STATUS "Using extra NVCC flags:${FORMATTED_COMPUTE_CAPABILITY}")

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${FORMATTED_COMPUTE_CAPABILITY})
