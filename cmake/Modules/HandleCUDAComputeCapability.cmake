# This module determines which compute capability / SM version
# we should be compiling our CUDA code for, and adds the appropriate
# switch to the NVCC compiler flags - so that you don't have to worry
# about it.
#
# TODO:
# * Be willing to take CUDA_CC, CUDA_TARGET_COMPUTE_CAPABILITY,
#   CUDA_TARGET_COMPUTE or CUDA_TARGET_COMPUTE_CAP and maybe even
#   those without the CUDA_ prefix
# * Support for CMake versions under 3.8 (shouldn't be difficult,
#   just different variable names
# * Support "roll-up" of redundant existing nvcc flags
# * Support clang instead of nvcc
#
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

if (NOT CUDA_TARGET_COMPUTE_CAPABILITY)
	if (CMAKE_CUDA_COMPILER_LOADED)
		set(QUERY_CUDA_COMPUTE_CAPABILITY_SOURCE ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/query_gpu_compute_capability.cu)
		file(WRITE ${QUERY_CUDA_COMPUTE_CAPABILITY_SOURCE}
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
		try_run(
			QUERY_CUDA_COMPUTE_CAPABILITY_RUN_RESULT
			QUERY_CUDA_COMPUTE_CAPABILITY_COMPILE_RESULT
			${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}
			${QUERY_CUDA_COMPUTE_CAPABILITY_SOURCE}
			RUN_OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITIES_
			COMPILE_OUTPUT_VARIABLE QUERY_CUDA_COMPUTE_CAPABILITY_COMPILE_OUTPUT)
		if(NOT "${QUERY_CUDA_COMPUTE_CAPABILITY_COMPILE_RESULT}")
			message(SEND_ERROR "CUDA device compute capability query: Compilation failure")
			message(SEND_ERROR "${QUERY_CUDA_COMPUTE_CAPABILITY_COMPILE_OUTPUT}")
		elseif(NOT ("${QUERY_CUDA_COMPUTE_CAPABILITY_RUN_RESULT}" EQUAL "0"))
			message(SEND_ERROR "CUDA device compute capability query: Runtime error")
			message(SEND_ERROR "${CUDA_TARGET_COMPUTE_CAPABILITIES_}")
		endif()
	endif()
endif()

set(CUDA_TARGET_COMPUTE_CAPABILITY "${CUDA_TARGET_COMPUTE_CAPABILITIES_}" CACHE STRING "List of CUDA compute capabilities of the targeted \
	CUDA devices in X.Y format; list items separated by semicolons; see table of features and capabilities by \
	capability X.Y value at https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications")

string(REPLACE ";" ", " COMPUTE_CAPABILITIES_FORMATTED_FOR_PRINTING "${CUDA_TARGET_COMPUTE_CAPABILITY}")
message(STATUS "CUDA compilation target architecture(s): ${COMPUTE_CAPABILITIES_FORMATTED_FOR_PRINTING}")

set(NVCC_TARGET_COMPUTE_CAPABILITY_FLAGS "")
foreach(COMPUTE_CAPABILITY ${CUDA_TARGET_COMPUTE_CAPABILITY})
	string(REPLACE "." "" COMPUTE_CAPABILITY "${COMPUTE_CAPABILITY}")
	# nvcc's documentation is rather confusing regarding what we should actually set these two variables, arch and code, to. It
	# seems they've unified the set of possible values, so we're just going to go with something simplistic which works.
	string(APPEND NVCC_TARGET_COMPUTE_CAPABILITY_FLAGS " -gencode arch=compute_${COMPUTE_CAPABILITY},code=compute_${COMPUTE_CAPABILITY}")
endforeach(COMPUTE_CAPABILITY)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_TARGET_COMPUTE_CAPABILITY_FLAGS}")
