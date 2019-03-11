# This module determines which compute capability / SM version
# we should be compiling our CUDA code for, and adds the appropriate
# switch to the NVCC compiler flags - so that you don't have to worry
# about it.
#
# TODO: Be willing to take CUDA_CC, CUDA_TARGET_COMPUTE_CAPABILITY, 
# CUDA_TARGET_COMPUTE or CUDA_TARGET_COMPUTE_CAP and maybe even 
# those without the CUDA_ prefix

if (NOT CUDA_TARGET_COMPUTE_CAPABILITY)
	if(WIN32)
		find_package(CUDA)
		if (CUDA_FOUND)
			set(CUDAFILE ${CMAKE_CURRENT_SOURCE_DIR}/examples/check_cuda.cu)
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
	else()
		if("$ENV{CUDA_SM}" STREQUAL "")
			set(ENV{CUDA_INCLUDE_DIRS} "${CUDA_INCLUDE_DIRS}")
			set(ENV{CUDA_CUDART_LIBRARY} "${CUDA_CUDART_LIBRARY}")
			set(ENV{CMAKE_CXX_COMPILER} "${CMAKE_CXX_COMPILER}")
			execute_process(COMMAND bash -c "${CMAKE_CURRENT_SOURCE_DIR}/scripts/get_cuda_sm.sh"  OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY_)
		else()
			set(CUDA_TARGET_COMPUTE_CAPABILITY_ $ENV{CUDA_SM})
		endif()

		execute_process(COMMAND bash -c "echo -n $(echo ${CUDA_TARGET_COMPUTE_CAPABILITY})" OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY)
		execute_process(COMMAND bash -c "echo ${CUDA_TARGET_COMPUTE_CAPABILITY} | sed 's/^\\([0-9]\\)\\([0-9]\\)/\\1.\\2/;' | xargs echo -n" OUTPUT_VARIABLE FORMATTED_COMPUTE_CAPABILITY)
	endif()
endif()

set(CUDA_TARGET_COMPUTE_CAPABILITY "${CUDA_TARGET_COMPUTE_CAPABILITY_}" CACHE STRING "CUDA compute capability of the (first) \
	CUDA device on the system, in XY format (like the X.Y format but no dot); see table of features and capabilities by \
	capability X.Y value at https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications")

message(STATUS "CUDA device-side code will assume compute capability ${FORMATTED_COMPUTE_CAPABILITY}")

set(CUDA_GENCODE "arch=compute_${CUDA_TARGET_COMPUTE_CAPABILITY},code=compute_${CUDA_TARGET_COMPUTE_CAPABILITY}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode ${CUDA_GENCODE} )

