# This module determines which compute capability / SM version
# we should be compiling our CUDA code for, and adds the appropriate
# switch to the NVCC compiler flags - so that you don't have to worry
# about it.
#
# TODO: Be willing to take CUDA_CC, CUDA_TARGET_COMPUTE_CAPABILITY, 
# CUDA_TARGET_COMPUTE or CUDA_TARGET_COMPUTE_CAP and maybe even 
# those without the CUDA_ prefix

if("$ENV{CUDA_SM}" STREQUAL "")
        set(ENV{CUDA_INCLUDE_DIRS} "${CUDA_INCLUDE_DIRS}")
        set(ENV{CUDA_CUDART_LIBRARY} "${CUDA_CUDART_LIBRARY}")
        set(ENV{CMAKE_CXX_COMPILER} "${CMAKE_CXX_COMPILER}")
        execute_process(COMMAND bash -c "${CMAKE_CURRENT_SOURCE_DIR}/scripts/get_cuda_sm.sh"  OUTPUT_VARIABLE CUDA_TARGET_COMPUTE_CAPABILITY) 
else("$ENV{CUDA_SM}" STREQUAL "")
        set(CUDA_TARGET_COMPUTE_CAPABILITY $ENV{CUDA_SM})
endif("$ENV{CUDA_SM}" STREQUAL "")

execute_process(COMMAND bash -c "echo -n $(echo ${CUDA_TARGET_COMPUTE_CAPABILITY}) | sed 's/^\\([0-9]\\)\\([0-9]\\)/\\1.\\2/;' | xargs echo -n" OUTPUT_VARIABLE FORMATTED_COMPUTE_CAPABILITY) 

message("Building for Compute Capability ${FORMATTED_COMPUTE_CAPABILITY}.")

set(CUDA_GENCODE "arch=compute_${CUDA_TARGET_COMPUTE_CAPABILITY},code=compute_${CUDA_TARGET_COMPUTE_CAPABILITY}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode ${CUDA_GENCODE} )

