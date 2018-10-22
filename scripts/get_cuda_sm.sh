#!/bin/bash
#
# Prints the compute capability of the first CUDA device installed
# on the system, or alternatively the device whose index is the
# first command-line argument

device_index=${1:-0}
timestamp=$(date +%s.%N)
gcc_binary=${CMAKE_CXX_COMPILER:-$(which c++)}
cuda_root=${CUDA_DIR:-/usr/local/cuda}
nvcc_default_path=$(which nvcc:-"/usr/local/cuda/bin/nvcc")
cuda_root=${CUDA_DIR:-${nvcc_default_path:0:-4}..}
cuda_include_dirs=${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES:-${cuda_root}/include}
cuda_link_directories=${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES:-${cuda_root}/lib64}
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${cuda_link_directories}"
cuda_link_libraries=${CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES:--lcudart}
cudart_library=${CUDA_CUDART_LIBRARY:-${cuda_link_directory}/libcudart.so}
generated_binary="/tmp/cuda-compute-version-helper-$$-$timestamp"
# create a 'here document' that is code we compile and use to probe the card
source_code="$(cat << EOF 
#include <stdio.h>
#include <cuda_runtime_api.h>

int main()
{
	cudaDeviceProp prop;
	cudaError_t status;
	int device_count;
	status = cudaGetDeviceCount(&device_count);
	if (status != cudaSuccess) { 
		fprintf(stderr,"cudaGetDeviceCount() failed: %s\n", cudaGetErrorString(status)); 
		return -1;
	}
	if (${device_index} >= device_count) {
		fprintf(stderr, "Specified device index %d exceeds the maximum (the device count on this system is %d)\n", ${device_index}, device_count);
		return -1;
	}
	status = cudaGetDeviceProperties(&prop, ${device_index});
	if (status != cudaSuccess) { 
		fprintf(stderr,"cudaGetDeviceProperties() for device ${device_index} failed: %s\n", cudaGetErrorString(status)); 
		return -1;
	}
	int v = prop.major * 10 + prop.minor;
	printf("%d\\n", v);
}
EOF
)"
echo "$source_code" | $gcc_binary -x c++ -I"$cuda_include_dirs" -o "$generated_binary" - -x none -L${cuda_link_directories} ${cuda_link_libraries}

# probe the card and cleanup

$generated_binary
rm $generated_binary
