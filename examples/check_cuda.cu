#include <stdio.h>
#include <cuda_runtime_api.h>

int main()
{
	cudaDeviceProp prop;
	cudaError_t status;
	int device_count;
	status = cudaGetDeviceCount(&device_count);
	int device_index = 0;
	if (status != cudaSuccess) { 
		fprintf(stderr,"cudaGetDeviceCount() failed: %s\n", cudaGetErrorString(status)); 
		return -1;
	}
	if (device_index >= device_count) {
		fprintf(stderr, "Specified device index %d exceeds the maximum (the device count on this system is %d)\n", device_index, device_count);
		return -1;
	}
	status = cudaGetDeviceProperties(&prop, device_index);
	if (status != cudaSuccess) { 
		fprintf(stderr,"cudaGetDeviceProperties() for device ${device_index} failed: %s\n", cudaGetErrorString(status)); 
		return -1;
	}
	printf("%d%d %d.%d", prop.major, prop.minor, prop.major, prop.minor);
	return 0;
}
