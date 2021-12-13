/**
 * Derived from the nVIDIA CUDA 8.0 samples by
 *
 *   Eyal Rozenberg
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 *
 * This version differs from the other vectorAdd example in that mapped memory is
 * used instead of regular host and device memory.
 */

#include <cuda/runtime_api.hpp>

#include <iostream>
#include <memory>
#include <algorithm>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements) { C[i] = A[i] + B[i]; }
}

int main()
{
	if (cuda::device::count() == 0) {
		std::cerr << "No CUDA devices on this system" << "\n";
		exit(EXIT_FAILURE);
	}

	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	std::cout << "[Vector addition of " << numElements << " elements]\n";

	auto device = cuda::device::current::get();
	auto buffer_A = cuda::memory::mapped::allocate(device, size);
	auto buffer_B = cuda::memory::mapped::allocate(device, size);
	auto buffer_C = cuda::memory::mapped::allocate(device, size);

	auto h_A = (float*) buffer_A.host_side; auto d_A = (float*) buffer_A.device_side;
	auto h_B = (float*) buffer_B.host_side; auto d_B = (float*) buffer_B.device_side;
	auto h_C = (float*) buffer_C.host_side; auto d_C = (float*) buffer_C.device_side;

	auto generator = []() { return rand() / (float) RAND_MAX; };
	std::generate(h_A, h_A + numElements, generator);
	std::generate(h_B, h_B + numElements, generator);

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	std::cout
		<< "CUDA kernel launch with " << blocksPerGrid
		<< " blocks of " << threadsPerBlock << " threads\n";

	cuda::launch(
		vectorAdd,
		cuda::make_launch_config( blocksPerGrid, threadsPerBlock ),
		d_A, d_B, d_C, numElements
	);

	// Synchronization is necessary here despite the synchronous nature of the default stream -
	// since the copying-back of data is not something we've waited for
	device.synchronize();

	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i) {
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)  {
			std::cerr << "Result verification failed at element " << i << "\n";
			exit(EXIT_FAILURE);
		}
	}

	cuda::memory::mapped::free(buffer_A);
	cuda::memory::mapped::free(buffer_B);
	cuda::memory::mapped::free(buffer_C);

	std::cout << "Test PASSED\n";
	std::cout << "SUCCESS\n";
	return 0;
}

