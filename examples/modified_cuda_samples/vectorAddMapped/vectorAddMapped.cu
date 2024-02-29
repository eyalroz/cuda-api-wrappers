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

#include <cuda/api.hpp>

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

	auto a = buffer_A.as_spans<float>();
	auto b = buffer_B.as_spans<float>();
	auto c = buffer_C.as_spans<float>();

	auto generator = []() { return rand() / (float) RAND_MAX; };
	std::generate(a.host_side.begin(), b.host_side.end(), generator);

	// Launch the Vector Add CUDA Kernel

	auto launch_config = cuda::launch_config_builder()
		.overall_size(numElements)
		.block_size(256)
		.build();

	std::cout
		<< "CUDA kernel launch with " << launch_config.dimensions.grid.volume()
		<< " blocks of " << launch_config.dimensions.block.volume() << " threads\n";

	cuda::launch(
		vectorAdd, launch_config,
		a.device_side.data(), b.device_side.data(), c.device_side.data(), numElements);

	// Synchronization is necessary here despite the synchronous nature of the default stream -
	// since the copying-back of data is not something we've waited for
	device.synchronize();


	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i) {
		if (fabs(a.host_side[i] + b.host_side[i] - c.host_side[i]) > 1e-5)  {
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

