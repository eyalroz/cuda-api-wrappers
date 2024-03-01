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

	// If we could rely on C++14, we would  use std::make_unique
	auto h_A = std::unique_ptr<float[]>(new float[numElements]);
	auto h_B = std::unique_ptr<float[]>(new float[numElements]);
	auto h_C = std::unique_ptr<float[]>(new float[numElements]);

	auto generator = []() { return rand() / (float) RAND_MAX; };
	std::generate(h_A.get(), h_A.get() + numElements, generator);
	std::generate(h_B.get(), h_B.get() + numElements, generator);

	auto device = cuda::device::current::get();
	auto d_A = cuda::memory::make_unique<float[]>(device, numElements);
	auto d_B = cuda::memory::make_unique<float[]>(device, numElements);
	auto d_C = cuda::memory::make_unique<float[]>(device, numElements);

	cuda::memory::copy(d_A.get(), h_A.get(), size);
	cuda::memory::copy(d_B.get(), h_B.get(), size);

	auto launch_config = cuda::launch_config_builder()
		.overall_size(numElements)
		.block_size(256)
		.build();

	std::cout
		<< "CUDA kernel launch with " << launch_config.dimensions.grid.x
		<< " blocks of " << launch_config.dimensions.block.x << " threads each\n";

	cuda::launch(
		vectorAdd, launch_config,
		d_A.get(), d_B.get(), d_C.get(), numElements
	);

	cuda::memory::copy(h_C.get(), d_C.get(), size);

	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i) {
		if (fabs(h_A.get()[i] + h_B.get()[i] - h_C.get()[i]) > 1e-5)  {
			std::cerr << "Result verification failed at element " << i << "\n";
			exit(EXIT_FAILURE);
		}
	}

	std::cout << "Test PASSED\n";
	std::cout << "SUCCESS\n";
}

