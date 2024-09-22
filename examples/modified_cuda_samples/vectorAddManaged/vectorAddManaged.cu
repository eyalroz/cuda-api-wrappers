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
 * This version differs from the other vectorAdd example in that managed memory is
 * used instead of regular host and device memory.
 */

#include <cuda/api.hpp>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <random>

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
	std::cout << "[Vector addition of " << numElements << " elements]\n";

	auto buffer_A = cuda::memory::managed::make_unique_span<float>(numElements);
	auto buffer_B = cuda::memory::managed::make_unique_span<float>(numElements);
	auto buffer_C = cuda::memory::managed::make_unique_span<float>(numElements);

	auto generator = []() {
		static std::random_device random_device;
		static std::mt19937 randomness_generator { random_device() };
		static std::uniform_real_distribution<float> distribution { 0.0, 1.0 };
		return distribution(randomness_generator);
	};
	std::generate(buffer_A.begin(), buffer_A.end(), generator);
	std::generate(buffer_B.begin(), buffer_B.end(), generator);

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
		buffer_A.data(), buffer_B.data(), buffer_C.data(), numElements
	);

	// Synchronization is necessary here despite the synchronous nature of the default stream -
	// since the copying-back of data is not something we've waited for
	cuda::device::current::get().synchronize();

	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i) {
		if (std::fabs(buffer_A[i] + buffer_B[i] - buffer_C[i]) > 1e-5f)  {
			std::cerr << "Result verification failed at element " << i << "\n";
			exit(EXIT_FAILURE);
		}
	}

	std::cout << "Test PASSED\n";
	std::cout << "SUCCESS\n";
	return 0;
}

