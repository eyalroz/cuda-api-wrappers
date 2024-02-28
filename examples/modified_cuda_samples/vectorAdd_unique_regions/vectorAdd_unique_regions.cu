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

#include "../../common.hpp"

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
	std::cout << "[Vector addition of " << numElements << " elements]\n";

	auto h_A = std::vector<float>(numElements);
	auto h_B = std::vector<float>(numElements);
	auto h_C = std::vector<float>(numElements);

	auto generator = []() { return rand() / (float) RAND_MAX; };
	std::generate(h_A.begin(), h_A.end(), generator);
	std::generate(h_B.begin(), h_B.end(), generator);

	auto device = cuda::device::current::get();

	auto d_A = cuda::memory::make_unique_region(device, numElements * sizeof(float));
	auto d_B = cuda::memory::make_unique_region(device, numElements * sizeof(float));
	auto d_C = cuda::memory::make_unique_region(device, numElements * sizeof(float));
	auto sp_A = d_A.as_span<float>();
	auto sp_B = d_B.as_span<float>();
	auto sp_C = d_C.as_span<float>();

	cuda::memory::copy(sp_A, h_A.data());
	cuda::memory::copy(sp_B, h_B.data());

	auto launch_config = cuda::launch_config_builder()
		.overall_size(numElements)
		.block_size(256)
		.build();

	std::cout
		<< "CUDA kernel launch with " << launch_config.dimensions.grid.x
		<< " blocks of " << launch_config.dimensions.block.x << " threads each\n";

	cuda::launch(
		vectorAdd, launch_config,
		sp_A.data(), sp_B.data(), sp_C.data(), numElements
	);

	cuda::memory::copy(h_C.data(), sp_C);

	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i) {
		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)  {
			std::cerr << "Result verification failed at element " << i << "\n";
			exit(EXIT_FAILURE);
		}
	}

	std::cout << "Test PASSED\n";
	std::cout << "SUCCESS\n";
}

