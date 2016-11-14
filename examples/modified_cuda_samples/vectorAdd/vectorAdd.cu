/**
 * Derived from the nVIDIA CUDA 8.0 samples by
 *
 *   Eyal Rozenberg <E.Rozenberg@cwi.nl>
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 */

#include "cuda/api_wrappers.h"

#include <iostream>
#include <memory>
#include <algorithm>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements) { C[i] = A[i] + B[i]; }
}

int main(void)
{
	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	std::cout << "[Vector addition of " << numElements << " elements]\n";

	// If we could rely on C++14, we would  use std::make_unique
	auto h_A = std::unique_ptr<float>(new float[size]);
	auto h_B = std::unique_ptr<float>(new float[size]);
	auto h_C = std::unique_ptr<float>(new float[size]);

	auto generator = []() { return rand() / (float) RAND_MAX; };
	std::generate(h_A.get(), h_A.get() + numElements, generator);
	std::generate(h_B.get(), h_B.get() + numElements, generator);

	auto d_A = cuda::memory::device::make_unique<float[]>(numElements);
	auto d_B = cuda::memory::device::make_unique<float[]>(numElements);
	auto d_C = cuda::memory::device::make_unique<float[]>(numElements);

	cuda::memory::copy(d_A.get(), h_A.get(), size);
	cuda::memory::copy(d_B.get(), h_B.get(), size);

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	std::cout
		<< "CUDA kernel launch with " << blocksPerGrid
		<< " blocks of " << threadsPerBlock << " threads\n";

	cuda::launch(
		vectorAdd,
		{ blocksPerGrid, threadsPerBlock },
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
	return 0;
}

