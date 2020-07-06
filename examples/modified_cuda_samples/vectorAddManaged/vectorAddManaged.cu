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

	auto buffer_A = cuda::memory::managed::make_unique<float[]>(numElements);
	auto buffer_B = cuda::memory::managed::make_unique<float[]>(numElements);
	auto buffer_C = cuda::memory::managed::make_unique<float[]>(numElements);

	auto generator = []() { return rand() / (float) RAND_MAX; };
	std::generate(buffer_A.get(), buffer_A.get() + numElements, generator);
	std::generate(buffer_B.get(), buffer_B.get() + numElements, generator);

	// Launch the Vector Add CUDA Kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
	std::cout
		<< "CUDA kernel launch with " << blocksPerGrid
		<< " blocks of " << threadsPerBlock << " threads\n";

	cuda::launch(
		vectorAdd,
		cuda::make_launch_config( blocksPerGrid, threadsPerBlock ),
		buffer_A.get(), buffer_B.get(), buffer_C.get(), numElements
	);

	// Synchronization is necessary here despite the synchronous nature of the default stream -
	// since the copying-back of data is not something we've waited for
	cuda::device::current::get().synchronize();

	// Verify that the result vector is correct
	for (int i = 0; i < numElements; ++i) {
		if (fabs(buffer_A.get()[i] + buffer_B.get()[i] - buffer_C.get()[i]) > 1e-5)  {
			std::cerr << "Result verification failed at element " << i << "\n";
			exit(EXIT_FAILURE);
		}
	}

	std::cout << "Test PASSED\n";
	std::cout << "SUCCESS\n";
	return 0;
}

