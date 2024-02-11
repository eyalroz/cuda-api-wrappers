/**
 * Derived from the nVIDIA CUDA 10.0 samples by
 *
 *   Eyal Rozenberg <eyalroz@technion.ac.il>
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

#include <cuda/rtc.hpp>
#include <cuda/api.hpp>

#include <iostream>
#include <cmath>
#include <random>

const char* vectorAdd_source = R"(

/**
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements) { C[i] = A[i] + B[i]; }
}

)";

int main(void)
{
	int numElements = 50000;
	size_t size = numElements * sizeof(float);
	auto kernel_name = "vectorAdd";

	std::cout << "[Vector addition of " << numElements << " elements]\n";

	auto device = cuda::device::current::get();
	auto compilation_output = cuda::rtc::program_t<cuda::cuda_cpp>(kernel_name)
		.set_source(vectorAdd_source)
		.add_registered_global(kernel_name)
		.set_target(device).compile();
	auto mangled_kernel_name = compilation_output.get_mangling_of(kernel_name);

	auto context = cuda::device::current::get().primary_context();
	auto module = cuda::module::create(context, compilation_output);
	auto vectorAdd = module.get_kernel(mangled_kernel_name);

	// If we could rely on C++14, we would  use std::make_unique
	auto h_A = std::unique_ptr<float[]>(new float[numElements]);
	auto h_B = std::unique_ptr<float[]>(new float[numElements]);
	auto h_C = std::unique_ptr<float[]>(new float[numElements]);

	auto generator = []() {
		static std::random_device random_device;
		static std::mt19937 randomness_generator { random_device() };
		static std::uniform_real_distribution<> distribution { 0.0, 1.0 };
		return distribution(randomness_generator);
	};
	std::generate(h_A.get(), h_A.get() + numElements, generator);
	std::generate(h_B.get(), h_B.get() + numElements, generator);

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
		if (std::fabs(h_A.get()[i] + h_B.get()[i] - h_C.get()[i]) > 1e-5)  {
			std::cerr << "Result verification failed at element " << i << "\n";
			exit(EXIT_FAILURE);
		}
	}

    std::cout << "Test PASSED\n";
    std::cout << "SUCCESS\n";
}


