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

#include <cuda/api_wrappers.hpp>
#include "ptx.cuh"

#include <iostream>
#include <memory>

[[noreturn]] void die_(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}


__global__ void sequence_gpu(int *d_ptr, int length)
{
	int elemID = blockIdx.x * blockDim.x + threadIdx.x;

	if (elemID < length)
	{
		d_ptr[elemID] = ptx::special_registers::laneid();
	}
}

void sequence_cpu(int *h_ptr, int length)
{
	for (int elemID=0; elemID<length; elemID++)
	{
		h_ptr[elemID] = elemID % cuda::warp_size;
	}
}

int main(int argc, char **argv)
{
	if (cuda::device::count() == 0) {
		die_("No CUDA devices on this system");
	}


	const int N = 1000;

	cuda::device::current::set_to_default();
	auto current_device = cuda::device::current::get();

	auto d_ptr = cuda::memory::device::make_unique<int[]>(current_device, N);
	auto h_ptr = cuda::memory::host::make_unique<int[]>(N);

	std::cout << "Generating data on CPU\n";

	sequence_cpu(h_ptr.get(), N);

	cuda::grid::block_dimensions_t cudaBlockSize(256,1,1);
	cuda::grid::dimensions_t cudaGridSize((N + cudaBlockSize.x - 1) / cudaBlockSize.x, 1, 1);
	current_device.launch(
		sequence_gpu,
		{ cudaGridSize, cudaBlockSize },
		d_ptr.get(), N
	);

	cuda::outstanding_error::ensure_none();
	current_device.synchronize();

	auto h_d_ptr = cuda::memory::host::make_unique<int[]>(N);
	cuda::memory::copy(h_d_ptr.get(), d_ptr.get(), N * sizeof(int));

	bool bValid = true;

	for (int i=0; i<N && bValid; i++)
	{
		if (h_ptr.get()[i] != h_d_ptr.get()[i])
		{
			bValid = false;
		}
	}

	std::cout << (bValid ? "SUCCESS" : "FAILURE") << "\n";
	return bValid ? EXIT_SUCCESS: EXIT_FAILURE;
}
