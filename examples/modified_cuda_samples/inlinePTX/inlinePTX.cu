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

#include "ptx.cuh"

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

int main(int, char **)
{
	if (cuda::device::count() == 0) {
		die_("No CUDA devices on this system");
	}

	const int N = 1000;

	cuda::device::current::set_to_default();
	auto device = cuda::device::current::get();

	auto d_ptr = cuda::memory::device::make_unique<int[]>(device, N);
	auto h_ptr = cuda::memory::host::make_unique<int[]>(N);

	std::cout << "Generating data on CPU\n";

	sequence_cpu(h_ptr.get(), N);

	auto block_size = 256;
	auto grid_size = div_rounding_up(N, block_size);
	auto launch_config = cuda::make_launch_config(grid_size, block_size);
	device.launch(sequence_gpu, launch_config, d_ptr.get(), N);

	cuda::outstanding_error::ensure_none();
	device.synchronize();

	auto h_d_ptr = cuda::memory::host::make_unique<int[]>(N);
	cuda::memory::copy(h_d_ptr.get(), d_ptr.get(), N * sizeof(int));

	auto results_are_correct =	std::equal(h_ptr.get(), h_ptr.get() + N, h_d_ptr.get());
	if (not results_are_correct) {
		die_("Results check failed.");
	}
	std::cout << "SUCCESS\n";
}
