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

	auto d_span = cuda::memory::make_unique_span<int>(device, N);
	auto h_span = cuda::memory::host::make_unique_span<int>(N);

	std::cout << "Generating data on CPU\n";

	sequence_cpu(h_span.data(), h_span.size());

	auto launch_config = cuda::launch_config_builder()
		.overall_size(N)
		.block_size(256)
		.build();
	device.launch(sequence_gpu, launch_config, d_span.data(), (int) d_span.size());

	cuda::outstanding_error::ensure_none();
	device.synchronize();

	auto h_d_span = cuda::memory::host::make_unique_span<int>(N);
	cuda::memory::copy(h_d_span, d_span);

	auto results_are_correct =	std::equal(h_span.begin(), h_span.end(), h_d_span.begin());
	if (not results_are_correct) {
		die_("Results check failed.");
	}
	std::cout << "SUCCESS\n";
}
