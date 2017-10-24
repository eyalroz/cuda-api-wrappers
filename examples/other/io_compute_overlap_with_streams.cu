/**
 * An example program utilizing most/all calls from the CUDA
 * Runtime API module:
 *
 *   Stream Management
 *
 */
#include "cuda/api_wrappers.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <unistd.h>

using element_t = float;
constexpr size_t num_elements    = 1e5;
constexpr size_t num_kernels     = 5;
constexpr size_t num_repetitions = 3;

using clock_value_t = long long;

__device__ void gpu_sleep(clock_value_t sleep_cycles)
{
    clock_value_t start = clock64();
    clock_value_t cycles_elapsed;
    do { cycles_elapsed = clock64() - start; }
    while (cycles_elapsed < sleep_cycles);
}

template <typename T>
__global__ void add(
	const T* __restrict__  lhs,
	const T* __restrict__  rhs,
	T* __restrict__        result,
	size_t                 length)
{
	auto global_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (global_index < length) {
		result[global_index] = lhs[global_index] + rhs[global_index];
		gpu_sleep(50000);
	}
}

/*
 * Produce a launch configuration with one thread covering each element
 */
cuda::launch_configuration_t make_linear_launch_config(
	const cuda::device_t<>  device,
	size_t                  length)
{
	auto threads_per_block = device.properties().max_threads_per_block();
	cuda::grid_dimension_t num_blocks =
		(length / threads_per_block) +
		(length % threads_per_block == 0 ? 0 : 1);
	return cuda::make_launch_config(num_blocks, threads_per_block, cuda::no_shared_memory);
}

struct buffer_set_t {
	std::unique_ptr<element_t[]> host_lhs;
	std::unique_ptr<element_t[]> host_rhs;
	std::unique_ptr<element_t[]> host_result;
	cuda::memory::device::unique_ptr<element_t[]> device_lhs;
	cuda::memory::device::unique_ptr<element_t[]> device_rhs;
	cuda::memory::device::unique_ptr<element_t[]> device_result;
};

std::vector<buffer_set_t> generate_buffers(const cuda::device_t<>  device)
{
	// TODO: This should be an std::array, but generating
	// it is a bit tricky and I don't want to burden the example
	// with template wizardry
	std::vector<buffer_set_t> buffers;
	std::generate_n(std::back_inserter(buffers), num_kernels,
		[&]() {
			return buffer_set_t {
				// Sticking to C++11 here...
				std::unique_ptr<element_t[]>(new element_t[num_elements]),
				std::unique_ptr<element_t[]>(new element_t[num_elements]),
				std::unique_ptr<element_t[]>(new element_t[num_elements]),
				cuda::memory::device::make_unique<element_t[]>(device.id(), num_elements),
				cuda::memory::device::make_unique<element_t[]>(device.id(), num_elements),
				cuda::memory::device::make_unique<element_t[]>(device.id(), num_elements)
			};
		}
	);

	std::random_device random_device{};
	std::mt19937 random_engine{random_device()};
	std::uniform_real_distribution<float>distribution{0.0, 10.0};
	auto data_generator = [&]() { return (element_t) distribution(random_engine); };

	for(auto& buffer_set : buffers) {
		std::generate_n(buffer_set.host_lhs.get(), num_elements, data_generator);
		std::generate_n(buffer_set.host_rhs.get(), num_elements, data_generator);
	}
	return buffers;
}

int main(int argc, char **argv)
{
	auto device = cuda::device::current::get();
	std::cout << "Using CUDA device " << device.name() << " (having ID " << device.id() << ")\n";

	std::cout << "Generating host buffers... " << std::flush;
	std::vector<buffer_set_t> buffers = generate_buffers(device);
	std::cout << "done.\n" << std::flush;

	std::vector<cuda::stream_t<> > streams;
	streams.reserve(num_kernels);
	std::generate_n(std::back_inserter(streams), num_kernels,
		[&]() { return device.create_stream(cuda::stream::async); });

	auto common_launch_config = make_linear_launch_config(device, num_elements);
	auto buffer_size = num_elements * sizeof(element_t);

	auto work_can_start = cuda::event::create(device);

	for(size_t r = 0; r < num_repetitions; r++) {
		std::cout
			<< "Preparing another run of " << num_kernels << " kernels in parallel (run "
			<< r+1 << " of " << num_repetitions << ')' << std::endl;
		// Unfortunately, we need to use indices here - unless we
		// had access to a zip iterator (e.g. boost::zip_iterator)
		for(size_t k = 0; k < num_kernels; k++) {
			auto& stream = streams[k];
			auto& buffer_set = buffers[k];
			stream.enqueue.wait(work_can_start.id());
			std::cout << "Enqueueing actual work on stream " << k+1 << " of " << num_kernels << std::endl;
			stream.enqueue.copy(buffer_set.device_lhs.get(), buffer_set.host_lhs.get(), buffer_size);
			stream.enqueue.copy(buffer_set.device_rhs.get(), buffer_set.host_rhs.get(), buffer_size);
			stream.enqueue.kernel_launch(
				add<element_t>,
				common_launch_config,
				buffer_set.device_lhs.get(),
				buffer_set.device_rhs.get(),
				buffer_set.device_result.get(),
				num_elements);
			stream.enqueue.copy(buffer_set.host_result.get(), buffer_set.device_result.get(), buffer_size);
			stream.enqueue.callback(
				[k, r](cuda::stream::id_t stream_id, cuda::status_t status) {
					std::cout
						<< "Stream " << k+1 << " of " << num_kernels << " has concluded all work "
						<< "for run " << r+1 << " of " << num_repetitions << std::endl;
				}
			);
		}
		cuda::outstanding_error::ensure_none();
		usleep(25000);

		// We expect the streams to schedule _no_work_ until we tell them to. But were
		// they busy while we slept?

		auto num_streams_with_remaining_work =
			std::count_if(std::begin(streams), std::end(streams),
				[](const cuda::stream_t<>& stream) { return stream.has_work_remaining(); }
			);
		if (num_streams_with_remaining_work != num_kernels) {
			std::stringstream ss;
			std::cout
				<< "Some streams (" << num_kernels - num_streams_with_remaining_work
				<< " of " << num_kernels << ") have concluded their work before the "
				<< "event allowing any work to commence has fired; that should not "
				<< "have happened." << std::endl;

			std::cout << "\nFAILURE" << std::endl;
			exit(EXIT_FAILURE);
		}

		work_can_start.fire();

		std::cout << "Synchronizing all streams after having allowing them to start working." << std::endl;
		for(auto& stream : streams) { stream.synchronize(); }
	}


	// TODO: Check for correctness here

	std::cout << "\nSUCCESS" << std::endl;
}
