/**
 * @brief A program running a simple elementwise computation kernel,
 * but splitting the data into multiple chunks, each processed on
 * a separate stream. Intended to demonstrate simple multi-stream
 * scheduling behavior on NVIDIA GPUs (through the use of a profiler
 * such as NSight-System), as well as demonstrating the use of some
 * new(ish) C++20 features: full-fledged spans (with subspans) and
 * templated lambdas with auto parameters.
 *
 * @note This is a rewrite of a program appearing in the Stackoverflow
 * question: https://stackoverflow.com/q/79675777/1593077 , so it may
 * or may not be subject to the terms of the CC-AT-SA 4.0 license,
 * available at @url https://creativecommons.org/licenses/by-sa/4.0/
 *
 * @todo Perhaps check the results, to ensure everything is close to 1?
 */
#include <cuda/api.hpp>
#include <cstring>
#include <iostream>

__global__ void kernel(float *a)
{
	size_t i = threadIdx.x + blockIdx.x*blockDim.x;
	float x = static_cast<float>(i);
	float s = sinf(x);
	float c = cosf(x);
	a[i] += sqrtf(s*s + c*c);
}

int main(int argc, char* argv[])
{
	size_t launch_grid_block_size = 256;
	size_t chunk_size = 10 * launch_grid_block_size * 1024;
	size_t num_chunks = 8;
	auto overall_elements = num_chunks * chunk_size;

	auto host_data = cuda_::memory::host::make_unique_span<float>(overall_elements);
	// Fill with zeros as dummy data
	std::memset(host_data.data(), 0, host_data.size());

	auto device_id = (argc > 1) ? std::atoi(argv[1]) : 0;
	auto device = cuda_::device::get(device_id);
	std::cout << "Device: " << device.name() << '\n';

	auto device_data = cuda_::make_unique_span<float>(device, overall_elements);

	auto launch_config = cuda_::launch_config_builder()
		.overall_size(chunk_size)
		.block_size(launch_grid_block_size)
		.build();

	auto streams = cuda_::generate_unique_span(
		num_chunks, [&](size_t) { return device.create_stream(); });

	for (size_t i = 0; i < num_chunks; ++i) {
		static const auto get_chunk = [&](auto const& span_, size_t) { return span_.subspan(i * chunk_size, chunk_size); };
		auto host_chunk = get_chunk(host_data.get(), i);
		auto device_chunk = get_chunk(device_data.get(), i);
		auto const& stream = streams[i];
		stream.enqueue.copy(device_chunk, host_chunk);
		stream.enqueue.kernel_launch(kernel, launch_config, device_data.data());
		stream.enqueue.copy(host_chunk, device_chunk);
	}
	device.synchronize();
	std::cout << "SUCCESS";
}

