/**
 * An example program utilizing most/all calls from the CUDA
 * Runtime API module:
 *
 *   Stream Management
 *
 */
#include "cuda/api_wrappers.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>

using std::printf;

template <typename T, size_t N>
struct poor_mans_array {
	T data[N];
	__host__ __device__ operator T*() { return data; }
	__host__ __device__ operator const T*() const { return data; }
	__host__ __device__ T& operator [](off_t offset) { return data[offset]; }
	__host__ __device__ const T& operator [](off_t offset) const { return data[offset]; }
};

template <size_t N>
poor_mans_array<char, N> message(const char* message_str)
{
	poor_mans_array<char, N> a;
	assert(std::strlen(message_str) < N);
	std::strcpy(a.data, message_str);
	return a;
}

template <size_t N>
poor_mans_array<char, N> message(const std::string& message_str)
{
	return message<N>(message_str.c_str());
}

template <size_t N, unsigned Index>
__global__ void print_message(poor_mans_array<char, N> message)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("Kernel no. %u says: %s\n", Index, (const char*) message);
	}
}

__host__ __device__ void print_first_char(const char* __restrict__ data)
{
	printf("data[0] = '%c' (0x%02x)\n", data[0], (unsigned) data[0]);
}

__global__ void print_first_char_kernel(const char* __restrict__ data)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		print_first_char(data);
	}
}

__global__ void increment(char* data, size_t length)
{
	size_t global_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (global_index < length)
	data[global_index]++;
}

int main(int argc, char **argv)
{
	static constexpr size_t N = 40;

	// Being very cavalier about our command-line arguments here...
	cuda::device::id_t device_id =  (argc > 1) ?
		std::stoi(argv[1]) : cuda::device::default_device_id;

	cuda::device::current::set(device_id);
	auto device = cuda::device::current::get();

	std::cout << "Using CUDA device " << device.name() << " (having device ID " << device.id() << ")\n";

	// Stream creation and destruction, stream flags
	//------------------------------------------------
	{
		auto stream = cuda::device::current::get().create_stream();
		std::cout
			<< "A new CUDA stream with no options specified defaults to having priority "
			<< stream.priority() << " and synchronizes by "
			<< (stream.synchronizes_with_default_stream() ? "blocking" : "busy-waiting")
			<< ".\n";
		stream.enqueue.kernel_launch(print_message<N,1>, { 1, 1 }, message<N>("I can see my house!"));
		stream.synchronize();
	}

	// Everything else - Enqueueing kernels, events, callbacks
	// and memory attachments, recording and waiting on events
	//--------------------------------------------------------------

	auto stream_1 = cuda::device::current::get().create_stream(
		cuda::stream::default_priority + 1,
		cuda::stream::no_implicit_synchronization_with_default_stream);
	auto stream_2 = cuda::device::current::get().create_stream(
		cuda::stream::default_priority + 1,
		cuda::stream::no_implicit_synchronization_with_default_stream);


	constexpr auto buffer_size = 12345678;
	auto buffer = cuda::memory::managed::make_unique<char[]>(
		buffer_size,
		device.supports_concurrent_managed_access() ?
			cuda::memory::managed::initial_visibility_t::to_supporters_of_concurrent_managed_access:
			cuda::memory::managed::initial_visibility_t::to_all_devices);
	print_first_char(buffer.get());
	std::fill(buffer.get(), buffer.get() + buffer_size, 'a');
	print_first_char(buffer.get());

	auto event_1 = cuda::event::make(cuda::event::sync_by_blocking);
	stream_1.enqueue.kernel_launch(print_message<N,2>, { 1, 1 }, message<N>("I'm on stream 1"));
	stream_1.enqueue.memset(buffer.get(), 'b', buffer_size);
	stream_1.enqueue.callback(
		[&buffer](cuda::stream::id_t stream_id, cuda::status_t status) {
			std::cout << "Callback from stream 1!... ";
			print_first_char(buffer.get());
		}
	);
	auto threads_per_block = cuda::device_function_t(increment).attributes().maxThreadsPerBlock;
	auto num_blocks = (buffer_size + threads_per_block - 1) / threads_per_block;
	auto launch_config = cuda::make_launch_config(num_blocks, threads_per_block);
	// TODO: The following doesn't have much of a meaningful effect; we should modify this example
	// so that the attachment has some observable effect
	stream_1.enqueue.memory_attachment(buffer.get());
	stream_1.enqueue.kernel_launch(increment, launch_config, buffer.get(), buffer_size);
	event_1.record(stream_1.id());
	stream_1.enqueue.kernel_launch(print_message<N,4>, { 1, 1 }, message<N>("I'm on stream 1"));
	stream_2.wait_on(event_1.id());
	stream_2.enqueue.kernel_launch(print_first_char_kernel, launch_config , buffer.get());
	stream_2.enqueue.kernel_launch(print_message<N,5>, { 1, 1 }, message<N>("I'm on stream 2"));
	bool idleness_1 = stream_2.has_work();
	device.synchronize();
	print_first_char(buffer.get());
	// cuda::memory::managed::free(buffer);
	bool idleness_2 = stream_2.has_work();
	std::cout << std::boolalpha
		<< "Did stream 2 have work before device-level synchronization? " << (idleness_1 ? "yes" : "no") << "\n"
		<< "Did stream 2 have work after  device-level synchronization? " << (idleness_2 ? "yes" : "no") << "\n";
	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
