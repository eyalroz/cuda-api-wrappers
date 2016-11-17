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

/*
[[noreturn]] void die(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}
*/

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
__global__ void print(poor_mans_array<char, N> message)
{
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("Kernel no. %u says: %s\n", Index, (const char*) message);
	}
}


int main(int argc, char **argv)
{
	static constexpr size_t N = 40;

	// Being very cavalier about our command-line arguments here...
	cuda::device::id_t device_id =  (argc > 1) ?
		std::stoi(argv[1]) : cuda::device::default_device_id;

	cuda::device::current::set(device_id);
	auto device = cuda::device::current::get();

	std::cout << "Working with CUDA device " << device.name() << " (having ID " << device.id() << ")\n";

	{
		auto stream = cuda::device::current::get().create_stream();
		std::cout
			<< "A new CUDA stream with no options specified defaults to having priority "
			<< stream.priority() << " and synchronizes by "
			<< (stream.synchronizes_with_default_stream() ? "blocking" : "busy-waiting")
			<< ".\n";
		stream.enqueue_launch(print<N,1>, { 1, 1 }, message<N>("I can see my house!"));
		stream.synchronize();
	}

	auto stream_1 = cuda::device::current::get().create_stream(
		cuda::stream::default_priority + 1,
		cuda::stream::no_implicit_synchronization_with_default_stream);
	auto stream_2 = cuda::device::current::get().create_stream(
		cuda::stream::default_priority + 1,
		cuda::stream::no_implicit_synchronization_with_default_stream);


	cuda::event_t event_1(cuda::event::sync_by_blocking);
	stream_1.enqueue_launch(print<N,2>, { 1, 1 }, message<N>("I'm on stream 1"));
	event_1.record(stream_1.id());
	stream_1.enqueue_launch(print<N,3>, { 1, 1 }, message<N>("I'm on stream 1"));

	stream_2.wait_on(event_1.id());
	stream_2.enqueue_launch(print<N,4>, { 1, 1 }, message<N>("I'm on stream 2"));

	bool idleness_1 = stream_2.has_work();

	device.synchronize();

	bool idleness_2 = stream_2.has_work();

	std::cout << std::boolalpha
		<< "Did stream 2 have work before device-level synchronization? " << (idleness_1 ? "yes" : "no") << "\n"
		<< "Did stream 2 have work after  device-level synchronization? " << (idleness_2 ? "yes" : "no") << "\n";

	// Remain to be tested: add_callback, enqueue_memory_attachment

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
