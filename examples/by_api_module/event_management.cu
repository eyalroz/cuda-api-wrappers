/**
 * An example program utilizing most/all calls from the CUDA
 * Runtime API module:
 *
 *   Event Management
 *
 */
#include "../common.hpp"

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

__global__ void increment(char* data, size_t length)
{
	size_t global_index = threadIdx.x + blockIdx.x * blockDim.x;
	if (global_index < length)
	data[global_index]++;
}

inline void report_occurrence(
	const std::string& prefix_message,
	const cuda_::event_t& e1,
	const cuda_::event_t& e2)
{
	std::cout
		<< prefix_message << ": "
		<< "Event 1 has " << (e1.has_occurred() ? "" : "not ") << "occurred; "
		<< "event 2 has " << (e2.has_occurred() ? "" : "not ") << "occurred.\n";

}

int main(int argc, char **argv)
{
	if (cuda_::device::count() == 0) {
		die_("No CUDA devices on this system");
	}

	static constexpr size_t N = 40;

	// Being very cavalier about our command-line arguments here...
	auto device_id =  (argc > 1) ? std::stoi(argv[1]) : cuda_::device::default_device_id;
	auto device = cuda_::device::get(device_id);

	std::cout << "Working with CUDA device " << device.name() << " (having ID " << device.id() << ")\n";

	// Everything else - Enqueueing kernels, events, callbacks
	// and memory attachments, recording and waiting on events
	//--------------------------------------------------------------

	auto stream = device.create_stream(cuda_::stream::async);

	{ auto event = cuda_::event::create(device); }
	auto event_1 = cuda_::event::create(
		device,
		cuda_::event::sync_by_blocking,
		cuda_::event::do_record_timings,
		cuda_::event::not_interprocess);
	auto event_2 = cuda_::event::create(
		device,
		cuda_::event::sync_by_blocking,
		cuda_::event::do_record_timings,
		cuda_::event::not_interprocess);
	auto event_3 = device.create_event(
		cuda_::event::sync_by_blocking,
		cuda_::event::do_record_timings,
		cuda_::event::not_interprocess);

	auto buffer = cuda_::memory::managed::make_unique_span<char>(
		device, 12345678, cuda_::memory::managed::initial_visibility_t::to_all_devices);
	auto wrapped_kernel = cuda_::kernel::get(device, increment);
	auto launch_config = cuda_::launch_config_builder()
		.kernel(&wrapped_kernel)
		.overall_size(buffer.size())
		.use_maximum_linear_block()
		.build();

	stream.enqueue.kernel_launch(print_message<N,1>, { 1, 1 }, message<N>("I am launched before the first event"));
	stream.enqueue.event(event_1);
	auto first_callback = [&] {
		report_occurrence("In first callback (enqueued after first event but before first kernel)", event_1, event_2);
	};
	stream.enqueue.host_invokable(first_callback);
	stream.enqueue.kernel_launch(increment, launch_config, buffer.data(), buffer.size());
	auto second_callback = [&] {
		report_occurrence("In second callback (enqueued after the first kernel but before the second event)",
			event_1, event_2);
	};
	stream.enqueue.host_invokable(second_callback);
	stream.enqueue.event(event_2);
	stream.enqueue.kernel_launch(print_message<N,3>, { 1, 1 }, message<N>("I am launched after the second event"));
	stream.enqueue.event(event_3);
	stream.enqueue.kernel_launch(print_message<N,4>, { 1, 1 }, message<N>("I am launched after the third event"));

	try {
		cuda_::event::time_elapsed_between(event_1, event_2);
		std::cerr << "Attempting to obtain the elapsed time between two events on a"
			"stream which does not auto-sync with the default stream and has not been "
			"synchronized should fail - but it didn't\n";
		exit(EXIT_FAILURE);

	} catch(cuda_::runtime_error& e) {
		(void) e; // This avoids a spurious warning in MSVC 16.11
		assert(e.code() == cuda_::status::async_dependency_ops_not_yet_completed);
	}
	event_2.synchronize();
	report_occurrence("After synchronizing on event_2, but before synchronizing on the stream", event_1, event_2);
	std::cout
		<< cuda_::event::time_elapsed_between(event_1, event_2).count() << " msec have elapsed, "
		<< "executing the second kernel (\"increment\") on a buffer of " << buffer.size()
		<< " chars and triggering two callbacks.\n";
	// ... and this should make the third kernel execute
	stream.synchronize();

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
