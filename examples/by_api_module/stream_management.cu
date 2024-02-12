/**
 * An example program utilizing most/all calls from the CUDA
 * Runtime API module:
 *
 *   Stream Management
 *
 */
#include "../common.hpp"

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

#if CUDA_VERSION >= 11000
const char* get_policy_name(cuda::stream::synchronization_policy_t policy)
{
	switch(policy) {
		case cuda::stream::automatic: return "automatic";
		case cuda::stream::spin: return "spin";
		case cuda::stream::yield: return "yield";
		case cuda::stream::block: return "block";
		default:
			return "unknown policy";
	}
}
#endif // CUDA_VERSION >= 11000

int main(int argc, char **argv)
{
	constexpr const size_t N = 50;
	cuda::launch_configuration_t single_thread_config { 1, 1 };

	// Being very cavalier about our command-line arguments here...
	cuda::device::id_t device_id =  (argc > 1) ?
		std::stoi(argv[1]) : cuda::device::default_device_id;

	if (cuda::device::count() == 0) {
		die_("No CUDA devices on this system");
	}

	auto device = cuda::device::get(device_id).make_current();

	std::cout << "Using CUDA device " << device.name() << " (having device ID " << device.id() << ")\n";

	// Stream creation and destruction, stream flags
	//------------------------------------------------
	{
		auto stream = cuda::device::current::get().create_stream(cuda::stream::sync);
		std::cout
			<< "A new CUDA stream with no priority specified defaults to having priority "
			<< stream.priority() << ".\n";
		stream.enqueue.kernel_launch(print_message<N,1>, single_thread_config, message<N>("I can see my house!"));
		stream.synchronize();
	}

	// Use of the default stream as an rvalue
	// ---------------------------------------------
	cuda::enqueue_launch(
		print_message<N,2>, device.default_stream(), single_thread_config, 
		message<N>("I was launched on the default stream.")
	);

	// Everything else - Enqueueing kernel or host-function launches, events
	// and memory attachments, recording and waiting on events
	//--------------------------------------------------------------

	auto stream_1 = cuda::device::current::get().create_stream(
		cuda::stream::default_priority + 1,
		cuda::stream::no_implicit_synchronization_with_default_stream);
	auto stream_2 = cuda::device::current::get().create_stream(
		cuda::stream::default_priority + 1,
		cuda::stream::no_implicit_synchronization_with_default_stream);

#if CUDA_VERSION >= 11000
	// Stream synchronization policy and attribute copying

	auto initial_policy = stream_1.synchronization_policy();
	std::cout
		<< "Initial stream synchronization policy is "
		<< get_policy_name(initial_policy) << " (numeric value: " << (int) initial_policy << ")\n";
	if (initial_policy != stream_2.synchronization_policy()) {
		throw std::logic_error("Different synchronization policies for streams created the same way");
	}
	cuda::stream::synchronization_policy_t alt_policy =
		(initial_policy == cuda::stream::yield) ? cuda::stream::block : cuda::stream::yield;
	stream_2.set_synchronization_policy(alt_policy);
	auto new_s2_policy = stream_2.synchronization_policy();
	if (alt_policy != new_s2_policy) {
		std::stringstream ss;
		ss
			<< "Got a different synchronization policy (" << get_policy_name(new_s2_policy) << ")"
			<< " than the one we set the stream to (" << get_policy_name(alt_policy) << ")\n";
		throw std::logic_error(ss.str());
	}
	std::cout << "Overwriting all attributes of stream 1 with those of stream 2.\n";
	cuda::copy_attributes(stream_1, stream_2);
	auto s1_policy_after_copy = stream_1.synchronization_policy();
	if (alt_policy != s1_policy_after_copy) {
		std::stringstream ss;
		ss
			<< "Got a different synchronization policy (" << get_policy_name(s1_policy_after_copy) << ")"
			<< " than the one we expected after attribute-copying (" << get_policy_name(alt_policy) << ")\n";
		throw std::logic_error(ss.str());
	}
#endif

	constexpr auto buffer_size = 12345678;
	auto buffer = cuda::memory::managed::make_unique_span<char>(
		buffer_size,
		device.supports_concurrent_managed_access() ?
			cuda::memory::managed::initial_visibility_t::to_supporters_of_concurrent_managed_access:
			cuda::memory::managed::initial_visibility_t::to_all_devices);
	print_first_char(buffer.data());
	std::fill(buffer.begin(), buffer.end(), 'a');
	print_first_char(buffer.data());

	auto event_1 = cuda::event::create(device, cuda::event::sync_by_blocking);
	stream_1.enqueue.kernel_launch(print_message<N,3>, single_thread_config, message<N>("I'm on stream 1"));
	stream_1.enqueue.memset(buffer, 'b');
	auto callback = [&]() {
		std::cout << "Callback from stream 1!... \n";
		print_first_char(buffer.data());
	};
	stream_1.enqueue.host_invokable(callback);
	auto threads_per_block = cuda::kernel::get(device, increment).get_attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
	auto num_blocks = div_rounding_up(buffer_size, threads_per_block);
	auto launch_config = cuda::launch_configuration_t{num_blocks, threads_per_block};
	// TODO: The following doesn't have much of a meaningful effect; we should modify this example
	// so that the attachment has some observable effect
	stream_1.enqueue.attach_managed_region(buffer.get());
	stream_1.enqueue.kernel_launch(increment, launch_config, buffer.data(), buffer_size);
	event_1.record(stream_1);
	stream_1.enqueue.kernel_launch(print_message<N,4>, single_thread_config, message<N>("I'm on stream 1"));
	stream_2.enqueue.wait(event_1);
	stream_2.enqueue.kernel_launch(print_first_char_kernel, launch_config , buffer.data());
	stream_2.enqueue.kernel_launch(print_message<N,5>, single_thread_config, message<N>("I'm on stream 2"));
	bool idleness_1 = stream_2.has_work_remaining();
	device.synchronize();
	print_first_char(buffer.data());
	// cuda::memory::managed::free(buffer);
	bool idleness_2 = stream_2.has_work_remaining();
	std::cout << std::boolalpha
		<< "Did stream 2 have work before device-level synchronization? " << (idleness_1 ? "yes" : "no") << "\n"
		<< "Did stream 2 have work after  device-level synchronization? " << (idleness_2 ? "yes" : "no") << "\n";
	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
