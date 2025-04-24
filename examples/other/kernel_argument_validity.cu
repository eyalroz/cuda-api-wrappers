/**
 * @file
 *
 * @brief An example program demonstrating the effects and use of the @ref
 * cuda::is_valid_kernel_argument trait
 */
#include "../common.hpp"
#include <cuda/api/kernel_launch.hpp>

using std::printf;

class NonTriviallyCopyable {
public:
	NonTriviallyCopyable() = default;
	NonTriviallyCopyable(const char *name) {
		printf("NonTriviallyCopyable constructor called\n");
		std::strncpy(name_, name, sizeof(name_));
		name_[sizeof(name_) - 1] = '\0';
	}

	NonTriviallyCopyable(NonTriviallyCopyable &&) = default;

	NonTriviallyCopyable(const NonTriviallyCopyable &other) {
		printf("NonTriviallyCopyable copy constructor called\n");
		std::strncpy(name_, other.name_, sizeof(name_));
		name_[sizeof(name_) - 1] = '\0';
	}

	~NonTriviallyCopyable() {
		printf("NonTriviallyCopyable destructor called\n");
	}

	__host__ __device__ NonTriviallyCopyable &
	operator=(const NonTriviallyCopyable &) = default;
	NonTriviallyCopyable &operator=(NonTriviallyCopyable &&) = default;
	// Non-trivial destructor

	__host__ __device__ const char *name() const noexcept { return name_; }

private:
	char name_[10];
};
static_assert(
	not std::is_trivially_copyable<NonTriviallyCopyable>::value,
	"NonTriviallyCopyable should not be trivially copyable");

template <>
struct cuda::is_valid_kernel_argument<NonTriviallyCopyable> : ::std::true_type {};

__global__ void print_message(NonTriviallyCopyable message) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		printf("Kernel says: %s\n", (const char *)message.name());
	}
}

__global__ void copy_me(NonTriviallyCopyable in, NonTriviallyCopyable *out) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*out = in;
		printf("Kernel copied me: %s\n", (const char *)out->name());
	}
}

int main(int argc, char **argv) {

	auto device = [&]() {
		// Being very cavalier about our command-line arguments here...
		cuda::device::id_t device_id = (argc > 1) ? std::stoi(argv[1]) : cuda::device::default_device_id;

		if (cuda::device::count() == 0) {
			die_("No CUDA devices on this system");
		}
		return cuda::device::get(device_id).make_current();
	}();

	std::cout << "Using CUDA device " << device.name() << " (having device ID " << device.id() << ")\n";

	auto stream = cuda::device::current::get().create_stream(cuda::stream::sync);
	cuda::launch_configuration_t single_thread_config{1, 1};

	std::cout
		<< "A new CUDA stream with no priority specified defaults to having priority " << stream.priority() << ".\n";
	stream.enqueue.kernel_launch(print_message, single_thread_config, NonTriviallyCopyable("It works!"));

	auto dest_mem = cuda::memory::device::make_unique_region(device, sizeof(NonTriviallyCopyable));
	auto dest_span = dest_mem.as_span<NonTriviallyCopyable>();

	NonTriviallyCopyable src("Source");
	NonTriviallyCopyable dest;
	stream.enqueue.kernel_launch(copy_me, single_thread_config, src, dest_span.data());
	stream.enqueue.copy(&dest, dest_span);
	stream.synchronize();
	assert(strcmp(dest.name(), src.name()) == 0);
	printf("Host says: %s\n", (const char *)dest.name());

	std::cout << "\nSUCCESS\n";
}
