/**
 * An example program utilizing most/all calls
 * from the CUDA Runtime API module:
 *
 *   Execution control
 *
 * but excluding the part of this module dealing with parameter buffers:
 *
 *   cudaGetParameterBuffer
 *   cudaGetParameterBufferV2
 *   cudaSetDoubleForDevice
 *   cudaSetDoubleForHost
 *
 */
#include "cuda/api_wrappers.h"

#include <cuda_runtime_api.h>
#if __CUDACC_VER_MAJOR__ >= 9
#include <cooperative_groups.h>
#endif

#include <iostream>
#include <string>
#include <cassert>

[[noreturn]] void die_(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

__global__ void foo(int bar)
{
	if (threadIdx.x == 0) {
		printf("Block %u is executing (with v = %d)\n", blockIdx.x, bar);
	}
}

#ifdef _CG_HAS_GRID_GROUP
__global__ void grid_cooperating_foo(int bar)
{
	auto g = cooperative_groups::this_grid();
	g.sync();
	if (threadIdx.x == 0) {
		printf("Block %u is executing (with v = %d)\n", blockIdx.x, bar);
	}
}
#endif

std::ostream& operator<<(std::ostream& os, cuda::device::compute_capability_t cc)
{
	return os << cc.major << '.' << cc.minor;
}

int main(int argc, char **argv)
{
	if (cuda::device::count() == 0) {
		die_("No CUDA devices on this system");
	}

	const auto kernel = foo;
	const auto kernel_name = "foo"; // no reflection, sadly...

	// Being very cavalier about our command-line arguments here...
	cuda::device::id_t device_id =  (argc > 1) ?
		std::stoi(argv[1]) : cuda::device::default_device_id;

	if (cuda::device::count() <= device_id) {
		die_("No CUDA device with ID " + std::to_string(device_id));
	}

	auto device = cuda::device::get(device_id);
	std::cout << "Using CUDA device " << device.name() << " (having device ID " << device.id() << ")\n";
	cuda::device_function_t device_function(kernel);

	// ------------------------------------------
	//  Attributes without a specific API call
	// ------------------------------------------

	auto attributes = device_function.attributes();
	std::cout
		<< "The PTX version used in compiling device function " << kernel_name
		<< " is " << attributes.ptx_version() << ".\n";

	std::string cache_preference_names[] = {
		"No preference",
		"Equal L1 and shared memory",
		"Prefer shared memory over L1",
		"Prefer L1 over shared memory",
	};

	// --------------------------------------------------------------
	//  Attributes with a specific API call:
	//  L1/shared memory size preference and shared memory bank size
	// --------------------------------------------------------------

	device_function.cache_preference(
		cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory);

	device_function.shared_memory_bank_size(
		cuda::multiprocessor_shared_memory_bank_size_option_t::four_bytes_per_bank);

	// You may be wondering why we're only setting these "attributes' but not
	// obtaining their existing values. Well - we can't! The runtime doesn't expose
	// API calls for that (as of CUDA v8.0).

	// ------------------
	//  Kernel launching
	// ------------------

	const int bar = 123;
	const unsigned num_blocks = 3;
	auto launch_config = cuda::make_launch_config(num_blocks, attributes.maxThreadsPerBlock);
	cuda::device::current::set(device_id);
	std::cout
		<< "Launching kernel " << kernel_name
		<< " with " << num_blocks << " blocks, using cuda::launch()\n" << std::flush;
	cuda::launch(kernel, launch_config, bar);
	cuda::device::current::get().synchronize();

	// Let's do the same, but when the kernel is wrapped in a device_function_t
	std::cout
		<< "Launching kernel " << kernel_name
		<< " wrapped in a device_function_t strcture,"
		<< " with " << num_blocks << " blocks, using cuda::launch()\n" << std::flush;

	cuda::launch(device_function, launch_config, bar);
	cuda::device::current::get().synchronize();

	// But there's more than one way to launch! we can also do
	// it via the device proxy, using the default stream:

	std::cout
		<< "Launching kernel " << kernel_name
		<< " with " << num_blocks << " blocks, using device.launch()\n" << std::flush;
	device.launch(kernel, launch_config, bar);
	device.synchronize();

	// or via a stream:

	auto stream = cuda::device::current::get().create_stream(
		cuda::stream::no_implicit_synchronization_with_default_stream);

	std::cout
		<< "Launching kernel " << kernel_name
		<< " with " << num_blocks << " blocks, using stream.launch()\n" << std::flush;
	stream.enqueue.kernel_launch(kernel, launch_config, bar);
	stream.synchronize();

#ifdef _CG_HAS_GRID_GROUP
	// And finally, some "cooperative" vs ""uncooperative"  kernel launches
	if (cuda::version_numbers::runtime() >= 9) {
		const auto kernel = grid_cooperating_foo;
		std::cout
			<< "Launching kernel " << kernel_name
			<< " with " << num_blocks << " blocks, cooperatively, using stream.launch()\n"
			<< "(but note it does not actually check the cooperativeness)." << std::flush;
		stream.enqueue.kernel_launch(cuda::thread_blocks_may_cooperate, kernel, launch_config, bar);
		stream.synchronize();

		std::cout
			<< "Launching kernel " << kernel_name
			<< " with " << num_blocks << " blocks, un-cooperatively, using stream.launch()\n" << std::flush;
		stream.enqueue.kernel_launch(cuda::thread_blocks_cant_cooperate, kernel, launch_config, bar);
		stream.synchronize();
	}
#endif

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
