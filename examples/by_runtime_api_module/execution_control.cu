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
#include <cuda/api_wrappers.hpp>

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

#if __CUDACC_VER_MAJOR__ >= 9
__global__ void grid_cooperating_foo(int bar)
{
#ifdef _CG_HAS_GRID_GROUP
	auto g = cooperative_groups::this_grid();
#else
	auto g = cooperative_groups::this_thread_block();
#endif
	g.sync();

	if (threadIdx.x == 0) {
		printf("Block %u is executing (with v = %d)\n", blockIdx.x, bar);
	}
}
#endif

std::ostream& operator<<(std::ostream& os, cuda::device::compute_capability_t cc)
{
	return os << cc.major() << '.' << cc.minor();
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

	auto device = cuda::device::get(device_id).make_current();
	std::cout << "Using CUDA device " << device.name() << " (having device ID " << device.id() << ")\n";
	cuda::device_function_t device_function(device, kernel);

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

	device_function.set_cache_preference(
		cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory);

	device_function.set_shared_memory_bank_size(
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

#if __CUDACC_VER_MAJOR__ >= 9
	try {
		auto kernel = grid_cooperating_foo;
		auto kernel_name = "grid_cooperating_foo";
#else
	{
#endif
		// And finally, some "cooperative" vs ""uncooperative"  kernel launches:

		auto can_launch_cooperatively =
#if __CUDACC_VER_MAJOR__ >= 9
			(cuda::device::current::get().get_attribute(cudaDevAttrCooperativeLaunch) > 0);
#else
			false; // This is not strictly true, since the device might support it, but
			       // 1. We can't check and
			       // 2. We don't have a cooperative launch API before CUDA 9
			       // so "false" is good enough.
#endif
		if (can_launch_cooperatively) {
			std::cout
				<< "Launching kernel" << kernel_name
				<< " with " << num_blocks << " blocks, cooperatively, using stream.launch()\n"
				<< "(but note this does not actually check that cooperation takes place).\n" << std::flush;
			stream.enqueue.kernel_launch(cuda::thread_blocks_may_cooperate, kernel, launch_config, bar);
			stream.synchronize();

/*			// same, but with a device_function_t
			std::cout
				<< "Launching kernel" << kernel_name
				<< " with " << num_blocks << " blocks, cooperatively, using stream.launch()\n"
				<< "(but note it does not actually check the cooperativeness).\n" << std::flush;
			stream.enqueue.kernel_launch(cuda::thread_blocks_may_cooperate, device_function, launch_config, bar);
			stream.synchronize();
*/
			// Same, but using cuda::enqueue_launch
			// both options
			std::cout
				<< "Launching kernel " << kernel_name
				<< " wrapped in a device_function_t strcture,"
				<< " with " << num_blocks << " blocks, using cuda::enqueue_launch(),"
				<< " and allowing thread block cooperation\n"
				<< "(but note this does not actually check that cooperation takes place).\n" << std::flush;

			cuda::enqueue_launch((bool) cuda::thread_blocks_may_cooperate, device_function, stream, launch_config, bar);
			cuda::device::current::get().synchronize();

		}
		else {
			std::cout
				<< "Skipping launch of kernel" << kernel_name
				<< ", since our CUDA device doesn't support cooperative launches.\n" << std::flush;
		}

	}
#if __CUDACC_VER_MAJOR__ >= 9
	catch(cuda::runtime_error& e) {
		if (not (e.code() == cuda::status::not_supported)) {
			throw e;
		}
		cuda::outstanding_error::clear();
	}
#endif

	std::cout
		<< "Launching kernel " << kernel_name
		<< " with " << num_blocks << " blocks, un-cooperatively, using stream.launch()\n" << std::flush;
	stream.enqueue.kernel_launch(cuda::thread_blocks_may_not_cooperate, kernel, launch_config, bar);
	stream.synchronize();

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
