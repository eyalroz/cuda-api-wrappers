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
#include "../common.hpp"

#ifdef USE_COOPERATIVE_GROUPS
#if __CUDACC_VER_MAJOR__ < 9
#error "Can't use the cooperative groups header with CUDA versions before 9.x!"
#endif
#include <cooperative_groups.h>
#endif

__global__ void foo(int bar)
{
	if (threadIdx.x == 0) {
		printf("Block %u is executing (with v = %d)\n", blockIdx.x, bar);
	}
}

#if USE_COOPERATIVE_GROUPS
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

int main(int argc, char **argv)
{
	auto device_id = choose_device(argc, argv);
	auto device = cuda::device::get(device_id).make_current();

	const auto kernel_function = foo;
	const auto kernel_name = "foo"; // no reflection, sadly...
#if CUDA_VERSION >= 12030
	const auto full_kernel_name = "foo(int)"; // no reflection, sadly...
#endif

	auto kernel = cuda::kernel::get(device, kernel_function);

#if CUDA_VERSION >= 12030
	cuda::kernel_t const& base_ref = kernel;
	auto mangled_name_from_kernel = base_ref.mangled_name();
#ifdef __GNUC__
	auto demangled_name = demangle(mangled_name_from_kernel);
	if (strcmp(demangled_name.c_str(), full_kernel_name) != 0) {
		std::cerr
			<< "CUDA reports a different name for kernel \"" << kernel_name
			<< "\" via its handle: \"" << demangled_name << "\"\n"
			<< "FAILURE\n";
		return EXIT_FAILURE;
	}
#endif
#endif

	// ------------------------------------------
	//  Attributes without a specific API call
	// ------------------------------------------

	std::cout
		<< "The PTX version used in compiling device function \"" << kernel_name
		<< "\" is " << kernel.ptx_version() << ".\n";

	// --------------------------------------------------------------
	//  Attributes with a specific API call:
	//  L1/shared memory size preference and shared memory bank size
	// --------------------------------------------------------------

	kernel.set_cache_preference(
		cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory);

#if CUDA_VERSION < 12030
	kernel.set_shared_memory_bank_size(
		cuda::multiprocessor_shared_memory_bank_size_option_t::four_bytes_per_bank);
#endif // CUDA_VERSION < 12030

	// ------------------
	//  Kernel launching
	// ------------------

	const int bar = 123;
	const unsigned num_blocks = 3;
	std::cout << "Getting kernel attribute CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK" << std::endl;
	auto max_threads_per_block = kernel.get_attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
	auto launch_config = cuda::launch_configuration_t(num_blocks, max_threads_per_block);
	std::cout
		<< "Launching kernel " << kernel_name
		<< " with " << num_blocks << " blocks, using cuda::launch()" << std::endl;
	{
		// Copy and move construction and assignment of launch configurations
		auto launch_config_2 = cuda::launch_configuration_t{2, 2};
		auto launch_config_3 = cuda::launch_configuration_t{3, 3};
		cuda::launch_configuration_t launch_config_4{launch_config};
		(void) launch_config_4;
		launch_config_4 = launch_config_2;
		launch_config_4 = std::move(launch_config_3);
		cuda::launch_configuration_t launch_config_5{std::move(launch_config_2)};
		(void) launch_config_4;
		(void) launch_config_5;
		// In case the `[[maybe_unused]]` attribute and the void-casting is ignored,
		// let's try to trick the compiler
    	// into thinking we're actually using launch_config_4.
    	launch_config_4.dimensions == launch_config.dimensions;
	}

	cuda::launch(kernel_function, launch_config, bar);
	cuda::device::current::get().synchronize();

	// Let's do the same, but when the kernel is wrapped in a kernel_t
	std::cout
		<< "Launching kernel " << kernel_name
		<< " wrapped in a kernel_t structure,"
		<< " with " << num_blocks << " blocks, using cuda::launch()\n" << std::flush;

	cuda::launch(kernel, launch_config, bar);
	cuda::device::current::get().synchronize();

	// But there's more than one way to launch! we can also do
	// it via the device proxy, using the default stream:

	std::cout
		<< "Launching kernel " << kernel_name
		<< " with " << num_blocks << " blocks, using device.launch()\n" << std::flush;
	device.launch(kernel_function, launch_config, bar);
	device.synchronize();

	// or via a stream:

	auto stream = cuda::device::current::get().create_stream(cuda::stream::async);

	std::cout
		<< "Launching kernel " << kernel_name
		<< " with " << num_blocks << " blocks, using stream.launch()\n" << std::flush;
	stream.enqueue.kernel_launch(kernel, launch_config, bar);
	stream.synchronize();

#if TEST_COOPERATIVE_GROUPS
	try {
		auto can_launch_cooperatively = stream.device().supports_block_cooperation();
		if (can_launch_cooperatively) {
			auto cooperative_kernel_function = grid_cooperating_foo;
			auto cooperative_kernel_name = "grid_cooperating_foo";
			auto cooperative_config = launch_config;
			cooperative_config.block_cooperation = true;
			std::cout
			<< "Launching kernel" << cooperative_kernel_name
				<< " with " << num_blocks << " blocks, cooperatively, using stream.launch()\n"
				<< "(but note this does not actually check that cooperation takes place).\n" << std::flush;
			stream.enqueue.kernel_launch(cooperative_kernel_function, cooperative_config, bar);
			stream.synchronize();

			// Same, but using cuda::enqueue_launch
			std::cout
				<< "Launching kernel " << cooperative_kernel_name
				<< " wrapped in a kernel_t structure,"
				<< " with " << num_blocks << " blocks, using cuda::enqueue_launch(),"
				<< " and allowing thread block cooperation\n"
				<< "(but note this does not actually check that cooperation takes place).\n" << std::flush;

			cuda::enqueue_launch(cooperative_kernel_function, stream, cooperative_config, bar);
			cuda::device::current::get().synchronize();
		}
		else {
			std::cout
				<< "Skipping launch of kernel" << kernel_name
				<< ", since our CUDA device doesn't support cooperative launches.\n" << std::flush;
		}

	}
	catch(cuda::runtime_error& e) {
		if (not (e.code() == cuda::status::not_supported)) {
			throw e;
		}
		// We should really not have a sticky error at this point, but lets' make
		// extra sure.
		cuda::outstanding_error::ensure_none();
	}
#endif
	auto non_cooperative_kernel = cuda::kernel::get(device, kernel_function);
	auto non_cooperative_config = launch_config;
	non_cooperative_config.block_cooperation = true;
	std::cout
		<< "Launching kernel " << kernel_name << " with "
		<< num_blocks << " blocks, un-cooperatively, using stream.launch()\n" << std::flush;
	stream.enqueue.kernel_launch(non_cooperative_kernel, non_cooperative_config, bar);
	stream.synchronize();

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
