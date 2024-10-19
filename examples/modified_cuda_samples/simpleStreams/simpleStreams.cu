/**
 * Derived from the nVIDIA CUDA 8.0 samples by
 *
 *   Eyal Rozenberg
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the deriving author.
 *
 *
 * This sample illustrates the usage of CUDA streams for overlapping
 * kernel execution with device/host memcopies.  The kernel is used to
 * initialize an array to a specific value, after which the array is
 * copied to the host (CPU) memory.  To increase performance, multiple
 * kernel/memcopy pairs are launched asynchronously, each pair in its
 * own stream.  Devices with Compute Capability 1.1 can overlap a kernel
 * and a memcopy as long as they are issued in different streams.  Kernels
 * are serialized.  Thus, if n pairs are launched, streamed approach
 * can reduce the memcopy cost to the (1/n)th of a single copy of the entire
 * data set.
 *
 * Additionally, this sample uses CUDA events to measure elapsed time for
 * CUDA calls.  Events are a part of CUDA API and provide a system independent
 * way to measure execution times on CUDA devices with approximately 0.5
 * microsecond precision.
 *
 * Elapsed times are averaged over nreps repetitions (10 by default).
 *
 */

const char *sSDKsample = "simpleStreams";

const char *sEventSyncMethod[] =
{
	"cudaEventDefault",
	"cudaEventBlockingSync",
	"cudaEventDisableTiming",
	NULL
};

// helper functions and utilities to work with CUDA
#include "../../common.hpp"

#include <cstdlib>

#include <vector>
#include <iostream>
#include <algorithm>

using sync_policy_type = cuda::context::host_thread_sync_scheduling_policy_t;



// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

__global__ void init_array(int *g_data, const int *factor, int num_iterations)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i=0; i<num_iterations; i++)
	{
		g_data[idx] += *factor;    // non-coalesced on purpose, to burn time
	}
}

template <typename Container>
bool check_resulting_data(Container const & container, const int c)
{
	for (size_t i = 0; i < container.size(); i++) {
		if (container[i] != c) {
			std::cerr << i << ": " << container[i] << " " << c << "\n";
			return false;
		}
	}
	return true;
}

void printHelp()
{
	std::cout
		<< "Usage: " << sSDKsample << " [options below]\n"
		<< "\t--sync_method (" << (int) sync_policy_type::default_ << ") for CPU thread synchronization with GPU work."
		<< "\t             Possible values: " << (int) sync_policy_type::heuristic << ", "
		<< (int) sync_policy_type::spin << ", "
		<< (int) sync_policy_type::yield << ", "
		<< (int) sync_policy_type::block << ".\n"
		<< "\t--use_generic_memory (default) use generic page-aligned host memory allocation\n"
		<< "\t--use_cuda_malloc_host (optional) use pinned host memory allocation\n";
}

struct simple_streams_params_t
{
	int n;       // number of ints in the data set; originally 2^16
	int num_iterations;
	float scale_factor;
	unsigned faux_cores_per_sm;
	unsigned faux_cores_overall;
};

void run_simple_streams_example(
	const cuda::device_t& device,
	simple_streams_params_t params,
	const cuda::context::host_thread_sync_scheduling_policy_t sync_policy)
{
	int nstreams = 4;               // number of streams for CUDA calls
	int nreps = 10;                // number of times each experiment is repeated; originally 10
	std::size_t nbytes = params.n * sizeof(int);   // number of data bytes
	int c = 5;                      // value to which the array will be initialized

	// Allocate Host memory
	auto h_a = cuda::memory::host::make_unique_span<int>(params.n);

	// allocate device memory
	// pointers to data and init value in the device memory
	auto d_a = cuda::memory::make_unique_span<int>(device, params.n);
	auto d_c = cuda::memory::make_unique_span<int>(device, 1);
	cuda::memory::copy_single(d_c.data(), &c);

	std::cout << "\nStarting Test\n";

	// allocate and initialize an array of stream handles
	std::vector<cuda::stream_t> streams;
	std::generate_n(
		std::back_inserter(streams), nstreams,
		[&device]() {
			// Note: we could omit the specific requirement of synchronization
			// with the default stream, since that's the CUDA default - but I
			// think it's important to state that's the case
			return device.create_stream(cuda::stream::implicitly_synchronizes_with_default_stream);
		}
	);

	// create CUDA event handles
	// use blocking sync
	auto use_blocking_sync = (sync_policy == cudaDeviceBlockingSync);

	auto start_event = cuda::event::create(device, use_blocking_sync);
	auto stop_event = cuda::event::create(device, use_blocking_sync);

	// time memcpy from device
	start_event.record(); // record on the default stream, to ensure that all previous CUDA calls have completed
	cuda::memory::copy(h_a.get(), d_a, streams[0]);
	stop_event.record();
	stop_event.synchronize(); // block until the event is actually recorded
	auto time_memcpy = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << "memcopy:\t" << time_memcpy.count() << "\n";

	// time kernel
	auto launch_config = cuda::launch_config_builder()
		.overall_size(params.n)
		.block_size(512)
		.build();
	start_event.record();
	streams[0].enqueue.kernel_launch(init_array, launch_config, d_a.data(), d_c.data(), params.num_iterations);
	stop_event.record();
	stop_event.synchronize();
	auto time_kernel = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << "kernel:\t\t" << time_kernel.count() << "\n";

	//////////////////////////////////////////////////////////////////////
	// time non-streamed execution for reference
	launch_config = cuda::launch_config_builder()
		.overall_size(params.n)
		.block_size(512)
		.build();
	start_event.record();

	for (int k = 0; k < nreps; k++)
	{
		device.launch(init_array, launch_config, d_a.data(), d_c.data(), params.num_iterations);
		cuda::memory::copy(h_a.get(), d_a);
	}

	stop_event.record();
	stop_event.synchronize();
	auto elapsed_time = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << "non-streamed:\t" << elapsed_time.count() / nreps << "\n";

	//////////////////////////////////////////////////////////////////////
	// time execution with nstreams streams
	launch_config = cuda::launch_config_builder()
		.overall_size(params.n/nstreams)
		.block_size(512)
		.build();
	// TODO: Avoid need to push and pop here
	std::fill(h_a.begin(), h_a.end(), 255);     // set host memory bits to all 1s, for testing correctness
	// This instruction is actually the only one in our program
	// for which the device.make_current() command was necessary.
	// TODO: Avoid having to do that altogether...
	cuda::memory::device::zero(d_a); // set device memory to all 0s, for testing correctness
	start_event.record();

	for (int k = 0; k < nreps; k++)
	{
		// asynchronously launch nstreams kernels, each operating on its own portion of data
		for (int i = 0; i < nstreams; i++)
		{
			streams[i].enqueue.kernel_launch(
				init_array, launch_config, d_a.data() + i * params.n / nstreams, d_c.data(), params.num_iterations);
		}

		// asynchronously launch nstreams memcopies.  Note that memcopy in stream x will only
		//   commence executing when all previous CUDA calls in stream x have completed
		for (int i = 0; i < nstreams; i++)
		{
			cuda::memory::copy(
				h_a.data() + i * params.n / nstreams,
				d_a.data() + i * params.n / nstreams, nbytes / nstreams,
				streams[i]);
		}
	}

	stop_event.record();
	stop_event.synchronize();
	elapsed_time = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << nstreams <<" streams:\t" << elapsed_time.count() / (float) nreps << "\n";

	// check whether the output is correct
	std::cout << "-------------------------------\n";
	if (not check_resulting_data(h_a, c * nreps * params.num_iterations)) {
		die_("Result check FAILED.");
	}
}


simple_streams_params_t determine_params(const cuda::device_t& device)
{
	simple_streams_params_t result;
	result.n = 2 * 1024 * 1024;       // number of ints in the data set; originally 2^16
	result.scale_factor = 1.0f;

	// Checking for compute capabilities
	auto properties = device.properties();
	auto compute_capability = properties.compute_capability();

	if (compute_capability < cuda::device::compute_capability_t{1, 1} ) {
		std::cout << properties.name << " does not have Compute Capability 1.1 or newer. Reducing workload.\n";
	}
	// number of iterations for the loop inside the example kernel
	result.num_iterations = (compute_capability >= cuda::device::make_compute_capability(1,2)) ? 5 : 1;

	// Check if GPU can map host memory (Generic Method), if not then we override bPinGenericMemory to be false
	std::cout << "Device: <" << properties.name << "> canMapHostMemory: "
			  << (properties.canMapHostMemory ? "Yes" : "No") << "\n";

	if (not properties.can_map_host_memory())
	{
		std::cout << "Cannot allocate pinned memory (and map GPU device memory to it); waiving this example.\n";
		exit(EXIT_SUCCESS);
	}

	// Anything that is less than 32 Cores will have scaled down workload
	result.faux_cores_per_sm = compute_capability.max_in_flight_threads_per_processor();
	result.faux_cores_overall = properties.max_in_flight_threads_on_device();
	result.scale_factor = std::max((32.0f / result.faux_cores_overall), 1.0f);
	result.n = (int)rint((float)result.n / result.scale_factor);

	std::cout << "> CUDA Capable: SM " << compute_capability.major() << "." << compute_capability.minor() << " hardware\n";
	std::cout
		<< "> " << properties.multiProcessorCount << " Multiprocessor(s)"
		<< " x " << result.faux_cores_per_sm << " (Cores/Multiprocessor) = "
		<< result.faux_cores_overall << " (Cores)\n";

	std::cout << "> scale_factor = " << 1.0f/result.scale_factor << "\n";
	std::cout << "> array_size   = " << result.n << "\n\n";
	return result;
}

int main(int argc, char **argv)
{
	auto device = cuda::device::get(choose_device(argc, argv));

	device.make_current();
	// This is "necessary", for now, for the memory operations whose API is context-unaware,
	// but which would actually fail if the appropriate context is not the current one

	auto params = determine_params(device);

	for (const auto sync_policy : {
		sync_policy_type::heuristic,
		sync_policy_type::spin,
		sync_policy_type::yield,
		sync_policy_type::block })
	{
		std::cout << "> Running example using CPU/GPU Device Synchronization method " << sync_policy << '\n';
		device.set_sync_scheduling_policy(sync_policy);
		run_simple_streams_example(device, params, sync_policy);
		std::cout << '\n';
	}
	std::cout << "SUCCESS\n";
}
