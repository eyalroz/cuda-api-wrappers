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
#include "../helper_string.h"
#include "../../common.hpp"

#include <cstdlib>

#include <vector>
#include <iostream>
#include <algorithm>

using synch_policy_type = cuda::context::host_thread_synch_scheduling_policy_t;



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

bool check_resulting_data(const int *a, const int n, const int c)
{
	for (int i = 0; i < n; i++) {
		if (a[i] != c) {
			std::cerr << i << ": " << a[i] << " " << c << "\n";
			return false;
		}
	}
	return true;
}

void printHelp()
{
	std::cout
		<< "Usage: " << sSDKsample << " [options below]\n"
		<< "\t--sync_method (" << (int) synch_policy_type::default_ << ") for CPU thread synchronization with GPU work."
		<< "\t             Possible values: " << (int) synch_policy_type::heuristic << ", "
		<< (int) synch_policy_type::spin << ", "
		<< (int) synch_policy_type::yield << ", "
		<< (int) synch_policy_type::block << ".\n"
		<< "\t--use_generic_memory (default) use generic page-aligned host memory allocation\n"
		<< "\t--use_cuda_malloc_host (optional) use pinned host memory allocation\n";
}

int main(int argc, char **argv)
{
	int nstreams = 4;               // number of streams for CUDA calls
	int nreps = 5;                 // number of times each experiment is repeated; originally 10
	int n = 2 * 1024 * 1024;       // number of ints in the data set; originally 2^16
	std::size_t nbytes = n * sizeof(int);   // number of data bytes
	dim3 threads, blocks;           // kernel launch configuration
	float scale_factor = 1.0f;

	// allocate generic memory and pin it laster instead of using cudaHostAlloc()

	auto synch_policy = synch_policy_type::block;

	int niterations;    // number of iterations for the loop inside the kernel

	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	{
		printHelp();
		return EXIT_SUCCESS;
	}

	auto synch_policy_arg = getCmdLineArgumentInt(argc, (const char **)argv, "sync_method");
	if (   synch_policy_arg == synch_policy_type::heuristic
		|| synch_policy_arg == synch_policy_type::spin
		|| synch_policy_arg == synch_policy_type::yield
		|| synch_policy_arg == synch_policy_type::block)
	{
		synch_policy = (synch_policy_type) synch_policy_arg;
			std::cout << "Device synchronization method set to: " <<
				  (synch_policy_type) synch_policy << "\n";
		std::cout << "Setting reps to 100 to demonstrate steady state\n";
		nreps = 100;
	}
	else
	{
		std::cout << "Invalid command line option sync_method \"" << synch_policy << "\"\n";
		return EXIT_FAILURE;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "use_cuda_malloc_host"))
	{
		std::cout
			<< "To simplify this example, support for using cuda_malloc_host instead of "
			<< "pinned memory has been dropped.\n";
		return EXIT_FAILURE;
	}

	std::cout << "\n> ";
	auto device = cuda::device::get(choose_device(argc, argv));
	device.make_current();
		// This is "necessary", for now, for the memory operations whose API is context-unaware,
		// but which would actually fail if the appropriate context is not the current one

	// Checking for compute capabilities
	auto properties = device.properties();
	auto compute_capability = properties.compute_capability();

	if (compute_capability < cuda::device::compute_capability_t({1, 1}) ) {
		std::cout << properties.name << " does not have Compute Capability 1.1 or newer. Reducing workload.\n";
	}

	if (compute_capability.major() >= 2) {
		niterations = 5;
	} else {
		if (compute_capability.minor() > 1) {
			niterations = 5;
		} else {
			niterations = 1; // reduced workload for compute capability 1.0 and 1.1
		}
	}

	// Check if GPU can map host memory (Generic Method), if not then we override bPinGenericMemory to be false
	std::cout << "Device: <" << properties.name << "> canMapHostMemory: "
			<< (properties.canMapHostMemory ? "Yes" : "No") << "\n";

	if (not properties.can_map_host_memory())
	{
		std::cout << "Cannot allocate pinned memory (and map GPU device memory to it); aborting.\n";
		return EXIT_FAILURE;
	}

	// Anything that is less than 32 Cores will have scaled down workload
	auto faux_cores_per_sm = compute_capability.max_in_flight_threads_per_processor();
	auto faux_cores_overall = properties.max_in_flight_threads_on_device();
	scale_factor = std::max((32.0f / faux_cores_overall), 1.0f);
	n = (int)rint((float)n / scale_factor);

	std::cout << "> CUDA Capable: SM " << compute_capability.major() << "." << compute_capability.minor() << " hardware\n";
	std::cout
		<< "> " << properties.multiProcessorCount << " Multiprocessor(s)"
		<< " x " << faux_cores_per_sm << " (Cores/Multiprocessor) = "
		<< faux_cores_overall << " (Cores)\n";

	std::cout << "> scale_factor = " << 1.0f/scale_factor << "\n";
	std::cout << "> array_size   = " << n << "\n\n";

	// enable use of blocking sync, to reduce CPU usage
	std::cout << "> Using CPU/GPU Device Synchronization method " << synch_policy << std::endl;

	synch_policy_type  policy;
	switch(synch_policy) {
	case 0: policy = synch_policy_type::heuristic; break;
	case 1: policy = synch_policy_type::spin;      break;
	case 2: policy = synch_policy_type::yield;     break;
	case 4: policy = synch_policy_type::block;     break;
	default: // should not be able to get here
		exit(EXIT_FAILURE);
	}

	device.set_synch_scheduling_policy(policy);
	// Not necessary: Since CUDA 3.2 (which is below the minimum supported
	// version for the API wrappers, all contexts allow such mapping.
	// device.enable_mapping_host_memory();

	int c = 5;                      // value to which the array will be initialized

	// Allocate Host memory
	auto h_a = cuda::memory::host::make_unique<int[]>(n);

	// allocate device memory
	// pointers to data and init value in the device memory
	auto d_a = cuda::memory::device::make_unique<int[]>(device, n);
	auto d_c = cuda::memory::device::make_unique<int>(device);
	cuda::memory::copy_single(d_c.get(), &c);

	std::cout << "\nStarting Test\n";

	// allocate and initialize an array of stream handles
	std::vector<cuda::stream_t> streams;
	std::generate_n(
		std::back_inserter(streams), nstreams,
		[&device]() {
			// Note: we could omit the specific requirement of synchronization
			// with the default stream, since that's the CUDA default - but I
			// think it's important to state that's the case
			return device.create_stream(
				cuda::stream::implicitly_synchronizes_with_default_stream);
		}
	);

	// create CUDA event handles
	// use blocking sync
	auto use_blocking_sync = (synch_policy == cudaDeviceBlockingSync);

	auto start_event = cuda::event::create(device, use_blocking_sync);
	auto stop_event = cuda::event::create(device, use_blocking_sync);

	// time memcopy from device
	start_event.record(); // record on the default stream, to ensure that all previous CUDA calls have completed
	cuda::memory::async::copy(h_a.get(), d_a.get(), nbytes, streams[0]);
	stop_event.record();
	stop_event.synchronize(); // block until the event is actually recorded
	auto time_memcpy = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << "memcopy:\t" << time_memcpy.count() << "\n";

	// time kernel
	threads=dim3(512, 1);
	assert_(n % threads.x == 0);
	blocks=dim3(n / threads.x, 1);
	auto launch_config = cuda::make_launch_config(blocks, threads);
	start_event.record();
	streams[0].enqueue.kernel_launch(init_array, launch_config, d_a.get(), d_c.get(), niterations);
	stop_event.record();
	stop_event.synchronize();
	auto time_kernel = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << "kernel:\t\t" << time_kernel.count() << "\n";

	//////////////////////////////////////////////////////////////////////
	// time non-streamed execution for reference
	threads=dim3(512, 1);
	blocks=dim3(n / threads.x, 1);
	launch_config = cuda::make_launch_config(blocks, threads);
	start_event.record();

	for (int k = 0; k < nreps; k++)
	{
		device.launch(init_array, launch_config, d_a.get(), d_c.get(), niterations);
		cuda::memory::copy(h_a.get(), d_a.get(), nbytes);
	}

	stop_event.record();
	stop_event.synchronize();
	auto elapsed_time = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << "non-streamed:\t" << elapsed_time.count() / nreps << "\n";

	//////////////////////////////////////////////////////////////////////
	// time execution with nstreams streams
	threads=dim3(512,1);
	blocks=dim3(n/(nstreams*threads.x),1);
	launch_config = cuda::make_launch_config(blocks, threads);
	// TODO: Avoid need to push and pop here
	memset(h_a.get(), 255, nbytes);     // set host memory bits to all 1s, for testing correctness
	// This instruction is actually the only one in our program
	// for which the device.make_current() command was necessary.
	// TODO: Avoid having to do that altogether...
	cuda::memory::device::zero(cuda::memory::region_t{d_a.get(), nbytes}); // set device memory to all 0s, for testing correctness
	start_event.record();

	for (int k = 0; k < nreps; k++)
	{
		// asynchronously launch nstreams kernels, each operating on its own portion of data
		for (int i = 0; i < nstreams; i++)
		{
			streams[i].enqueue.kernel_launch(
				init_array, launch_config, d_a.get() + i *n / nstreams, d_c.get(), niterations);
		}

		// asynchronously launch nstreams memcopies.  Note that memcopy in stream x will only
		//   commence executing when all previous CUDA calls in stream x have completed
		for (int i = 0; i < nstreams; i++)
		{
			cuda::memory::async::copy(
				h_a.get() + i * n / nstreams,
				d_a.get() + i * n / nstreams, nbytes / nstreams,
				streams[i]);
		}
	}

	stop_event.record();
	stop_event.synchronize();
	elapsed_time = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << nstreams <<" streams:\t" << elapsed_time.count() / (float) nreps << "\n";

	// check whether the output is correct
	std::cout << "-------------------------------\n";
	if (not check_resulting_data(h_a.get(), n, c * nreps * niterations)) {
		die_("Result check FAILED.");
	}
	std::cout << "\nSUCCESS\n";
}
