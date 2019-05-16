/**
 * Derived from the nVIDIA CUDA 8.0 samples by
 *
 *   Eyal Rozenberg <E.Rozenberg@cwi.nl>
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

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

const char *sSDKsample = "simpleStreams";

const char *sEventSyncMethod[] =
{
	"cudaEventDefault",
	"cudaEventBlockingSync",
	"cudaEventDisableTiming",
	NULL
};

const char *sDeviceSyncMethod[] =
{
	"cudaDeviceScheduleAuto",
	"cudaDeviceScheduleSpin",
	"cudaDeviceScheduleYield",
	"INVALID",
	"cudaDeviceScheduleBlockingSync",
	NULL
};

// System includes

// CUDA runtime
#include "cuda_runtime.h"

// helper functions and utilities to work with CUDA
#include "../helper_cuda.h"

#include <cuda/api_wrappers.hpp>

#ifndef WIN32
#include <sys/mman.h> // for mmap() / munmap()
#endif

#include <cstdlib>

#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>


// Macro to aligned up to the memory size in question
#define MEMORY_ALIGNMENT  4096
#define ALIGN_UP(x,size) ( ((size_t)x+(size-1))&(~(size-1)) )

__global__ void init_array(int *g_data, int *factor, int num_iterations)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i=0; i<num_iterations; i++)
	{
		g_data[idx] += *factor;    // non-coalesced on purpose, to burn time
	}
}

bool correct_data(int *a, const int n, const int c)
{
	for (int i = 0; i < n; i++) {
		if (a[i] != c) {
			std::cout << i << ": " << a[i] << " " << c << "\n";
			return false;
		}
	}
	return true;
}

static const char *sSyncMethod[] =
{
	"0 (Automatic Blocking)",
	"1 (Spin Blocking)",
	"2 (Yield Blocking)",
	"3 (Undefined Blocking Method)",
	"4 (Blocking Sync Event) = low CPU utilization",
	NULL
};

void printHelp()
{
	std::cout
		<< "Usage: " << sSDKsample << " [options below]\n"
		<< "\t--sync_method=n for CPU/GPU synchronization\n"
		<< "\t             n=" << sSyncMethod[0] << "\n"
		<< "\t             n=" << sSyncMethod[1] << "\n"
		<< "\t             n=" << sSyncMethod[2] << "\n"
		<< "\t   <Default> n=" << sSyncMethod[4] << "\n"
		<< "\t--use_generic_memory (default) use generic page-aligned for system memory\n"
		<< "\t--use_cuda_malloc_host (optional) use cudaMallocHost to allocate system memory\n";
}

int main(int argc, char **argv)
{
	int cuda_device_id = 0;
	int nstreams = 4;               // number of streams for CUDA calls
	int nreps = 10;                 // number of times each experiment is repeated
	int n = 16 * 1024 * 1024;       // number of ints in the data set
	int nbytes = n * sizeof(int);   // number of data bytes
	dim3 threads, blocks;           // kernel launch configuration
	float scale_factor = 1.0f;

	// allocate generic memory and pin it laster instead of using cudaHostAlloc()

	int  device_sync_method = cudaDeviceBlockingSync; // by default we use BlockingSync

	int niterations;    // number of iterations for the loop inside the kernel

	if (checkCmdLineFlag(argc, (const char **)argv, "help"))
	{
		printHelp();
		return EXIT_SUCCESS;
	}

	if ((device_sync_method = getCmdLineArgumentInt(argc, (const char **)argv, "sync_method")) >= 0)
	{
		if (device_sync_method == 0 || device_sync_method == 1 || device_sync_method == 2 || device_sync_method == 4)
		{
			std::cout << "Device synchronization method set to = " << sSyncMethod[device_sync_method] << "\n";
			std::cout << "Setting reps to 100 to demonstrate steady state\n";
			nreps = 100;
		}
		else
		{
			std::cout << "Invalid command line option sync_method=\"" << device_sync_method << "\"\n";
			return EXIT_FAILURE;
		}
	}
	else
	{
		printHelp();
		return EXIT_SUCCESS;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "use_cuda_malloc_host"))
	{
		std::cout
			<< "To simplify this example, support for using cuda_malloc_host instead of "
			<< "pinned memory has been dropped.\n";
		return EXIT_FAILURE;
	}

	std::cout << "\n> ";
	cuda_device_id = findCudaDevice(argc, (const char **)argv);

	// check the compute capability of the device
	auto num_devices = cuda::device::count();

	if ( 0 == num_devices)
	{
		std::cerr << "your system does not have a CUDA capable device, waiving test...\n";
		return EXIT_WAIVED;
	}

	// check if the command-line chosen device ID is within range, exit if not
	if (cuda_device_id >= num_devices)
	{
		std::cout
			<< "cuda_device=" << cuda_device_id << " is invalid, "
			<< "must choose device ID between 0 and " << num_devices-1 << "\n";
		return EXIT_FAILURE;
	}

	cuda::device::current::set(cuda_device_id);
	auto current_device = cuda::device::current::get();

	// Checking for compute capabilities
	auto properties = current_device.properties();
	auto compute_capability = properties.compute_capability();

	if (compute_capability < cuda::device::compute_capability_t({1, 1}) ) {
		std::cout << properties.name << " does not have Compute Capability 1.1 or newer.  Reducing workload.\n";
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
	scale_factor = max((32.0f / faux_cores_overall), 1.0f);
	n = (int)rint((float)n / scale_factor);

	std::cout << "> CUDA Capable: SM " << compute_capability.major() << "." << compute_capability.minor() << " hardware\n";
	std::cout
		<< "> " << properties.multiProcessorCount << " Multiprocessor(s)"
		<< " x " << faux_cores_per_sm << " (Cores/Multiprocessor) = "
		<< faux_cores_overall << " (Cores)\n";

	std::cout << "> scale_factor = " << 1.0f/scale_factor << "\n";
	std::cout << "> array_size   = " << n << "\n\n";

	// enable use of blocking sync, to reduce CPU usage
	std::cout << "> Using CPU/GPU Device Synchronization method " << sDeviceSyncMethod[device_sync_method] << "\n";

	cuda::host_thread_synch_scheduling_policy_t policy;
	switch(device_sync_method) {
	case 0: policy = cuda::heuristic; break;
	case 1: policy = cuda::spin;      break;
	case 2: policy = cuda::yield;     break;
	case 4: policy = cuda::block;     break;
	default: // should not be able to get here
		exit(EXIT_FAILURE);
	}
	current_device.set_synch_scheduling_policy(policy);
	current_device.enable_mapping_host_memory();

	// allocate host memory
	int c = 5;                      // value to which the array will be initialized

	// Allocate Host memory
	auto h_a = cuda::memory::host::make_unique<int[]>(n);

	// allocate device memory
	// pointers to data and init value in the device memory
	auto d_a = cuda::memory::device::make_unique<int[]>(cuda_device_id, n);
	auto d_c = cuda::memory::device::make_unique<int>(cuda_device_id);
	cuda::memory::copy_single(d_c.get(), &c);

	std::cout << "\nStarting Test\n";

	// allocate and initialize an array of stream handles
	std::vector<cuda::stream_t<>> streams;
	std::generate_n(
		std::back_inserter(streams), nstreams,
		[&current_device]() {
			// Note: we could omit the specific requirement of synchronization
			// with the default stream, since that's the CUDA default - but I
			// think it's important to state that's the case
			return current_device.create_stream(
				cuda::stream::implicitly_synchronizes_with_default_stream);
		}
	);

	// create CUDA event handles
	// use blocking sync
	auto use_blocking_sync = (device_sync_method == cudaDeviceBlockingSync);

	auto start_event = cuda::event::create(current_device, use_blocking_sync);
	auto stop_event = cuda::event::create(current_device, use_blocking_sync);

	// time memcopy from device
	start_event.record(cuda::stream::default_stream_id); // record in stream-0, to ensure that all previous CUDA calls have completed
	cuda::memory::async::copy(h_a.get(), d_a.get(), nbytes, streams[0].id());
	stop_event.record(cuda::stream::default_stream_id); // record in stream-0, to ensure that all previous CUDA calls have completed
	stop_event.synchronize(); // block until the event is actually recorded
	auto time_memcpy = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << "memcopy:\t" << time_memcpy.count() << "\n";

	// time kernel
	threads=dim3(512, 1);
	blocks=dim3(n / threads.x, 1);
	start_event.record(cuda::stream::default_stream_id);
	init_array<<<blocks, threads, 0, streams[0].id()>>>(d_a.get(), d_c.get(), niterations);
	stop_event.record(cuda::stream::default_stream_id);
	stop_event.synchronize();
	auto time_kernel = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << "kernel:\t\t" << time_kernel.count() << "\n";

	//////////////////////////////////////////////////////////////////////
	// time non-streamed execution for reference
	threads=dim3(512, 1);
	blocks=dim3(n / threads.x, 1);
	start_event.record(cuda::stream::default_stream_id);

	for (int k = 0; k < nreps; k++)
	{
		init_array<<<blocks, threads>>>(d_a.get(), d_c.get(), niterations);
		cuda::memory::copy(h_a.get(), d_a.get(), nbytes);
	}

	stop_event.record(cuda::stream::default_stream_id);
	stop_event.synchronize();
	auto elapsed_time = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << "non-streamed:\t" << elapsed_time.count() / nreps << "\n";

	//////////////////////////////////////////////////////////////////////
	// time execution with nstreams streams
	threads=dim3(512,1);
	blocks=dim3(n/(nstreams*threads.x),1);
	memset(h_a.get(), 255, nbytes);     // set host memory bits to all 1s, for testing correctness
	cuda::memory::device::zero(d_a.get(), nbytes); // set device memory to all 0s, for testing correctness
	start_event.record(cuda::stream::default_stream_id);

	for (int k = 0; k < nreps; k++)
	{
		// asynchronously launch nstreams kernels, each operating on its own portion of data
		for (int i = 0; i < nstreams; i++)
		{
			init_array<<<blocks, threads, 0, streams[i].id()>>>(d_a.get() + i *n / nstreams, d_c.get(), niterations);
		}

		// asynchronously launch nstreams memcopies.  Note that memcopy in stream x will only
		//   commence executing when all previous CUDA calls in stream x have completed
		for (int i = 0; i < nstreams; i++)
		{
			cuda::memory::async::copy(
				h_a.get() + i * n / nstreams,
				d_a.get() + i * n / nstreams, nbytes / nstreams,
				streams[i].id());
		}
	}

	stop_event.record(cuda::stream::default_stream_id);
	stop_event.synchronize();
	elapsed_time = cuda::event::time_elapsed_between(start_event, stop_event);
	std::cout << nstreams <<" streams:\t" << elapsed_time.count() / nreps << "\n";

	// check whether the output is correct
	std::cout << "-------------------------------\n";
	bool bResults = correct_data(h_a.get(), n, c*nreps*niterations);

	std::cout << (bResults ? "SUCCESS" : "FAILURE") << "\n";
	return bResults ? EXIT_SUCCESS : EXIT_FAILURE;
}
