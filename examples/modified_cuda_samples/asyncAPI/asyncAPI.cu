////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

//
// This sample illustrates the usage of CUDA events for both GPU timing and
// overlapping CPU and GPU execution.  Events are inserted into a stream
// of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
// perform computations while GPU is executing (including DMA memcopies
// between the host and device).  CPU can query CUDA events to determine
// whether GPU has completed tasks.
//

// includes, system
#include <stdio.h>

// includes, project
#include "../helper_cuda.hpp"

using datum = int;

__global__ void increment_kernel(datum*g_data, datum inc_value)
{
	int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
	g_data[global_idx] = g_data[global_idx] + inc_value;
}

bool correct_output(int *data, const int n, const int x)
{
	for (int i = 0; i < n; i++)
		if (data[i] != x)
		{
			printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
			return false;
		}
	return true;
}

int main(int, char **)
{
	std::cout << "asyncAPI Starting...\n";

	// This will pick the best possible CUDA capable device
	// int devID = findCudaDevice(argc, (const char **)argv);

	auto device = cuda::device::current::get();

	std::cout << "CUDA device [" <<  device.name() << "]\n";

	int n = 16 * 1024 * 1024;
	int nbytes = n * sizeof(datum);
	int value = 26;

	// allocate host memory
	auto a = cuda::memory::host::make_unique<datum[]>(n);
	cuda::memory::host::zero(a.get(), nbytes);

	auto d_a = cuda::memory::device::make_unique<datum[]>(device, n);

	auto threads = cuda::grid::block_dimensions_t(512, 1);
	assert_(n % threads.x == 0);
	auto blocks  = cuda::grid::dimensions_t(n / threads.x, 1);
	auto launch_config = cuda::make_launch_config(blocks, threads);

	// create cuda event handles
	auto start_event = cuda::event::create(
		device,
		cuda::event::sync_by_blocking,
		cuda::event::do_record_timings,
		cuda::event::not_interprocess);
	auto end_event = cuda::event::create(
		device,
		cuda::event::sync_by_blocking,
		cuda::event::do_record_timings,
		cuda::event::not_interprocess);

	auto stream = device.default_stream(); // device.create_stream(cuda::stream::async);
	auto cpu_time_start = std::chrono::high_resolution_clock::now();
	stream.enqueue.event(start_event);
	stream.enqueue.copy(d_a.get(), a.get(), nbytes);
	stream.enqueue.kernel_launch(increment_kernel, launch_config, d_a.get(), value);
	stream.enqueue.copy(a.get(), d_a.get(), nbytes);
	stream.enqueue.event(end_event);
	auto cpu_time_end = std::chrono::high_resolution_clock::now();

	// have CPU do some work while waiting for stage 1 to finish
	unsigned long int counter=0;

	while (not end_event.has_occurred())
	{
		counter++;
	}

	std::cout << "time spent executing by the GPU: " << std::setprecision(2)
		<< cuda::event::time_elapsed_between(start_event, end_event).count() << '\n';
	std::cout << "time spent by CPU in CUDA calls: " << std::setprecision(2)<< (cpu_time_end - cpu_time_start).count() << '\n';
	std::cout << "CPU executed " << counter << " iterations while waiting for GPU to finish\n";

	auto bFinalResults = correct_output(a.get(), n, value);

	std::cout << (bFinalResults ? "SUCCESS" : "FAILURE") << '\n';

	exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}
