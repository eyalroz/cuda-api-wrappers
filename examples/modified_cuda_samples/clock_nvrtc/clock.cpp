/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


// This example shows how to use the clock function to measure the performance of
// block of threads of a kernel accurately.
//

// Blocks are executed in parallel and out of order. Since there's no synchronization
// mechanism between blocks, we measure the clock once for each block. The clock
// samples are written to device memory.

// System includes
#include <sstream>
#include <memory>
#include <utility>

#include "../../rtc_common.hpp"

#include <cuda/api.hpp>
#include <cuda/nvrtc.hpp>

namespace clock_kernel {

constexpr const char *name = "timedReduction";

constexpr const char *source =
	R"(extern "C" __global__  void timedReduction(const float *input, float *output, clock_t *timer)
{
    // __shared__ float shared[2 * blockDim.x];
    extern __shared__ float shared[];

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;

    if (tid == 0) timer[bid] = clock();

    // Copy input.
    shared[tid] = input[tid];
    shared[tid + blockDim.x] = input[tid + blockDim.x];

    // Perform reduction to find minimum.
    for (int d = blockDim.x; d > 0; d /= 2)
    {
        __syncthreads();

        if (tid < d)
        {
            float f0 = shared[tid];
            float f1 = shared[tid + d];

            if (f1 < f0)
            {
                shared[tid] = f1;
            }
        }
    }

    // Write result.
    if (tid == 0) output[bid] = shared[0];

    __syncthreads();

    if (tid == 0) timer[bid+gridDim.x] = clock();
}
)";

} // namespace kernel


// It's interesting to change the number of blocks and the number of threads to
// understand how to keep the hardware busy.
//
//
// Here are some numbers I get on my G80:
//    blocks - clocks
//    1 - 3096
//    8 - 3232
//    16 - 3364
//    32 - 4615
//    64 - 9981
//
//
// With less than 16 blocks some of the multiprocessors of the device are idle. With
// more than 16 you are using all the multiprocessors, but there's only one block per
// multiprocessor and that doesn't allow you to hide the latency of the memory. With
// more than 32 the speed scales linearly.

long double compute_average_elapsed_clocks(const clock_t* timers, std::size_t num_blocks)
{
	long double offset_sum = 0;

	for (std::size_t block_idx = 0; block_idx < num_blocks; block_idx++)
	{
		offset_sum += (long double) (timers[block_idx + num_blocks] - timers[block_idx]);
	}

	return offset_sum / num_blocks;
}

cuda::dynarray<char> compile_to_cubin(
	const char* kernel_source,
	const char* kernel_name,
	cuda::device_t target_device)
{
	auto program = cuda::rtc::program::create(kernel_name)
		.set_source(kernel_source).set_target(target_device);
		// I wonder if using the same name for the program and the kernel is a good idea

	auto output = program.compile();
	auto log = output.log();

	if (log.size() >= 1) {
		std::cerr << "\n compilation log ---\n";
		std::cerr << log.data();
		std::cerr << "\n end log ---\n";
	}
	return output.cubin();
}


int main()
{
	std::cout << "CUDA Clock sample\n";

	auto device_id { 0 }; // Not bothering with supporting a command-line argument here
	auto device = cuda::device::get(device_id);
	auto cubin = compile_to_cubin(clock_kernel::source, clock_kernel::name, device);
	auto module = cuda::module::create(device, cubin);
	auto kernel_in_module = module.get_kernel(clock_kernel::name);

	cuda::grid::dimension_t num_blocks { 64 };
	cuda::grid::block_dimension_t num_threads_per_block { 256 };
	std::size_t num_timers { num_blocks * 2 };
	std::size_t input_size { num_threads_per_block * 2 };
	std::unique_ptr<clock_t[]> timers(new clock_t[input_size]);
		// Strangely, it seems CUDA's clock() intrinsic uses the C standard library's clock_t. Oh well.
	std::unique_ptr<float[]> input(new float[input_size]);

	auto generator = []() {
		static size_t v = 0;
		return (float) v++;
	};
	std::generate_n(input.get(), input_size, generator);
	// TODO: Too bad we don't have a generator variant which passes the element index
	{
		const auto dynamic_shmem_size = sizeof(float) * 2 * num_threads_per_block;

		auto d_input = cuda::memory::device::make_unique<float[]>(device, input_size);
		auto d_output = cuda::memory::device::make_unique<float[]>(device, num_blocks);
			// Note: We won't actually be checking the output...
		auto d_timers = cuda::memory::device::make_unique<clock_t []>(device, num_timers);
		cuda::memory::copy(d_input.get(), input.get(), input_size * sizeof(float));

		auto launch_config = cuda::make_launch_config(num_blocks, num_threads_per_block, dynamic_shmem_size);
		cuda::launch(kernel_in_module, launch_config, d_input.get(), d_output.get(), d_timers.get());
		device.synchronize();
		cuda::memory::copy(timers.get(), d_timers.get(), num_timers * sizeof(clock_t));
	} // The allcoated device buffers are released here
	long double average_elapsed_clock_ticks_per_block = compute_average_elapsed_clocks(timers.get(), num_blocks);

	std::cout << "Average clocks/block: " << average_elapsed_clock_ticks_per_block << '\n';

	std::cout << "\nSUCCESS\n";
}

