/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2023, Eyal Rozenberg <eyalroz1@gmx.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample demonstrates stream ordered memory allocation on a GPU using
 * asynchronous memory allocations and the memory-pool-related API wrappers.
 *
 * basicStreamOrderedAllocation(): demonstrates stream ordered allocation using
 * the stream wrapper class.
 *
 * streamOrderedAllocationPostSync(): demonstrates if there's a synchronization
 * in between allocations, then setting the release threshold on the pool will
 * make sure the synchronize will not free memory back to the OS.
 */

#include <climits>
#include <algorithm>
#include <iostream>

#include <cuda/api.hpp>

#define MAX_ITER 20

#if __cplusplus >= 201712L
using std::span;
#else
using cuda::span;
#endif

/* Add two vectors on the GPU */

__global__ void vectorAddGPU(const float *a, const float *b, float *c, size_t N)
{
    size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {
		c[idx] = a[idx] + b[idx];
	}
}

bool check_results(
	const char* method,
	span<const float> a,
	span<const float> b,
	span<float      > c)
{
	assert(a.size() == c.size() and b.size() == c.size());

	float errorNorm, refNorm, ref, diff;

	std::cout << "> Checking the results from vectorAddGPU() ...\n";
	errorNorm = 0.f;
	refNorm = 0.f;

	for (size_t n = 0; n < c.size(); n++) {
		ref = a[n] + b[n];
		diff = c[n] - ref;
		errorNorm += diff * diff;
		refNorm += ref * ref;
	}

	errorNorm = (float) std::sqrt((double) errorNorm);
	refNorm = (float) std::sqrt((double) refNorm);
	bool error_small_enough = (errorNorm / refNorm < 1.e-6f);
	if (not error_small_enough) {
		std::cerr << "Excessive error with method " << method << ": " << errorNorm / refNorm << '\n';
	}
	return error_small_enough;
}

int basicStreamOrderedAllocation(
	const cuda::device_t& device,
	span<const float> a,
	span<const float> b,
	span<float      > c)
{
	static constexpr const char* method = "basicStreamOrderedAllocation";
	assert(a.size() == c.size() and b.size() == c.size());

	auto launch_config = cuda::launch_config_builder()
		.block_size(256)
		.overall_size(c.size())
		.no_dynamic_shared_memory().build();

	std::cout << "Starting " << method << "\n";
	auto stream = device.create_stream(cuda::stream::async);

	auto d_a = span<float>(stream.enqueue.allocate(a.size() * sizeof(float)));
	auto d_b = span<float>(stream.enqueue.allocate(b.size() * sizeof(float)));
	auto d_c = span<float>(stream.enqueue.allocate(c.size() * sizeof(float)));
	stream.enqueue.copy(d_a, a);
	stream.enqueue.copy(d_b, b);
	stream.enqueue.kernel_launch(vectorAddGPU, launch_config, d_a.data(), d_b.data(), d_c.data(), (int) c.size());
	stream.enqueue.free(d_a);
	stream.enqueue.free(d_b);
	stream.enqueue.copy(c, d_c);
	stream.enqueue.free(d_c);
	stream.synchronize();

	return check_results(method, a, b, c);
}

// streamOrderedAllocationPostSync(): demonstrates If the application wants the
// memory to persist in the pool beyond synchronization, then it sets the
// release threshold on the pool. This way, when the application reaches the
// "steady state", it is no longer allocating/freeing memory from the OS.
int streamOrderedAllocationPostSync(
	const cuda::device_t& device,
	span<const float> a,
	span<const float> b,
	span<float      > c)
{
	static constexpr const char* method = "streamOrderedAllocationPostSync";

	auto launch_config = cuda::launch_config_builder()
		.block_size(256)
		.overall_size(c.size())
		.no_dynamic_shared_memory().build();

	std::cout << "Starting " << method << "\n";
	auto stream = device.create_stream(cuda::stream::async);

	// set high release threshold on the default pool so that cudaFreeAsync will
	// not actually release memory to the system. By default, the release
	// threshold for a memory pool is set to zero. This implies that the CUDA
	// driver is allowed to release a memory chunk back to the system as long as
	// it does not contain any active suballocations.
	device.default_memory_pool().set_release_threshold(ULONG_MAX);

	// Record the start event
	auto start_event = stream.enqueue.event();
	for (int i = 0; i < MAX_ITER; i++) {
		// Not: Not using unique_span's,
		auto d_a = cuda::span<float>(stream.enqueue.allocate(a.size() * sizeof(float)));
		auto d_b = cuda::span<float>(stream.enqueue.allocate(b.size() * sizeof(float)));
		auto d_c = cuda::span<float>(stream.enqueue.allocate(c.size() * sizeof(float)));
		stream.enqueue.copy(d_a, a);
		stream.enqueue.copy(d_b, b);
		stream.enqueue.kernel_launch(vectorAddGPU, launch_config, d_a.data(), d_b.data(), d_c.data(), c.size());
		stream.enqueue.free(d_a);
		stream.enqueue.free(d_b);
		stream.enqueue.copy(c, d_c);
		stream.enqueue.free(d_c);
		stream.synchronize();
	}
	auto end_event = stream.enqueue.event();
	end_event.synchronize();

	auto elapsed_time_msec = cuda::event::time_elapsed_between(start_event, end_event);
	std::cout
		<< "Total elapsed time for method " << method << " = "
		<< elapsed_time_msec.count() << " msec over " << MAX_ITER << " iterations.\n";
	return check_results(method, a, b, c);
}

int main(int argc, char **argv)
{
	cuda::device::id_t device_id = (argc > 1) ? std::stoi(argv[1]) : cuda::device::default_device_id;
	auto device = cuda::device::get(device_id);

	if (not device.supports_memory_pools()) {
		printf("Waiving execution as device does not support Memory Pools\n");
		exit(EXIT_SUCCESS);
	}

	size_t nelem = 1048576;

	auto a = std::unique_ptr<float[]>(new float[nelem]);
	auto b = std::unique_ptr<float[]>(new float[nelem]);
	auto c = std::unique_ptr<float[]>(new float[nelem]);
	std::generate_n(a.get(), nelem, [&] { return (float) rand() / (float) RAND_MAX; });
	std::generate_n(b.get(), nelem, [&] { return (float) rand() / (float) RAND_MAX; });

	auto a_sp = span<const float>{a.get(), nelem};
	auto b_sp = span<const float>{b.get(), nelem};
	auto c_sp = span<float      >{c.get(), nelem};

	auto b1 = basicStreamOrderedAllocation(device, a_sp, b_sp, c_sp);
	auto b2 = streamOrderedAllocationPostSync(device, a_sp, b_sp, c_sp);

	std::cout << '\n' << ((b1 and b2) ? "SUCCESS" : "FAILURE") << '\n';
}
