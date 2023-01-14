/*
 * Original code Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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

#include <cooperative_groups.h>
#include <cuda/api.hpp>
#include <vector>
#include <iostream>
#include <numeric>

#if __cplusplus >= 202001L
using span = std::span;
#else
using cuda::span;
#endif

namespace cg = cooperative_groups;

#define THREADS_PER_BLOCK 512
#define GRAPH_LAUNCH_ITERATIONS 3

__global__ void reduce(float *inputVec, double *outputVec, size_t inputSize, size_t outputSize)
{
	__shared__ double tmp[THREADS_PER_BLOCK];

	cg::thread_block cta = cg::this_thread_block();
	size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

	double temp_sum = 0.0;
	for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
		temp_sum += (double) inputVec[i];
	}
	tmp[cta.thread_rank()] = temp_sum;

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	double beta = temp_sum;
	double temp;

	for (int i = tile32.size() / 2; i > 0; i >>= 1) {
		if (tile32.thread_rank() < i) {
			temp = tmp[cta.thread_rank() + i];
			beta += temp;
			tmp[cta.thread_rank()] = beta;
		}
		cg::sync(tile32);
	}
	cg::sync(cta);

	if (cta.thread_rank() == 0 && blockIdx.x < outputSize) {
		beta = 0.0;
		for (int i = 0; i < cta.size(); i += tile32.size()) {
			beta += tmp[i];
		}
		outputVec[blockIdx.x] = beta;
	}
}

__global__ void reduceFinal(double *inputVec, double *result, size_t inputSize)
{
	__shared__ double tmp[THREADS_PER_BLOCK];

	cg::thread_block cta = cg::this_thread_block();
	size_t globaltid = blockIdx.x * blockDim.x + threadIdx.x;

	double temp_sum = 0.0;
	for (int i = globaltid; i < inputSize; i += gridDim.x * blockDim.x) {
		temp_sum += (double) inputVec[i];
	}
	tmp[cta.thread_rank()] = temp_sum;

	cg::sync(cta);

	cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

	// do reduction in shared mem
	if ((blockDim.x >= 512) && (cta.thread_rank() < 256)) {
		tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 256];
	}

	cg::sync(cta);

	if ((blockDim.x >= 256) && (cta.thread_rank() < 128)) {
		tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 128];
	}

	cg::sync(cta);

	if ((blockDim.x >= 128) && (cta.thread_rank() < 64)) {
		tmp[cta.thread_rank()] = temp_sum = temp_sum + tmp[cta.thread_rank() + 64];
	}

	cg::sync(cta);

	if (cta.thread_rank() < 32) {
		// Fetch final intermediate sum from 2nd warp
		if (blockDim.x >= 64) temp_sum += tmp[cta.thread_rank() + 32];
		// Reduce final warp using shuffle
		for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
			temp_sum += tile32.shfl_down(temp_sum, offset);
		}
	}
	// write result for this block to global mem
	if (cta.thread_rank() == 0) result[0] = temp_sum;
}

void init_input(float *a, size_t size) {
	auto generator = []() {  return(rand() & 0xFF) / (float)RAND_MAX; };
	std::generate_n(a, size, generator);
}

void myRealHostNodeCallback(char const *graph_construction_mode, double result)
{
	std::cout << "Host callback in graph constructed by " << graph_construction_mode << ": result = " << result << std::endl;
	result = 0.0;  // reset the result
}

void CUDART_CB myHostNodeCallback(void *type_erased_data)
{
	auto *data = reinterpret_cast<std::pair<const char*, double*>*>(type_erased_data);
	auto graph_construction_mode = data->first;
	auto result = data->second;
	myRealHostNodeCallback(graph_construction_mode, *result);
}

void use(const cuda::device_t &device, const cuda::graph::template_t &graph, const char* how_created)
{
	std::cout << '\n'
		<< "Attempting use of a CUDA graph created using " << how_created << '\n'
		<< "----------------------------------------------------------------\n";
	std::cout << "Number of graph nodes = " << graph.num_nodes() << '\n';

	auto instance = cuda::graph::instantiate(graph);

	auto cloned = graph.clone();
	auto cloned_graph_instance = cuda::graph::instantiate(graph);

	auto stream_for_graph = cuda::stream::create(device, cuda::stream::async);

	for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
		std::cout
			<< "Launching an instance of the original graph: launch "
			<< (i+1) << " of " << GRAPH_LAUNCH_ITERATIONS << std::endl;
		instance.launch(stream_for_graph);
	}

	for (int i = 0; i < GRAPH_LAUNCH_ITERATIONS; i++) {
		std::cout
			<< "Launching an instance of the cloned graph: launch "
			<< (i+1) << " of " << GRAPH_LAUNCH_ITERATIONS << std::endl;
		cloned_graph_instance.launch(stream_for_graph);
	}
	std::cout << std::endl;
	stream_for_graph.synchronize();
}

void cudaGraphsManual(
	const cuda::device_t& device,
	span<float>  inputVec_h,
	span<float>  inputVec_d,
	span<double> outputVec_d,
	span<double> result_d)
{
	const char* graph_construction_mode = "explicit node and edge insertion calls";
	double result_h = 0.0;

	using node_kind_t = cuda::graph::node::kind_t;
	auto graph = cuda::graph::create();

	auto memcpy_node = [&] {
		cuda::memory::copy_parameters_t<3> copy_params;
		// TODO: Have the copy_parameters_t class be more like a builder.
		// And - accept sizes with dimensionality upto the copy params dimensionality
		copy_params.set_source<float>(inputVec_h);
		copy_params.set_destination<float>(inputVec_d);
		// TODO: Need to tweak the copy parameters class so that these next few lines are not necessary;
		// and to make sure we don't use the params without everything necessary being set
		copy_params.set_extent<float>(inputVec_h.size());
		copy_params.clear_offsets();
		copy_params.clear_rest();
		auto current_context = cuda::context::current::get();
		return graph.insert.node<node_kind_t::memory_copy>(current_context, copy_params);
	}();

	auto memset_node = [&] {
		cuda::graph::node::parameters_t<node_kind_t::memory_set> params;
		params.value = 0;
		params.width_in_bytes = 4;
		params.region = outputVec_d;
		return graph.insert.node<node_kind_t::memory_set>(params);
	}();

	auto reduce_node = [&] {
		auto reduce_kernel = cuda::kernel::get(device, reduce);
		auto launch_config = cuda::launch_config_builder()
			.grid_size(outputVec_d.size())
			.block_size(THREADS_PER_BLOCK)
			.build();
		auto kernel_arg_pointers = cuda::graph::make_kernel_argument_pointers(
			inputVec_d.data(), outputVec_d.data(), inputVec_d.size(), outputVec_d.size());
		auto kernel_node_args = cuda::graph::make_launch_primed_kernel(reduce_kernel, launch_config, kernel_arg_pointers);
		return graph.insert.node<node_kind_t::kernel_launch>(kernel_node_args);
	}();

	graph.insert.edge(memcpy_node, reduce_node);
	graph.insert.edge(memset_node, reduce_node);

	auto memset_result_node = [&] {
		cuda::graph::node::parameters_t<node_kind_t::memory_set> params;
		params.value = 0;
		params.width_in_bytes = 4;
		params.region = result_d;
		return graph.insert.node<node_kind_t::memory_set>(params);
	}();

	auto reduce_final_node = [&] {
		auto kernel = cuda::kernel::get(device, reduceFinal);
		auto launch_config = cuda::launch_config_builder()
			.grid_size(1)
			.block_size(THREADS_PER_BLOCK)
			.build();
		auto arg_ptrs = cuda::graph::make_kernel_argument_pointers(outputVec_d.data(), result_d.data(), outputVec_d.size());
		return graph.insert.node<node_kind_t::kernel_launch>(kernel, launch_config, arg_ptrs);
	}();

	graph.insert.edge(reduce_node, reduce_final_node);
	graph.insert.edge(memset_result_node, reduce_final_node);

	auto memcpy_result_node = [&] {
		cuda::memory::copy_parameters_t<3> copy_params;
		// TODO: Have the copy_parameters_t class be more like a builder.
		// And - accept sizes with dimensionality upto the copy params dimensionality
		copy_params.set_source<double>(result_d);
		copy_params.set_destination<double>(&result_h, 1);
		copy_params.set_extent<double>(1);
		copy_params.clear_offsets();
		copy_params.clear_rest();
		return graph.insert.node<node_kind_t::memory_copy>(copy_params);
	}();

	graph.insert.edge(reduce_final_node, memcpy_result_node);

	auto host_function_data = std::make_pair(graph_construction_mode, &result_h);
	auto host_function_node = graph.insert.node<node_kind_t::host_function_call>(myHostNodeCallback, &host_function_data);

	graph.insert.edge(memcpy_result_node, host_function_node);

	use(device, graph, graph_construction_mode);
}

void cudaGraphsUsingStreamCapture(
	const cuda::device_t& device,
	span<float>  inputVec_h,
	span<float>  inputVec_d,
	span<double> outputVec_d,
	span<double> result_d)
{
	const char* graph_construction_mode = "stream capture";
	double result_h = 0.0;

	using cuda::stream::async;
	auto stream_1 = cuda::stream::create(device, async);
	auto stream_2 = cuda::stream::create(device, async);
	auto stream_3 = cuda::stream::create(device, async);

	auto fork_stream_event = cuda::event::create(device);
	auto reduce_output_memset_event = cuda::event::create(device);
	auto final_result_memset_event = cuda::event::create(device);

	stream_1.begin_capture(cuda::stream::capture::mode_t::global);

	stream_1.enqueue.event(fork_stream_event);
	stream_2.enqueue.wait(fork_stream_event);
	stream_3.enqueue.wait(fork_stream_event);

	stream_1.enqueue.copy(inputVec_d, inputVec_h);
	stream_2.enqueue.memzero(outputVec_d);

	stream_2.enqueue.event(reduce_output_memset_event);
	stream_3.enqueue.memzero(result_d);
	stream_3.enqueue.event(final_result_memset_event);

	stream_1.enqueue.wait(reduce_output_memset_event);

	auto launch_config = cuda::launch_config_builder()
		.grid_dimensions(outputVec_d.size())
		.block_dimensions(THREADS_PER_BLOCK)
		.build();

	stream_1.enqueue.kernel_launch(reduce, launch_config,
		inputVec_d.data(), outputVec_d.data(), inputVec_d.size(), outputVec_d.size());

	stream_1.enqueue.wait(final_result_memset_event);

	launch_config =  cuda::launch_config_builder()
		.grid_dimensions(1)
		.block_dimensions(THREADS_PER_BLOCK)
		.build();
	stream_1.enqueue.kernel_launch(reduceFinal, launch_config,
		outputVec_d.data(), result_d.data(), outputVec_d.size());

	stream_1.enqueue.copy(&result_h, result_d);

	auto callback = [&]() { myRealHostNodeCallback(graph_construction_mode, result_h); };
	stream_1.enqueue.host_invokable(callback);

	auto graph = stream_1.end_capture();

	use(device, graph, graph_construction_mode);
}

[[noreturn]] bool die_(const ::std::string& message)
{
	::std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	size_t size { 1 << 24 }; // number of elements to reduce
	size_t maxBlocks { 512 };

	if (cuda::device::count() == 0) {
		die_("No CUDA devices on this system");
	}

	// Being very cavalier about our command-line arguments here...
	cuda::device::id_t device_id = (argc > 1) ?
		::std::stoi(argv[1]) : cuda::device::default_device_id;

	auto device = cuda::device::get(device_id);

	std::cout
		<< size << " elements\n"
		<< "threads per block  = " << THREADS_PER_BLOCK << '\n'
		<< "Graph Launch iterations = " << GRAPH_LAUNCH_ITERATIONS << '\n'
		<< std::flush;

	auto inputVec_h = cuda::memory::host::make_unique<float[]>(size);
	auto inputVec_d = cuda::memory::device::make_unique<float[]>(device, size);
	auto outputVec_d = cuda::memory::device::make_unique<double[]>(device, maxBlocks);
	auto result_d = cuda::memory::device::make_unique<double>(device);

	init_input(inputVec_h.get(), size);

	auto result_verification = ::std::accumulate(
#if __cplusplus >= 201712L
		::std::execution::par_unseq
#endif
		inputVec_h.get(), inputVec_h.get() + size, 0.0);
	std::cout << "Expected result = " << result_verification << '\n';

	device.synchronize();

	cudaGraphsManual(
		device,
		{ inputVec_h.get(), size },
		{ inputVec_d.get(), size },
		{ outputVec_d.get(), maxBlocks },
		{ result_d.get(), 1 }
		);

	cudaGraphsUsingStreamCapture(
		device,
		{ inputVec_h.get(), size },
		{ inputVec_d.get(), size },
		{ outputVec_d.get(), maxBlocks },
		{ result_d.get(), 1 }
		);

	std::cout << "\n\nSUCCESS\n";
}
