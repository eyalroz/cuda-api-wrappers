#include "jacobi_kernels.cuh"
#include "jacobi.h"

#include <vector>
#include <iomanip>
#include <iostream>
#include <cuda/api.hpp>

static void finalize_error(
	const cuda::stream_t& stream, span<double> d_sum, const cuda::launch_configuration_t& launch_config,
	double& sum, int k, const span<double> x_to_overwrite)
{
	stream.enqueue.memzero(d_sum);
	auto final_error_launch_config = launch_config;
	final_error_launch_config.dimensions.grid.x = (N_ROWS / final_error_launch_config.dimensions.block.x) + 1;
	auto warps_per_block = final_error_launch_config.dimensions.block.x / cuda::warp_size;
	final_error_launch_config.dynamic_shared_memory_size = (warps_per_block + 1) * sizeof(double);
	// TODO: Double-check the original source to ensure we're using the right x here
	stream.enqueue.kernel_launch(finalError, final_error_launch_config, x_to_overwrite.data(), d_sum.data());
	stream.enqueue.copy(&sum, d_sum);
	stream.synchronize();
	report_error_sum("GPU", k + 1, sum);
}

template<>
double do_jacobi_inner<computation_method_t::graph_with_set_kernel_params>(
	const cuda::device_t &device,
	const cuda::stream_t &stream,
	span<float  const> A,
	span<double const> b,
	float convergence_threshold,
	int num_iterations,
	span<double> x,
	span<double> x_new,
	span<double> d_sum)
{
	auto launch_config = cuda::launch_config_builder()
		.block_size(256)
		.grid_dimensions((N_ROWS / ROWS_PER_CTA) + 2, 1, 1)
		.build();

	double sum;

	auto graph = cuda::graph::create();

	using cuda::graph::node::kind_t;

	auto memset_node = [&] {
		cuda::graph::node::parameters_t<kind_t::memory_set> params;
		params.value = 0;
		params.width_in_bytes = 4;
		params.region = d_sum;
		return graph.insert.node<kind_t::memory_set>(params);
	}();

	auto jacobi_kernel = cuda::kernel::get(device, JacobiMethod);
	struct { cuda::graph::node::parameters_t<kind_t::kernel_launch>  odd, even; } kernel_params = {
		{ jacobi_kernel, launch_config, cuda::graph::make_kernel_argument_pointers(A, b, convergence_threshold, x, x_new, d_sum) },
		{ jacobi_kernel, launch_config, cuda::graph::make_kernel_argument_pointers(A, b, convergence_threshold, x_new, x, d_sum) },
	};
	auto jacobi_kernel_node = graph.insert.node<kind_t::kernel_launch>(kernel_params.even);

	graph.insert.edge(memset_node, jacobi_kernel_node);

	auto memcpy_node = [&] {
		cuda::memory::copy_parameters_t<3> params;
		params.set_source(d_sum);
		params.set_destination(&sum, 1);
		params.set_extent<double>(1);
		params.clear_offsets();
		params.clear_rest();
		return graph.insert.node<cuda::graph::node::kind_t::memcpy>(params);
	}();

	graph.insert.edge(jacobi_kernel_node, memcpy_node);


	cuda::graph::instance_t instance = graph.instantiate();

// 	::std::cout << "settings node params for the kernel node with k ==  " << k << " and params.marshalled_arguments.size() = "
//			  << params.marshalled_arguments.size() << std::endl;

	for (int k = 0; k < num_iterations; k++) {
		instance.launch(stream);
		stream.synchronize();

		if (sum <= convergence_threshold) {
			auto x_to_overwrite = ((k & 1) == 0) ? x : x_new;
			finalize_error(stream, d_sum, launch_config, sum, k, x_to_overwrite);
			break;
		}
		// Odd iterations have an even value of k, since we start with k == 0;
		// but - here we sent
		const auto& next_iteration_params = ((k & 1) == 0) ? kernel_params.even : kernel_params.odd;
		instance.set_node_parameters<kind_t::kernel_launch>(jacobi_kernel_node, next_iteration_params);
	}
	return sum;
}

template<>
double do_jacobi_inner<computation_method_t::graph_with_exec_update>(
	const cuda::device_t &,
	const cuda::stream_t &stream,
	span<float  const> A,
	span<double const> b,
	float convergence_threshold,
	int num_iterations,
	span<double> x,
	span<double> x_new,
	span<double> d_sum)
{
	auto launch_config = cuda::launch_config_builder()
		.block_size(256)
		.grid_dimensions((N_ROWS / ROWS_PER_CTA) + 2, 1, 1)
		.build();

	::std::unique_ptr<cuda::graph::instance_t> instance_ptr{};

	double sum = 0.0;
	for (int k = 0; k < num_iterations; k++) {
		stream.begin_capture(cuda::stream::capture::mode_t::global);
		stream.enqueue.memzero(d_sum);
		auto x_to_read = ((k & 1) == 0) ? x : x_new;
		auto x_to_overwrite = ((k & 1) == 0) ? x_new : x;
		stream.enqueue.kernel_launch(JacobiMethod, launch_config,
			A.data(), b.data(), convergence_threshold, x_to_read.data(), x_to_overwrite.data(), d_sum.data());
		stream.enqueue.copy(&sum, d_sum);
		auto graph = stream.end_capture();

		if (instance_ptr == nullptr) {
			auto instance = graph.instantiate();
			instance_ptr.reset(new cuda::graph::instance_t{::std::move(instance)});
		}
		else {
			instance_ptr->update(graph);
			// Note: The original code tried to re-instantiate if the update
			// of the instance failed, we don't do this.
		}
		stream.enqueue.graph_launch(*instance_ptr);
		stream.synchronize();

		if (sum <= convergence_threshold) {
			finalize_error(stream, d_sum, launch_config, sum, k, x_to_overwrite);
			break;
		}
	}

	return sum;
}

template<>
double do_jacobi_inner<computation_method_t::non_graph_gpu>(
	const cuda::device_t &,
	const cuda::stream_t &stream,
	span<float const> A,
	span<double const> b,
	float convergence_threshold,
	int num_iterations,
	span<double> x,
	span<double> x_new,
	span<double> d_sum)
{
	auto launch_config = cuda::launch_config_builder()
		.block_size(256)
		.grid_dimensions((N_ROWS / ROWS_PER_CTA) + 2, 1, 1)
		.build();

	double sum;
	for (int k = 0; k < num_iterations; k++) {
		stream.enqueue.memzero(d_sum);
		auto x_to_read = ((k & 1) == 0) ? x : x_new;
		auto x_to_overwrite = ((k & 1) == 0) ? x_new : x;
		stream.enqueue.kernel_launch(JacobiMethod, launch_config,
			A.data(), b.data(), convergence_threshold, x_to_read.data(), x_to_overwrite.data(), d_sum.data());
		stream.enqueue.copy(&sum, d_sum);
		stream.synchronize();

		if (sum <= convergence_threshold) {
			finalize_error(stream, d_sum, launch_config, sum, k, x_to_overwrite);
			break;
		}
	}

	return sum;
}

