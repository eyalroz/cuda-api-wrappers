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

// This sample demonstrates Instantiated CUDA Graph Update
// with Jacobi Iterative Method in 3 different methods:
// 1 - JacobiMethodGpuCudaGraphExecKernelSetParams() - CUDA Graph with
// cudaGraphExecKernelNodeSetParams() 2 - JacobiMethodGpuCudaGraphExecUpdate() -
// CUDA Graph with cudaGraphExecUpdate() 3 - JacobiMethodGpu() - Non CUDA Graph
// method

// Jacobi method on a linear system A*x = b,
// where A is diagonally dominant and the exact solution consists
// of all ones.
// The dimension N_ROWS is included in jacobi.h

#include <cuda/api.hpp>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include "jacobi.h"

[[noreturn]] bool die_(const ::std::string& message = "")
{
	if (not message.empty()) { ::std::cerr << message << std::endl; }
	exit(EXIT_FAILURE);
}

// creates N_ROWS x N_ROWS matrix A with N_ROWS+1 on the diagonal and 1
// elsewhere. The elements of the right hand side b all equal 2*n, hence the
// exact solution x to A*x = b is a vector of ones.
void createLinearSystem(span<float> A, span<double> b)
{
	int i, j;
	for (i = 0; i < N_ROWS; i++) {
		b[i] = 2.0 * N_ROWS;
		for (j = 0; j < N_ROWS; j++) A[i * N_ROWS + j] = 1.0;
		A[i * N_ROWS + i] = N_ROWS + 1.0;
	}
}

// Run the Jacobi method for A*x = b on CPU.
std::pair<int, double> do_jacobi_on_cpu(span<float> A, span<double> b, float convergence_threshold, int max_iterations)
{
	auto x_ = std::array<double, N_ROWS>{};
	span<double> x = { x_.data(), N_ROWS };
	auto x_new = std::array<double, N_ROWS> { 0 };
	int k;

	for (k = 0; k < max_iterations; k++) {
		double sum = 0.0;
		for (int i = 0; i < N_ROWS; i++) {
			double temp_dx = b[i];
			for (int j = 0; j < N_ROWS; j++) temp_dx -= A[i * N_ROWS + j] * x[j];
			temp_dx /= A[i * N_ROWS + i];
			x_new[i] += temp_dx;
			sum += ::std::fabs(temp_dx);
		}
		std::copy(x_new.cbegin(), x_new.cend(), x.begin());

		if (sum <= convergence_threshold) break;
	}

	double sum = std::accumulate(std::begin(x), std::end(x), 0.0,
		[](double accumulation_so_far, double element) {
			return accumulation_so_far + ::std::fabs(element - 1.0);
		} );
//	sdkStopTimer(&timerCPU);

	report_error_sum("CPU", k+1, sum);
	std::cout << '\n';

	return {k + 1, sum};
}

template <computation_method_t Method>
bool do_gpu_jacobi(
	const cuda::device_t& device,
	const cuda::stream_t& stream,
	span<float  const> A,
	span<double const> b,
	span<float      > d_A,
	span<double     > d_b,
	float convergence_threshold,
	int max_iterations,
	span<double     > d_x,
	span<double     > d_x_new,
	span<double     > d_sum,
	double sum_on_cpu
)
{
	stream.enqueue.memzero(d_x);
	stream.enqueue.memzero(d_x_new);
	stream.enqueue.copy(d_A, A);
	stream.enqueue.copy(d_b, b);

//	sdkCreateTimer(&timerGpu);
//	sdkStartTimer(&timerGpu);

	std::cout << "Jacobi computation with method " << method_name(Method) << ":\n";
	double sum = do_jacobi_inner<Method>(device, stream, d_A, d_b, convergence_threshold, max_iterations, d_x, d_x_new, d_sum);

	bool success = ::std::fabs(sum_on_cpu - sum) < convergence_threshold;
	std::cout << (success ? "PASSED" : "FAILED") << "\n\n";
	return success;
}

void report_error_sum(const char* where, int num_iterations, double sum_on_cpu)
{
	std::cout << where << " iterations : " << num_iterations << '\n';
	auto cout_flags (std::cout.flags());
	std::cout << where << " error : " << std::setprecision(3) << std::scientific << sum_on_cpu << '\n';
	std::cout.setf(cout_flags);
}

int main(int argc, char **argv)
{
	// Being very cavalier about our command-line arguments here...
	cuda::device::id_t device_id = (argc > 1) ? ::std::stoi(argv[1]) : cuda::device::default_device_id;
	auto device = cuda::device::get(device_id);

	auto b_ = cuda::memory::host::make_unique<double[]>(N_ROWS);
	auto A_ = cuda::memory::host::make_unique<float[]>(N_ROWS * N_ROWS);

	span<double> b = { b_.get(), N_ROWS };
	span<float> A = { A_.get(), N_ROWS * N_ROWS };

	createLinearSystem(A, b);

	float convergence_threshold = 1.0e-2;
	int max_num_iterations = 4 * N_ROWS * N_ROWS;

	// create timer
//	StopWatchInterface *timerCPU = NULL, *timerGpu = NULL;
//	sdkCreateTimer(&timerCPU);

//	sdkStartTimer(&timerCPU);
	auto num_iterations_and_sum = do_jacobi_on_cpu(A, b, convergence_threshold, max_num_iterations);
	auto num_iterations = num_iterations_and_sum.first;
	auto sum_on_cpu = num_iterations_and_sum.second;

//	printf("CPU Processing time: %f (ms)\n", sdkGetTimerValue(&timerCPU));

	auto d_A_ = cuda::memory::device::make_unique<float[]>(device, N_ROWS * N_ROWS);
	auto d_b_ = cuda::memory::device::make_unique<double[]>(device, N_ROWS);
	auto d_x_ = cuda::memory::device::make_unique<double[]>(device, N_ROWS);
	auto d_x_new_ = cuda::memory::device::make_unique<double[]>(device, N_ROWS);
	auto d_sum_ = cuda::memory::device::make_unique<double>(device);
	span<float> d_A = { d_A_.get(), N_ROWS * N_ROWS};
	span<double> d_b = { d_b_.get(), N_ROWS };
	span<double> d_x = { d_x_.get(), N_ROWS };
	span<double> d_x_new = { d_x_new_.get(), N_ROWS };
	span<double> d_sum = { d_sum_.get(), 1 };
	auto stream = cuda::stream::create(device, cuda::stream::async);

	do_gpu_jacobi<graph_with_set_kernel_params>(
		device,	stream, A, b, d_A, d_b, convergence_threshold, num_iterations, d_x, d_x_new, d_sum, sum_on_cpu) or die_();
	do_gpu_jacobi<graph_with_exec_update>(
		device,	stream, A, b, d_A, d_b, convergence_threshold, num_iterations, d_x, d_x_new, d_sum, sum_on_cpu) or die_();
	do_gpu_jacobi<non_graph_gpu>(
		device,	stream, A, b, d_A, d_b, convergence_threshold, num_iterations, d_x, d_x_new, d_sum, sum_on_cpu) or die_();
}

