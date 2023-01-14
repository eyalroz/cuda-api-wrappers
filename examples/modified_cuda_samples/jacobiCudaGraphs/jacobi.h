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

#ifndef JACOBI_H
#define JACOBI_H

#define N_ROWS 512

#include <cuda/api.hpp>

#if __cplusplus >= 202001L
using span = std::span;
#else
using cuda::span;
#endif

#define N_ROWS 512

enum computation_method_t {
	graph_with_set_kernel_params = 0,
	graph_with_exec_update = 1,
	non_graph_gpu = 2,
	cpu = 3
};

inline const char* method_name(computation_method_t method)
{
	static const char* method_names[] = {
		"graph_with_set_kernel_params",
		"graph_with_exec_update",
		"non_graph_gpu",
		"cpu"
	};
	return method_names[method];
}

void report_error_sum(const char* where, int num_iterations, double sum_on_cpu);

template <computation_method_t Method>
double do_jacobi_inner(
	const cuda:: device_t& device,
	const cuda::stream_t &stream,
	span<float  const> A,
	span<double const> b,
	float conv_threshold,
	int num_iterations,
	span<double> x,
	span<double> x_new,
	span<double> d_sum);


#endif // JACOBI_H