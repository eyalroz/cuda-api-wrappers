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

#include "jacobi.h"

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// 8 Rows of square-matrix A processed by each CTA.
// This can be max 32 and only power of 2 (i.e., 2/4/8/16/32).
#define ROWS_PER_CTA 8

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double *address, double val) {
  unsigned long long int *address_as_ull = (unsigned long long int *)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

static __global__ void JacobiMethod(
	float const *A,
	double const *b,
    float const convergence_threshold,
	double const *x,
    double *x_new,
	double *sum)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ double x_shared[N_ROWS];  // N_ROWS == n
  __shared__ double b_shared[ROWS_PER_CTA + 1];

  for (int i = threadIdx.x; i < N_ROWS; i += blockDim.x) {
    x_shared[i] = x[i];
  }

  if (threadIdx.x < ROWS_PER_CTA) {
    int k = threadIdx.x;
    for (int i = k + (blockIdx.x * ROWS_PER_CTA);
         (k < ROWS_PER_CTA) && (i < N_ROWS);
         k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
      b_shared[i % (ROWS_PER_CTA + 1)] = b[i];
    }
  }

  cg::sync(cta);

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  for (int k = 0, i = blockIdx.x * ROWS_PER_CTA;
       (k < ROWS_PER_CTA) && (i < N_ROWS); k++, i++) {
    double rowThreadSum = 0.0;
    for (int j = threadIdx.x; j < N_ROWS; j += blockDim.x) {
      rowThreadSum += (A[i * N_ROWS + j] * x_shared[j]);
    }

    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      rowThreadSum += tile32.shfl_down(rowThreadSum, offset);
    }

    if (tile32.thread_rank() == 0) {
      atomicAdd(&b_shared[i % (ROWS_PER_CTA + 1)], -rowThreadSum);
    }
  }

  cg::sync(cta);

  if (threadIdx.x < ROWS_PER_CTA) {
    cg::thread_block_tile<ROWS_PER_CTA> tile8 =
        cg::tiled_partition<ROWS_PER_CTA>(cta);
    double temp_sum = 0.0;

    int k = threadIdx.x;

    for (int i = k + (blockIdx.x * ROWS_PER_CTA);
         (k < ROWS_PER_CTA) && (i < N_ROWS);
         k += ROWS_PER_CTA, i += ROWS_PER_CTA) {
      double dx = b_shared[i % (ROWS_PER_CTA + 1)];
      dx /= A[i * N_ROWS + i];

      x_new[i] = (x_shared[i] + dx);
      temp_sum += fabs(dx);
    }

    for (int offset = tile8.size() / 2; offset > 0; offset /= 2) {
      temp_sum += tile8.shfl_down(temp_sum, offset);
    }

    if (tile8.thread_rank() == 0) {
      atomicAdd(sum, temp_sum);
    }
  }
}

// Thread block size for finalError kernel should be multiple of 32
static __global__ void finalError(double const *x, double *g_sum) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  extern __shared__ double warpSum[];
  double sum = 0.0;

  int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = globalThreadId; i < N_ROWS; i += blockDim.x * gridDim.x) {
    double d = x[i] - 1.0;
    sum += fabs(d);
  }

  cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

  for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
    sum += tile32.shfl_down(sum, offset);
  }

  if (tile32.thread_rank() == 0) {
    warpSum[threadIdx.x / warpSize] = sum;
  }

  cg::sync(cta);

  double blockSum = 0.0;
  if (threadIdx.x < (blockDim.x / warpSize)) {
    blockSum = warpSum[threadIdx.x];
  }

  if (threadIdx.x < 32) {
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
      blockSum += tile32.shfl_down(blockSum, offset);
    }
    if (tile32.thread_rank() == 0) {
      atomicAdd(g_sum, blockSum);
    }
  }
}
