/**
 * @file nvtx.hpp
 *
 * @brief A single file which includes, in turn, all of the CUDA
 * NVTX library API wrappers and related headers.
 */
#pragma once
#ifndef CUDA_NVTX_WRAPPERS_HPP_
#define CUDA_NVTX_WRAPPERS_HPP_

static_assert(__cplusplus >= 201103L, "The CUDA NVTX API wrappers can only be compiled with C++11 or a later version of the C++ language standard");

#include <cuda/nvtx/profiling.hpp>

#endif // CUDA_NVTX_WRAPPERS_HPP_
