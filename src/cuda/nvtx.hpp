/**
 * @file
 *
 * @brief A single file which includes, in turn, all of the CUDA
 * NVTX library API wrappers and related headers.
 */
#pragma once
#ifndef CUDA_NVTX_WRAPPERS_HPP_
#define CUDA_NVTX_WRAPPERS_HPP_

#if (__cplusplus < 201103L && (!defined(_MSVC_LANG) || _MSVC_LANG < 201103L))
#error "The CUDA API headers can only be compiled with C++11 or a later version of the C++ language standard"
#endif

#include <cuda/nvtx/profiling.hpp>

#endif // CUDA_NVTX_WRAPPERS_HPP_
