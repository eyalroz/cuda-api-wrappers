/**
 * @file nvrtc.hpp
 *
 * @brief A single file which triggers inclusion all of the CUDA
 * NVRTC library API wrappers, and in turns, many of the CUDA runtime
 * and driver API headers.
 */
#pragma once
#ifndef CUDA_NVRTC_WRAPPERS_HPP_
#define CUDA_NVRTC_WRAPPERS_HPP_

static_assert(__cplusplus >= 201103L, "The CUDA NVTX API wrappers can only be compiled with C++11 or a later version of the C++ language standard");

#include <cuda/nvrtc/error.hpp>
#include <cuda/nvrtc/compilation_options.hpp>
#include <cuda/nvrtc/versions.hpp>
#include <cuda/nvrtc/program.hpp>

#endif // CUDA_NVRTC_WRAPPERS_HPP_
