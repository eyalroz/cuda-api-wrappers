/**
 * @file
 *
 * @brief A single file which triggers inclusion all of the CUDA
 * NVRTC library API wrappers, and in turns, many of the CUDA runtime
 * and driver API headers.
 */
#pragma once
#ifndef CUDA_NVRTC_WRAPPERS_HPP_
#define CUDA_NVRTC_WRAPPERS_HPP_

#if (__cplusplus < 201103L && (!defined(_MSVC_LANG) || _MSVC_LANG < 201103L))
#error "The CUDA API headers can only be compiled with C++11 or a later version of the C++ language standard"
#endif

#include "rtc/error.hpp"
#include "rtc/compilation_options.hpp"
#include "rtc/versions.hpp"
#include "rtc/compilation_output.hpp"
#include "rtc/program.hpp"

#endif // CUDA_NVRTC_WRAPPERS_HPP_
