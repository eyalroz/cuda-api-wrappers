/**
 * @file
 *
 * @brief A single file which includes, in turn, the CUDA
 * fatbin creator library API wrappers and related headers.
 */
#pragma once
#ifndef CUDA_FATBIN_WRAPPERS_HPP_
#define CUDA_FATBIN_WRAPPERS_HPP_

#if CUDA_VERSION >= 12040

#include "fatbin/types.hpp"
#include "fatbin/error.hpp"
#include "fatbin/builder_options.hpp"
#include "fatbin/builder.hpp"

#endif // CUDA_VERSION >= 12040

#endif // CUDA_FATBIN_WRAPPERS_HPP_
