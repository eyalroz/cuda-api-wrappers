/**
 * @file
 *
 * @brief Type definitions used in CUDA real-time compilation work wrappers.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_FATBIN_BUILDER_TYPES_HPP_
#define CUDA_API_WRAPPERS_FATBIN_BUILDER_TYPES_HPP_

#if CUDA_VERSION >= 12040

#include "../api/types.hpp"

#include <nvFatbin.h>

namespace cuda {

namespace fatbin_builder {

using handle_t = nvFatbinHandle;
using status_t = nvFatbinResult;

} // namespace fatbin_builder

} // namespace cuda

#endif // CUDA_VERSION >= 12040

#endif /* CUDA_API_WRAPPERS_FATBIN_BUILDER_TYPES_HPP_ */
