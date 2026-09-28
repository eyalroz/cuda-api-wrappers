/**
 * @file
 *
 * @brief Type definitions used in relation to creating fatbin files using
 * NVIDIA's fatbin creating library (nvFatbin).
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_FATBIN_BUILDER_TYPES_HPP_
#define CUDA_API_WRAPPERS_FATBIN_BUILDER_TYPES_HPP_

#if CUDA_VERSION >= 12040

#include "../api/types.hpp"

#include <nvFatbin.h>

namespace cuda_ {

namespace fatbin_builder {

using handle_t = nvFatbinHandle;
using status_t = nvFatbinResult;

} // namespace fatbin_builder

} // namespace cuda_

#endif // CUDA_VERSION >= 12040

#endif /* CUDA_API_WRAPPERS_FATBIN_BUILDER_TYPES_HPP_ */
