/**
 * @file
 *
 * @brief Type definitions used in relation to NVML, NVIDIA's
 * (GPU device) Management Library.
 */
#pragma once
#ifndef SRC_CUDA_NVML_TYPES_HPP_
#define SRC_CUDA_NVML_TYPES_HPP_

#include "../api/types.hpp"

#include <nvml.h>

#if __cplusplus >= 201703L
#include <string_view>
namespace cuda {
using string_view = ::std::string_view;
}
#else
#include <cuda/rtc/detail/string_view.hpp>
namespace cuda {
using string_view = bpstd::string_view;
}
#endif

namespace cuda {

namespace nvml {

using device_t = nvmlDevice_t;

/// Return status of an NVML API call
using status_t =  nvmlReturn_t;

} // namespace nvml

} // namespace cuda

#endif /* SRC_CUDA_NVML_TYPES_HPP_ */
