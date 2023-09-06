/**
 * @file
 *
 * @brief Type definitions used in relation to NVML, NVIDIA's
 * (GPU device) Management Library.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVML_TYPES_HPP_
#define CUDA_API_WRAPPERS_NVML_TYPES_HPP_

#include "../api/types.hpp"

#include <nvml.h>

/*
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
*/

namespace cuda {

namespace nvml {

namespace device {

namespace clock {

/// This is how NVML takes MHz frequency values
using mhz_frequency_t = unsigned int;

struct mhz_frequency_range_t {
	mhz_frequency_t min;
	mhz_frequency_t max;
};

enum class clocked_entity_t {
	graphics = NVML_CLOCK_GRAPHICS,
	cores = NVML_CLOCK_SM,
	memory = NVML_CLOCK_MEM,
	video = NVML_CLOCK_VIDEO,
	sm = cores,
	symmetric_multiprocessors = sm,
};

enum class scope_t {
	global_and_immediate,
	gpu = global_and_immediate,
	future_new_contexts,
	application = future_new_contexts
};

struct mem_and_sm_frequencies_t {
	mhz_frequency_t memory;
	mhz_frequency_t sm;
};

nvmlClockId_t raw_clock_id_of(scope_t scope)
{
	return static_cast<nvmlClockId_t>(scope);
}


enum class kind_t {
	current             = NVML_CLOCK_ID_CURRENT,             /// Current actual clock value
	application         = NVML_CLOCK_ID_APP_CLOCK_TARGET,    /// Target application clock <- TODO: What does that mean?
	default_application = NVML_CLOCK_ID_APP_CLOCK_DEFAULT,   /// Default application clock target
	customer_boost_max  = NVML_CLOCK_ID_CUSTOMER_BOOST_MAX,  /// OEM-defined maximum clock rate <- TODO: What does that mean?
};

} // namespace clock

using handle_t = nvmlDevice_t;

enum : const cuda::device::id_t { none = NVML_VALUE_NOT_AVAILABLE };

constexpr const handle_t no_handle = nullptr;

} // namespace device

/// The return status of an NVML API call
using status_t =  nvmlReturn_t;

} // namespace nvml

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVML_TYPES_HPP_ 
