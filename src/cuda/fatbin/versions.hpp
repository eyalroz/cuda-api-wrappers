/**
 * @file
 *
 * @brief Wrappers for Runtime API functions involving versions -
 * of the CUDA runtime and of the CUDA driver. Also defines a @ref cuda::version_t
 * class for working with such versions (as they are not really single
 * numbers) - which is what the wrappers return.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_FATBIN_VERSIONS_HPP_
#define CUDA_API_WRAPPERS_FATBIN_VERSIONS_HPP_

#include "error.hpp"

#if CUDA_VERSION >= 12040
#include <nvFatbin.h>
#endif

#include <limits>

namespace cuda {
namespace version_numbers {

#if CUDA_VERSION >= 12040

inline version_t fatbin() {
	unsigned int major { 0 }, minor { 0 };

	auto status = nvFatbinVersion(&major, &minor);
	throw_if_error_lazy(status, "Failed obtaining the nvfatbin library version");
#ifndef NDEBUG
	if (   (major == 0) or (major > ::std::numeric_limits<int>::max())
		or (minor == 0) or (minor > ::std::numeric_limits<int>::max())) {
		throw ::std::logic_error("Invalid version encountered: ("
			+ ::std::to_string(major) + ", " + ::std::to_string(minor) + ')' );
	}
#endif
	return version_t{ static_cast<int>(major), static_cast<int>(minor) };
}
#endif // CUDA_VERSION >= 12040

} // namespace version_numbers
} // namespace cuda

#endif // CUDA_API_WRAPPERS_FATBIN_VERSIONS_HPP_
