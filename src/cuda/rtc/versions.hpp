/**
 * @file
 *
 * @brief Complementing file for `cuda/api/versions.hpp` for obtaining
 * a version number for the NVRTC library
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVRTC_VERSIONS_HPP_
#define CUDA_API_WRAPPERS_NVRTC_VERSIONS_HPP_

#include <cuda/rtc/error.hpp>
#include <cuda/api/versions.hpp>

namespace cuda {

namespace version_numbers {

/**
 * Obtain the NVRTC library version
 */
inline version_t nvrtc() {
	version_t version;
	auto status = nvrtcVersion(&version.major, &version.minor);
	throw_if_error(status, "Failed obtaining the NVRTC library version");
	return version;
}

} // namespace version_numbers
} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_VERSIONS_HPP_
