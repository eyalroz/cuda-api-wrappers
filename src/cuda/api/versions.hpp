#pragma once
#ifndef CUDA_API_WRAPPERS_VERSIONS_HPP_
#define CUDA_API_WRAPPERS_VERSIONS_HPP_

#include "cuda/api/error.hpp"

namespace cuda {

using cuda_version_t = int;

enum : cuda_version_t { no_driver_installed = 0 };

namespace version_numbers {

/**
 * @return If an nVIDIA GPU driver is installed on this system,
 * the maximum CUDA version it supports is returned. If no nVIDIA
 * GPU driver is installed, {@ref no_version_supported}} is returned.
 */
cuda_version_t maximum_supported_by_driver() {
	cuda_version_t version;
	auto status = cudaDriverGetVersion(&version);
	throw_if_error(status, "Failed obtaining the maximum CUDA version supported by the nVIDIA GPU driver");
	return version;
}

/**
 * @note unlike {@ref maximum_supported_by_driver()}, 0 cannot be returned,
 * as we are actually using the runtime to obtain the version, so it does
 * have _some_ version.
 */
cuda_version_t runtime() {
	cuda_version_t version;
	auto status = cudaRuntimeGetVersion(&version);
	throw_if_error(status, "Failed obtaining the CUDA runtime version");
	return version;
}

} // namespace version_numbers
} // namespace cuda {

#endif /* CUDA_API_WRAPPERS_VERSIONS_HPP_ */
