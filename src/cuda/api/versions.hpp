#pragma once
#ifndef CUDA_RUNTIME_API_WRAPPERS_VERSIONS_HPP_
#define CUDA_RUNTIME_API_WRAPPERS_VERSIONS_HPP_

#include <cuda/api/error.hpp>

namespace cuda {

using driver_version_t = int;
enum : driver_version_t { no_driver_installed = 0 };
using runtime_version_t = int;

namespace version_numbers {

/**
 * @return 0 is no CUDA driver is installed; the driver version number otherwise
 */
driver_version_t driver() {
	driver_version_t version;
	auto status = cudaDriverGetVersion(&version);
	throw_if_error(status, "Failed obtaining the CUDA driver version");
	return version;
}

/**
 * @note unlike {@ref driver()}, 0 cannot be returned, as if a runtime was
 * not installed its version could not be checked.
 */
runtime_version_t runtime() {
	runtime_version_t version;
	auto status = cudaRuntimeGetVersion(&version);
	throw_if_error(status, "Failed obtaining the CUDA runtime version");
	return version;
}

} // namespace version_numbers
} // namespace cuda {

#endif /* CUDA_RUNTIME_API_WRAPPERS_VERSIONS_HPP_ */
