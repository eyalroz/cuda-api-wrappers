/**
 * @file device_count.hpp
 *
 * @brief A wrapper for the function counting the number of
 * available devices.
 *
 * @note This probably should not merit its own file, but I haven't
 * found another file to put it in - unless we also want to pull
 * in {@ref device.hpp}, which would be overkill I think.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DEVICE_COUNT_HPP_
#define CUDA_API_WRAPPERS_DEVICE_COUNT_HPP_

#include <cuda/api/types.hpp>
#include <cuda_runtime_api.h>

namespace cuda {
namespace device {

/**
 * Get the number of CUDA devices usable on the system (with the current CUDA
 * library and kernel driver)
 *
 * @note This _should_ be returning an unsigned value; unfortunately, device::id_t  is
 * signed in CUDA for some reason and we maintain compatibility (although this might
 * change in the future). So... the returned type is the same as in cudaGetDeviceCount,
 * a signed integer.
 *
 * @return the number of CUDA devices on this system
 * @throws cuda::error if the device count could not be obtained
 */
inline device::id_t  count()
{
	int device_count = 0; // Initializing, just to be on the safe side
	status_t result = cudaGetDeviceCount(&device_count);
	if (result == cudaErrorNoDevice) { return 0; }
	else {
		throw_if_error(result, "Failed obtaining the number of CUDA devices on the system");
	}
	if (device_count < 0) {
		throw std::logic_error("cudaGetDeviceCount() reports an invalid number of CUDA devices");
	}
	return device_count;
}

} // namespace device
} // namespace cuda


#endif // CUDA_API_WRAPPERS_DEVICE_COUNT_HPP_
