/**
 * @file miscellany.hpp
 *
 * @brief Miscellaneous functionality which does not fit in another file,
 * and does not depend on the main proxy classes
 *
 */
#ifndef CUDA_API_WRAPPERS_MISCELLANY_HPP_
#define CUDA_API_WRAPPERS_MISCELLANY_HPP_

#include <cuda/api/error.hpp>
#include <cuda/common/types.hpp>

#include <cuda_runtime_api.h>

namespace cuda {

/**
 * @brief Ensures the CUDA runtime has fully initialized
 *
 * @note The CUDA runtime uses lazy initialization, so that until you perform
 * certain actions, the CUDA driver is not used to create a context, nothing
 * is done on the device etc. This function forces this initialization to
 * happen immediately, while not having any other effect.
 */
inline
void force_runtime_initialization()
{
	// nVIDIA's Robin Thoni (https://www.rthoni.com/) guarantees
	// the following code "does the trick"
	auto status = cudaFree(nullptr);
	throw_if_error(status, "Forcing CUDA runtime initialization");
}

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
	if (result == status::no_device) {
		return 0;
	}
	else {
		throw_if_error(result, "Failed obtaining the number of CUDA devices on the system");
	}
	if (device_count < 0) {
		throw ::std::logic_error("cudaGetDeviceCount() reports an invalid number of CUDA devices");
	}
	return device_count;
}

} // namespace device


} // namespace cuda

#endif // CUDA_API_WRAPPERS_MISCELLANY_HPP_
