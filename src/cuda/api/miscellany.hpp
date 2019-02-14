/**
 * @file miscellany.hpp
 *
 * @brief Miscellaneous functionality which does not fit in another file
 *
 */
#ifndef CUDA_API_WRAPPERS_MISCELLANY_HPP_
#define CUDA_API_WRAPPERS_MISCELLANY_HPP_

#include "error.hpp"

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

} // namespace cuda

#endif // CUDA_API_WRAPPERS_MISCELLANY_HPP_
