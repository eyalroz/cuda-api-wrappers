/**
 * @file
 *
 * @brief Miscellaneous functionality which does not fit in another file,
 * and does not depend on the main proxy classes
 *
 */
#ifndef CUDA_API_WRAPPERS_NVML_MISCELLANY_HPP_
#define CUDA_API_NVML_MISCELLANY_HPP_

#include "types.hpp"
#include "error.hpp"

#include "../api/miscellany.hpp"

#include <cuda.h>
#include <ostream>
#include <utility>

namespace cuda {

namespace nvml {

enum {
    do_attach_gpus       = false,
    dont_attach_gpus     = true,
    failure_with_no_gpus = false,
    success_with_no_gpus = true,
};

/**
 * Obtains the CUDA Runtime version
 *
 * @note unlike {@ref maximum_supported_by_driver()}, 0 cannot be returned,
 * as we are actually using the runtime to obtain the version, so it does
 * have _some_ version.
 */
inline void initialize(
    bool attach_gpus = false,
    bool accept_no_gpus = failure_with_no_gpus)
{
	static constexpr const unsigned flags {
        attach_gpus    ? 0 : NVML_INIT_FLAG_NO_ATTACH :
        accept_no_gpus ? 0 : NVML_INIT_FLAG_NO_GPUS
    };
	auto status = nvmlInitWithFlags(flags);
	throw_if_nvml_error_lazy(status, "Failed initializing the CUDA driver");
}

inline void ensure_initialization(
    bool attach_gpus = false,
    bool accept_no_gpus = failure_with_no_gpus)
{
	thread_local bool nvml_known_to_be_initialized { false };
	if (not nvml_known_to_be_initialized) {
        cuda::ensure_driver_is_initialized();
        initialize_driver();
        nvml_known_to_be_initialized = true;
	}
}

} // namespace nvml

} // namespace cuda

#endif // CUDA_API_NVML_MISCELLANY_HPP_
