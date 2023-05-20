/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * of runtime-linking-related functions.
 */
#ifndef MULTI_WRAPPER_IMPLS_LINK_HPP_
#define MULTI_WRAPPER_IMPLS_LINK_HPP_

#include "../link.hpp"
#include "../device.hpp"

namespace cuda {

inline cuda::device_t link_t::device() const
{
	return cuda::device::get(device_id());
}

inline cuda::context_t link_t::context() const
{
	static constexpr const bool dont_take_ownership { false };
	return context::wrap(device_id(), context_handle_, dont_take_ownership);
}

} // namespace cuda

#endif //MULTI_WRAPPER_IMPLS_LINK_HPP_
