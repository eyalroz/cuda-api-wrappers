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

namespace cuda_ {

inline device_t link_t::device() const
{
	return device::get(device_id());
}

inline cuda_::context_t link_t::context() const
{
	static constexpr bool dont_take_ownership { false };
	return context::wrap(device_id(), context_handle_, dont_take_ownership);
}

} // namespace cuda_

#endif //MULTI_WRAPPER_IMPLS_LINK_HPP_
