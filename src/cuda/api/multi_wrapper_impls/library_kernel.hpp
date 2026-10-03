/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard (non-contextualized) library kernels.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_MULTI_WRAPPER_LIBRARY_KERNEL_HPP
#define CUDA_API_WRAPPERS_MULTI_WRAPPER_LIBRARY_KERNEL_HPP

#if CUDA_VERSION >= 12000

#include "kernel.hpp"
#include "../library.hpp"
#include "../kernels/in_library.hpp"

namespace cuda_ {

namespace library {

namespace kernel {

inline attribute_value_t get_attribute(
	library::kernel_t const&  library_kernel,
	kernel::attribute_t       attribute,
	device_t const&           device)
{
	return detail::get_attribute(library_kernel.handle(), device.id(), attribute);
}

inline void set_attribute(
	library::kernel_t const&  library_kernel,
	kernel::attribute_t       attribute,
	device_t const&           device,
	attribute_value_t         value)
{
	detail::set_attribute(library_kernel.handle(), device.id(), attribute, value);
}

cuda_::kernel_t contextualize(kernel_t const& kernel, context_t const& context)
{
	auto new_handle = detail::contextualize(kernel.handle(), context.handle());
	using cuda_::kernel::wrap;
	return wrap(context.device_id(), context.handle(), new_handle, do_not_hold_primary_context_refcount_unit);
}

} // namespace kernel

} // namespace library

} // namespace cuda_

#endif // CUDA_VERSION >= 12000

#endif // CUDA_API_WRAPPERS_MULTI_WRAPPER_LIBRARY_KERNEL_HPP
