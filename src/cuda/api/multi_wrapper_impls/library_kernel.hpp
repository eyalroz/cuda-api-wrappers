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

namespace cuda {

namespace library {

namespace kernel {

inline attribute_value_t get_attribute(
	const library::kernel_t&  library_kernel,
	kernel::attribute_t       attribute,
	const device_t&           device)
{
	return detail_::get_attribute(library_kernel.handle(), device.id(), attribute);
}

inline void set_attribute(
	const library::kernel_t&  library_kernel,
	kernel::attribute_t       attribute,
	const device_t&           device,
	attribute_value_t         value)
{
	detail_::set_attribute(library_kernel.handle(), device.id(), attribute, value);
}

cuda::kernel_t contextualize(const kernel_t& kernel, const context_t& context)
{
	auto new_handle = detail_::contextualize(kernel.handle(), context.handle());
	using cuda::kernel::wrap;
	return wrap(context.device_id(), context.handle(), new_handle, do_not_hold_primary_context_refcount_unit);
}

} // namespace kernel

} // namespace library

} // namespace cuda

#endif // CUDA_VERSION >= 12000

#endif // CUDA_API_WRAPPERS_MULTI_WRAPPER_LIBRARY_KERNEL_HPP
