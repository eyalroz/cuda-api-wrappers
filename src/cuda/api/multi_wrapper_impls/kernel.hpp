/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard kernels. Specifically:
 *
 * 1. Functions in the `cuda::kernel` namespace.
 * 2. Methods of @ref cuda::kernel_t and possibly some relates classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_KERNEL_HPP_
#define MULTI_WRAPPER_IMPLS_KERNEL_HPP_

#include "../device.hpp"
#include "../pointer.hpp"
#include "../primary_context.hpp"
#include "../kernel.hpp"
#include "../module.hpp"

namespace cuda {

inline context_t kernel_t::context() const noexcept
{
	constexpr bool dont_take_ownership { false };
	return context::wrap(device_id_, context_handle_, dont_take_ownership);
}

inline device_t kernel_t::device() const noexcept
{
return device::get(device_id_);
}

inline void kernel_t::set_attribute(kernel::attribute_t attribute, kernel::attribute_value_t value) const
{
	kernel::detail_::set_attribute_in_current_context(handle_, attribute, value);
}

#if CUDA_VERSION >= 12030
inline module_t kernel_t::module() const
{
	auto module_handle = kernel::detail_::get_module(context_handle_, handle_);
	return module::detail_::wrap(device_id_, context_handle_, module_handle, do_not_take_ownership);
}
#endif

namespace detail_ {

template<typename Kernel>
device::primary_context_t get_implicit_primary_context(Kernel)
{
	return device::current::get().primary_context();
}

template<>
inline device::primary_context_t get_implicit_primary_context<kernel_t>(kernel_t kernel)
{
	auto context = kernel.context();
	auto device = context.device();
	auto primary_context = device.primary_context();
	if (context != primary_context) {
		throw ::std::logic_error("Attempt to launch a kernel associated with a non-primary context without specifying a stream associated with that context.");
	}
	return primary_context;
}

} // namespace detail_

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_KERNEL_HPP_

