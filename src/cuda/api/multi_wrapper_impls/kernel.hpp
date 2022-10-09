/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard kernels. Specifically:
 *
 * 1. Functions in the `cuda::kernel` namespace.
 * 2. Methods of @ref `cuda::kernel_t` and possibly some relates classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_KERNEL_HPP_
#define MULTI_WRAPPER_IMPLS_KERNEL_HPP_

#include "../device.hpp"
#include "../pointer.hpp"
#include "../primary_context.hpp"
#include "../kernel.hpp"
#include "../apriori_compiled_kernel.hpp"
#include "../current_context.hpp"

namespace cuda {

namespace kernel {

template<typename KernelFunctionPtr>
apriori_compiled_kernel_t get(context_t context, KernelFunctionPtr function_ptr)
{
	static_assert(
	::std::is_pointer<KernelFunctionPtr>::value
	and ::std::is_function<typename ::std::remove_pointer<KernelFunctionPtr>::type>::value,
	"function_ptr must be a bona fide pointer to a kernel (__global__) function");

	auto ptr_ = reinterpret_cast<const void*>(function_ptr);
#if CAN_GET_APRIORI_KERNEL_HANDLE
	auto handle = detail_::get_handle(ptr_);
#else
	auto handle = nullptr;
#endif
	return detail_::wrap(context.device_id(), context.handle(), handle, ptr_);
}

/**
 * @note The returned kernel proxy object will keep the device's primary
 * context active while the kernel exists.
 */
template<typename KernelFunctionPtr>
apriori_compiled_kernel_t get(const device_t& device, KernelFunctionPtr function_ptr)
{
	auto primary_context = device.primary_context(do_hold_primary_context_refcount_unit);
	return get<KernelFunctionPtr>(primary_context, function_ptr);
}

} // namespace kernel

inline context_t kernel_t::context() const noexcept
{
constexpr bool dont_take_ownership { false };
return context::detail_::from_handle(context_handle_, dont_take_ownership);
}

inline device_t kernel_t::device() const noexcept
{
return device::get(device_id_);
}

inline void kernel_t::set_attribute(kernel::attribute_t attribute, kernel::attribute_value_t value) const
{
#if CUDA_VERSION >= 9000
	context::current::detail_::scoped_override_t set_context_for_this_context(context_handle_);
	auto result = cuFuncSetAttribute(handle_, static_cast<CUfunction_attribute>(attribute), value);
	throw_if_error(result,
		"Setting CUDA device function attribute " +
		::std::string(kernel::detail_::attribute_name(attribute)) +
		" to value " + ::std::to_string(value)	);
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif
}

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

template<>
inline device::primary_context_t get_implicit_primary_context<apriori_compiled_kernel_t>(apriori_compiled_kernel_t kernel)
{
	const kernel_t& kernel_ = kernel;
	return get_implicit_primary_context(kernel_);
}

} // namespace detail_

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_KERNEL_HPP_

