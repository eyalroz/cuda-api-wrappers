/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard modules. Specifically:
 *
 * 1. Functions in the `cuda::module` namespace.
 * 2. Methods of @ref `cuda::module_t` and possibly some relates classes.
 * 3. The `context_t::create_module()` methods; see issue #320 on the issue tracker.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_MODULE_HPP_
#define MULTI_WRAPPER_IMPLS_MODULE_HPP_

#include "../device.hpp"
#include "../context.hpp"
#include "../module.hpp"

namespace cuda {

// Moved over from context.hpp
template <typename ContiguousContainer,
cuda::detail_::enable_if_t<detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool>>
module_t context_t::create_module(ContiguousContainer module_data) const
{
	return module::create<ContiguousContainer>(*this, module_data);
}

template <typename ContiguousContainer,
cuda::detail_::enable_if_t<detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool>>
module_t context_t::create_module(ContiguousContainer module_data, link::options_t link_options) const
{
	return module::create<ContiguousContainer>(*this, module_data, link_options);
}


// These API calls are not really the way you want to work.
inline cuda::kernel_t module_t::get_kernel(const char* name) const
{
	context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle_);
	kernel::handle_t kernel_function_handle;
	auto result = cuModuleGetFunction(&kernel_function_handle, handle_, name);
	throw_if_error(result, ::std::string("Failed obtaining function ") + name
						   + " from " + module::detail_::identify(*this));
	return kernel::wrap(context::detail_::get_device_id(context_handle_), context_handle_, kernel_function_handle);
}


namespace module {

namespace detail_{

template <typename Creator>
module_t create(const context_t& context, const void* module_data, Creator creator_function)
{
	context::current::scoped_override_t set_context_for_this_scope(context);
	handle_t new_module_handle;
	auto status = creator_function(new_module_handle, module_data);
	throw_if_error(status, ::std::string(
	"Failed loading a module from memory location ")
						   + cuda::detail_::ptr_as_hex(module_data)
						   + " within " + context::detail_::identify(context));
	bool do_take_ownership { true };
	// TODO: Make sure the default-constructed options correspond to what cuModuleLoadData uses as defaults
	return detail_::construct(
	context.device_id(), context.handle(), new_module_handle,
	link::options_t{}, do_take_ownership);
}


inline device::primary_context_t get_context_for(device_t& locus) { return locus.primary_context(); }

} // namespace detail_

} // namespace module

inline context_t module_t::context() const { return context::detail_::from_handle(context_handle_); }
inline device_t module_t::device() const { return device::get(context::detail_::get_device_id(context_handle_)); }

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_MODULE_HPP_

