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

#include "context.hpp"
#include "../device.hpp"
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

namespace detail_ {

template <typename Creator>
module_t create(const context_t& context, const void* module_data, Creator creator_function)
{
	context::current::scoped_override_t set_context_for_this_scope(context);
	handle_t new_module_handle;
	auto status = creator_function(new_module_handle, module_data);
	throw_if_error(status, ::std::string("Failed loading a module from memory location ")
		+ cuda::detail_::ptr_as_hex(module_data)
		+ " within " + context::detail_::identify(context));
	bool do_take_ownership { true };
	bool doesnt_hold_pc_refcount_unit { false };
		// TODO: Do we want to allow holding a refcount unit here, if context is
		// the primary context?

	// TODO: Make sure the default-constructed options correspond to what cuModuleLoadData uses as defaults
	return detail_::wrap(
		context.device_id(), context.handle(), new_module_handle,
		link::options_t{}, do_take_ownership, doesnt_hold_pc_refcount_unit);
}

// TODO: Consider adding create_module() methods to context_t
inline module_t create(const context_t& context, const void* module_data, const link::options_t& link_options)
{
	auto creator_function =
		[&link_options](handle_t& new_module_handle, const void* module_data) {
			auto marshalled_options = marshal(link_options);
			return cuModuleLoadDataEx(
				&new_module_handle,
				module_data,
				marshalled_options.count(),
				const_cast<link::option_t *>(marshalled_options.options()),
				const_cast<void **>(marshalled_options.values())
			);
		};
	return detail_::create(context, module_data, creator_function);
}

inline module_t create(const context_t& context, const void* module_data)
{
	auto creator_function =
		[](handle_t& new_module_handle, const void* module_data) {
			return cuModuleLoadData(&new_module_handle, module_data);
		};
	return detail_::create(context, module_data, creator_function);
}



inline device::primary_context_t get_context_for(device_t& locus) { return locus.primary_context(); }

} // namespace detail_

inline module_t load_from_file(
	const device_t&  device,
	const char*      path,
	link::options_t  link_options)
{
	auto pc = device.primary_context();
	device::primary_context::detail_::increase_refcount(device.id());
	return load_from_file(pc, path, link_options);
}

inline module_t load_from_file(
	const char*      path,
	link::options_t  link_options)
{
	return load_from_file(device::current::get(), path, link_options);
}

} // namespace module

inline context_t module_t::context() const { return context::detail_::from_handle(context_handle_); }
inline device_t module_t::device() const { return device::get(context::detail_::get_device_id(context_handle_)); }

inline CUsurfref module_t::get_surface(const char* name) const
{
	context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle_);
	CUsurfref raw_surface_reference;
	auto status = cuModuleGetSurfRef(&raw_surface_reference, handle_, name);
	throw_if_error(status, ::std::string("Failed obtaining a reference to surface \"") + name + "\" from "
		+ module::detail_::identify(*this));
	return raw_surface_reference;
}

inline CUtexref module_t::get_texture_reference(const char* name) const
{
	context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle_);
	CUtexref raw_texture_reference;
	auto status = cuModuleGetTexRef(&raw_texture_reference, handle_, name);
	throw_if_error(status, ::std::string("Failed obtaining a reference to texture \"") + name + "\" from "
		+ module::detail_::identify(*this));
	return raw_texture_reference;
}


} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_MODULE_HPP_

