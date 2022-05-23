/**
 * @file
 *
 * @brief Wrappers for working with modules of compiled CUDA code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_MODULE_HPP_
#define CUDA_API_WRAPPERS_MODULE_HPP_

#include <cuda/api/context.hpp>
#include <cuda/api/primary_context.hpp>
#include <cuda/api/kernel.hpp>
#include <cuda/api/memory.hpp>
#include <cuda/api/array.hpp>
#include <cuda/api/link_options.hpp>
#include <cuda.h>
#include <array>

#if __cplusplus >= 201703L
#include <filesystem>
#endif

namespace cuda {

///@cond
class device_t;
class context_t;
class module_t;
class kernel_t;
///@endcond

namespace module {

using handle_t = CUmodule;

namespace detail_ {

inline module_t construct(
	device::id_t device_id,
	context::handle_t context_handle,
	handle_t handle,
	link::options_t options,
	bool take_ownership = false,
	bool holds_primary_context_refcount_unit = false) noexcept;

inline ::std::string identify(const module::handle_t &handle)
{
	return std::string("module ") + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(const module::handle_t &handle, context::handle_t context_handle)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle);
}

inline ::std::string identify(const module::handle_t &handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " in " + context::detail_::identify(context_handle, device_id);
}

::std::string identify(const module_t &module);

} // namespace detail_

/**
 * Create a CUDA driver module from raw module image data.
 *
 * @param[inout] context The CUDA context into which the module data will be loaded (and
 *     in which the module contents may be used)
 * @param[inout] primary_context The CUDA context, being primary on its device, into which
 *     the module data will be loaded (and in which the module contents may be used); this is
 *     handled distinctly from a regular context, in that the primary context must be kept
 *     alive until the module has been destroyed.
 * @param[in] module_data the opaque, raw binary data for the module - in a contiguous container
 *     such as a span, a cuda::dynarray etc..
 */
///@{
template <typename Locus, typename ContiguousContainer,
	cuda::detail_::enable_if_t<cuda::detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool> = true >
module_t create(
	Locus&&              locus,
	ContiguousContainer  module_data,
	link::options_t      link_options);

template <typename Locus, typename ContiguousContainer,
	cuda::detail_::enable_if_t<cuda::detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool> = true >
module_t create(
	Locus&&          locus,
	ContiguousContainer  module_data);
///@}

} // namespace module

/**
 * Wrapper class for a CUDA code module
 *
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to the module is a const-respecting operation on this class.
 */
class module_t {

public: // getters

	module::handle_t handle() const { return handle_; }
	context::handle_t context_handle() const { return context_handle_; }
	device::id_t device_id() const { return device_id_; }
	context_t context() const;

	device_t device() const;

	// These API calls are not really the way you want to work.
	cuda::kernel_t get_kernel(const char* name) const;

	memory::region_t get_global_region(const char* name) const
	{
		CUdeviceptr dptr;
		size_t size;
		auto result = cuModuleGetGlobal(&dptr, &size, handle_, name);
		throw_if_error(result, "Obtaining the address and size of a named global object");
		return { memory::as_pointer(dptr), size };
	}

	// TODO: Implement a surface reference and texture reference class rather than these raw pointers.

	CUsurfref* get_surface(const char* name) const;
	CUtexref* get_texture_reference(const char* name) const;

protected: // constructors

	module_t(
		device::id_t device_id,
		context::handle_t context,
		module::handle_t handle,
		link::options_t options,
		bool owning,
		bool holds_primary_context_refcount_unit)
	noexcept
		: device_id_(device_id), context_handle_(context), handle_(handle), options_(options), owning_(owning),
		  holds_pc_refcount_unit(holds_primary_context_refcount_unit)
	{ }

public: // friendship

	friend module_t module::detail_::construct(
		device::id_t, context::handle_t, module::handle_t, link::options_t, bool, bool) noexcept;


public: // constructors and destructor

	module_t(const module_t&) = delete;

	module_t(module_t&& other) noexcept :
		module_t(
			other.device_id_,
			other.context_handle_,
			other.handle_,
			other.options_,
			other.owning_,
			other.holds_pc_refcount_unit)
	{
		other.owning_ = false;
		other.holds_pc_refcount_unit = false;
	};

	// Note: It is up to the user of this class to ensure that it is destroyed _before_ the context
	// in which it was created; and one needs to be particularly careful about this point w.r.t.
	// primary contexts
	~module_t()
	{
		if (owning_) {
			context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle_);
			auto status = cuModuleUnload(handle_);
			throw_if_error(status, "Failed unloading " + module::detail_::identify(*this));
		}
		// TODO: DRY
		if (holds_pc_refcount_unit) {
#ifdef NDEBUG
			cuDevicePrimaryCtxRelease(device_id_);
				// Note: "Swallowing" any potential error to avoid std::terminate(); also,
				// because a failure probably means the primary context is inactive already
#else
			device::primary_context::detail_::decrease_refcount(device_id_);
#endif
		}
	}

public: // operators

	module_t& operator=(const module_t& other) = delete;
	module_t& operator=(module_t&& other) = delete;

protected: // data members
	device::id_t       device_id_;
	context::handle_t  context_handle_;
	module::handle_t   handle_;
	link::options_t    options_;
	bool               owning_;
		// this field is mutable only for enabling move construction; other
		// than in that case it must not be altered
	bool holds_pc_refcount_unit;
		// When context_handle_ is the handle of a primary context, this module
		// may be "keeping that context alive" through the refcount - in which
		// case it must release its refcount unit on destruction
};

namespace module {

using handle_t = CUmodule;

namespace detail_ {

inline module_t load_from_file_in_current_context(
	device::id_t current_context_device_id,
	context::handle_t current_context_handle,
	const char *path,
	link::options_t link_options,
	bool holds_primary_context_refcount_unit = false)
{
	handle_t new_module_handle;
	auto status = cuModuleLoad(&new_module_handle, path);
	throw_if_error(status, ::std::string("Failed loading a module from file ") + path);
	bool do_take_ownership{true};
	return construct(
		current_context_device_id,
		current_context_handle,
		new_module_handle,
		link_options,
		do_take_ownership,
		holds_primary_context_refcount_unit);
}

} // namespace detail_


/**
 * Load a module from an appropriate compiled or semi-compiled file, allocating all
 * relevant resources for it.
 *
 * @param path of a cubin, PTX, or fatbin file constituting the module to be loaded.
 * @return the loaded module
 *
 * @note this covers cuModuleLoadFatBinary() even though that's not directly used
 *
 * @todo: consider adding load_module methods to context_t
 * @todo: When switching to the C++17 standard, use string_view's instead of the const char*
 * and std::string reference
 */
///@{
inline module_t load_from_file(
	const context_t&  context,
	const char*       path,
	link::options_t   link_options = {})
{
	context::current::detail_::scoped_override_t set_context_for_this_scope(context.handle());
	return detail_::load_from_file_in_current_context(
		context.device_id(), context.handle(), path, link_options);
}

inline module_t load_from_file(
	const context_t&      context,
	const ::std::string&  path,
	link::options_t       link_options = {})
{
	return load_from_file(context, path.c_str(), link_options);
}

inline module_t load_from_file(
	const device_t&  device,
	const char*      path,
	link::options_t  link_options = {})
{
	auto pc = device.primary_context();
	device::primary_context::detail_::increase_refcount(device.id());
	return load_from_file(pc, path, link_options);
}

inline module_t load_from_file(
	const device_t&       device,
	const ::std::string*  path,
	link::options_t       link_options = {})
{
	return load_from_file(device, path->c_str(), link_options);
}

inline module_t load_from_file(
	const char*      path,
	link::options_t  link_options = {})
{
	return load_from_file(device::current::get(), path, link_options);
}

inline module_t load_from_file(
	const ::std::string&  path,
	link::options_t       link_options)
{
	return load_from_file(device::current::get(), path.c_str(), link_options);
}


#if __cplusplus >= 201703L

inline module_t load_from_file(
	const device_t&                 device,
	const ::std::filesystem::path&  path,
	link::options_t                 link_options = {})
{
	return load_from_file(device, path.c_str(), link_options);
}

inline module_t load_from_file(
	const ::std::filesystem::path&  path,
	link::options_t                 link_options = {})
{
	return load_from_file(device::current::get(), path, link_options);
}

#endif
///@}

namespace detail_ {

// This might have been called "wrap", if we had not needed to take care
// of primary context reference counting
inline module_t construct(
	device::id_t device_id,
	context::handle_t context_handle,
	handle_t module_handle,
	link::options_t options,
	bool take_ownership,
	bool hold_pc_refcount_unit
) noexcept
{
	return module_t{device_id, context_handle, module_handle, options, take_ownership, hold_pc_refcount_unit};
}

template <typename Creator>
module_t create(const context_t& context, const void* module_data, Creator creator_function);

// TODO: Consider adding create_module() methods to context_t
inline module_t create(const context_t& context, const void* module_data, const link::options_t& link_options)
{
	auto creator_function =
		[&link_options](handle_t& new_module_handle, const void* module_data) {
			auto marshalled_options = link_options.marshal();
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

} // namespace detail_

// TODO: Use an optional to reduce the number of functions here... when the
// library starts requiring C++14.

namespace detail_ {

inline ::std::string identify(const module_t& module)
{
	return identify(module.handle(), module.context_handle(), module.device().id());
}

inline context_t get_context_for(const context_t& locus) { return locus; }
inline device::primary_context_t get_context_for(device_t& locus);

} // namespace detail_

// Note: The following may create the primary context of a device!
template <typename Locus, typename ContiguousContainer,
	cuda::detail_::enable_if_t<cuda::detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool>>
module_t create(
	Locus&&  locus,
	ContiguousContainer module_data)
{
	auto context = detail_::get_context_for(locus);
	return detail_::create(context, module_data.data());
}

// Note: The following may create the primary context of a device!
template <typename Locus, typename ContiguousContainer,
	cuda::detail_::enable_if_t<cuda::detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool>>
module_t create(
	Locus&&              locus,
	ContiguousContainer  module_data,
	link::options_t      link_options)
{
	auto context = detail_::get_context_for(locus);
	return detail_::create(context, module_data.data(), link_options);
}

} // namespace module

} // namespace cuda

#endif // CUDA_API_WRAPPERS_MODULE_HPP_
