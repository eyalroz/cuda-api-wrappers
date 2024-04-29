/**
 * @file
 *
 * @brief Wrappers for working with modules of compiled CUDA code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_MODULE_HPP_
#define CUDA_API_WRAPPERS_MODULE_HPP_

#include "context.hpp"
#include "primary_context.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "array.hpp"
#include "link_options.hpp"
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

// The CUDA driver's raw handle for modules
using handle_t = CUmodule;

namespace detail_ {

/// Construct a module proxy object - for an existing module - from the class'
/// constituent fields
inline module_t wrap(
	device::id_t            device_id,
	context::handle_t       context_handle,
	handle_t                handle,
	bool                    take_ownership = false,
	bool                    holds_primary_context_refcount_unit = false) noexcept;

inline ::std::string identify(const module::handle_t &handle)
{
	return ::std::string("module ") + cuda::detail_::ptr_as_hex(handle);
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

inline void destroy(handle_t handle, context::handle_t context_handle, device::id_t device_id);

} // namespace detail_

/**
 * Create a CUDA driver module from raw module image data.
 *
 * @param[in] locus either a @ref context_t or a @ref device_t - an entity for which one
 *     can obtain a CUDA context. This is the context into which the module data is to
 *     be loaded (and in which the module contents may be used)
 * @param[in] module_data the opaque, raw binary data for the module - in a contiguous container
 *     such as a span, a cuda::unique_span etc..
 * @param link_options Potential options for the PTX compilation and linking of the compiled
 *     device-side code.
 */
///@{
template <typename Locus, typename ContiguousContainer,
	cuda::detail_::enable_if_t<cuda::detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool> = true >
module_t create(
	Locus&&                 locus,
	ContiguousContainer     module_data,
	const link::options_t&  link_options);

template <typename Locus, typename ContiguousContainer,
	cuda::detail_::enable_if_t<cuda::detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool> = true >
module_t create(
	Locus&&              locus,
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
	/// Getters for the module object's raw constituent fields
	///@{
	module::handle_t handle() const { return handle_; }
	context::handle_t context_handle() const { return context_handle_; }
	device::id_t device_id() const { return device_id_; }
	///@}

	/// @returns the context in which this module exists
	context_t context() const;

	/// @returns the device with which this module is associated
	device_t device() const;

	/**
	 * Obtains a kernel constituting part of this module
	 *
	 * @param name The function name, in case of a C-style function,
	 * or the mangled function signature, in case of a C++-style
	 * function.
	 *
	 * @return An enqueable kernel proxy object for the requested kernel.
	 */
	cuda::kernel_t get_kernel(const char* name) const;

	/// @copydoc get_kernel(const char*) const
	cuda::kernel_t get_kernel(const ::std::string& name) const
	{
		return get_kernel(name.c_str());
	}

	/// Get the mapping of a named memory region in this module to actual memory
	memory::region_t get_global_region(const char* name) const
	{
		CUdeviceptr dptr;
		size_t size;
		auto result = cuModuleGetGlobal(&dptr, &size, handle_, name);
		throw_if_error_lazy(result, "Obtaining the address and size of a named global object");
		return { memory::as_pointer(dptr), size };
	}

	// TODO: Implement a surface reference and texture reference class rather than these raw pointers.

#if CUDA_VERSION < 12000
	/// A "plug" of a method regarding surfaces, which modules support but our wrappers don't
	/// really cater to the use of
	CUsurfref get_surface(const char* name) const;

	/// A "plug" of a method regarding texture references
	///
	/// TODO: Beef this up into a proper method and consider relations between texture references,
	/// texture objects and texture views
	CUtexref get_texture_reference(const char* name) const;
#endif

protected: // constructors

	module_t(
		device::id_t device_id,
		context::handle_t context,
		module::handle_t handle,
		bool owning,
		bool holds_primary_context_refcount_unit)
	noexcept
		: device_id_(device_id), context_handle_(context), handle_(handle), owning_(owning),
		  holds_pc_refcount_unit_(holds_primary_context_refcount_unit)
	{ }

public: // friendship

	friend module_t module::detail_::wrap(device::id_t, context::handle_t, module::handle_t, bool, bool) noexcept;

public: // constructors and destructor

	module_t(const module_t&) = delete;

	module_t(module_t&& other) noexcept :
		module_t(
			other.device_id_,
			other.context_handle_,
			other.handle_,
			other.owning_,
			other.holds_pc_refcount_unit_)
	{
		other.owning_ = false;
		other.holds_pc_refcount_unit_ = false;
	};

	// Note: It is up to the user of this class to ensure that it is destroyed _before_ the context
	// in which it was created; and one needs to be particularly careful about this point w.r.t.
	// primary contexts
	~module_t() noexcept(false)
	{
		if (owning_) {
			module::detail_::destroy(handle_, context_handle_, device_id_);
		}
		// TODO: DRY
		if (holds_pc_refcount_unit_) {
#ifdef NDEBUG
			device::primary_context::detail_::decrease_refcount_nothrow(device_id_);
				// Note: "Swallowing" any potential error to avoid ::std::terminate(); also,
				// because a failure probably means the primary context is inactive already
#else
			device::primary_context::detail_::decrease_refcount(device_id_);
#endif
		}
	}

public: // operators

	module_t& operator=(const module_t&) = delete;
	module_t& operator=(module_t&& other) noexcept
	{
		::std::swap(device_id_, other.device_id_);
		::std::swap(context_handle_, other.context_handle_);
		::std::swap(handle_, other.handle_);
		::std::swap(owning_, other.owning_);
		::std::swap(holds_pc_refcount_unit_, holds_pc_refcount_unit_);
		return *this;
	}

protected: // data members
	device::id_t       device_id_;
	context::handle_t  context_handle_;
	module::handle_t   handle_;
	bool               owning_;
		// this field is mutable only for enabling move construction; other
		// than in that case it must not be altered
	bool holds_pc_refcount_unit_;
		// When context_handle_ is the handle of a primary context, this module
		// may be "keeping that context alive" through the refcount - in which
		// case it must release its refcount unit on destruction
};

namespace module {

namespace detail_ {

inline module_t load_from_file_in_current_context(
	device::id_t            current_context_device_id,
	context::handle_t       current_context_handle,
	const char *            path,
	bool                    holds_primary_context_refcount_unit = false)
{
	handle_t new_module_handle;
	auto status = cuModuleLoad(&new_module_handle, path);
	throw_if_error_lazy(status, ::std::string("Failed loading a module from file ") + path);
	bool do_take_ownership{true};
	return wrap(
		current_context_device_id,
		current_context_handle,
		new_module_handle,
		do_take_ownership,
		holds_primary_context_refcount_unit);
}

} // namespace detail_


/**
 * Load a module from an appropriate compiled or semi-compiled file, allocating all
 * relevant resources for it.
 *
 * @param path of a cubin, PTX, or fatbin file constituting the module to be loaded.
 * @param context The context in which to create a module containing the loaded data
 * @return the loaded module
 *
 * @note this covers cuModuleLoadFatBinary() even though that's not directly used
 *
 * @todo consider adding load_module methods to context_t
 * @todo When switching to the C++17 standard, use string_view's instead of the const char*
 * and ::std::string reference
 */
inline module_t load_from_file(
	const context_t&        context,
	const char*             path)
{
	CAW_SET_SCOPE_CONTEXT(context.handle());
	return detail_::load_from_file_in_current_context(context.device_id(), context.handle(), path);
}

/// @copydoc load_from_file(const context_t&, const char*)
inline module_t load_from_file(
	const context_t&        context,
	const ::std::string&    path)
{
	return load_from_file(context, path.c_str());
}

/**
 * @copydoc load_from_file(const context_t&, const char*)
 *
 * @param device The device in whose primary context to create a module containing
 * the loaded data
 */
module_t load_from_file(
	const device_t&         device,
	const char*             path);

/**
 * @copydoc load_from_file(const context_t&, const char*)
 *
 * @param device The device in whose primary context to create a module containing
 * the loaded data
 */
inline module_t load_from_file(
	const device_t&         device,
	const ::std::string&    path)
{
	return load_from_file(device, path.c_str());
}

/**
 * @copydoc load_from_file(const context_t&, const char*)
 *
 * @note loads the module into the primary context of the device of the current context
 * (although not necessarily in the current context!)
 */
module_t load_from_file(const char* path);

/// @copydoc load_from_file(const char*)
inline module_t load_from_file(const ::std::string& path)
{
	return load_from_file(path.c_str());
}

#if __cplusplus >= 201703L
/// @copydoc load_from_file(device_t, const char*)
inline module_t load_from_file(
	const device_t&                 device,
	const ::std::filesystem::path&  path)
{
	return load_from_file(device, path.c_str());
}

/// @copydoc load_from_file(const char*)
inline module_t load_from_file(
	const ::std::filesystem::path&  path)
{
	return load_from_file(device::current::get(), path);
}
#endif

namespace detail_ {

inline module_t wrap(
	device::id_t            device_id,
	context::handle_t       context_handle,
	handle_t                module_handle,
	bool                    take_ownership,
	bool                    hold_pc_refcount_unit
) noexcept
{
	return module_t{device_id, context_handle, module_handle, take_ownership, hold_pc_refcount_unit};
}

/**
 * Creates a new module in a context using raw compiled code
 *
 * @param context The module will exist within this GPU context, i.e. the globals (functions,
 * variable) of the module would be usable within that constant.
 * @param module_data The raw compiled code for the module.
 * @param link_options Potential options for the PTX compilation and linking of the compiled
 *     device-side code.
 */
module_t create(const context_t& context, const void* module_data, const link::options_t& link_options);

/// @copydoc create(const context_t&, const void*, const link::options_t&)
module_t create(const context_t& context, const void* module_data);

inline void destroy(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	auto status = cuModuleUnload(handle);
	throw_if_error_lazy(status, "Failed unloading " + identify(handle, context_handle, device_id));
}

} // namespace detail_

// TODO: Use an optional to reduce the number of functions here... when the
// library starts requiring C++14.

namespace detail_ {

inline ::std::string identify(const module_t& module)
{
	return identify(module.handle(), module.context_handle(), module.device_id());
}

inline context_t get_context_for(const context_t& locus) { return locus; }
inline device::primary_context_t get_context_for(device_t& locus);

} // namespace detail_

/**
 * Create a new module - in a specified context or in a device's primary context,
 * using raw module data in memory.
 *
 * @tparam Locus Either a @ref cuda::device_t or a {@ref cuda::context_t}.
 * @tparam ContiguousContainer A span, a vector, a unique_span, or similar type
 * @param locus Where the new module should be created
 * @param module_data The raw data for the module in locus-accessible memory.
 *
 * @note This function may create/allocate resources for the primary context of a device!
 */
template <typename Locus, typename ContiguousContainer,
	cuda::detail_::enable_if_t<cuda::detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool>>
module_t create(
	Locus&&             locus,
	ContiguousContainer module_data)
{
	auto context = detail_::get_context_for(locus);
	return detail_::create(context, module_data.data());
}

/**
 * @copydoc create(Locus&&,	ContiguousContainer)
 *
 * @param link_options Options for PTX compilation and for linking the module data,
 * eventually.
 *
 * @return
 */
// Note: The following may create the primary context of a device!
template <typename Locus, typename ContiguousContainer,
	cuda::detail_::enable_if_t<cuda::detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool>>
module_t create(
	Locus&&                 locus,
	ContiguousContainer     module_data,
	const link::options_t&  link_options)
{
	auto context = detail_::get_context_for(locus);
	return detail_::create(context, module_data.data(), link_options);
}

} // namespace module

} // namespace cuda

#endif // CUDA_API_WRAPPERS_MODULE_HPP_
