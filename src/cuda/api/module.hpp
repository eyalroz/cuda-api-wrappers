/**
 * @file module.hpp
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

namespace detail_ {

inline module_t construct(
	device::id_t device_id,
	context::handle_t context_handle,
	handle_t handle,
	link::options_t options,
	bool take_ownership = false,
	bool hold_primary_context_reference = false) noexcept;

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
 * Load a module from an appropriate compiled or semi-compiled file, allocating all
 * relevant resources for it.
 *
 * @param path of a cubin, PTX, or fatbin file constituting the module to be loaded.
 * @return the loaded module
 *
 * @note this covers cuModuleLoadFatBinary() even though that's not directly used
 */
module_t load_from_file(const char *path, link::options_t link_options = {});

module_t load_from_file(const ::std::string &path, link::options_t link_options = {});

#if __cplusplus >= 201703L
module_t load_from_file(const ::std::filesystem::path& path, link::options_t options = {});
#endif

/**
 * Create a CUDA driver module from raw module image data.
 *
 * @param[inout] context The CUDA context into which the module data will be loaded (and
 *     in which the module contents may be used)
 *     @parem[in
 * The pointer may be obtained by mapping a cubin or PTX or fatbin file, passing a cubin or PTX or fatbin file as a NULL-terminated text string, or incorporating a cubin or fatbin object into the executable resources and using operating system calls such as Windows FindResource() to obtain the pointer.
 */
///@{
module_t create(context_t context, const void* module_data, link::options_t link_options);
module_t create(context_t context, const void* module_data);
module_t create(device_t device, const void* module_data, link::options_t link_options);
module_t create(device_t device, const void* module_data);
template <typename ContiguousContainer>
module_t create(context_t context, ContiguousContainer module_data);
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
	cuda::kernel_t get_kernel(const char* name) const
	{
		context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle_);
		kernel::handle_t kernel_function_handle;
		auto result = cuModuleGetFunction(&kernel_function_handle, handle_, name);
		throw_if_error(result, ::std::string("Failed obtaining function ") + name
			+ " from " + module::detail_::identify(*this));
		return kernel::detail_::wrap(
			context::detail_::get_device_id(context_handle_), context_handle_, kernel_function_handle);
	}

	cuda::memory::region_t get_global_object(const char* name) const;

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
		bool hold_primary_context_reference)
#ifdef NDEBUG
		noexcept
#endif
		: device_id_(device_id), context_handle_(context), handle_(handle), options_(options), owning_(owning),
		  holds_primary_context_refcount_unit_(hold_primary_context_reference)
	{
#ifndef NDEBUG
		if (not owning and hold_primary_context_reference) {
			throw std::invalid_argument("A non-owning module proxy should not try to hold its own primary context refcount unit");
		}
		if (hold_primary_context_reference and not context::detail_::is_primary(context_handle_))
		{
			throw std::invalid_argument("A module in a non-primary context should not presume to hold a primary context refcount unit");
		}
#endif
		if (owning and hold_primary_context_reference) {
			device::primary_context::detail_::increase_refcount(device_id);
		}
	}

	module_t(device::id_t device_id, context::handle_t context, module::handle_t handle, link::options_t options, bool owning) noexcept
	: module_t(device_id, context, handle, options, owning, false)
	{ }

public: // friendship

	friend module_t module::detail_::construct(device::id_t, context::handle_t, module::handle_t, link::options_t, bool, bool) noexcept;


public: // constructors and destructor

	module_t(const module_t&) = delete;

	module_t(module_t&& other) noexcept :
		module_t(other.device_id_, other.context_handle_, other.handle_, other.options_, other.owning_)
	{
		other.owning_ = false;
	};

	~module_t()
	{
		if (owning_) {
			context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle_);
			auto status = cuModuleUnload(handle_);
		 	throw_if_error(status, "Failed unloading " + module::detail_::identify(*this));

			if (holds_primary_context_refcount_unit_) {
				device::primary_context::detail_::decrease_refcount(device_id_);
			}
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
	bool               holds_primary_context_refcount_unit_;
};

namespace module {

using handle_t = CUmodule;

/**
* Loads a populated module from a file on disk
*
* @param path Filesystem path of a fatbin, cubin or PTX file
*
* @todo: Do we really need link options here?
 * @todo: Make this take a context_t; and consider adding load_module methods to context_t
*/
inline module_t load_from_file(const char* path, link::options_t link_options)
{
	handle_t new_module_handle;
	auto status = cuModuleLoad(&new_module_handle, path);
	throw_if_error(status, ::std::string("Failed loading a module from file ") + path);
	bool do_take_ownership { true };
	auto current_context_handle = context::current::detail_::get_handle();
	auto current_device_id = context::detail_::get_device_id(current_context_handle);
	return detail_::construct(current_device_id, current_context_handle, new_module_handle, link_options,
		do_take_ownership);
}

inline module_t load_from_file(const ::std::string& path, link::options_t link_options)
{
	return load_from_file(path.c_str(), link_options);
}

#if __cplusplus >= 201703L
inline module_t load_from_file(const ::std::filesystem::path& path)
{
	return load_from_file(path.c_str());
}
#endif

namespace detail_ {

// This might have been called "wrap", if we had not needed to take care
// of primary context reference counting
inline module_t construct(
	device::id_t device_id,
	context::handle_t context_handle,
	handle_t module_handle,
	link::options_t options,
	bool take_ownership,
	bool hold_primary_context_reference) noexcept
{
	return module_t{device_id, context_handle, module_handle, options, take_ownership, hold_primary_context_reference};
}

template <typename Creator>
inline module_t create(const context_t& context, const void* module_data, Creator creator_function, bool hold_pc_reference)
{
	context::current::scoped_override_t set_context_for_this_scope(context);
	handle_t new_module_handle;
	auto status = creator_function(new_module_handle, module_data);
	throw_if_error(status, ::std::string(
		"Failed loading a module from memory location ") + cuda::detail_::ptr_as_hex(module_data) +
		"within " + context::detail_::identify(context));
	bool do_take_ownership { true };
	// TODO: Make sure the default-constructed options correspond to what cuModuleLoadData uses as defaults
	return detail_::construct(context.device_id(), context.handle(), new_module_handle,
		link::options_t{}, do_take_ownership, hold_pc_reference);
}

// TODO: Consider adding create_module() methods to context_t
inline module_t create(const context_t& context, const void* module_data, const link::options_t& link_options, bool hold_pc_reference)
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
	return detail_::create(context, module_data, creator_function, hold_pc_reference);
}

inline module_t create(const context_t& context, const void* module_data, bool hold_pc_reference)
{
	auto creator_function =
		[](handle_t& new_module_handle, const void* module_data) {
			return cuModuleLoadData(&new_module_handle, module_data);
		};
	return detail_::create(context, module_data, creator_function, hold_pc_reference);
}

} // namespace detail_

// TODO: Use an optional to reduce the number of functions here... when the
// library starts requiring C++14.

inline module_t create(context_t context, const void* module_data)
{
	return detail_::create(context, module_data, false);
}

inline module_t create(context_t context, const void* module_data, link::options_t link_options)
{
	return detail_::create(context, module_data, link_options, false);
}

inline module_t create(device::primary_context_t primary_context, const void* module_data)
{
	constexpr const bool do_hold_primary_context_reference {true };
	const context_t& context = primary_context;
	return detail_::create(context, module_data, do_hold_primary_context_reference);
}

inline module_t create(device::primary_context_t primary_context, const void* module_data, link::options_t link_options)
{
	constexpr const bool do_hold_primary_context_reference {true };
	const context_t& context = primary_context;
	return detail_::create(context, module_data, link_options, do_hold_primary_context_reference);
}

namespace detail_ {

inline ::std::string identify(const module_t& module)
{
	return identify(module.handle(), module.context_handle(), module.device().id());
}

} // namespace detail_

} // namespace module

} // namespace cuda

#endif // CUDA_API_WRAPPERS_MODULE_HPP_
