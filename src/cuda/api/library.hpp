/**
 * @file
 *
 * @brief Wrappers for working with "libraries" of compiled CUDA code (which are similar
 * to modules, but not associated with any CUDA context).
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_LIBRARY_HPP_
#define CUDA_API_WRAPPERS_LIBRARY_HPP_

#if CUDA_VERSION >= 12000

#include "module.hpp"
#include "error.hpp"

#if __cplusplus >= 201703L
#include <filesystem>
#endif

namespace cuda {

///@cond
class context_t;
class module_t;
class library_t;
class kernel_t;
///@endcond

namespace library {

using handle_t = CUlibrary;

namespace kernel {

using handle_t = CUkernel; // Don't be confused; a context-associated kernel is a CUfunction :-(

} // namespace kernel

namespace detail_ {

using option_t = CUlibraryOption;

} // namespace detail_

class kernel_t; // A kernel stored within a library; strangely, a context-associated kernel is a CUfunction.

namespace detail_ {

inline library_t wrap(
	handle_t                handle,
	bool                    take_ownership = false) noexcept;

inline ::std::string identify(const library::handle_t &handle)
{
	return ::std::string("library ") + cuda::detail_::ptr_as_hex(handle);
}

::std::string identify(const library_t &library);

inline status_t unload_nothrow(handle_t handle) noexcept
{
	return cuLibraryUnload(handle);
}

inline void unload(handle_t handle)
{
	auto status = unload_nothrow(handle);
	throw_if_error_lazy(status, ::std::string{"Failed unloading "}
		+ library::detail_::identify(handle));
}

} // namespace detail_

/**
 * Create a CUDA driver library of compiled code from raw image data.
 *
 * @param[in] module_data the opaque, raw binary data for the module - in a contiguous container
 *     such as a span, a cuda::unique_span etc..
 */
///@{
template <typename ContiguousContainer,
	cuda::detail_::enable_if_t<cuda::detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool> = true >
library_t create(
	ContiguousContainer        library_data,
	optional<link::options_t>  link_options,
	bool                       code_is_preserved);
///@}


namespace detail_ {

inline kernel::handle_t get_kernel_in_current_context(handle_t library_handle, const char* name)
{
	library::kernel::handle_t kernel_handle;
	auto status = cuLibraryGetKernel(&kernel_handle, library_handle, name);
	throw_if_error_lazy(status, ::std::string{"Failed obtaining kernel "}
		 + name + "' from " + library::detail_::identify(library_handle));
	return kernel_handle;
}

inline kernel::handle_t get_kernel(context::handle_t context_handle, handle_t library_handle, const char* name)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	return get_kernel_in_current_context(library_handle, name);
}

} // namespace detail_

inline kernel_t get_kernel(const library_t& library, const char* name);
inline kernel_t get_kernel(context_t& context, const library_t& library, const char* name);

} // namespace library

memory::region_t get_global(const context_t& context, const library_t& library, const char* name);
memory::region_t get_managed_region(const library_t& library, const char* name);

namespace module {

module_t create(const context_t& context, const library_t& library);
module_t create(const library_t& library);

} // namespace module

void* get_unified_function(const context_t& context, const library_t& library, const char* symbol);

/**
 * Wrapper class for a CUDA compiled code library (like a @ref module_t , but not associated
 * with a context)
 */
class library_t {

public: // getters

	library::handle_t handle() const { return handle_; }

	/**
	 * Obtains an already-compiled kernel previously associated with
	 * this library, in the current context.
	 *
	 * @param name The function name, in case of a C-style function,
	 * or the mangled function signature, in case of a C++-style
	 * function.
	 *
	 * @return An enqueable kernel proxy object for the requested kernel,
	 * in the current context.
	 */
	library::kernel_t get_kernel(const context_t& context, const char* name) const;
	library::kernel_t get_kernel(const context_t& context, const ::std::string& name) const;
	library::kernel_t get_kernel(const char* name) const;
	library::kernel_t get_kernel(const ::std::string& name) const;

	memory::region_t get_global(const char* name) const
	{
		return cuda::get_global(context::current::get(), *this, name);
	}

	memory::region_t get_global(const ::std::string& name) const
	{
		return get_global(name.c_str());
	}

	memory::region_t get_managed(const char* name) const
	{
		return cuda::get_managed_region(*this, name);
	}

	memory::region_t get_managed(const ::std::string& name) const
	{
		return get_managed(name.c_str());
	}

protected: // constructors

	library_t(library::handle_t handle, bool owning) noexcept
		: handle_(handle), owning_(owning)
	{ }

public: // friendship

	friend library_t library::detail_::wrap(library::handle_t, bool) noexcept;

public: // constructors and destructor

	library_t(const library_t&) = delete;

	library_t(library_t&& other) noexcept : library_t(other.handle_,  other.owning_)
	{
		other.owning_ = false;
	};

	~library_t() DESTRUCTOR_EXCEPTION_SPEC
	{
		if (not owning_) { return; }
#ifdef THROW_IN_DESTRUCTORS
		library::detail_::unload(handle_);
#else
		library::detail_::unload_nothrow(handle_);
#endif
	}

public: // operators

	library_t& operator=(const library_t&) = delete;
	library_t& operator=(library_t&& other) noexcept
	{
		::std::swap(handle_, other.handle_);
		::std::swap(owning_, other.owning_);
		return *this;
	}

protected: // data members
	library::handle_t   handle_;
	bool                owning_;
		// this field is mutable only for enabling move construction; other
		// than in that case it must not be altered
};

inline memory::region_t get_global(const context_t& context, const library_t& library, const char* name)
{
	CUdeviceptr dptr;
	size_t size;
	auto result = cuLibraryGetGlobal(&dptr, &size, library.handle(), name);
	throw_if_error_lazy(result,
		::std::string("Obtaining the memory address and size for the global object '") + name + "' from "
		+ library::detail_::identify(library) + " in context " + context::detail_::identify(context));
	return { memory::as_pointer(dptr), size };
	// Note: Nothing is holding a PC refcount unit here!
}

// More library item getters
namespace library {

} // namespace library

inline memory::region_t get_managed_region(const library_t& library, const char* name)
{
	memory::device::address_t region_start;
	size_t region_size;
	auto status = cuLibraryGetManaged(&region_start, &region_size, library.handle(), name);
	throw_if_error_lazy(status, ::std::string("Failed obtaining the managed memory region '") + name
		+ "' from " + library::detail_::identify(library));
	return { memory::as_pointer(region_start), region_size };
}

namespace module {

/**
 * Create an in-context module from the compiled code within a loaded library
 */
inline module_t create(const context_t& context, const library_t& library)
{
	CAW_SET_SCOPE_CONTEXT(context.handle());
	module::handle_t new_handle;
	auto status = cuLibraryGetModule(&new_handle, library.handle());
	throw_if_error_lazy(status, ::std::string("Failed creating a module '") +
		+ "' from " + library::detail_::identify(library) + " in " + context::detail_::identify(context));
	constexpr const bool is_owning { true };
	return module::detail_::wrap(context.device_id(), context.handle(), new_handle,
		is_owning, do_hold_primary_context_refcount_unit);
	// TODO: We could consider adding a variant of this function taking a context&&, and using that
	// to decide whether or not to hold a PC refcount unit
}

} // namespace module

// I really have no idea what this does!
inline void* get_unified_function(const context_t& context, const library_t& library, const char* symbol)
{
	CAW_SET_SCOPE_CONTEXT(context.handle());
	void* function_ptr;
	auto status = cuLibraryGetUnifiedFunction(&function_ptr, library.handle(), symbol);
	throw_if_error_lazy(status, ::std::string("Failed obtaining a pointer for function '") + symbol
		+ "' from " + library::detail_::identify(library) + " in " + context::detail_::identify(context));
	return function_ptr;
}

namespace library {

namespace detail_ {

template <typename Creator, typename DataSource, typename ErrorStringGenerator>
library_t create(
	Creator                 creator,
	DataSource              data_source,
	ErrorStringGenerator    error_string_generator,
	const link::options_t&  link_options = {},
	bool                    code_is_preserved = false)
{
	handle_t new_lib_handle;
	auto raw_link_opts = link::detail_::marshal(link_options);
	struct {
		detail_::option_t options[1];
		void* values[1];
		unsigned count;
	} raw_opts = { { CU_LIBRARY_BINARY_IS_PRESERVED }, { &code_is_preserved }, 1 };
	auto status = creator(
		&new_lib_handle, data_source,
		const_cast<link::detail_::option_t*>(raw_link_opts.options()),
		const_cast<void**>(raw_link_opts.values()), raw_link_opts.count(),
		raw_opts.options, raw_opts.values, raw_opts.count
	);
	throw_if_error_lazy(status,
		::std::string("Failed loading a compiled CUDA code library from ") + error_string_generator());
	bool do_take_ownership{true};
	return detail_::wrap(new_lib_handle, do_take_ownership);
}

} // namespace detail_

/**
 * Load a library from an appropriate compiled or semi-compiled file, allocating all
 * relevant resources for it.
 *
 * @param path of a cubin, PTX, or fatbin file constituting the module to be loaded.
 * @return the loaded library
 *
 * @note this covers cuModuleLoadFatBinary() even though that's not directly used
 *
 * @todo: When switching to the C++17 standard, use string_view's instead of the const char*
 */
///@{
inline library_t load_from_file(
	const char*                path,
	const link::options_t&     link_options = {},
	bool                       code_is_preserved = false)
{
	return detail_::create(
		cuLibraryLoadFromFile, path,
		[path]() { return ::std::string("file ") + path; },
		link_options, code_is_preserved);
}

inline library_t load_from_file(
	const ::std::string&    path,
	const link::options_t&  link_options = {},
	bool                    code_is_preserved = false)
{
	return load_from_file(path.c_str(), link_options, code_is_preserved);
}

#if __cplusplus >= 201703L

inline library_t load_from_file(
	const ::std::filesystem::path&  path,
	const link::options_t&          link_options = {},
	bool                            code_is_preserved = false)
{
	return load_from_file(path.c_str(), link_options, code_is_preserved);
}

#endif
///@}

namespace detail_ {

inline library_t wrap(handle_t handle, bool take_ownership) noexcept
{
	return library_t{handle, take_ownership};
}

} // namespace detail_

/**
 * Creates a new module in a context using raw compiled code
 *
 * @param module_data The raw compiled code for the module.
 * @param link_options Potential options for the PTX compilation and device linking of the code.
 * @param code_is_preserved See @ref
 */
inline library_t create(
	const void*             module_data,
	const link::options_t&  link_options = {},
	bool                    code_is_preserved = false)
{
	return detail_::create(
		cuLibraryLoadData, module_data,
		[module_data]() { return ::std::string("data at ") + cuda::detail_::ptr_as_hex(module_data); },
		link_options, code_is_preserved);
}


// TODO: Use an optional to reduce the number of functions here... when the
// library starts requiring C++14.

namespace detail_ {

inline ::std::string identify(const library_t& library)
{
	return identify(library.handle());
}

} // namespace detail_

template <typename ContiguousContainer,
	cuda::detail_::enable_if_t<cuda::detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool> >
library_t create(
	ContiguousContainer        library_data,
	optional<link::options_t>  link_options,
	bool                       code_is_preserved)
{
	return create(library_data.data(), link_options, code_is_preserved);
}

} // namespace library

} // namespace cuda

#endif // CUDA_VERSION >= 12000

#endif // CUDA_API_WRAPPERS_LIBRARY_HPP_
