/**
 * @file
 *
 * @brief The cuda::library::kernel_t class and related code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_IN_LIBRARY_KERNEL_HPP_
#define CUDA_API_WRAPPERS_IN_LIBRARY_KERNEL_HPP_

#if CUDA_VERSION >= 12000

#include "../library.hpp"

#include <type_traits>

namespace cuda {

///@cond
class kernel_t;
class context_t;
///@endcond

namespace library {

///@cond
class kernel_t;
///@endcond

} // namespace library

namespace detail_ {

template <typename Kernel>
struct is_library_kernel : ::std::is_same<typename ::std::decay<Kernel>::type, library::kernel_t> { };

} // namespace detail_

// TODO: Avoid the copy?
kernel_t contextualize(const library::kernel_t& kernel, const context_t& context);

namespace library {

namespace kernel {

using handle_t = CUkernel;
using cuda::kernel::attribute_t;
using cuda::kernel::attribute_value_t;
// using cuda::kernel::apriori_compiled::attributes_t;

namespace detail_ {

// Note: library kernels never hold a PC refcount unit, nor do they own anything;
// only the library wrapper owns (and it's not associated with the kernel).
kernel_t wrap(library::handle_t library_handle, kernel::handle_t handle);

inline ::std::string identify(kernel::handle_t handle)
{
	return "library kernel at " + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(library::handle_t library_handle, kernel::handle_t handle)
{
	return identify(handle) + " within " + library::detail_::identify(library_handle);
}

::std::string identify(const kernel_t &kernel);

inline ::std::pair<cuda::kernel::handle_t, status_t> contextualize_in_current_context(
	const kernel::handle_t& library_kernel_handle)
{
	cuda::kernel::handle_t contextualized_kernel_handle;
	auto status = cuKernelGetFunction(&contextualized_kernel_handle, library_kernel_handle);
	return {contextualized_kernel_handle, status};
}

inline cuda::kernel::handle_t contextualize(
	const handle_t& kernel_handle,
	const context::handle_t context_handle)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	auto handle_and_status = contextualize_in_current_context(kernel_handle);
	throw_if_error_lazy(handle_and_status.second, "Failed placing " + identify(kernel_handle) + " in "
		+ context::detail_::identify(context_handle));
	return handle_and_status.first;
}

inline attribute_value_t get_attribute(
	handle_t             library_kernel_handle,
	device::id_t         device_id,
	kernel::attribute_t  attribute)
{
	attribute_value_t value;
	auto status = cuKernelGetAttribute(&value, attribute, library_kernel_handle, device_id);
	throw_if_error_lazy(status, ::std::string("Failed getting attribute ")
		+ cuda::kernel::detail_::attribute_name(attribute) + " for " + identify(library_kernel_handle)
		+ " on " + device::detail_::identify(device_id));
	return value;
}

inline void set_attribute(
	kernel::handle_t     library_kernel_handle,
	device::id_t         device_id,
	kernel::attribute_t  attribute,
	attribute_value_t    value)
{
	auto status = cuKernelSetAttribute(attribute, value, library_kernel_handle, device_id);
	throw_if_error_lazy(status, ::std::string("Failed setting attribute ")
								+ cuda::kernel::detail_::attribute_name(attribute) + " value to " + ::std::to_string(value)
								+ " for " + identify(library_kernel_handle) + " on " + device::detail_::identify(device_id));
}

} // namespace detail

attribute_value_t get_attribute(
	const library::kernel_t&  library_kernel,
	kernel::attribute_t       attribute,
	const device_t&           device);

inline void set_attribute(
	const library::kernel_t&  library_kernel,
	kernel::attribute_t       attribute,
	const device_t&           device,
	attribute_value_t         value);

} // namespace kernel

/**
 * @brief A proxy class for compiled kernels in a loaded library, which are
 * unassociated with a device and a context.
 */
class kernel_t {
public: // getters
	kernel::handle_t handle() const noexcept { return handle_; }
	library::handle_t library_handle() const noexcept { return library_handle_; }
	library_t library() const noexcept { return library::detail_::wrap(library_handle_); }

public: // type_conversions

public: // non-mutators

#if CUDA_VERSION >= 12300
	/**
	 * Return the kernel function name as registered within its library
	 *
	 * @note This may return a mangled name if the kernel function was not declared as having C linkage.
	 */
	const char* name() const
	{
		if (name_ != nullptr) { return name_; }
		const char* result;
		auto status = cuKernelGetName(&result, handle_);
		throw_if_error_lazy(status, "Retrieving the name of " + kernel::detail_::identify(*this));
		name_ = result;
		return name_;
	}
#endif
	cuda::kernel_t contextualize(const context_t& context) const;

protected: // ctors & dtor
	kernel_t(library::handle_t library_handle, kernel::handle_t handle)
	:
	library_handle_(library_handle), handle_(handle) {}

public: // ctors & dtor
	kernel_t(const kernel_t &) = default;
	kernel_t(kernel_t&& other) = default;

public: // friends
	friend kernel_t kernel::detail_::wrap(library::handle_t, kernel::handle_t);

protected: // data members
	library::handle_t library_handle_;
	kernel::handle_t handle_;
	mutable const char* name_ { nullptr }; // The name is cached after having been retrieved for the first time
}; // kernel_t

namespace kernel {
namespace detail_ {

inline kernel_t wrap(library::handle_t library_handle, kernel::handle_t handle)
{
	return {library_handle, handle};
}

inline ::std::string identify(const kernel_t& library_kernel)
{
	return identify(library_kernel.library_handle(), library_kernel.handle());
}

} // namespace detail_

inline kernel_t get(const library_t& library, const char* name)
{
	auto kernel_handle = library::detail_::get_kernel(library.handle(), name);
	return kernel::detail_::wrap(library.handle(), kernel_handle);
}

} // namespace kernel

} // namespace library

inline library::kernel_t library_t::get_kernel(const char* name) const
{
	return library::kernel::get(*this, name);
}

inline library::kernel_t library_t::get_kernel(const ::std::string& name) const
{
	return get_kernel(name.c_str());
}

} // namespace cuda

#endif // CUDA_VERSION >= 12000

#endif // CUDA_API_WRAPPERS_IN_LIBRARY_KERNEL_HPP_

