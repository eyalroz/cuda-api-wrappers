/**
 * @file
 *
 * @brief Wrappers for getting and setting CUDA's choice of
 * which device is 'current'
 *
 * CUDA has one device set as 'current'; and much of the Runtime API
 * implicitly refers to that device only. This file contains wrappers
 * for getting and setting it - as standalone functions - and
 * a RAII class which can be used for setting it for the duration of
 * a scope, popping back the old setting as the scope is exited.
 *
 * @note that code for getting the current device as a CUDA device
 * proxy class is found in @ref device.hpp
 *
 * @note the scoped device setter is used extensively throughout
 * this CUDA API wrapper library.
 *
 */
#ifndef CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_
#define CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_

#include "constants.hpp"
#include "miscellany.hpp"
#include "current_context.hpp"
#include "primary_context.hpp"

#include <cuda_runtime_api.h>

namespace cuda {

///@cond
class device_t;
///@endcond

namespace device {

namespace current {

namespace detail_ {

/**
 * Obtains the numeric id of the device set as current for the CUDA Runtime API
 */
inline id_t get_id()
{
	static constexpr const id_t default_device_id { 0 };
	context::handle_t current_context_handle;
	auto status = cuCtxGetCurrent(&current_context_handle);
	if (status == CUDA_ERROR_NOT_INITIALIZED) {
		initialize_driver();
		// Should we activate and push the default device's context? probably not.
		return default_device_id;
	}
	throw_if_error_lazy(status,
		"Failed obtaining the current context for determining which device is active");

	if (current_context_handle == context::detail_::none) {
		// Should we activate and push the default device's context? probably not.
		return default_device_id;
	}
	return cuda::context::current::detail_::get_device_id();
	// ... which is the equivalent of doing:
	//
//	handle_t  device_id;
//	auto status = cudaGetDevice(&device_id);
//	throw_if_error_lazy(status, "Failure obtaining current device id");
//	return device_id;
}

/**
 * Set a device as the current one for the CUDA Runtime API (so that API calls
 * not specifying a device apply to it.)
 *
 * @note This replaces the current CUDA context (rather than pushing a context
 * onto the stack), so use with care.
 *
 * @param[in] device Numeric ID of the device to make current
 */
///@{

/**
 * @note: The primary context reference count will be increased by calling this
 * function, except if the following conditions are met:
 *
 * 1. The primary context handle was specified as a parameter (i.e.
 *    we got something other than @ref detail_::none was for nit).
 * 2. The current context is the desired device's primary context.
 *
 * USE WITH EXTRA CARE!
 */
inline context::handle_t set_with_aux_info(
	id_t device_id,
	bool driver_is_initialized,
	context::handle_t current_context_handle = context::detail_::none,
	context::handle_t device_pc_handle = context::detail_::none)
{
	if (not driver_is_initialized) {
		initialize_driver();
		device_pc_handle = device::primary_context::detail_::obtain_and_increase_refcount(device_id);
		context::current::detail_::set(device_pc_handle);
		return device_pc_handle;
	}
	if (current_context_handle != context::detail_::none) {
		if (current_context_handle == device_pc_handle) {
			return device_pc_handle;
		}
	}
	device_pc_handle = device::primary_context::detail_::obtain_and_increase_refcount(device_id);
	if (current_context_handle == device_pc_handle) {
		return device_pc_handle;
	}
	context::current::detail_::set(device_pc_handle); // Remember: This _replaces_ the current context
	return device_pc_handle;
}

/**
 * @brief Ensures activation of a device's primary context and makes that
 * context current, placing it at the top of the context stack - and
 * replacing the previous top stack element if one existed.
 *
 * @note This causes a primary context for the device to be created
 * ("activated"), if it doesn't already exist - in which case it also "leaks"
 * a reference count unit, setting the refcount at 1. On the other hand,
 * if the primary context was already active, the reference count is _not_
 * increased - regardless of whether the primary context was the current
 * context or not.
 *
 * @note This should be equivalent to `cudaSetDevice(device_id)` + error
 * checking.
 */
inline void set(id_t device_id)
{
	context::handle_t current_context_handle;
	auto status = cuCtxGetCurrent(&current_context_handle);
	bool driver_initialized = (status == CUDA_ERROR_NOT_INITIALIZED);
	set_with_aux_info(device_id, driver_initialized, current_context_handle);
	// Note: We can safely assume the refcount was increased.
}
///@}

/**
 * Set the first possible of several devices to be the current one for the CUDA Runtime API.
 *
 * @param[in] device_ids Numeric IDs of the devices to try and make current, in order
 * @param[in] num_devices The number of device IDs pointed to by @device_ids
 *
 * @note this replaces the current CUDA context (rather than pushing a context
 * onto the stack), so use with care.
 */
inline void set(const id_t *device_ids, size_t num_devices)
{
	if (num_devices > static_cast<size_t>(cuda::device::count())) {
		throw cuda::runtime_error(status::invalid_device, "More devices listed than exist on the system");
	}
	auto result = cudaSetValidDevices(const_cast<int *>(device_ids), static_cast<int>(num_devices));
	throw_if_error_lazy(result,
		"Failure setting the current device to any of the list of "
		+ ::std::to_string(num_devices) + " devices specified");
}

} // namespace detail

/**
 * Tells the CUDA runtime API to consider the specified device as the current one.
 *
 * @note this will replace the top of the context stack, if the stack isn't empty;
 * and will create/activate the device's primary context if it isn't already active.
 */
void set(const device_t& device);

/**
 * Reset the CUDA Runtime API's current device to its default value - the default device
 */
inline void set_to_default() { return detail_::set(device::default_device_id); }

/**
 * This macro will set the current device for the remainder of the scope in which it is
 * invoked, and will change it back to the previous value when exiting the scope. Use
 * it as an opaque command, which does not explicitly expose the variable defined under
 * the hood to effect this behavior.
 */
#define CUDA_DEVICE_FOR_THIS_SCOPE(_cuda_device) \
::cuda::device::current::scoped_override_t scoped_device_override{ _cuda_device }


} // namespace current
} // namespace device
} // namespace cuda

#endif // CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_
