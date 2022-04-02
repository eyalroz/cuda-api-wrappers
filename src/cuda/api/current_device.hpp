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

#include <cuda/api/constants.hpp>
#include <cuda/api/miscellany.hpp>
#include <cuda/api/current_context.hpp>
#include <cuda/api/primary_context.hpp>

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
	constexpr const id_t default_device_id { 0 };
	context::handle_t current_context_handle;
	auto status = cuCtxGetCurrent(&current_context_handle);
	if (status == CUDA_ERROR_NOT_INITIALIZED) {
		initialize_driver();
		// Should we activate and push the default device's context? probably not.
		return default_device_id;
	}
	throw_if_error(status, "Failed obtaining the current context for determining which "
		"device is active");

	if (current_context_handle == context::detail_::none) {
		// Should we activate and push the default device's context? probably not.
		return default_device_id;
	}
	return cuda::context::current::detail_::get_device_id();
	// ... which is the equivalent of doing:
	//
//	handle_t  device_id;
//	auto status = cudaGetDevice(&device_id);
//	throw_if_error(status, "Failure obtaining current device id");
//	return device_id;
}

/**
 * Set a device as the current one for the CUDA Runtime API (so that API calls
 * not specifying a device apply to it.)
 *
 * @note This replaces the current CUDA context (rather than pushing a context
 * onto the stack), so use with care.
 *
 * @note This causes a primary context for the device to be created, if it
 * doesn't already exist. I'm not entirely sure regarding the conditions under
 * which it will be destroyed, however.
 *
 * @param[in] device Numeric ID of the device to make current
 */
inline void set(id_t device_id)
{
	context::handle_t current_context_handle;
	bool have_current_context;
	auto status = cuCtxGetCurrent(&current_context_handle);
	if (status == CUDA_ERROR_NOT_INITIALIZED) {
		initialize_driver();
		// Should we activate and PUSH the default device's context? probably not.
		have_current_context = false;
	}
	else {
		have_current_context = (current_context_handle != context::detail_::none);
	}
	if (have_current_context) {
		auto current_context_device_id = context::detail_::get_device_id(current_context_handle);
		if (current_context_device_id == device_id) {
			return;
		}
	}
	auto device_pc_is_active = device::primary_context::detail_::is_active(device_id);
	bool need_refcount_increase = not device_pc_is_active;
	auto dev_pc_handle = device::primary_context::detail_::get_handle(device_id, need_refcount_increase);
	context::current::detail_::set(dev_pc_handle);


	// ... which is the equivalent of doing:
	// auto status = cudaSetDevice(device_id);
	// throw_if_error(status, "Failure setting current device to " + ::std::to_string(device_id));
}

/**
 * Set the first possible of several devices to be the current one for the CUDA Runtime API.
 *
 * @param[in] device_ids Numeric IDs of the devices to try and make current, in order
 * @param[in] num_devices The number of device IDs pointed to by @device_ids
 *
 * @note this replaces the current CUDA context (rather than pushing a context
 * onto the stack), so use with care.
 */
inline void set(const id_t* device_ids, size_t num_devices)
{
	if (num_devices > static_cast<size_t>(cuda::device::count())) {
		throw cuda::runtime_error(status::invalid_device, "More devices listed than exist on the system");
	}
	auto result = cudaSetValidDevices(const_cast<int*>(device_ids), (int) num_devices);
	throw_if_error(result, "Failure setting the current device to any of the list of "
		+ ::std::to_string(num_devices) + " devices specified");
}

/**
 * @note See the out-of-`detail_::` version of this class.
 *
 * @note Perhaps it would be better to keep a copy of the current context ID in a
 * member of this class, instead of on the stack?
 *
 * @note we have no guarantee that the context stack is not altered during
 * the lifetime of this object; but - we assume it wasn't, and it's up to the users
 * of this class to assure that's the case or face the consequences.
 *
 * @note We don't want to use the cuda::context::detail_scoped_override_t
 * as the implementation, since we're not simply pushing and popping
 */

class scoped_context_override_t {
public:
	explicit scoped_context_override_t(id_t device_id) :
		device_id_(device_id),
		refcount_was_nonzero(device::primary_context::detail_::is_active(device_id))
	{
		auto top_of_context_stack = context::current::detail_::get_handle();
		if (top_of_context_stack != context::detail_::none) {
			context::current::detail_::push(top_of_context_stack); // Yes, we're pushing a copy of the same context
		}
		device::current::detail_::set(device_id); // ... which now gets overwritten at the top of the stack
		primary_context_handle = device::primary_context::detail_::obtain_and_increase_refcount(device_id);

//		auto top_of_context_stack = context::current::detail_::get_handle();
//		device::current::detail_::set(device_id); // ... which now gets overwritten at the top of the stack
//		primary_context = device::primary_context::detail_::get_handle(device_id);
//		context::current::detail_::push(primary_context);
	}
	~scoped_context_override_t() {
		context::current::detail_::pop();
//#else
//		auto popped_context_handle = context::current::detail_::pop();
//		if (popped_context_handle != primary_context_handle) {
//			throw ::std::logic_error("Expected the top of the context stack to hold the primary context of "
//				+ device::detail_::identify(device_id_));
//		}
//#endif
		if (refcount_was_nonzero) {
			device::primary_context::detail_::decrease_refcount(device_id_);
			// We intentionally "leak" a refcount, as otherwise, the primary context
			// gets destroyed after we have created it - and we don't want that happening.
		}

	}
	device::id_t device_id_;
	primary_context::handle_t primary_context_handle;
	bool refcount_was_nonzero;
};


} // namespace detail_

/**
 * Reset the CUDA Runtime API's current device to its default value - the default device
 */
inline void set_to_default() { return detail_::set(device::default_device_id); }

void set(const device_t& device);

/**
 * A RAII-like mechanism for setting the CUDA Runtime API's current device for
 * what remains of the current scope, and changing it back to its previous value
 * when exiting the scope.
 *
 * @note The description says "RAII-like" because the reality is more complex. The
 * runtime API sets a device by overwriting the current
 */
class scoped_override_t : private detail_::scoped_context_override_t {
protected:
	using parent = detail_::scoped_context_override_t;
public:
	scoped_override_t(const device_t& device);
	scoped_override_t(device_t&& device);
	~scoped_override_t() = default;
};

/**
 * This macro will set the current device for the remainder of the scope in which it is
 * invoked, and will change it back to the previous value when exiting the scope. Use
 * it as an opaque command, which does not explicitly expose the variable defined under
 * the hood to effect this behavior.
 */
#define CUDA_DEVICE_FOR_THIS_SCOPE(_cuda_device_ctor_argument) \
	::cuda::device::current::scoped_override_t scoped_device_override( ::cuda::device_t(_cuda_device_ctor_argument) )


} // namespace current
} // namespace device
} // namespace cuda

#endif // CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_
