/**
 * @file current_device.hpp
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
#include <cuda/api/error.hpp>
#include <cuda/api/miscellany.hpp>
#include <cuda/common/types.hpp>

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
	id_t  device;
	status_t result = cudaGetDevice(&device);
	throw_if_error(result, "Failure obtaining current device index");
	return device;
}

/**
 * Set a device as the current one for the CUDA Runtime API (so that API calls
 * not specifying a device apply to it.)
 *
 * @param[in] device Numeric ID of the device to make current
 */
inline void set(id_t device)
{
	status_t result = cudaSetDevice(device);
	throw_if_error(result, "Failure setting current device to " + ::std::to_string(device));
}

/**
 * Set the first possible of several devices to be the current one for the CUDA Runtime API.
 *
 * @param[in] device_ids Numeric IDs of the devices to try and make current, in order
 * @param[in] num_devices The number of device IDs pointed to by @device_ids
 */
inline void set(const id_t* device_ids, size_t num_devices)
{
	if (num_devices > static_cast<size_t>(cuda::device::count())) {
		throw cuda::runtime_error(status::invalid_device, "More devices listed than exist on the system");
	}
	auto result = cudaSetValidDevices(const_cast<int*>(device_ids), num_devices);
	throw_if_error(result, "Failure setting the current device to any of the list of "
		+ ::std::to_string(num_devices) + " devices specified");
}

/**
 * @note See the out-of-`detail_::` version of this class.
 */
class scoped_override_t {
protected:
	static id_t  replace(id_t new_device_id)
	{
		id_t previous_device_id = device::current::detail_::get_id();
		device::current::detail_::set(new_device_id);
		return previous_device_id;
	}

public:
	scoped_override_t(id_t new_device_id) : previous_device_id(replace(new_device_id)) { }
	~scoped_override_t() {
		// Note that we have no guarantee that the current device was not
		// already replaced while this object was in scope; but - that's life.
		replace(previous_device_id);
	}
private:
	id_t  previous_device_id;
};


} // namespace detail_

/**
 * Reset the CUDA Runtime API's current device to its default value - the default device
 */
inline void set_to_default() { return detail_::set(device::default_device_id); }

void set(device_t device);

/**
 * A RAII-based mechanism for setting the CUDA Runtime API's current device for
 * what remains of the current scope, and changing it back to its previous value
 * when exiting the scope.
 */
class scoped_override_t : private detail_::scoped_override_t {
protected:
	using parent = detail_::scoped_override_t;
public:
	scoped_override_t(device_t& device);
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
