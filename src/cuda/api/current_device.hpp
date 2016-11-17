#ifndef CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_
#define CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_

#include "cuda/api/types.h"
#include "cuda/api/constants.h"
#include "cuda/api/error.hpp"

#include <cuda_runtime_api.h>

namespace cuda {
namespace device {
namespace current {

inline device::id_t get_id()
{
	device::id_t  device;
	status_t result = cudaGetDevice(&device);
	throw_if_error(result, "Failure obtaining current device index");
	return device;
}

inline void set(device::id_t  device)
{
	status_t result = cudaSetDevice(device);
	throw_if_error(result, "Failure setting current device to " + std::to_string(device));
}

inline void set_to_default() { return set(device::default_device_id); }

template <bool AssumedCurrent = false> class scoped_override_t;

template <>
class scoped_override_t<detail::do_not_assume_device_is_current> {
protected:
	// Note the previous device and the current one might be one and the same;
	// in that case, the push is idempotent (but who guarantees this? Hmm.)
	static inline device::id_t  push(device::id_t new_device)
	{
		device::id_t  previous_device = device::current::get_id();
		device::current::set(new_device);
		return previous_device;
	}
	static inline void pop(device::id_t  old_device) { device::current::set(old_device); }

public:
	scoped_override_t(device::id_t  device) { previous_device = push(device); }
	~scoped_override_t() { pop(previous_device); }
private:
	device::id_t  previous_device;
};

template <>
class scoped_override_t<detail::assume_device_is_current> {
public:
	scoped_override_t(device::id_t  device) { }
	~scoped_override_t() { }
};


// In user code, this could be useful. Maybe.
#define CUDA_DEVICE_FOR_THIS_SCOPE(_device_id) \
	::cuda::device::current::scoped_override_t<::cuda::detail::do_not_assume_device_is_current> scoped_device_override(_device_id)


} // namespace current
} // namespace device
} // namespace cuda

#endif /* CUDA_API_WRAPPERS_CURRENT_DEVICE_HPP_ */
