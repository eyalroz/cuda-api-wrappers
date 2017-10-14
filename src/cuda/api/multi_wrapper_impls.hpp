/**
 * @file multi_wrapper_impls.hpp
 *
 * @brief Implementations of methods or functions requiring the definitions of
 * multiple CUDA entity proxy classes. In some cases these are declared in the
 * individual proxy class files, with the other classes forward-declared.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_HPP_
#define MULTI_WRAPPER_IMPLS_HPP_

#include <cuda/api/stream.hpp>
#include <cuda/api/device.hpp>
#include <cuda/api/event.hpp>

namespace cuda {

// device_t methods

template <bool AssumedCurrent>
inline void device_t<AssumedCurrent>::synchronize(event_t& event)
{
	return synchronize_event(event.id());
}

template <bool AssumedCurrent>
inline void device_t<AssumedCurrent>::synchronize(stream_t<detail::do_not_assume_device_is_current>& stream)
{
	return synchronize_stream(stream.id());
}

template <bool AssumedCurrent>
inline stream_t<AssumedCurrent> device_t<AssumedCurrent>::default_stream() const
{
	// TODO: Perhaps support not-knowing our ID here as well, somehow?
	return stream_t<AssumedCurrent>(id(), stream::default_stream_id);
}

template <bool AssumedCurrent>
inline stream_t<detail::do_not_assume_device_is_current>
device_t<AssumedCurrent>::create_stream(
	bool                will_synchronize_with_default_stream,
	stream::priority_t  priority)
{
	device::current::scoped_override_t<AssumedCurrent> set_device_for_this_scope(id_);
	return stream_t<>(id(), stream::detail::create_on_current_device(
		will_synchronize_with_default_stream, priority));
}

// event_t methods

inline device_t<detail::do_not_assume_device_is_current>
event_t::device() const { return cuda::device::get(device_id_); }

// stream_t methods

template <bool AssumesDeviceIsCurrent>
inline device_t<detail::do_not_assume_device_is_current>
stream_t<AssumesDeviceIsCurrent>::device() const { return cuda::device::get(device_id_); }

namespace event {


/**
 * @brief creates a new execution stream on a device.
 *
 * @note The CUDA API runtime defaults to creating 
 * which you synchronize on by busy-waiting. This function does
 * the same for compatibility.
 *
 * @param device The device on which to create the new stream
 * @param uses_blocking_sync When synchronizing on this new evet,
 * shall a thread busy-wait for it, or
 * @param records_timing Can this event be used to record time
 * values (e.g. duration between events)
 * @param interprocess Can multiple processes work with the constructed
 * event?
 * @return The constructed event proxy class
 */
template <bool DeviceAssumedCurrent>
inline event_t create(
	device_t<DeviceAssumedCurrent>  device,
	bool                            uses_blocking_sync = sync_by_busy_waiting,
	bool                            records_timing     = do_record_timings,
	bool                            interprocess       = not_interprocess)
{
	return create(device.id(), uses_blocking_sync, records_timing, interprocess);
}

} // namespace event_t

} // namespace cuda

#endif /* MULTI_WRAPPER_IMPLS_HPP_ */
