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
#include <cuda/api/pointer.hpp>

namespace cuda {

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
inline stream_t<AssumedCurrent> device_t<AssumedCurrent>::default_stream() const noexcept
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
	constexpr const auto take_ownership = true;
	return stream::wrap(id(), stream::detail::create_on_current_device(
		will_synchronize_with_default_stream, priority), take_ownership);
}

template <bool AssumedCurrent>
inline event_t device_t<AssumedCurrent>::create_event(
	bool          uses_blocking_sync,
	bool          records_timing,
	bool          interprocess)
{
	// The current implementation of event::create is not super-smart,
	// but it's probably not worth it trying to improve just this function
	return event::create<AssumedCurrent>(*this, uses_blocking_sync, records_timing, interprocess);
}

// event_t methods

inline device_t<detail::do_not_assume_device_is_current>
event_t::device() const { return cuda::device::get(device_id_); }


// stream_t methods

template <bool AssumesDeviceIsCurrent>
inline device_t<detail::do_not_assume_device_is_current>
stream_t<AssumesDeviceIsCurrent>::device() const { return cuda::device::get(device_id_); }

template <bool AssumesDeviceIsCurrent>
inline void stream_t<AssumesDeviceIsCurrent>::enqueue_t::wait(const event_t& event_)
{
#ifndef NDEBUG
	if (event_.device_id() != device_id_) {
		throw std::invalid_argument("Attempt to have a stream on CUDA device "
			+ std::to_string(device_id_) + " wait for an event on another device ("
			"device " + std::to_string(event_.device_id()) + ")");
	}
#endif

	// Required by the CUDA runtime API; the flags value is
	// currently unused
	constexpr const unsigned int  flags = 0;

	auto status = cudaStreamWaitEvent(stream_id_, event_.id(), flags);
	throw_if_error(status,
		std::string("Failed scheduling a wait for event ") + cuda::detail::ptr_as_hex(event_.id())
		+ " on stream " + cuda::detail::ptr_as_hex(stream_id_)
		+ " on CUDA device " + std::to_string(device_id_));

}

template <bool AssumesDeviceIsCurrent>
inline event_t& stream_t<AssumesDeviceIsCurrent>::enqueue_t::event(event_t& existing_event)
{
#ifndef NDEBUG
	if (existing_event.device_id() != device_id_) {
		throw std::invalid_argument("Attempt to have a stream on CUDA device "
			+ std::to_string(device_id_) + " wait for an event on another device ("
			"device " + std::to_string(existing_event.device_id()) + ")");
	}
#endif
	auto status = cudaEventRecord(existing_event.id(), stream_id_);
	throw_if_error(status,
		"Failed scheduling event " + cuda::detail::ptr_as_hex(existing_event.id()) + " to occur"
		+ " on stream " + cuda::detail::ptr_as_hex(stream_id_)
		+ " on CUDA device " + std::to_string(device_id_));
	return existing_event;
}

template <bool AssumesDeviceIsCurrent>
inline event_t stream_t<AssumesDeviceIsCurrent>::enqueue_t::event(
    bool          uses_blocking_sync,
    bool          records_timing,
    bool          interprocess)
{
	event_t ev {event::create(device_id_, uses_blocking_sync, records_timing, interprocess)};
	// so far, we've created an event which is not associated with this stream; we
	// must specifically enqueue it:
	this->event(ev);
	return ev;
}

namespace memory {

template <typename T>
inline device_t<cuda::detail::do_not_assume_device_is_current>
pointer_t<T>::device() const 
{ 
	return cuda::device::get(attributes().device); 
}

} // namespace memory

namespace device_function {

inline grid_dimension_t maximum_active_blocks_per_multiprocessor(
	device_t<>                device,
	const device_function_t&  device_function,
	grid_block_dimension_t    num_threads_per_block,
	memory::shared::size_t      dynamic_shared_memory_per_block,
	bool                      disable_caching_override)
{
	device::current::scoped_override_t<> set_device_for_this_context(device.id());
	int result;
	unsigned int flags = disable_caching_override ?
		cudaOccupancyDisableCachingOverride : cudaOccupancyDefault;
	auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
		&result, device_function.ptr(), num_threads_per_block,
		dynamic_shared_memory_per_block, flags);
	throw_if_error(status, "Failed calculating the maximum occupancy "
		"of device function blocks per multiprocessor");
	return result;
}

} // namespace device_function

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_HPP_
