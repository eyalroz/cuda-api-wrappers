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
#include <cuda/api/unique_ptr.hpp>
#include <cuda/api/array.hpp>
#include <cuda/api/kernel_launch.cuh>

#include <type_traits>

namespace cuda {

namespace array {

namespace detail {

template<typename T>
cudaArray* allocate(device_t& device, array::dimensions_t<3> dimensions)
{
	device::current::scoped_override_t<> set_device_for_this_scope(device.id());
	return allocate_on_current_device<T>(dimensions);
}

template<typename T>
cudaArray* allocate(device_t& device, array::dimensions_t<2> dimensions)
{
	device::current::scoped_override_t<> set_device_for_this_scope(device.id());
	return allocate_on_current_device<T>(dimensions);
}

} // namespace detail

} // namespace array

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
inline event_t create(
	device_t&  device,
	bool       uses_blocking_sync,
	bool       records_timing,
	bool       interprocess)
{
	auto device_id = device.id();
		// Yes, we need the ID explicitly even on the current device,
		// because event_t's don't have an implicit device ID.
	return event::detail::create(device_id , uses_blocking_sync, records_timing, interprocess);
}

namespace ipc {

inline handle_t export_(event_t& event)
{
	return detail::export_(event.id());
}

inline event_t import(device_t& device, const handle_t& handle)
{
	bool do_not_take_ownership { false };
	return event::detail::wrap(device.id(), detail::import(handle), do_not_take_ownership);
}

} // namespace ipc

} // namespace event


// device_t methods

inline void device_t::synchronize(event_t& event)
{
	scoped_setter_t set_device_for_this_scope(id_);
	auto status = cudaEventSynchronize(event.id());
	throw_if_error(status, "Failed synchronizing the event with id "
		+ detail::ptr_as_hex(event.id()) + " on   " + device_id_as_str());
}

inline void device_t::synchronize(stream_t& stream)
{
	return synchronize_stream(stream.id());
}

inline stream_t device_t::default_stream() const noexcept
{
	// TODO: Perhaps support not-knowing our ID here as well, somehow?
	return stream_t(id(), stream::default_stream_id);
}

inline stream_t
device_t::create_stream(
	bool                will_synchronize_with_default_stream,
	stream::priority_t  priority)
{
	device::current::scoped_override_t<> set_device_for_this_scope(id_);
	constexpr const auto take_ownership = true;
	return stream::detail::wrap(id(), stream::detail::create_on_current_device(
		will_synchronize_with_default_stream, priority), take_ownership);
}

namespace detail {

} // namespace detail

template <typename KernelFunction, typename ... KernelParameters>
void device_t::launch(
	bool thread_block_cooperativity,
	const KernelFunction& kernel_function, launch_configuration_t launch_configuration,
	KernelParameters ... parameters)
{
	return default_stream().enqueue.kernel_launch(
		thread_block_cooperativity, kernel_function, launch_configuration, parameters...);
}

inline event_t device_t::create_event(
	bool          uses_blocking_sync,
	bool          records_timing,
	bool          interprocess)
{
	// The current implementation of event::create is not super-smart,
	// but it's probably not worth it trying to improve just this function
	return event::create(*this, uses_blocking_sync, records_timing, interprocess);
}

// event_t methods

inline device_t event_t::device() const { return cuda::device::get(device_id_); }

inline void event_t::record(stream_t& stream)
{
	// Note:
	// TODO: Perhaps check the device ID here, rather than
	// have the Runtime API call fail?
	event::detail::enqueue(stream.id(), id_);
}

inline void event_t::fire(stream_t& stream)
{
	record(stream);
	stream.synchronize();
}


// stream_t methods

inline device_t stream_t::device() const
{
	return cuda::device::get(device_id_);
}

inline void stream_t::enqueue_t::wait(const event_t& event_)
{
#ifndef NDEBUG
	if (event_.device_id() != associated_stream.device_id_) {
		throw std::invalid_argument("Attempt to have a stream on CUDA device "
			+ std::to_string(associated_stream.device_id_) + " wait for an event on another device ("
			"device " + std::to_string(event_.device_id()) + ")");
	}
#endif

	// Required by the CUDA runtime API; the flags value is
	// currently unused
	constexpr const unsigned int  flags = 0;

	auto status = cudaStreamWaitEvent(associated_stream.id_, event_.id(), flags);
	throw_if_error(status,
		std::string("Failed scheduling a wait for event ") + cuda::detail::ptr_as_hex(event_.id())
		+ " on stream " + cuda::detail::ptr_as_hex(associated_stream.id_)
		+ " on CUDA device " + std::to_string(associated_stream.device_id_));

}

inline event_t& stream_t::enqueue_t::event(event_t& existing_event)
{
#ifndef NDEBUG
	if (existing_event.device_id() != associated_stream.device_id_) {
		throw std::invalid_argument("Attempt to have a stream on CUDA device "
			+ std::to_string(associated_stream.device_id_) + " wait for an event on another device ("
			"device " + std::to_string(existing_event.device_id()) + ")");
	}
#endif
	auto status = cudaEventRecord(existing_event.id(), associated_stream.id_);
	throw_if_error(status,
		"Failed scheduling event " + cuda::detail::ptr_as_hex(existing_event.id()) + " to occur"
		+ " on stream " + cuda::detail::ptr_as_hex(associated_stream.id_)
		+ " on CUDA device " + std::to_string(associated_stream.device_id_));
	return existing_event;
}

inline event_t stream_t::enqueue_t::event(
    bool          uses_blocking_sync,
    bool          records_timing,
    bool          interprocess)
{
	event_t ev { event::detail::create(associated_stream.device_id_, uses_blocking_sync, records_timing, interprocess) };
	// Note that, at this point, the event is not associated with this enqueue object's stream.
	this->event(ev);
	return ev;
}

namespace memory {

template <typename T>
inline device_t pointer_t<T>::device() const
{ 
	return cuda::device::get(attributes().device); 
}

namespace async {

inline void copy(void *destination, const void *source, size_t num_bytes, stream_t& stream)
{
	detail::copy(destination, source, num_bytes, stream.id());
}

template <typename T, size_t NumDimensions>
inline void copy(array_t<T, NumDimensions>& destination, const void *source, stream_t& stream) {
	detail::copy(destination, source, stream.id());
}

template <typename T, size_t NumDimensions>
inline void copy(void* destination, const array_t<T, NumDimensions>& source, stream_t& stream) {
	detail::copy(destination, source, stream.id());
}

template <typename T>
inline void copy_single(T& destination, const T& source, stream_t& stream)
{
	detail::copy(&destination, &source, sizeof(T), stream.id());
}

} // namespace async

namespace device {

inline void* allocate(cuda::device_t device, size_t size_in_bytes)
{
	return memory::device::allocate(device, size_in_bytes);
}

namespace async {

inline void set(void* start, int byte_value, size_t num_bytes, stream_t& stream)
{
	detail::set(start, byte_value, num_bytes, stream.id());
}

inline void zero(void* start, size_t num_bytes, stream_t& stream)
{
	detail::zero(start, num_bytes, stream.id());
}

} // namespace async

} // namespace device

namespace managed {

namespace async {

inline void prefetch(
	const void*      managed_ptr,
	size_t           num_bytes,
	cuda::device_t   destination,
	cuda::stream_t&  stream)
{
	detail::prefetch(managed_ptr, num_bytes, destination.id(), stream.id());
}

} // namespace async


inline void* allocate(
	cuda::device_t        device,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility)
{
	return detail::allocate(device.id(), num_bytes, initial_visibility);
}


} // namespace managed

namespace mapped {

inline region_pair allocate(
	cuda::device_t                   device,
	size_t                           size_in_bytes,
	region_pair::allocation_options  options)
{
	return cuda::memory::mapped::allocate(device.id(), size_in_bytes, options);
}

} // namespace mapped

} // namespace memory


/**
 * @brief Sets a device function's preference of either having more L1 cache or
 * more shared memory space when executing on some device
 *
 * @param device_id the CUDA device for execution on which the preference is set
 * @param preference value to set for the device function (more cache, more L1 or make the equal)
 */
inline void device_function_t::cache_preference(
	device_t                           device,
	multiprocessor_cache_preference_t  preference)
{
	device::current::scoped_override_t<> set_device_for_this_context(device.id());
	cache_preference(preference);
}

inline void device_function_t::opt_in_to_extra_dynamic_memory(
	cuda::memory::shared::size_t  maximum_shared_memory_required_by_kernel,
	device_t                      device)
{
	device::current::scoped_override_t<> set_device_for_this_context(device.id());
	opt_in_to_extra_dynamic_memory(maximum_shared_memory_required_by_kernel);
}

inline void device_function_t::set_shared_mem_to_l1_cache_fraction(
	unsigned  shared_mem_percentage,
	device_t  device)
{
	device::current::scoped_override_t<> set_device_for_this_context(device.id());
	set_shared_mem_to_l1_cache_fraction(shared_mem_percentage);
}

namespace device_function {

inline grid::dimension_t maximum_active_blocks_per_multiprocessor(
	const device_t            device,
	const device_function_t&  device_function,
	grid::block_dimension_t   num_threads_per_block,
	memory::shared::size_t    dynamic_shared_memory_per_block,
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

namespace stream {

inline stream_t create(
	device_t    device,
	bool        synchronizes_with_default_stream,
	priority_t  priority)
{
	return detail::create(device.id(), synchronizes_with_default_stream, priority);
}

} // namespace stream

template<typename KernelFunction, typename... KernelParameters>
inline void enqueue_launch(
	bool                        thread_block_cooperation,
	KernelFunction              kernel_function,
	stream_t&                   stream,
	launch_configuration_t      launch_configuration,
	KernelParameters&&...       parameters)
{
	auto unwrapped_kernel_function =
		device_function::unwrap<
			KernelFunction,
			detail::kernel_parameter_decay_t<KernelParameters>...
		>(kernel_function);
		// Note: This helper function is necessary since we may have gotten a
		// device_function_t as KernelFunction, which is type-erased - in
		// which case we need both to obtain the raw function pointer, and determine
		// its type, i.e. un-type-erase it. Luckily, we have the KernelParameters pack
		// which - if we can trust the user - contains more-or-less the function's
		// parameter types; and kernels return `void`, which settles the whole signature.
		//
		// I say "more or less" because the KernelParameter pack may contain some
		// references, arrays and so on - which CUDA kernels cannot accept; so
		// we massage those a bit.

	detail::enqueue_launch(
		thread_block_cooperation,
		unwrapped_kernel_function,
		stream.id(),
		launch_configuration,
		std::forward<KernelParameters>(parameters)...);
}

template<typename KernelFunction, typename... KernelParameters>
inline void launch(
	KernelFunction              kernel_function,
	launch_configuration_t      launch_configuration,
	KernelParameters&&...       parameters)
{
	stream_t stream = device::current::get().default_stream();
	enqueue_launch(
		kernel_function,
		stream,
		launch_configuration,
		std::forward<KernelParameters>(parameters)...);
}

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_HPP_
