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

#include <cuda/api/array.hpp>
#include <cuda/api/device.hpp>
#include <cuda/api/event.hpp>
#include <cuda/api/kernel_launch.hpp>
#include <cuda/api/pointer.hpp>
#include <cuda/api/stream.hpp>
#include <cuda/api/unique_ptr.hpp>
#include <cuda_runtime.h>

#include <type_traits>
#include <vector>
#include <algorithm>

namespace cuda {

namespace array {

namespace detail_ {

template<typename T>
inline cudaArray* allocate(device_t& device, array::dimensions_t<3> dimensions)
{
	device::current::detail_::scoped_override_t set_device_for_this_scope(device.id());
	return allocate_on_current_device<T>(dimensions);
}

template<typename T>
inline cudaArray* allocate(device_t& device, array::dimensions_t<2> dimensions)
{
	device::current::detail_::scoped_override_t set_device_for_this_scope(device.id());
	return allocate_on_current_device<T>(dimensions);
}

} // namespace detail_

} // namespace array

namespace event {

inline event_t create(
	device_t&  device,
	bool       uses_blocking_sync,
	bool       records_timing,
	bool       interprocess)
{
	auto device_id = device.id();
		// Yes, we need the ID explicitly even on the current device,
		// because event_t's don't have an implicit device ID.
	return event::detail_::create(device_id , uses_blocking_sync, records_timing, interprocess);
}

namespace ipc {

inline handle_t export_(event_t& event)
{
	return detail_::export_(event.id());
}

inline event_t import(device_t& device, const handle_t& handle)
{
	return event::detail_::wrap(device.id(), detail_::import(handle), do_not_take_ownership);
}

} // namespace ipc

} // namespace event


// device_t methods

inline stream_t device_t::default_stream() const noexcept
{
	return stream::detail_::wrap(id(), stream::default_stream_id);
}

inline stream_t
device_t::create_stream(
	bool                will_synchronize_with_default_stream,
	stream::priority_t  priority)
{
	device::current::detail_::scoped_override_t set_device_for_this_scope(id_);
	return stream::detail_::wrap(id(), stream::detail_::create_on_current_device(
		will_synchronize_with_default_stream, priority), do_take_ownership);
}

namespace device {
namespace current {

inline scoped_override_t::scoped_override_t(device_t& device) : parent(device.id()) { }
inline scoped_override_t::scoped_override_t(device_t&& device) : parent(device.id()) { }

} // namespace current
} // namespace device


namespace detail_ {

} // namespace detail_

template <typename KernelFunction, typename ... KernelParameters>
void device_t::launch(
	bool thread_block_cooperativity,
	KernelFunction kernel_function, launch_configuration_t launch_configuration,
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

inline device_t event_t::device() const noexcept
{
	return cuda::device::get(device_id_);
}

inline void event_t::record(const stream_t& stream)
{
	// Note:
	// TODO: Perhaps check the device ID here, rather than
	// have the Runtime API call fail?
	event::detail_::enqueue(stream.id(), id_);
}

inline void event_t::fire(const stream_t& stream)
{
	record(stream);
	stream.synchronize();
}


// stream_t methods

inline device_t stream_t::device() const noexcept
{
	return cuda::device::get(device_id_);
}

inline void stream_t::enqueue_t::wait(const event_t& event_)
{
	auto device_id = associated_stream.device_id_;
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id);

	// Required by the CUDA runtime API; the flags value is currently unused
	constexpr const unsigned int flags = 0;

	auto status = cudaStreamWaitEvent(associated_stream.id_, event_.id(), flags);
	throw_if_error(status,
		::std::string("Failed scheduling a wait for event ") + cuda::detail_::ptr_as_hex(event_.id())
		+ " on stream " + cuda::detail_::ptr_as_hex(associated_stream.id_)
		+ " on CUDA device " + ::std::to_string(device_id));

}

inline event_t& stream_t::enqueue_t::event(event_t& existing_event)
{
	auto device_id = associated_stream.device_id_;
	if (existing_event.device_id() != device_id) {
		throw ::std::invalid_argument("Attempt to enqueue a CUDA event associated with device "
			+ ::std::to_string(existing_event.device_id()) + " to be triggered by a stream on CUDA device "
			+ ::std::to_string(device_id ) );
	}
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id);
	stream::detail_::record_event_on_current_device(device_id, associated_stream.id_, existing_event.id());
	return existing_event;
}

inline event_t stream_t::enqueue_t::event(
    bool          uses_blocking_sync,
    bool          records_timing,
    bool          interprocess)
{
	auto device_id = associated_stream.device_id_;
	device::current::detail_::scoped_override_t set_device_for_this_scope(device_id);

	event_t ev { event::detail_::create_on_current_device(device_id, uses_blocking_sync, records_timing, interprocess) };
	// Note that, at this point, the event is not associated with this enqueue object's stream.
	stream::detail_::record_event_on_current_device(device_id, associated_stream.id_, ev.id());
	return ev;
}

namespace memory {

template <typename T>
inline device_t pointer_t<T>::device() const noexcept
{
	return cuda::device::get(attributes().device);
}

namespace async {

inline void copy(void *destination, const void *source, size_t num_bytes, const stream_t& stream)
{
	detail_::copy(destination, source, num_bytes, stream.id());
}

template <typename T, dimensionality_t NumDimensions>
inline void copy(array_t<T, NumDimensions>& destination, const T* source, const stream_t& stream)
{
	detail_::copy(destination, source, stream.id());
}

template <typename T, dimensionality_t NumDimensions>
inline void copy(T* destination, const array_t<T, NumDimensions>& source, const stream_t& stream)
{
	detail_::copy(destination, source, stream.id());
}

template <typename T>
inline void copy_single(T& destination, const T& source, const stream_t& stream)
{
	detail_::copy_single(&destination, &source, sizeof(T), stream.id());
}

} // namespace async

namespace device {

inline region_t allocate(cuda::device_t device, size_t size_in_bytes)
{
	return detail_::allocate(device.id(), size_in_bytes);
}

namespace async {

inline region_t allocate(const stream_t& stream, size_t size_in_bytes)
{
	return detail_::allocate(stream.device().id(), stream.id(), size_in_bytes);
}

inline void set(void* start, int byte_value, size_t num_bytes, const stream_t& stream)
{
	detail_::set(start, byte_value, num_bytes, stream.id());
}

inline void zero(void* start, size_t num_bytes, const stream_t& stream)
{
	detail_::zero(start, num_bytes, stream.id());
}

} // namespace async

/**
 * @brief Create a variant of ::std::unique_pointer for an array in
 * device-global memory
 *
 * @tparam T  an array type; _not_ the type of individual elements
 *
 * @param device        on which to construct the array of elements
 * @param num_elements  the number of elements to allocate
 * @return an ::std::unique_ptr pointing to the constructed T array
 */
template<typename T>
inline unique_ptr<T> make_unique(device_t device, size_t num_elements)
{
	static_assert(::std::is_array<T>::value, "make_unique<T>(device, num_elements) can only be invoked for T being an array type, T = U[]");
	cuda::device::current::detail_::scoped_override_t set_device_for_this_scope(device.id());
	return cuda::memory::detail_::make_unique<T, detail_::allocator, detail_::deleter>(num_elements);
}

/**
 * @brief Create a variant of ::std::unique_pointer for a single value
 * in device-global memory
 *
 * @tparam T  the type of value to construct in device memory
 *
 * @param device  on which to construct the T element
 * @return an ::std::unique_ptr pointing to the allocated memory
 */
template <typename T>
inline unique_ptr<T> make_unique(device_t device)
{
	cuda::device::current::detail_::scoped_override_t set_device_for_this_scope(device.id());
	return cuda::memory::detail_::make_unique<T, detail_::allocator, detail_::deleter>();
}

} // namespace device

namespace managed {

namespace detail_ {

template <typename T>
inline device_t base_region_t<T>::preferred_location() const
{
	auto device_id = detail_::get_scalar_range_attribute<bool>(*this, cudaMemRangeAttributePreferredLocation);
	return cuda::device::get(device_id);
}

template <typename T>
inline void base_region_t<T>::set_preferred_location(device_t& device) const
{
	detail_::set_scalar_range_attribute(*this, (cudaMemoryAdvise) cudaMemAdviseSetPreferredLocation, device.id());
}

template <typename T>
inline void base_region_t<T>::clear_preferred_location() const
{
	detail_::set_scalar_range_attribute(*this, (cudaMemoryAdvise) cudaMemAdviseUnsetPreferredLocation);
}

} // namespace detail_


inline void advise_expected_access_by(const_region_t region, device_t& device)
{
	detail_::set_scalar_range_attribute(region, cudaMemAdviseSetAccessedBy, device.id());
}

inline void advise_no_access_expected_by(const_region_t region, device_t& device)
{
	detail_::set_scalar_range_attribute(region, cudaMemAdviseUnsetAccessedBy, device.id());
}

template <typename Allocator>
::std::vector<device_t, Allocator> accessors(const_region_t region, const Allocator& allocator)
{
	static_assert(sizeof(cuda::device::id_t) == sizeof(device_t), "Unexpected size difference between device IDs and their wrapper class, device_t");

	auto num_devices = cuda::device::count();
	::std::vector<device_t, Allocator> devices(num_devices, allocator);
	auto device_ids = reinterpret_cast<cuda::device::id_t *>(devices.data());


	auto status = cudaMemRangeGetAttribute(
		device_ids, sizeof(device_t) * devices.size(),
		cudaMemRangeAttributeAccessedBy, region.start(), region.size());
	throw_if_error(status, "Obtaining the IDs of devices with access to the managed memory range at " + cuda::detail_::ptr_as_hex(region.start()));
	auto first_invalid_element = ::std::lower_bound(device_ids, device_ids + num_devices, cudaInvalidDeviceId);
	// We may have gotten less results that the set of all devices, so let's whittle that down

	if (first_invalid_element - device_ids != num_devices) {
		devices.resize(first_invalid_element - device_ids);
	}

	return devices;
}

namespace async {

inline void prefetch(
	const_region_t   region,
	cuda::device_t   destination,
	const stream_t&  stream)
{
	detail_::prefetch(region, destination.id(), stream.id());
}

} // namespace async


inline region_t allocate(
	cuda::device_t        device,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility)
{
	return detail_::allocate(device.id(), num_bytes, initial_visibility);
}

} // namespace managed

namespace mapped {

inline region_pair allocate(
	cuda::device_t&     device,
	size_t              size_in_bytes,
	allocation_options  options)
{
	return cuda::memory::mapped::detail_::allocate(device.id(), size_in_bytes, options);
}

} // namespace mapped

} // namespace memory

// kernel_t methods

inline void kernel_t::set_attribute(cudaFuncAttribute attribute, int value)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	auto result = cudaFuncSetAttribute(ptr_, attribute, value);
	throw_if_error(result, "Setting CUDA device function attribute " + ::std::to_string(attribute) + " to value " + ::std::to_string(value));
}

inline void kernel_t::opt_in_to_extra_dynamic_memory(cuda::memory::shared::size_t amount_required_by_kernel)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
#if CUDART_VERSION >= 9000
	auto result = cudaFuncSetAttribute(ptr_, cudaFuncAttributeMaxDynamicSharedMemorySize, amount_required_by_kernel);
	throw_if_error(result,
		"Trying to opt-in to " + ::std::to_string(amount_required_by_kernel) + " bytes of dynamic shared memory, "
		"exceeding the maximum available on device " + ::std::to_string(device_id_) + " (consider the amount of static shared memory"
		"in use by the function).");
#else
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif
}

#if defined(__CUDACC__)
// Unfortunately, the CUDA runtime API does not allow for computation of the grid parameters for maximum occupancy
// from code compiled with a host-side-only compiler! See cuda_runtime.h for details

inline ::std::pair<grid::dimension_t, grid::block_dimension_t>
kernel_t::min_grid_params_for_max_occupancy(
	memory::shared::size_t   dynamic_shared_memory_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
#if CUDART_VERSION <= 10000
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#else
	int min_grid_size_in_blocks, block_size;
	auto result = cudaOccupancyMaxPotentialBlockSizeWithFlags(
		&min_grid_size_in_blocks, &block_size,
		ptr_,
		static_cast<::std::size_t>(dynamic_shared_memory_size),
		static_cast<int>(block_size_limit),
		disable_caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault
		);
	throw_if_error(result,
		"Failed obtaining parameters for a minimum-size grid for kernel " + detail_::ptr_as_hex(ptr_) +
		" on device " + ::std::to_string(device_id_) + ".");
	return { min_grid_size_in_blocks, block_size };
#endif
}

template <typename UnaryFunction>
::std::pair<grid::dimension_t, grid::block_dimension_t>
kernel_t::min_grid_params_for_max_occupancy(
	UnaryFunction            block_size_to_dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
#if CUDART_VERSION <= 10000
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#else
	int min_grid_size_in_blocks, block_size;
	auto result = cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
		&min_grid_size_in_blocks, &block_size,
		ptr_,
		block_size_to_dynamic_shared_mem_size,
		static_cast<int>(block_size_limit),
		disable_caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault
		);
	throw_if_error(result,
		"Failed obtaining parameters for a minimum-size grid for kernel " + detail_::ptr_as_hex(ptr_) +
		" on device " + ::std::to_string(device_id_) + ".");
	return { min_grid_size_in_blocks, block_size };
#endif
}
#endif

inline void kernel_t::set_preferred_shared_mem_fraction(unsigned shared_mem_percentage)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	if (shared_mem_percentage > 100) {
		throw ::std::invalid_argument("Percentage value can't exceed 100");
	}
#if CUDART_VERSION >= 9000
	auto result = cudaFuncSetAttribute(ptr_, cudaFuncAttributePreferredSharedMemoryCarveout, shared_mem_percentage);
	throw_if_error(result, "Trying to set the carve-out of shared memory/L1 cache memory");
#else
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif
}

inline kernel::attributes_t kernel_t::attributes() const
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	kernel::attributes_t function_attributes;
	auto status = cudaFuncGetAttributes(&function_attributes, ptr_);
	throw_if_error(status, "Failed obtaining attributes for a CUDA device function");
	return function_attributes;
}

inline void kernel_t::set_cache_preference(multiprocessor_cache_preference_t  preference)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	auto result = cudaFuncSetCacheConfig(ptr_, (cudaFuncCache) preference);
	throw_if_error(result,
		"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
		"CUDA device function");
}


inline void kernel_t::set_shared_memory_bank_size(
	multiprocessor_shared_memory_bank_size_option_t  config)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	auto result = cudaFuncSetSharedMemConfig(ptr_, (cudaSharedMemConfig) config);
	throw_if_error(result);
}

inline grid::dimension_t kernel_t::maximum_active_blocks_per_multiprocessor(
	grid::block_dimension_t   num_threads_per_block,
	memory::shared::size_t    dynamic_shared_memory_per_block,
	bool                      disable_caching_override)
{
	device::current::detail_::scoped_override_t set_device_for_this_context(device_id_);
	int result;
	unsigned int flags = disable_caching_override ?
		cudaOccupancyDisableCachingOverride : cudaOccupancyDefault;
	auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
		&result, ptr_, num_threads_per_block,
		dynamic_shared_memory_per_block, flags);
	throw_if_error(status, "Failed calculating the maximum occupancy "
		"of device function blocks per multiprocessor");
	return result;
}


template <typename DeviceFunction>
kernel_t::kernel_t(const device_t& device, DeviceFunction f, bool thread_block_cooperation)
: kernel_t(device.id(), reinterpret_cast<const void*>(f), thread_block_cooperation) { }

namespace stream {

inline stream_t create(
	device_t    device,
	bool        synchronizes_with_default_stream,
	priority_t  priority)
{
	return detail_::create(device.id(), synchronizes_with_default_stream, priority);
}

namespace detail_ {

inline void record_event_on_current_device(device::id_t device_id, stream::id_t stream_id, event::id_t event_id)
{
	auto status = cudaEventRecord(event_id, stream_id);
	throw_if_error(status,
		"Failed scheduling event " + cuda::detail_::ptr_as_hex(event_id) + " to occur"
		+ " on stream " + cuda::detail_::ptr_as_hex(stream_id)
		+ " on CUDA device " + ::std::to_string(device_id));
}
} // namespace detail_

} // namespace stream

template<typename Kernel, typename... KernelParameters>
inline void enqueue_launch(
	bool                    thread_block_cooperation,
	Kernel                  kernel_function,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	auto unwrapped_kernel_function =
		kernel::unwrap<
			Kernel,
			detail_::kernel_parameter_decay_t<KernelParameters>...
		>(kernel_function);
		// Note: This helper function is necessary since we may have gotten a
		// kernel_t as Kernel, which is type-erased - in
		// which case we need both to obtain the raw function pointer, and determine
		// its type, i.e. un-type-erase it. Luckily, we have the KernelParameters pack
		// which - if we can trust the user - contains more-or-less the function's
		// parameter types; and kernels return `void`, which settles the whole signature.
		//
		// I say "more or less" because the KernelParameter pack may contain some
		// references, arrays and so on - which CUDA kernels cannot accept; so
		// we massage those a bit.

#ifdef DEBUG
	assert(thread_block_cooperation == detail_::intrinsic_block_cooperation_value,
		"mismatched indications of whether thread block should be able to cooperate for a kernel");
#endif
	detail_::enqueue_launch(
		thread_block_cooperation,
		unwrapped_kernel_function,
		stream.id(),
		launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

template<typename Kernel, typename... KernelParameters>
inline void launch(
	Kernel                  kernel_function,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	stream_t stream = device::current::get().default_stream();
	enqueue_launch(
		kernel_function,
		stream,
		launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_HPP_

