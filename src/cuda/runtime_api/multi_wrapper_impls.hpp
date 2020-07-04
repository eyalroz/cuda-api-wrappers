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

#include <cuda/runtime_api/array.hpp>
#include <cuda/runtime_api/device.hpp>
#include <cuda/runtime_api/event.hpp>
#include <cuda/runtime_api/kernel_launch.hpp>
#include <cuda/runtime_api/pointer.hpp>
#include <cuda/runtime_api/stream.hpp>
#include <cuda/runtime_api/unique_ptr.hpp>

#include <cuda_runtime.h>

#include <type_traits>
#include <vector>
#include <algorithm>

namespace cuda {

namespace array {

namespace detail {

template<typename T>
inline cudaArray* allocate(device_t& device, array::dimensions_t<3> dimensions)
{
	device::current::detail::scoped_override_t<> set_device_for_this_scope(device.id());
	return allocate_on_current_device<T>(dimensions);
}

template<typename T>
inline cudaArray* allocate(device_t& device, array::dimensions_t<2> dimensions)
{
	device::current::detail::scoped_override_t<> set_device_for_this_scope(device.id());
	return allocate_on_current_device<T>(dimensions);
}

} // namespace detail

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
	return stream::detail::wrap(id(), stream::default_stream_id);
}

inline stream_t
device_t::create_stream(
	bool                will_synchronize_with_default_stream,
	stream::priority_t  priority)
{
	device::current::detail::scoped_override_t<> set_device_for_this_scope(id_);
	constexpr const auto take_ownership = true;
	return stream::detail::wrap(id(), stream::detail::create_on_current_device(
		will_synchronize_with_default_stream, priority), take_ownership);
}

namespace device {
namespace current {

inline scoped_override_t<cuda::detail::do_not_assume_device_is_current>::scoped_override_t(device_t& device) : parent(device.id()) { }
inline scoped_override_t<cuda::detail::do_not_assume_device_is_current>::scoped_override_t(device_t&& device) : parent(device.id()) { }

} // namespace current
} // namespace device


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

inline void copy(region_t destination, region_t source, stream_t& stream)
{
	detail::copy(destination, source, stream.id());
}

template <typename T, dimensionality_t NumDimensions>
inline void copy(array_t<T, NumDimensions>& destination, const T* source, stream_t& stream)
{
	detail::copy(destination, source, stream.id());
}

template <typename T, dimensionality_t NumDimensions>
inline void copy(T* destination, const array_t<T, NumDimensions>& source, stream_t& stream)
{
	detail::copy(destination, source, stream.id());
}

template <typename T>
inline void copy_single(T& destination, const T& source, stream_t& stream)
{
	detail::copy_single(&destination, &source, sizeof(T), stream.id());
}

} // namespace async

namespace device {

inline region_t allocate(cuda::device_t device, size_t size_in_bytes)
{
	return detail::allocate(device.id(), size_in_bytes);
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

inline device_t region_t::preferred_location() const
{
	auto device_id = detail::get_scalar_range_attribute<bool>(*this, cudaMemRangeAttributePreferredLocation);
	return cuda::device::get(device_id);
}

inline void region_t::set_preferred_location(device_t& device) const
{
	detail::set_scalar_range_attribute(*this, (cudaMemoryAdvise) cudaMemAdviseSetPreferredLocation, device.id());
}

inline void region_t::clear_preferred_location() const
{
	detail::set_scalar_range_attribute(*this, (cudaMemoryAdvise) cudaMemAdviseUnsetPreferredLocation);
}

inline void region_t::advise_expected_access_by(device_t& device) const
{
	detail::set_scalar_range_attribute(*this, cudaMemAdviseSetAccessedBy, device.id());
}

inline void region_t::advise_no_access_expected_by(device_t& device) const
{
	detail::set_scalar_range_attribute(*this, cudaMemAdviseUnsetAccessedBy, device.id());
}

template <typename Allocator>
std::vector<device_t, Allocator> region_t::accessors(region_t region, const Allocator& allocator) const
{
	static_assert(sizeof(cuda::device::id_t) == sizeof(device_t), "Unexpected size difference between device IDs and their wrapper class, device_t");

	auto num_devices = cuda::device::count();
	std::vector<device_t, Allocator> devices(num_devices, allocator);
	auto device_ids = reinterpret_cast<cuda::device::id_t *>(devices.data());


	auto status = cudaMemRangeGetAttribute(
		device_ids, sizeof(device_t) * devices.size(),
		cudaMemRangeAttributeAccessedBy, region.start, region.size_in_bytes);
	throw_if_error(status, "Obtaining the IDs of devices with access to the managed memory range at " + cuda::detail::ptr_as_hex(region.start));
	auto first_invalid_element = std::lower_bound(device_ids, device_ids + num_devices, cudaInvalidDeviceId);
	// We may have gotten less results that the set of all devices, so let's whittle that down

	if (first_invalid_element - device_ids != num_devices) {
		devices.resize(first_invalid_element - device_ids);
	}

	return devices;
}

namespace async {

inline void prefetch(
	region_t         region,
	cuda::device_t   destination,
	cuda::stream_t&  stream)
{
	detail::prefetch(region, destination.id(), stream.id());
}

} // namespace async


inline region_t allocate(
	cuda::device_t        device,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility)
{
	return detail::allocate(device.id(), num_bytes, initial_visibility);
}

} // namespace managed

namespace mapped {

inline region_pair allocate(
	cuda::device_t&     device,
	size_t              size_in_bytes,
	allocation_options  options)
{
	return cuda::memory::mapped::detail::allocate(device.id(), size_in_bytes, options);
}

} // namespace mapped

} // namespace memory

// kernel_t methods

template<typename... KernelParameters>
void kernel_t::launch(
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	auto device = cuda::device::get(device_id_);
	return device.launch(thread_block_cooperation_, ptr_, launch_configuration, parameters...);
}

template<typename... KernelParameters>
void kernel_t::enqueue_launch(
	stream_t&               stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	cuda::enqueue_launch(
		thread_block_cooperation_,
		ptr_,
		stream,
		launch_configuration,
		std::forward<KernelParameters>(parameters)...);
}

inline void kernel_t::set_attribute(cudaFuncAttribute attribute, int value)
{
	device::current::detail::scoped_override_t<> set_device_for_this_context(device_id_);
	auto result = cudaFuncSetAttribute(ptr_, attribute, value);
	throw_if_error(result, "Setting CUDA device function attribute " + std::to_string(attribute) + " to value " + std::to_string(value));
}

inline void kernel_t::opt_in_to_extra_dynamic_memory(cuda::memory::shared::size_t amount_required_by_kernel)
{
	device::current::detail::scoped_override_t<> set_device_for_this_context(device_id_);
#if CUDART_VERSION >= 9000
	auto result = cudaFuncSetAttribute(ptr_, cudaFuncAttributeMaxDynamicSharedMemorySize, amount_required_by_kernel);
	throw_if_error(result,
		"Trying to opt-in to " + std::to_string(amount_required_by_kernel) + " bytes of dynamic shared memory, "
		"exceeding the maximum available on device " + std::to_string(device_id_) + " (consider the amount of static shared memory"
		"in use by the function).");
#else
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif
}

#if defined(__CUDACC__)
// Unfortunately, the CUDA runtime API does not allow for computation of the grid parameters for maximum occupancy
// from code compiled with a host-side-only compiler! See cuda_runtime.h for details

std::pair<grid::dimension_t, grid::block_dimension_t>
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
		static_cast<std::size_t>(dynamic_shared_memory_size),
		static_cast<int>(block_size_limit),
		disable_caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault
		);
	throw_if_error(result,
		"Failed obtaining parameters for a minimum-size grid for kernel " + detail::ptr_as_hex(ptr_) +
		" on device " + std::to_string(device_id_) + ".");
	return { min_grid_size_in_blocks, block_size };
#endif
}

template <typename UnaryFunction>
std::pair<grid::dimension_t, grid::block_dimension_t>
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
		"Failed obtaining parameters for a minimum-size grid for kernel " + detail::ptr_as_hex(ptr_) +
		" on device " + std::to_string(device_id_) + ".");
	return { min_grid_size_in_blocks, block_size };
#endif
}
#endif

inline void kernel_t::set_preferred_shared_mem_fraction(unsigned shared_mem_percentage)
{
	device::current::detail::scoped_override_t<> set_device_for_this_context(device_id_);
	if (shared_mem_percentage > 100) {
		throw std::invalid_argument("Percentage value can't exceed 100");
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
	device::current::detail::scoped_override_t<> set_device_for_this_context(device_id_);
	kernel::attributes_t function_attributes;
	auto status = cudaFuncGetAttributes(&function_attributes, ptr_);
	throw_if_error(status, "Failed obtaining attributes for a CUDA device function");
	return function_attributes;
}

inline void kernel_t::set_cache_preference(multiprocessor_cache_preference_t  preference)
{
	device::current::detail::scoped_override_t<> set_device_for_this_context(device_id_);
	auto result = cudaFuncSetCacheConfig(ptr_, (cudaFuncCache) preference);
	throw_if_error(result,
		"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
		"CUDA device function");
}


inline void kernel_t::set_shared_memory_bank_size(
	multiprocessor_shared_memory_bank_size_option_t  config)
{
	device::current::detail::scoped_override_t<> set_device_for_this_context(device_id_);
	auto result = cudaFuncSetSharedMemConfig(ptr_, (cudaSharedMemConfig) config);
	throw_if_error(result);
}

inline grid::dimension_t kernel_t::maximum_active_blocks_per_multiprocessor(
	grid::block_dimension_t   num_threads_per_block,
	memory::shared::size_t    dynamic_shared_memory_per_block,
	bool                      disable_caching_override)
{
	device::current::detail::scoped_override_t<> set_device_for_this_context(device_id_);
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
	return detail::create(device.id(), synchronizes_with_default_stream, priority);
}

} // namespace stream

template<typename Kernel, typename... KernelParameters>
inline void enqueue_launch(
	bool                    thread_block_cooperation,
	Kernel                  kernel_function,
	stream_t&               stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	auto unwrapped_kernel_function =
		kernel::unwrap<
			Kernel,
			detail::kernel_parameter_decay_t<KernelParameters>...
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
	assert(thread_block_cooperation == detail::intrinsic_block_cooperation_value,
		"mismatched indications of whether thread block should be able to cooperate for a kernel");
#endif
	detail::enqueue_launch(
		thread_block_cooperation,
		unwrapped_kernel_function,
		stream.id(),
		launch_configuration,
		std::forward<KernelParameters>(parameters)...);
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
		std::forward<KernelParameters>(parameters)...);
}

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_HPP_
