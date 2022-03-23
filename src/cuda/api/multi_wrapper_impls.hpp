/**
 * @file multi_wrapper_impls.hpp
 *
 * @brief Implementations of methods or functions requiring the definitions of
 * multiple CUDA entity proxy classes. In most cases these are declared in the
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
#include <cuda/api/primary_context.hpp>
#include <cuda/api/kernel.hpp>
#include <cuda/api/apriori_compiled_kernel.hpp>
#include <cuda/api/module.hpp>
#include <cuda/api/virtual_memory.hpp>
#include <cuda/api/current_context.hpp>
#include <cuda/api/current_device.hpp>
#include <cuda/api/texture_view.hpp>
#include <cuda/api/peer_to_peer.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

#include <type_traits>
#include <vector>
#include <algorithm>

namespace cuda {

namespace detail_ {

template <typename... >
using void_t = void;

template<typename, template <typename> class, typename = void>
struct is_detected : ::std::false_type {};

template<typename T, template <typename> class Op>
struct is_detected<T, Op, void_t<Op<T>>> : ::std::true_type {};

template< class, class = void >
struct has_data : ::std::false_type { };

template< class T>
struct has_data<T, void_t<decltype(std::declval<T>().data())>>
: std::is_same<decltype(std::declval<T>().data()), void*>::type { };

} // namespace detail_

namespace array {

template <typename T, dimensionality_t NumDimensions>
array_t<T,NumDimensions> create(
	const context_t&             context,
	dimensions_t<NumDimensions>  dimensions)
{
	handle_t handle = detail_::create<T, NumDimensions>(context.handle(), dimensions);
	return wrap<T, NumDimensions>(context.device_id(), context.handle(), handle, dimensions);
}

template <typename T, dimensionality_t NumDimensions>
array_t<T,NumDimensions> create(
	device_t                     device,
	dimensions_t<NumDimensions>  dimensions)
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device.id());
	auto context_handle =  set_device_for_this_scope.primary_context_handle;
	handle_t handle = detail_::create<T, NumDimensions>(context_handle, dimensions);
	return wrap<T, NumDimensions>(device.id(), context_handle, handle, dimensions);
}

} // namespace array

namespace event {

inline event_t create(
	const context_t&  context,
	bool              uses_blocking_sync,
	bool              records_timing,
	bool              interprocess)
{
	// Yes, we need the ID explicitly even on the current device,
	// because event_t's don't have an implicit device ID.
	return event::detail_::create(context.device_id(), context.handle(), uses_blocking_sync, records_timing, interprocess);
}

inline event_t create(
	device_t&  device,
	bool       uses_blocking_sync,
	bool       records_timing,
	bool       interprocess)
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device.id());
	return event::detail_::create_in_current_context(
		device.id(),
		context::current::detail_::get_handle(),
		uses_blocking_sync, records_timing, interprocess);
}

namespace ipc {

inline handle_t export_(const event_t& event)
{
	return detail_::export_(event.handle());
}

inline event_t import(const context_t& context, const handle_t& event_ipc_handle)
{
	bool do_not_take_ownership { false };
	return event::wrap(context.device_id(), context.handle(), detail_::import(event_ipc_handle), do_not_take_ownership);
}


inline event_t import(const device_t& device, const handle_t& event_ipc_handle)
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device.id());
	auto handle = detail_::import(event_ipc_handle);
	return event::wrap(device.id(), context::current::detail_::get_handle(), handle, do_not_take_ownership);
}

} // namespace ipc

} // namespace event


// device_t methods

inline device::primary_context_t device_t::primary_context(bool scoped) const
{
    auto pc_handle = primary_context_handle();
    auto decrease_refcount_on_destruct = not scoped;
    if (not scoped) {
        device::primary_context::detail_::increase_refcount(id_);
            // Q: Why increase the refcount here, when `primary_context_handle()`
            //    ensured this has already happened for this object?
            // A: Because an unscoped primary_context_t needs its own refcount
            //    unit (e.g. in case this object gets destructed but the
            //    primary_context_t is still alive.
    }
    return device::primary_context::detail_::wrap(id_, pc_handle, decrease_refcount_on_destruct);
}

inline stream_t device_t::default_stream() const
{
    return stream::wrap(id(), primary_context_handle(), stream::default_stream_handle);
}

inline stream_t device_t::create_stream(
	bool                will_synchronize_with_default_stream,
	stream::priority_t  priority) const
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(id_);
	return stream::detail_::create(id_, primary_context_handle(), will_synchronize_with_default_stream, priority);
}

inline bool context_t::is_primary() const
{
	auto pc_handle = device::primary_context::detail_::obtain_and_increase_refcount(device_id_);
	device::primary_context::detail_::decrease_refcount(device_id_);
	return handle_ == pc_handle;
}

template <typename ContiguousContainer,
	cuda::detail_::enable_if_t<detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool>>
module_t context_t::create_module(ContiguousContainer module_data) const
{
	return module::create<ContiguousContainer>(*this, module_data);
}

template <typename ContiguousContainer,
	cuda::detail_::enable_if_t<detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool>>
module_t context_t::create_module(ContiguousContainer module_data, link::options_t link_options) const
{
	return module::create<ContiguousContainer>(*this, module_data, link_options);
}

inline void context_t::enable_access_to(const context_t& peer) const
{
	context::peer_to_peer::enable_access(*this, peer);
}

inline void context_t::disable_access_to(const context_t& peer) const
{
	context::peer_to_peer::disable_access(*this, peer);
}

inline device_t context_t::device() const
{
	return device::wrap(device_id_);
}

inline stream_t context_t::create_stream(
	bool                will_synchronize_with_default_stream,
	stream::priority_t  priority)
{
	return stream::detail_::create(device_id_, handle_, will_synchronize_with_default_stream, priority);
}

namespace device {

namespace primary_context {

inline bool is_active(const device_t& device)
{
	return detail_::is_active(device.id());
}

inline void destroy(const device_t& device)
{
	auto status = cuDevicePrimaryCtxReset(device.id());
	throw_if_error(status, "Failed destroying/resetting the primary context of device " + ::std::to_string(device.id()));
}

inline primary_context_t get(const device_t& device)
{
	auto pc_handle = detail_::get_handle(device.id(), true);
	return detail_::wrap( device.id(), pc_handle, true);
}


} // namespace primary_context

namespace peer_to_peer {

inline bool can_access(device_t accessor, device_t peer)
{
	return detail_::can_access(accessor.id(), peer.id());
}

inline void enable_access(device_t accessor, device_t peer)
{
	return context::peer_to_peer::enable_access(accessor.primary_context(), peer.primary_context());
}

inline void disable_access(device_t accessor, device_t peer)
{
#ifndef NDEBUG
	if (accessor == peer) {
		throw std::invalid_argument("A device cannot be used as its own peer");
	}
#endif
	context::peer_to_peer::disable_access(accessor.primary_context(), peer.primary_context());
}

inline bool can_access_each_other(device_t first, device_t second)
{
	return can_access(first, second) and can_access(second, first);
}

inline void enable_bidirectional_access(device_t first, device_t second)
{
#ifndef NDEBUG
	if (first == second) {
		throw std::invalid_argument("A device cannot be used as its own peer");
	}
#endif
	context::peer_to_peer::enable_bidirectional_access(first.primary_context(), second.primary_context());
}

inline void disable_bidirectional_access(device_t first, device_t second)
{
#ifndef NDEBUG
	if (first == second) {
		throw std::invalid_argument("A device cannot be used as its own peer");
	}
#endif
	context::peer_to_peer::disable_bidirectional_access(first.primary_context(), second.primary_context());
}

inline attribute_value_t get_attribute(attribute_t attribute, device_t first, device_t second)
{
#ifndef NDEBUG
	if (first == second) {
		throw std::invalid_argument("A device cannot be used as its own peer");
	}
#endif
	return detail_::get_attribute(attribute, first.id(), second.id());
}

} // namespace peer_to_peer

inline stream_t primary_context_t::default_stream() const noexcept
{
	return stream::wrap(device_id_, handle_, stream::default_stream_handle);
}

namespace current {


inline scoped_override_t::scoped_override_t(const device_t& device) : parent(device.id()) { }
inline scoped_override_t::scoped_override_t(device_t&& device) : parent(device.id()) { }

} // namespace current
} // namespace device

inline void synchronize(const device_t& device)
{
	auto pc = device.primary_context();
	context::current::detail_::scoped_override_t set_device_for_this_scope(pc.handle());
	context::current::detail_::synchronize(device.id(), pc.handle());
}

namespace detail_ {

} // namespace detail_

template <typename KernelFunction, typename ... KernelParameters>
void device_t::launch(
	KernelFunction kernel_function, launch_configuration_t launch_configuration,
	KernelParameters ... parameters) const
{
	return default_stream().enqueue.kernel_launch(
		kernel_function, launch_configuration, parameters...);
}

inline context_t device_t::create_context(
	context::host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
	bool                                            keep_larger_local_mem_after_resize) const
{
	return context::create(*this, synch_scheduling_policy, keep_larger_local_mem_after_resize);
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

inline device_t event_t::device() const
{
	return cuda::device::get(device_id());
}

inline context_t event_t::context() const
{
	constexpr const bool dont_take_ownership { false };
	return context::wrap(device_id(), context_handle_, dont_take_ownership);
}



inline void event_t::record(const stream_t& stream) const
{
	// Note:
	// TODO: Perhaps check the context match here, rather than have the Runtime API call fail?
	event::detail_::enqueue(stream.handle(), handle_);
}

inline void event_t::fire(const stream_t& stream) const
{
	record(stream);
	stream.synchronize();
}

// stream_t methods

inline device_t stream_t::device() const noexcept
{
	return cuda::device::wrap(device_id_);
}

inline context_t stream_t::context() const noexcept
{
	constexpr const bool dont_take_ownership { false };
	return context::wrap(device_id_, context_handle_, dont_take_ownership);
}

inline void stream_t::enqueue_t::wait(const event_t& event_)
{
	auto device_id = associated_stream.device_id_;
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device_id);

	// Required by the CUDA runtime API; the flags value is currently unused
	constexpr const unsigned int flags = 0;

	auto status = cuStreamWaitEvent(associated_stream.handle_, event_.handle(), flags);
	throw_if_error(status, "Failed scheduling a wait for " + event::detail_::identify(event_.handle())
		+ " on " + stream::detail_::identify(associated_stream));

}

inline event_t& stream_t::enqueue_t::event(event_t& existing_event)
{
	auto device_id = associated_stream.device_id_;
	auto context_handle = associated_stream.context_handle_;
	auto stream_context_handle_ = associated_stream.context_handle_;
	if (existing_event.context_handle() != stream_context_handle_) {
		throw ::std::invalid_argument("Attempt to enqueue "
			+ event::detail_::identify(existing_event)
			+ ", to be triggered by " + stream::detail_::identify(associated_stream));
	}
	context::current::detail_::scoped_override_t set_device_for_this_scope(context_handle);
	stream::detail_::record_event_in_current_context(device_id, context_handle,
		associated_stream.handle_,existing_event.handle());
	return existing_event;
}

inline event_t stream_t::enqueue_t::event(
    bool          uses_blocking_sync,
    bool          records_timing,
    bool          interprocess)
{
	auto context_handle = associated_stream.context_handle_;
	context::current::detail_::scoped_override_t set_device_for_this_scope(context_handle);

	event_t ev { event::detail_::create_in_current_context(
		associated_stream.device_id_, context_handle,
		uses_blocking_sync, records_timing, interprocess) };
	// Note that, at this point, the event is not associated with this enqueue object's stream.
	this->event(ev);
	return ev;
}

namespace memory {

template <typename T>
inline device_t pointer_t<T>::device() const
{
	cuda::device::id_t device_id = get_attribute<CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>();
	return cuda::device::get(device_id);
}
template <typename T>
inline pointer_t<T> pointer_t<T>::other_side_of_region_pair() const
{
	pointer::attribute_t attributes[] = {
		CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
		CU_POINTER_ATTRIBUTE_HOST_POINTER,
		CU_POINTER_ATTRIBUTE_DEVICE_POINTER
	};
	type_t memory_type;
	T* host_ptr;
	T* device_ptr;
	void* value_ptrs[] = { &memory_type, &host_ptr, &device_ptr };
	pointer::detail_::get_attributes(3, attributes, value_ptrs, ptr_);

#ifndef NDEBUG
	assert(host_ptr == ptr_ or device_ptr == ptr_);
#endif
	return { ptr_ == host_ptr ? device_ptr : host_ptr };
}


template <typename T>
inline context_t pointer_t<T>::context() const
{
	pointer::attribute_t attributes[] = {
		CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
		CU_POINTER_ATTRIBUTE_CONTEXT
	};
	cuda::device::id_t device_id;
	context::handle_t context_handle;
	void* value_ptrs[] = {&device_id, &context_handle};
	pointer::detail_::get_attributes(2, attributes, value_ptrs, ptr_);
	return context::wrap(device_id, context_handle);
}

namespace async {

inline void copy(void *destination, const void *source, size_t num_bytes, const stream_t& stream)
{
	detail_::copy(destination, source, num_bytes, stream.handle());
}

// Note: Assumes the source pointer is valid in the stream's context
template <typename T, dimensionality_t NumDimensions>
inline void copy(array_t<T, NumDimensions>& destination, const T* source, const stream_t& stream)
{
	detail_::copy<T, NumDimensions>(destination, source, stream.handle());
}

// Note: Assumes the destination, source and stream are all usable on the same content
template <typename T, dimensionality_t NumDimensions>
inline void copy(T* destination, const array_t<T, NumDimensions>& source, const stream_t& stream)
{
	if (stream.context_handle() != source.context_handle()) {
		throw std::invalid_argument("Attempt to copy an array in"
			+ context::detail_::identify(source.context_handle()) + " via "
			+ stream::detail_::identify(stream));
	}
	detail_::copy<T, NumDimensions>(destination, source, stream.handle());
}

template <typename T>
inline void copy_single(T& destination, const T& source, const stream_t& stream)
{
	detail_::copy_single(&destination, &source, sizeof(T), stream.handle());
}

} // namespace async

namespace device {

inline region_t allocate(const context_t& context, size_t size_in_bytes)
{
	return detail_::allocate(context.handle(), size_in_bytes);
}


inline region_t allocate(const device_t& device, size_t size_in_bytes)
{
	cuda::device::current::detail_::scoped_context_override_t set_device_for_this_scope{device.id()};
	return detail_::allocate_in_current_context(size_in_bytes);
}

namespace async {

inline region_t allocate(const stream_t& stream, size_t size_in_bytes)
{
	return detail_::allocate(stream.context().handle(), stream.handle(), size_in_bytes);
}

template <typename T>
inline void typed_set(T* start, const T& value, size_t num_elements, const stream_t& stream)
{
	detail_::set(start, value, num_elements, stream.handle());
}

inline void zero(void* start, size_t num_bytes, const stream_t& stream)
{
	detail_::zero(start, num_bytes, stream.handle());
}

} // namespace async


/**
 * @brief Create a variant of ::std::unique_pointer for an array in
 * device-global memory.
 *
 * @note CUDA's runtime API always has a current device; but -
 * there is not necessary a current context; so a primary context
 * for a device may be created through this call.
 *
 * @tparam T  an array type; _not_ the type of individual elements
 *
 * @param context       The CUDA device context in which to make the
 *                      allocation.
 * @param num_elements  the number of elements to allocate
 *
 * @return an ::std::unique_ptr pointing to the constructed T array
*/
template <typename T>
inline unique_ptr<T> make_unique(const context_t& context, size_t num_elements)
{
	static_assert(::std::is_array<T>::value, "make_unique<T>() can only be invoked for T being an array type, T = U[]");
	return memory::detail_::make_unique<T>(context.handle(), num_elements);
}

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
	static_assert(::std::is_array<T>::value, "make_unique<T>() can only be invoked for T being an array type, T = U[]");
	cuda::device::current::detail_::scoped_context_override_t set_device_for_this_scope(device.id());
	return memory::detail_::make_unique<T, device::detail_::allocator, device::detail_::deleter>(num_elements);
}

/**
 * @brief Create a variant of ::std::unique_pointer for an array in
 * device-global memory on the current device.
 *
 * @note The allocation will be made in the device's primary context -
 * which will be created if it has not yet been.
 *
 * @tparam T  an array type; _not_ the type of individual elements
 *
 * @param num_elements  the number of elements to allocate
 *
 * @return an ::std::unique_ptr pointing to the constructed T array
 */
template<typename T>
inline unique_ptr<T> make_unique(size_t num_elements)
{
	static_assert(::std::is_array<T>::value, "make_unique<T>() can only be invoked for T being an array type, T = U[]");
	auto device = cuda::device::current::get();
	make_unique<T>(device, num_elements);
}

/**
 * @brief Create a variant of ::std::unique_pointer for a single value
 * in device-global memory.
 *
 * @tparam T  the type of value to construct in device memory
 *
 * @param device  on which to construct the T element
 * @return an ::std::unique_ptr pointing to the allocated memory
 */
template <typename T>
inline unique_ptr<T> make_unique(const context_t& context)
{
	return cuda::memory::detail_::make_unique<T>(context.handle());
}

/**
 * @brief Create a variant of ::std::unique_pointer for a single value
 * in device-global memory.
 *
 * @tparam T  the type of value to construct in device memory
 *
 * @param device  on which to construct the T element
 * @return an ::std::unique_ptr pointing to the allocated memory
 */
template <typename T>
inline unique_ptr<T> make_unique(device_t device)
{
	cuda::device::current::detail_::scoped_context_override_t set_device_for_this_scope(device.id());
	return memory::detail_::make_unique<T, device::detail_::allocator, device::detail_::deleter>();
}

/**
 * @brief Create a variant of ::std::unique_pointer for a single value
 * in device-global memory, on the current device
 *
 * @note The allocation will be made in the device's primary context -
 * which will be created if it has not yet been.
 *
 * @tparam T  the type of value to construct in device memory
 *
 * @param num_elements  the number of elements to allocate
 *
 * @return an ::std::unique_ptr pointing to the allocated memory
 */
template<typename T>
inline unique_ptr<T> make_unique()
{
	auto device = cuda::device::current::get();
	make_unique<T>(device);
}

} // namespace device

namespace inter_context {

inline void copy(
	void *        destination_address,
	context_t     destination_context,
	const void *  source_address,
	context_t     source_context,
	size_t        num_bytes)
{
	return detail_::copy(
		destination_address, destination_context.handle(),
		source_address, source_context.handle(), num_bytes);
}

namespace async {

inline void copy(
	void *        destination_address,
	context_t     destination_context,
	const void *  source_address,
	context_t     source_context,
	size_t        num_bytes,
	stream_t      stream)
{
	return detail_::copy(
		destination_address, destination_context.handle(), source_address,
		source_context.handle(), num_bytes, stream.handle());
}

inline void copy(
	region_t        destination,
	context_t       destination_context,
	const_region_t  source,
	context_t       source_context,
	stream_t        stream)
{
#ifndef NDEBUG
	if (destination.size() < destination.size()) {
		throw ::std::invalid_argument(
			"Attempt to copy a region of " + ::std::to_string(source.size()) +
				" bytes into a region of size " + ::std::to_string(destination.size()) + " bytes");
	}
#endif
	copy(destination.start(), destination_context, source, source_context, stream);
}


inline void copy(
	void *           destination,
	context_t        destination_context,
	const_region_t   source,
	context_t        source_context,
	const stream_t&  stream)
{
	copy(destination, destination_context, source.start(), source_context, source.size(), stream);
}

} // namespace async

} // namespace inter_context


namespace managed {

namespace detail_ {

template <typename T>
inline device_t base_region_t<T>::preferred_location() const
{
	auto device_id = detail_::get_scalar_range_attribute<bool>(*this, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION);
	return cuda::device::get(device_id);
}

template <typename T>
inline void base_region_t<T>::set_preferred_location(device_t& device) const
{
	detail_::set_range_attribute(*this,CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION, device.id());
}

template <typename T>
inline void base_region_t<T>::clear_preferred_location() const
{
	detail_::unset_range_attribute(*this, CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION);
}

} // namespace detail_

template<typename T>
inline unique_ptr<T> make_unique(
	const context_t&      context,
	size_t                n,
	initial_visibility_t  initial_visibility)
{
	context::current::scoped_override_t set_device_for_this_scope(context);
	return detail_::make_unique_in_current_context<T>(n, initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
	const device_t&       device,
	size_t                n,
	initial_visibility_t  initial_visibility)
{
	cuda::device::current::detail_::scoped_context_override_t set_device_for_this_scope(device.id());
	return detail_::make_unique_in_current_context<T>(n, initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
	size_t                n,
	initial_visibility_t  initial_visibility)
{
	auto device = cuda::device::current::get();
	return make_unique<T>(device, n, initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
	const context_t&      context,
	initial_visibility_t  initial_visibility)
{
	context::current::scoped_override_t set_device_for_this_scope(context);
	return detail_::make_unique_in_current_context<T>(initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
	device_t              device,
	initial_visibility_t  initial_visibility)
{
	cuda::device::current::detail_::scoped_context_override_t set_device_for_this_scope(device.id());
	return detail_::make_unique_in_current_context<T>(initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
	initial_visibility_t  initial_visibility)
{
	auto device = cuda::device::current::get();
	return make_unique<T>(initial_visibility);
}


inline void advise_expected_access_by(const_region_t region, device_t& device)
{
	detail_::advise(region, CU_MEM_ADVISE_SET_ACCESSED_BY, device.id());
}

inline void advise_no_access_expected_by(const_region_t region, device_t& device)
{
	detail_::advise(region, CU_MEM_ADVISE_UNSET_ACCESSED_BY, device.id());
}

template <typename Allocator>
::std::vector<device_t, Allocator> accessors(const_region_t region, const Allocator& allocator)
{
	auto num_devices = cuda::device::count();
	::std::vector<device_t, Allocator> devices(num_devices, allocator);
	auto device_ids = reinterpret_cast<cuda::device::id_t *>(devices.data());

	auto status = cuMemRangeGetAttribute(
		device_ids, sizeof(device_t) * devices.size(),
		CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY, device::address(region.start()), region.size());
	throw_if_error(status, "Obtaining the IDs of devices with access to the managed memory range at "
		+ cuda::detail_::ptr_as_hex(region.start()));
	auto first_invalid_element = ::std::lower_bound(device_ids, device_ids + num_devices, cudaInvalidDeviceId);
	// We may have gotten less results that the set of all devices, so let's whittle that down

	if (first_invalid_element - device_ids != num_devices) {
		devices.resize(first_invalid_element - device_ids);
	}

	return devices;
}

namespace async {

inline void prefetch(
	const_region_t         region,
	const cuda::device_t&  destination,
	const stream_t&        stream)
{
	detail_::prefetch(region, destination.id(), stream.handle());
}

inline void prefetch_to_host(
	const_region_t   region,
	const stream_t&  stream)
{
	detail_::prefetch(region, CU_DEVICE_CPU, stream.handle());
}

} // namespace async

inline region_t allocate(
	const context_t&      context,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility)
{
	return detail_::allocate(context.handle(), num_bytes, initial_visibility);
}

inline region_t allocate(
	device_t              device,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility)
{
	cuda::device::current::detail_::scoped_context_override_t set_device_for_this_scope{device.id()};
	return detail_::allocate_in_current_context(num_bytes, initial_visibility);
}

inline region_t allocate(size_t num_bytes)
{
	auto context_handle = context::current::detail_::get_with_fallback_push();
	return allocate(context_handle, num_bytes, initial_visibility_t::to_all_devices);
}

} // namespace managed

namespace mapped {

inline region_pair allocate(
	cuda::device_t&     device,
	size_t              size_in_bytes,
	allocation_options  options)
{
    auto pc = device.primary_context();
	return cuda::memory::mapped::detail_::allocate(pc.handle(), size_in_bytes, options);
}


inline region_pair allocate(
	cuda::context_t&    context,
	size_t              size_in_bytes,
	allocation_options  options)
{
	return cuda::memory::mapped::detail_::allocate(context.handle(), size_in_bytes, options);
}

} // namespace mapped

} // namespace memory

// kernel_t methods

inline context_t kernel_t::context() const noexcept
{
	constexpr bool dont_take_ownership { false };
	return context::detail_::from_handle(context_handle_, dont_take_ownership);
}

inline device_t kernel_t::device() const noexcept
{
	return device::get(device_id_);
}

inline void kernel_t::set_attribute(kernel::attribute_t attribute, kernel::attribute_value_t value) const
{
#if CUDA_VERSION >= 9000
	context::current::detail_::scoped_override_t set_context_for_this_context(context_handle_);
	auto result = cuFuncSetAttribute(handle_, static_cast<CUfunction_attribute>(attribute), value);
	throw_if_error(result,
		"Setting CUDA device function attribute " +
#ifndef NDEBUG
		::std::string(kernel::detail_::attribute_name(attribute)) +
#else
		::std::to_string(static_cast<std::underlying_type<kernel::attribute_t>::type>(attribute)) +
#endif
		" to value " + ::std::to_string(value)	);
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif
}

/*
namespace kernel {

namespace occupancy {

inline grid::complete_dimensions_t min_grid_params_for_max_occupancy(
	const kernel_t&                   kernel,
	memory::shared::size_t            dynamic_shared_memory_size,
	grid::block_dimension_t           block_size_limit,
	bool                              disable_caching_override)
{
	return detail_::min_grid_params_for_max_occupancy(
		kernel.handle(), kernel.device().id(), dynamic_shared_memory_size, block_size_limit, disable_caching_override);
}

template <typename UnaryFunction>
grid::complete_dimensions_t
apriori_compiled_kernel_t::min_grid_params_for_max_occupancy(
	UnaryFunction            block_size_to_dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override) const
{
	return detail_::min_grid_params_for_max_occupancy(
		ptr_, device_id_, block_size_to_dynamic_shared_mem_size, block_size_limit, disable_caching_override);
}

} // namespace occupancy

} // namespace kernel
*/

namespace kernel {

template<typename KernelFunctionPtr>
apriori_compiled_kernel_t get(context_t context, KernelFunctionPtr function_ptr)
{
	static_assert(
		::std::is_pointer<KernelFunctionPtr>::value
			and ::std::is_function<typename ::std::remove_pointer<KernelFunctionPtr>::type>::value,
		"function_ptr must be a bona fide pointer to a kernel (__global__) function");

	auto ptr_ = reinterpret_cast<const void*>(function_ptr);
#if CAN_GET_APRIORI_KERNEL_HANDLE
	auto handle = detail_::get_handle(ptr_);
#else
	auto handle = nullptr;
#endif
	return detail_::wrap(context.device_id(), context.handle(), handle, ptr_);
}

template<typename KernelFunctionPtr>
apriori_compiled_kernel_t get(device_t device, KernelFunctionPtr function_ptr)
{
	return get<KernelFunctionPtr>(device.primary_context(), function_ptr);
}

} // namespace kernel


namespace stream {

namespace detail_ {

inline device::id_t device_id_of(stream::handle_t stream_handle)
{
	return context::detail_::get_device_id(context_handle_of(stream_handle));
}

inline void record_event_in_current_context(
	device::id_t       current_device_id,
	context::handle_t  current_context_handle_,
	stream::handle_t   stream_handle,
	event::handle_t    event_handle)
{
	auto status = cuEventRecord(event_handle, stream_handle);
	throw_if_error(status, "Failed scheduling " + event::detail_::identify(event_handle)
		+ " on " + stream::detail_::identify(stream_handle, current_context_handle_, current_device_id));
}

} // namespace detail_

inline stream_t create(
	const device_t&  device,
	bool             synchronizes_with_default_stream,
	priority_t       priority)
{
	cuda::device::current::detail_::scoped_context_override_t set_device_for_this_scope{device.id()};
	auto stream_handle = detail_::create_in_current_context(synchronizes_with_default_stream, priority);
	return stream::wrap(device.id(), context::current::detail_::get_handle(), stream_handle);
}

inline stream_t create(
	const context_t&  context,
	bool              synchronizes_with_default_stream,
	priority_t        priority)
{
	return detail_::create(context.device_id(), context.handle(), synchronizes_with_default_stream, priority);
}

} // namespace stream

namespace detail_ {

template<typename... KernelParameters>
void enqueue_launch_helper<apriori_compiled_kernel_t, KernelParameters...>::operator()(
	apriori_compiled_kernel_t  wrapped_kernel,
	const stream_t &           stream,
	launch_configuration_t     launch_configuration,
	KernelParameters &&...     parameters)
{
	using raw_kernel_t = typename kernel::detail_::raw_kernel_typegen<KernelParameters ...>::type;
	auto unwrapped_kernel_function = reinterpret_cast<raw_kernel_t>(const_cast<void *>(wrapped_kernel.ptr()));
	// Notes:
	// 1. The inner cast here is because we store the pointer as const void* - as an extra
	//    precaution against anybody trying to write through it. Now, function pointers
	//    can't get written through, but are still for some reason not considered const.
	// 2. We rely on the caller providing us with more-or-less the correct parameters -
	//    corresponding to the compiled kernel function's. I say "more or less" because the
	//    `KernelParameter` pack may contain some references, arrays and so on - which CUDA
	//    kernels cannot accept; so we massage those a bit.

	detail_::enqueue_raw_kernel_launch(
		unwrapped_kernel_function,
		stream.handle(),
		launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

template<typename... KernelParameters>
std::array<void*, sizeof...(KernelParameters)>
marshal_dynamic_kernel_arguments(KernelParameters&&... parameters)
{
	return ::std::array<void*, sizeof...(KernelParameters)> { &parameters... };
}

template<typename... KernelParameters>
struct enqueue_launch_helper<kernel_t, KernelParameters...> {

	void operator()(
		const kernel_t&                       wrapped_kernel,
		const stream_t &                      stream,
		launch_configuration_t                lc,
		KernelParameters &&...                parameters)
	{
		auto marshalled_arguments { marshal_dynamic_kernel_arguments(::std::forward<KernelParameters>(parameters)...) };
		auto function_handle = wrapped_kernel.handle();
		status_t status;
		if (lc.block_cooperation)
			status = cuLaunchCooperativeKernel(
				function_handle,
				lc.dimensions.grid.x,  lc.dimensions.grid.y,  lc.dimensions.grid.z,
				lc.dimensions.block.x, lc.dimensions.block.y, lc.dimensions.block.z,
				lc.dynamic_shared_memory_size,
				stream.handle(),
				marshalled_arguments.data()
			);
		else {
			constexpr const auto no_arguments_in_alternative_format = nullptr;
			// TODO: Consider passing arguments in the alternative format
			status = cuLaunchKernel(
				function_handle,
				lc.dimensions.grid.x,  lc.dimensions.grid.y,  lc.dimensions.grid.z,
				lc.dimensions.block.x, lc.dimensions.block.y, lc.dimensions.block.z,
				lc.dynamic_shared_memory_size,
				stream.handle(),
				marshalled_arguments.data(),
				no_arguments_in_alternative_format
			);
		}
		throw_if_error(status,
			(lc.block_cooperation ? "Cooperative " : "") +
				::std::string(" kernel launch failed for ") + kernel::detail_::identify(function_handle)
				+ " on " + stream::detail_::identify(stream));
	}

};

template<typename RawKernelFunction, typename... KernelParameters>
void enqueue_launch(
	::std::integral_constant<bool, false>, // Got a raw kernel function
	RawKernelFunction       kernel_function,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	detail_::enqueue_raw_kernel_launch<RawKernelFunction, KernelParameters...>(
		::std::forward<RawKernelFunction>(kernel_function), stream.handle(), launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

template<typename Kernel, typename... KernelParameters>
void enqueue_launch(
	::std::integral_constant<bool, true>, // a kernel wrapped in a kernel_t (sub)class
	Kernel                  kernel,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	enqueue_launch_helper<Kernel, KernelParameters...>{}(
		::std::forward<Kernel>(kernel), stream, launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

} // namespace detail_

#if CUDA_VERSION >= 10020
namespace memory {
namespace virtual_ {
namespace physical_allocation {

inline device_t properties_t::device() const
{
	return cuda::device::wrap(raw.location.id);
}

template<kind_t SharedHandleKind>
properties_t create_properties_for(cuda::device_t device)
{
	return detail_::create_properties<SharedHandleKind>(device.id());
}

template<kind_t SharedHandleKind>
inline physical_allocation_t create(size_t size, device_t device)
{
	auto properties = create_properties_for<SharedHandleKind>(device);
	return create(size, properties);
}

} // namespace physical_allocation

inline void set_access_mode(
	region_t fully_mapped_region,
	device_t device,
	access_mode_t access_mode)
{
	CUmemAccessDesc desc { { CU_MEM_LOCATION_TYPE_DEVICE, device.id() }, CUmemAccess_flags(access_mode) };
	constexpr const size_t count { 1 };
	auto result = cuMemSetAccess(fully_mapped_region.device_address(), fully_mapped_region.size(), &desc, count);
	throw_if_error(result, "Failed setting the access mode to the virtual memory mapping to the range of size "
		+ ::std::to_string(fully_mapped_region.size()) + " bytes at " + cuda::detail_::ptr_as_hex(fully_mapped_region.data()));
}

inline void set_access_mode(mapping_t mapping, device_t device, access_mode_t access_mode)
{
	set_access_mode(mapping.address_range(), device, access_mode);
}

template <template <typename... Ts> class Container>
inline void set_access_mode(
	region_t fully_mapped_region,
	const Container<device_t>& devices,
	access_mode_t access_mode)
{
	auto descriptors = new CUmemAccessDesc[devices.size()];
	for(std::size_t i = 0; i < devices.size(); i++) {
		descriptors[i] = {{CU_MEM_LOCATION_TYPE_DEVICE, devices[i].id()}, CUmemAccess_flags(access_mode)};
	}
	auto result = cuMemSetAccess(
		device::address(fully_mapped_region.start()), fully_mapped_region.size(), descriptors, devices.size());
	throw_if_error(result, "Failed setting the access mode to the virtual memory mapping to the range of size "
		+ ::std::to_string(fully_mapped_region.size()) + " bytes at " + cuda::detail_::ptr_as_hex(fully_mapped_region.data()));
}

template <template <typename... Ts> class Container>
inline void set_access_mode(
	region_t fully_mapped_region,
	Container<device_t>&& devices,
	access_mode_t access_mode)
{
	return set_access_mode(fully_mapped_region, devices, access_mode);
}

template <template <typename... Ts> class Container>
inline void set_access_mode(
	mapping_t mapping,
	const Container<device_t>&& devices,
	access_mode_t access_mode)
{
	set_access_mode(mapping.address_range(), devices, access_mode);
}

template <template <typename... Ts> class Container>
inline void set_access_mode(
	mapping_t mapping,
	Container<device_t>&& devices,
	access_mode_t access_mode)
{
	set_access_mode(mapping, devices, access_mode);
}

inline access_mode_t get_access_mode(region_t fully_mapped_region, device_t device)
{
	return detail_::get_access_mode(fully_mapped_region, device.id());
}

inline access_mode_t get_access_mode(mapping_t mapping, device_t device)
{
	return get_access_mode(mapping.address_range(), device);
}

inline access_mode_t mapping_t::get_access_mode(device_t device) const
{
	return virtual_::get_access_mode(*this, device);
}

inline void mapping_t::set_access_mode(device_t device, access_mode_t access_mode) const
{
	virtual_::set_access_mode(*this, device, access_mode);
}

template <template <typename... Ts> class ContiguousContainer>
void mapping_t::set_access_mode(
	const ContiguousContainer<device_t>& devices,
	access_mode_t access_mode) const
{
	virtual_::set_access_mode(*this, devices, access_mode);
}

template <template <typename... Ts> class ContiguousContainer>
void mapping_t::set_access_mode(
	ContiguousContainer<device_t>&& devices,
	access_mode_t access_mode) const
{
	virtual_::set_access_mode(*this, devices, access_mode);
}

} // namespace virtual_
} // namespace memory

#endif // CUDA_VERSION >= 10020

namespace context {

namespace detail_ {

inline handle_t get_primary_for_same_device(handle_t handle, bool increase_refcount)
{
	auto device_id = get_device_id(handle);
	return device::primary_context::detail_::get_handle(device_id, increase_refcount);
}

inline bool is_primary_for_device(handle_t handle, device::id_t device_id)
{
	auto context_device_id = context::detail_::get_device_id(handle);
	if (context_device_id != device_id) {
		return false;
	}
	constexpr const bool dont_increase_refcount { false };
	auto pc_handle = device::primary_context::detail_::get_handle(device_id, dont_increase_refcount);
	return handle == pc_handle;
}

} // namespace detail

inline bool is_primary(const context_t& context)
{
	return context::detail_::is_primary_for_device(context.handle(), context.device_id());
}

inline void synchronize(const context_t& context)
{
	return detail_::synchronize(context.device_id(), context.handle());
}

namespace current {

namespace detail_ {

/**
 * @todo This function is a bit shady, consider dropping it.s
 */
inline handle_t push_default_if_missing()
{
	auto handle = detail_::get_handle();
	if (handle != context::detail_::none) {
		return handle;
	}
	// TODO: consider using cudaSetDevice here instead
	auto current_device_id = device::current::detail_::get_id();
	auto pc_handle = device::primary_context::detail_::obtain_and_increase_refcount(current_device_id);
	push(pc_handle);
	return pc_handle;
}

/**
 * @note This specialized scope setter is used in API calls which aren't provided a context
 * as a parameter, and when there is no context that's current. Such API calls are necessarily
 * device-related (i.e. runtime-API-ish), and since there is always a current device, we can
 * (and in fact must) fall back on that device's primary context as what the user assumes we
 * would use. In these situations, we must also "leak" that device's primary context, in the
 * sense of adding to its reference count without ever decreasing it again - since the only
 * juncture at which we can decrease it is the scoped context setter's fallback; and if we
 * do that, we will actually trigger the destruction of that primary context. As a consequence,
 * if one _ever_ uses an API wrapper call which relies on this scoped context setter, the only
 * way for them to destroy the primary context is either via @ref device_t::reset() (or
 * manually decreasing the reference count to zero, which supposedly they will not do).
 *
 * @note not sure about how appropriate it is to pop the created primary context off
 */
class scoped_current_device_fallback_t {
public:
    device::id_t device_id_;
    context::handle_t pc_handle_ { context::detail_::none };

	explicit scoped_current_device_fallback_t()
	{
	    auto current_context_handle = get_handle();
	    if (current_context_handle  == context::detail_::none) {
	        device_id_ = device::current::detail_::get_id();
	        pc_handle_ = device::primary_context::detail_::obtain_and_increase_refcount(device_id_);
	        context::current::detail_::push(pc_handle_);
	    }
	}

	~scoped_current_device_fallback_t()
	{
//	    if (pc_handle_ != context::detail_::none) {
//            context::current::detail_::pop();
//	        device::primary_context::detail_::decrease_refcount(device_id_);
//	    }
	}
};

} // namespace detail_

inline scoped_override_t::scoped_override_t(const context_t &context) : parent(context.handle())
{}

inline scoped_override_t::scoped_override_t(context_t &&context) : parent(context.handle())
{}

} // namespace current

inline context_t create_and_push(
	device_t                               device,
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
	bool                                   keep_larger_local_mem_after_resize)
{
	auto handle = detail_::create_and_push(device.id(), synch_scheduling_policy, keep_larger_local_mem_after_resize);
	bool take_ownership = true;
	return context::wrap(device.id(), handle, take_ownership);
}

inline context_t create(
	device_t                               device,
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
	bool                                   keep_larger_local_mem_after_resize)
{
	auto created = create_and_push(device, synch_scheduling_policy, keep_larger_local_mem_after_resize);
	current::pop();
	return created;
}

namespace peer_to_peer {

inline bool can_access(context_t accessor, context_t peer)
{
	return device::peer_to_peer::detail_::can_access(accessor.device_id(), peer.device_id());
}

inline void enable_access(context_t accessor, context_t peer)
{
	detail_::enable_access(accessor.handle(), peer.handle());
}

inline void disable_access(context_t accessor, context_t peer)
{
	detail_::disable_access(accessor.handle(), peer.handle());
}

inline void enable_bidirectional_access(context_t first, context_t second)
{
	// Note: What happens when first and second are the same context? Or on the same device?
	enable_access(first,  second);
	enable_access(second, first );
}

inline void disable_bidirectional_access(context_t first, context_t second)
{
	// Note: What happens when first and second are the same context? Or on the same device?
	disable_access(first,  second);
	disable_access(second, first );
}


} // namespace peer_to_peer

namespace current {

namespace peer_to_peer {

inline void enable_access_to(const context_t &peer_context)
{
	context::peer_to_peer::detail_::enable_access_to(peer_context.handle());
}

inline void disable_access_to(const context_t &peer_context)
{
	context::peer_to_peer::detail_::disable_access_to(peer_context.handle());
}

} // namespace peer_to_peer

} // namespace current

} // namespace context

inline memory::region_t context_t::global_memory_type::allocate(size_t size_in_bytes)
{
	return cuda::memory::device::detail_::allocate(context_handle_, size_in_bytes);
}

inline device_t context_t::global_memory_type::associated_device() const
{
    return cuda::device::get(device_id_);
}

inline context_t context_t::global_memory_type::associated_context() const
{
    constexpr const bool non_owning { false };
    return cuda::context::wrap(device_id_, context_handle_, non_owning);
}


namespace detail_ {

template<typename Kernel>
device::primary_context_t get_implicit_primary_context(Kernel)
{
	return device::current::get().primary_context();
}

template<>
inline device::primary_context_t get_implicit_primary_context<kernel_t>(kernel_t kernel)
{
	auto context = kernel.context();
	auto device = context.device();
	auto primary_context = device.primary_context();
	if (context != primary_context) {
		throw std::logic_error("Attempt to launch a kernel associated with a non-primary context without specifying a stream associated with that context.");
	}
	return primary_context;
}

template<>
inline device::primary_context_t get_implicit_primary_context<apriori_compiled_kernel_t>(apriori_compiled_kernel_t kernel)
{
	const kernel_t& kernel_ = kernel;
	return get_implicit_primary_context(kernel_);
}

} // namespace detail_

template<typename Kernel, typename... KernelParameters>
inline void launch(
	Kernel                  kernel,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	auto primary_context = detail_::get_implicit_primary_context(kernel);
	auto stream = primary_context.default_stream();

	// Note: If Kernel is a kernel_t, and its associated device is different
	// than the current device, the next call will fail:

	enqueue_launch(
		kernel,
		stream,
		launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}


namespace memory {

namespace host {

inline void* allocate(
    size_t              size_in_bytes,
    allocation_options  options)
{
    context::current::detail_::scoped_current_device_fallback_t set_device_for_this_scope{};
    void* allocated = nullptr;
    auto flags = memory::detail_::make_cuda_host_alloc_flags(options);
    auto result = cuMemHostAlloc(&allocated, size_in_bytes, flags);
    if (is_success(result) && allocated == nullptr) {
        // Can this even happen? hopefully not
        result = static_cast<status_t>(status::named_t::unknown);
    }
    throw_if_error(result, "Failed allocating " + ::std::to_string(size_in_bytes) + " bytes of host memory");
    return allocated;
}

} // namespace host

namespace pointer {
namespace detail_ {

template<attribute_t attribute>
attribute_value_type_t <attribute> get_attribute(const void *ptr)
{
	context::current::detail_::scoped_current_device_fallback_t ensure_we_have_some_context;
	attribute_value_type_t <attribute> attribute_value;
	auto status = cuPointerGetAttribute(&attribute_value, attribute, device::address(ptr));
	throw_if_error(status, "Obtaining attribute " + ::std::to_string((int) attribute)
		+ " for pointer " + cuda::detail_::ptr_as_hex(ptr) );
	return attribute_value;
}

// TODO: Consider switching to a span with C++20
inline void get_attributes(unsigned num_attributes, pointer::attribute_t* attributes, void** value_ptrs, const void* ptr)
{
	context::current::detail_::scoped_current_device_fallback_t ensure_we_have_some_context;
	auto status = cuPointerGetAttributes( num_attributes, attributes, value_ptrs, device::address(ptr) );
	throw_if_error(status, "Obtaining multiple attributes for pointer " + cuda::detail_::ptr_as_hex(ptr));
}

} // namespace detail_
} // nasmespace pointer

} // namespace memory

// module_t methods

inline context_t module_t::context() const { return context::detail_::from_handle(context_handle_); }
inline device_t module_t::device() const { return device::get(context::detail_::get_device_id(context_handle_)); }

inline device_t texture_view::device() const
{
	return device::get(context::detail_::get_device_id(context_handle_));
}

inline context_t texture_view::context() const
{
	return context::detail_::from_handle(context_handle_);
}

template <typename T, dimensionality_t NumDimensions>
device_t array_t<T, NumDimensions>::device() const noexcept
{
	return device::get(device_id_);
}

template <typename T, dimensionality_t NumDimensions>
context_t array_t<T, NumDimensions>::context() const
{
	// TODO: Save the device id in the array_t as well.
	return context::detail_::from_handle(context_handle_);
}

namespace memory {

inline void copy(void *destination, const void *source, size_t num_bytes)
{
    context::current::detail_::scoped_current_device_fallback_t set_device_for_this_scope{};
    auto result = cuMemcpy(device::address(destination), device::address(source), num_bytes);
    // TODO: Determine whether it was from host to device, device to host etc and
    // add this information to the error string
    throw_if_error(result, "Synchronously copying data");
}

namespace device {

template <typename T>
inline void typed_set(T* start, const T& value, size_t num_elements)
{
    context::current::detail_::scoped_current_device_fallback_t set_device_for_this_scope{};
    static_assert(::std::is_trivially_copyable<T>::value, "Non-trivially-copyable types cannot be used for setting memory");
    static_assert(
        sizeof(T) == 1 or sizeof(T) == 2 or
        sizeof(T) == 4 or sizeof(T) == 8,
        "Unsupported type size - only sizes 1, 2 and 4 are supported");
    // TODO: Consider checking for alignment when compiling without NDEBUG
    status_t result {CUDA_SUCCESS};
    switch(sizeof(T)) {
        case(1): result = cuMemsetD8 (address(start), reinterpret_cast<const ::std::uint8_t& >(value), num_elements); break;
        case(2): result = cuMemsetD16(address(start), reinterpret_cast<const ::std::uint16_t&>(value), num_elements); break;
        case(4): result = cuMemsetD32(address(start), reinterpret_cast<const ::std::uint32_t&>(value), num_elements); break;
    }
    throw_if_error(result, "Setting global device memory bytes");
}

} // namespace device

} // namespace memory

namespace module {

namespace detail_{

inline device::primary_context_t get_context_for(device_t& locus) { return locus.primary_context(); }

} // namespace detail_

} // namespace module

namespace stream {

namespace detail_ {

inline ::std::string identify(const stream_t& stream)
{
	return identify(stream.handle(), stream.context().handle(), stream.device().id());
}

} // namespace detail_

} // namespace stream

#if CUDA_VERSION >= 11000

inline void copy_attributes(const stream_t &dest, const stream_t &src)
{
#ifndef NDEBUG
	if (dest.device() != src.device()) {
		throw ::std::invalid_argument("Attempt to copy attributes between streams on different devices");
	}
	if (dest.context() != src.context()) {
		throw ::std::invalid_argument("Attempt to copy attributes between streams on different contexts");
	}
#endif
	context::current::scoped_override_t set_device_for_this_scope(dest.context());
	auto status = cuStreamCopyAttributes(dest.handle(), src.handle());
	throw_if_error(status);
}

#endif // CUDA_VERSION >= 11000

#if ! CAN_GET_APRIORI_KERNEL_HANDLE

#if defined(__CUDACC__)

// Unfortunately, the CUDA runtime API does not allow for computation of the grid parameters for maximum occupancy
// from code compiled with a host-side-only compiler! See cuda_runtime.h for details

namespace detail_ {

template <typename UnaryFunction>
inline grid::complete_dimensions_t min_grid_params_for_max_occupancy(
	const void *             ptr,
	device::id_t             device_id,
	UnaryFunction            block_size_to_dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
#if CUDA_VERSION <= 10000
	throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#else
	int min_grid_size_in_blocks { 0 };
	int block_size { 0 };
		// Note: only initializing the values her because of a
		// spurious (?) compiler warning about potential uninitialized use.
	auto result = cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
		&min_grid_size_in_blocks, &block_size,
		ptr,
		block_size_to_dynamic_shared_mem_size,
		static_cast<int>(block_size_limit),
		disable_caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault
	);
	throw_if_error(result,
		"Failed obtaining parameters for a minimum-size grid for kernel " + detail_::ptr_as_hex(ptr) +
			" on device " + ::std::to_string(device_id) + ".");
	return { (grid::dimension_t) min_grid_size_in_blocks, (grid::block_dimension_t) block_size };
#endif // CUDA_VERSION <= 10000
}

inline grid::complete_dimensions_t min_grid_params_for_max_occupancy(
	const void *             ptr,
	device::id_t             device_id,
	memory::shared::size_t   dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
	auto always_need_same_shared_mem_size =
		[dynamic_shared_mem_size](::size_t) { return dynamic_shared_mem_size; };
	return min_grid_params_for_max_occupancy(
		ptr, device_id, always_need_same_shared_mem_size, block_size_limit, disable_caching_override);
}

} // namespace detail_

inline grid::complete_dimensions_t min_grid_params_for_max_occupancy(
	const apriori_compiled_kernel_t&          kernel,
	memory::shared::size_t   dynamic_shared_memory_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
	return detail_::min_grid_params_for_max_occupancy(
		kernel.ptr(), kernel.device().id(), dynamic_shared_memory_size, block_size_limit, disable_caching_override);
}

template <typename UnaryFunction>
grid::complete_dimensions_t min_grid_params_for_max_occupancy(
	const apriori_compiled_kernel_t& kernel,
	UnaryFunction            block_size_to_dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
	return detail_::min_grid_params_for_max_occupancy(
		kernel.ptr(), kernel.device_id(), block_size_to_dynamic_shared_mem_size, block_size_limit, disable_caching_override);
}

inline kernel::attributes_t apriori_compiled_kernel_t::attributes() const
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device_id_);
	kernel::attributes_t function_attributes;
	auto status = cudaFuncGetAttributes(&function_attributes, ptr_);
	throw_if_error(status, "Failed obtaining attributes for a CUDA device function");
	return function_attributes;
}

inline void apriori_compiled_kernel_t::set_cache_preference(multiprocessor_cache_preference_t preference) const
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device_id_);
	auto result = cudaFuncSetCacheConfig(ptr_, (cudaFuncCache) preference);
	throw_if_error(result,
		"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
		"CUDA device function");
}

inline void apriori_compiled_kernel_t::set_shared_memory_bank_size(
	multiprocessor_shared_memory_bank_size_option_t  config) const
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device_id_);
	auto result = cudaFuncSetSharedMemConfig(ptr_, (cudaSharedMemConfig) config);
	throw_if_error(result);
}

inline void apriori_compiled_kernel_t::set_attribute(kernel::attribute_t attribute, kernel::attribute_value_t value) const
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device_id_);
	cudaFuncAttribute runtime_attribute = [attribute]() {
		switch (attribute) {
			case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
				return cudaFuncAttributeMaxDynamicSharedMemorySize;
			case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
				return cudaFuncAttributePreferredSharedMemoryCarveout;
			default:
				throw cuda::runtime_error(status::not_supported,
					"Kernel attribute " + std::to_string(attribute) + " not supported (with CUDA version "
					+ std::to_string(CUDA_VERSION));
		}
	}();
	auto result = cudaFuncSetAttribute(ptr_, runtime_attribute, value);
	throw_if_error(result, "Setting CUDA device function attribute " + ::std::to_string(attribute) + " to value " + ::std::to_string(value));
}

kernel::attribute_value_t apriori_compiled_kernel_t::get_attribute(kernel::attribute_t attribute) const
{
	kernel::attributes_t attrs = attributes();
	switch(attribute) {
		case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
			return attrs.maxThreadsPerBlock;
		case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
			return attrs.sharedSizeBytes;
		case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
			return attrs.constSizeBytes;
		case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
			return attrs.localSizeBytes;
		case CU_FUNC_ATTRIBUTE_NUM_REGS:
			return attrs.numRegs;
		case CU_FUNC_ATTRIBUTE_PTX_VERSION:
			return attrs.ptxVersion;
		case CU_FUNC_ATTRIBUTE_BINARY_VERSION:
			return attrs.binaryVersion;
		case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA:
			return attrs.cacheModeCA;
		case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
			return attrs.maxDynamicSharedSizeBytes;
		case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
			return attrs.preferredShmemCarveout;
		default:
			throw cuda::runtime_error(status::not_supported,
				::std::string("Attribute ") +
#ifdef NDEBUG
					::std::to_string(static_cast<::std::underlying_type<kernel::attribute_t>::type>(attribute))
#else
					kernel::detail_::attribute_name(attribute)
#endif
				+ " cannot be obtained for apriori-compiled kernels before CUDA version 11.0"
			);
	}
}

#endif // defined(__CUDACC__)
#endif // ! CAN_GET_APRIORI_KERNEL_HANDLE

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_HPP_

