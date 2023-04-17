/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * in the `cuda::memory` namespace.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_MEMORY_HPP_
#define MULTI_WRAPPER_IMPLS_MEMORY_HPP_

#include "context.hpp"
#include "ipc.hpp"

#include "../array.hpp"
#include "../device.hpp"
#include "../event.hpp"
#include "../pointer.hpp"
#include "../stream.hpp"
#include "../unique_ptr.hpp"
#include "../primary_context.hpp"
#include "../kernel.hpp"
#include "../virtual_memory.hpp"
#include "../memory_pool.hpp"
#include "../current_device.hpp"

namespace cuda {

namespace memory {

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
		throw ::std::invalid_argument("Attempt to copy an array in"
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
	auto pc = device.primary_context();
	return allocate(pc, size_in_bytes);
}

namespace async {
#if CUDA_VERSION >= 11020
inline region_t allocate(const stream_t& stream, size_t size_in_bytes)
{
	return detail_::allocate(stream.context().handle(), stream.handle(), size_in_bytes);
}

inline void free(const stream_t& stream, void* region_start)
{
	return detail_::free(stream.context().handle(), stream.handle(), region_start);
}
#endif // CUDA_VERSION >= 11020

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
inline unique_ptr<T> make_unique(const device_t& device, size_t num_elements)
{
	static_assert(::std::is_array<T>::value, "make_unique<T>() can only be invoked for T being an array type, T = U[]");
	auto pc = device.primary_context();
	context::current::detail_::scoped_override_t set_context_for_this_scope(pc.handle());
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
	auto current_device_id = cuda::device::current::detail_::get_id();
	auto pc = cuda::device::primary_context::detail_::leaky_get(current_device_id);
	return make_unique<T>(pc, num_elements);
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
inline unique_ptr<T> make_unique(const device_t& device)
{
	auto pc = device.primary_context();
	context::current::detail_::scoped_override_t set_context_for_this_scope(pc.handle());
	return memory::detail_::make_unique<T, device::detail_::allocator, device::detail_::deleter>();
}

/**
 * @brief Create a variant of ::std::unique_pointer for a single value
 * in device-global memory, on the current device
 *
 * @note The allocation will be made in the device's primary context -
 * which will be created if it has not yet been.
 *
 * @note If called when the device' primary context is inactive,
 * this
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
	auto current_device_id = cuda::device::current::detail_::get_id();
	auto pc = cuda::device::primary_context::detail_::leaky_get(current_device_id);
	return make_unique<T>(pc);
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
	void *           destination_address,
	context_t        destination_context,
	const void *     source_address,
	context_t        source_context,
	size_t           num_bytes,
	const stream_t&  stream)
{
	return detail_::copy(
	destination_address, destination_context.handle(), source_address,
	source_context.handle(), num_bytes, stream.handle());
}

inline void copy(
	region_t         destination,
	context_t        destination_context,
	const_region_t   source,
	context_t        source_context,
	const stream_t&  stream)
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
	context::current::scoped_override_t set_context_for_this_scope(context);
	return detail_::make_unique_in_current_context<T>(n, initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
	const device_t&       device,
	size_t                n,
	initial_visibility_t  initial_visibility)
{
	auto pc = device.primary_context();
	return make_unique<T>(pc, n, initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
	size_t                n,
	initial_visibility_t  initial_visibility)
{
	auto current_device_id = cuda::device::current::detail_::get_id();
	auto pc = cuda::device::primary_context::detail_::leaky_get(current_device_id);
	return make_unique<T>(pc, n, initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
	const context_t&      context,
	initial_visibility_t  initial_visibility)
{
	context::current::scoped_override_t set_context_for_this_scope(context);
	return detail_::make_unique_in_current_context<T>(initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
	const device_t&       device,
	initial_visibility_t  initial_visibility)
{
	auto pc = device.primary_context();
	context::current::detail_::scoped_override_t set_context_for_this_scope(pc.handle());
	return detail_::make_unique_in_current_context<T>(initial_visibility);
}

template<typename T>
inline unique_ptr<T> make_unique(
	initial_visibility_t  initial_visibility)
{
	auto current_device_id = cuda::device::current::detail_::get_id();
	auto pc = cuda::device::primary_context::detail_::leaky_get(current_device_id);
	return make_unique<T>(pc, initial_visibility);
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
	throw_if_error_lazy(status, "Obtaining the IDs of devices with access to the managed memory range at "
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
	const device_t&       device,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility)
{
	auto pc = device.primary_context();
	return allocate(pc, num_bytes, initial_visibility);
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

namespace host {

/**
 * @note The allocation does not keep any device context alive/active; that is
 * the caller's responsibility. However, if there is no current context, it will
 * trigger the creation of a primary context on the default device, and "leak"
 * a refcount unit for it.
 */
inline region_t allocate(
	size_t              size_in_bytes,
	allocation_options  options)
{
	static constexpr const bool dont_decrease_pc_refcount_on_destruct { false };
	context::current::detail_::scoped_existence_ensurer_t ensure_we_have_a_context{
		dont_decrease_pc_refcount_on_destruct
	};
	void* allocated = nullptr;
	auto flags = memory::detail_::make_cuda_host_alloc_flags(options);
	auto result = cuMemHostAlloc(&allocated, size_in_bytes, flags);
	if (is_success(result) && allocated == nullptr) {
		// Can this even happen? hopefully not
		result = static_cast<status_t>(status::named_t::unknown);
	}
	throw_if_error_lazy(result, "Failed allocating " + ::std::to_string(size_in_bytes) + " bytes of host memory");
	return { allocated, size_in_bytes };
}

} // namespace host

namespace pointer {
namespace detail_ {

template<attribute_t attribute>
status_and_attribute_value<attribute> get_attribute_with_status(const void *ptr)
{
	context::current::detail_::scoped_existence_ensurer_t ensure_we_have_some_context;
	attribute_value_t <attribute> attribute_value;
	auto status = cuPointerGetAttribute(&attribute_value, attribute, device::address(ptr));
	return { status, attribute_value };
}


template<attribute_t attribute>
attribute_value_t<attribute> get_attribute(const void *ptr)
{
	auto status_and_attribute_value = get_attribute_with_status<attribute>(ptr);
	throw_if_error_lazy(status_and_attribute_value.status,
		"Obtaining attribute " + ::std::to_string((int) attribute)
		+ " for pointer " + cuda::detail_::ptr_as_hex(ptr) );
	return status_and_attribute_value.value;
}

// TODO: Consider switching to a span with C++20
inline void get_attributes(unsigned num_attributes, pointer::attribute_t* attributes, void** value_ptrs, const void* ptr)
{
	context::current::detail_::scoped_existence_ensurer_t ensure_we_have_some_context;
	auto status = cuPointerGetAttributes( num_attributes, attributes, value_ptrs, device::address(ptr) );
	throw_if_error_lazy(status, "Obtaining multiple attributes for pointer " + cuda::detail_::ptr_as_hex(ptr));
}

} // namespace detail_
} // namespace pointer

inline void copy(void *destination, const void *source, size_t num_bytes)
{
	context::current::detail_::scoped_existence_ensurer_t ensure_some_context{};
	auto result = cuMemcpy(device::address(destination), device::address(source), num_bytes);
	// TODO: Determine whether it was from host to device, device to host etc and
	// add this information to the error string
	throw_if_error_lazy(result, "Synchronously copying data");
}

namespace device {

template <typename T>
inline void typed_set(T* start, const T& value, size_t num_elements)
{
	context::current::detail_::scoped_existence_ensurer_t ensure_some_context{};
	static_assert(::std::is_trivially_copyable<T>::value, "Non-trivially-copyable types cannot be used for setting memory");
	static_assert(
	sizeof(T) == 1 or sizeof(T) == 2 or
	sizeof(T) == 4 or sizeof(T) == 8,
	"Unsupported type size - only sizes 1, 2 and 4 are supported");
	// TODO: Consider checking for alignment when compiling without NDEBUG
	status_t result {CUDA_SUCCESS};
	switch(sizeof(T)) {
		case 1: result = cuMemsetD8 (address(start), reinterpret_cast<const ::std::uint8_t& >(value), num_elements); break;
		case 2: result = cuMemsetD16(address(start), reinterpret_cast<const ::std::uint16_t&>(value), num_elements); break;
		case 4: result = cuMemsetD32(address(start), reinterpret_cast<const ::std::uint32_t&>(value), num_elements); break;
	}
	throw_if_error_lazy(result, "Setting global device memory bytes");
}

} // namespace device

inline void set(void* ptr, int byte_value, size_t num_bytes)
{
	switch ( type_of(ptr) ) {
	case device_:
//		case managed_:
	case unified_:
		memory::device::set(ptr, byte_value, num_bytes); break;
//		case unregistered_:
	case host_:
		::std::memset(ptr, byte_value, num_bytes); break;
	default:
		throw runtime_error(
			cuda::status::invalid_value,
			"CUDA returned an invalid memory type for the pointer 0x" + cuda::detail_::ptr_as_hex(ptr));
	}
}

#if CUDA_VERSION >= 10020
namespace physical_allocation {

inline device_t properties_t::device() const
{
	return cuda::device::wrap(raw.location.id);
}

template<shared_handle_kind_t SharedHandleKind>
properties_t create_properties_for(cuda::device_t device)
{
	return detail_::create_properties<SharedHandleKind>(device.id());
}

template<shared_handle_kind_t SharedHandleKind>
inline physical_allocation_t create(size_t size, device_t device)
{
	auto properties = create_properties_for<SharedHandleKind>(device);
	return create(size, properties);
}

} // namespace physical_allocation

namespace virtual_ {

inline void set_access_mode(
	region_t                     fully_mapped_region,
	device_t                     device,
	access_permissions_t         access_mode)
{
	CUmemAccessDesc desc { { CU_MEM_LOCATION_TYPE_DEVICE, device.id() }, CUmemAccess_flags(access_mode) };
	static constexpr const size_t count { 1 };
	auto result = cuMemSetAccess(fully_mapped_region.device_address(), fully_mapped_region.size(), &desc, count);
	throw_if_error_lazy(result, "Failed setting the access mode to the virtual memory mapping to the range of size "
						   + ::std::to_string(fully_mapped_region.size()) + " bytes at " + cuda::detail_::ptr_as_hex(fully_mapped_region.data()));
}

inline void set_access_mode(mapping_t mapping, device_t device, access_permissions_t access_mode)
{
	set_access_mode(mapping.address_range(), device, access_mode);
}

template <template <typename... Ts> class Container>
inline void set_access_mode(
	region_t                     fully_mapped_region,
	const Container<device_t>&   devices,
	access_permissions_t         access_mode)
{
	auto descriptors = ::std::unique_ptr<CUmemAccessDesc[]>(new CUmemAccessDesc[devices.size()]);
	for(::std::size_t i = 0; i < devices.size(); i++) {
		descriptors[i] = {{CU_MEM_LOCATION_TYPE_DEVICE, devices[i].id()}, CUmemAccess_flags(access_mode)};
	}
	auto result = cuMemSetAccess(
	device::address(fully_mapped_region.start()), fully_mapped_region.size(), descriptors.get(), devices.size());
	throw_if_error_lazy(result, "Failed setting the access mode to the virtual memory mapping to the range of size "
						   + ::std::to_string(fully_mapped_region.size()) + " bytes at " + cuda::detail_::ptr_as_hex(fully_mapped_region.data()));
}

template <template <typename... Ts> class Container>
inline void set_access_mode(
	region_t                     fully_mapped_region,
	Container<device_t>&&        devices,
	access_permissions_t         access_mode)
{
	return set_access_mode(fully_mapped_region, devices, access_mode);
}

template <template <typename... Ts> class Container>
inline void set_access_mode(
	mapping_t                    mapping,
	const Container<device_t>&&  devices,
	access_permissions_t         access_mode)
{
	set_access_mode(mapping.address_range(), devices, access_mode);
}

template <template <typename... Ts> class Container>
inline void set_access_mode(
	mapping_t                    mapping,
	Container<device_t>&&        devices,
	access_permissions_t         access_mode)
{
	set_access_mode(mapping, devices, access_mode);
}

inline access_permissions_t get_access_mode(region_t fully_mapped_region, device_t device)
{
	return detail_::get_access_mode(fully_mapped_region, device.id());
}

inline access_permissions_t get_access_mode(mapping_t mapping, device_t device)
{
	return get_access_mode(mapping.address_range(), device);
}

inline access_permissions_t mapping_t::get_access_mode(device_t device) const
{
	return virtual_::get_access_mode(*this, device);
}

inline void mapping_t::set_access_mode(device_t device, access_permissions_t access_mode) const
{
	virtual_::set_access_mode(*this, device, access_mode);
}

template <template <typename... Ts> class ContiguousContainer>
void mapping_t::set_access_mode(
	const ContiguousContainer<device_t>&  devices,
	access_permissions_t                  access_mode) const
{
	virtual_::set_access_mode(*this, devices, access_mode);
}

template <template <typename... Ts> class ContiguousContainer>
void mapping_t::set_access_mode(
	ContiguousContainer<device_t>&&       devices,
	access_permissions_t                  access_mode) const
{
	virtual_::set_access_mode(*this, devices, access_mode);
}

} // namespace virtual_
#endif // CUDA_VERSION >= 10020

#if CUDA_VERSION >= 11020
namespace pool {

template<shared_handle_kind_t SharedHandleKind>
pool_t create(const cuda::device_t& device)
{
	return detail_::create<SharedHandleKind>(device.id());
}


inline region_t allocate(const pool_t& pool, const stream_t &stream, size_t num_bytes)
{
	CUdeviceptr dptr;
	auto status = cuMemAllocFromPoolAsync(&dptr, num_bytes, pool.handle(), stream.handle());
	throw_if_error_lazy(status, "Failed scheduling an allocation of " + ::std::to_string(num_bytes)
		+ " bytes of memory from " + detail_::identify(pool) + ", on " + stream::detail_::identify(stream));
	return {as_pointer(dptr), num_bytes };
}

namespace ipc {

template <shared_handle_kind_t Kind>
shared_handle_t<Kind> export_(const pool_t& pool)
{
	shared_handle_t<Kind> result;
	static constexpr const unsigned long long flags { 0 };
	auto status = cuMemPoolExportToShareableHandle(&result, pool.handle(), (CUmemAllocationHandleType) Kind, flags);
	throw_if_error_lazy(status, "Exporting " + pool::detail_::identify(pool) +" for inter-process use");
	return result;
}

template <shared_handle_kind_t Kind>
pool_t import(const device_t& device, const shared_handle_t<Kind>& shared_pool_handle)
{
	auto handle = detail_::import<Kind>(shared_pool_handle);
	// TODO: MUST SUPPORT SAYING THIS POOL CAN'T ALLOCATE - NOT AN EXTRA FLAG IN THE POOL CLASS
	return memory::pool::wrap(device.id(), handle, do_not_take_ownership);
}

} // namespace ipc


} // namespace pool

inline region_t pool_t::allocate(const stream_t& stream, size_t num_bytes) const
{
	return pool::allocate(*this, stream, num_bytes);
}

inline cuda::device_t pool_t::device() const noexcept
{
	return cuda::device::wrap(device_id_);
}

inline pool::ipc::imported_ptr_t pool_t::import(const memory::pool::ipc::ptr_handle_t& exported_handle) const
{
	return pool::ipc::import_ptr(*this, exported_handle);
}

inline access_permissions_t access_permissions(const cuda::device_t& device, const pool_t& pool)
{
	return cuda::memory::detail_::access_permissions(device.id(), pool.handle());
}

inline void set_access_permissions(const cuda::device_t& device, const pool_t& pool, access_permissions_t permissions)
{
	if (pool.device_id() == device.id()) {
		throw ::std::invalid_argument("Cannot change the access permissions to a pool of the device "
			"on which the pool's memory is allocated (" + cuda::device::detail_::identify(device.id()) + ')');
	}
	cuda::memory::detail_::set_access_permissions(device.id(), pool.handle(), permissions);
}

template <typename DeviceRange>
void set_access_permissions(DeviceRange devices, const pool_t& pool, access_permissions_t permissions)
{
	cuda::dynarray<cuda::device::id_t> device_ids(devices.size());
	::std::transform(::std::begin(devices), ::std::end(devices), device_ids.begin());
	span<cuda::device::id_t> device_ids_span {device_ids.data(), device_ids.size()};
	cuda::memory::detail_::set_access_permissions(device_ids_span, pool.handle(), permissions);
}
#endif // #if CUDA_VERSION >= 11020

} // namespace memory

#if CUDA_VERSION >= 11020

template <memory::pool::shared_handle_kind_t Kind>
memory::pool_t device_t::create_memory_pool() const
{
	return cuda::memory::pool::detail_::create<Kind>(id_);
}

inline memory::region_t stream_t::enqueue_t::allocate(const memory::pool_t& pool, size_t num_bytes)
{
	return memory::pool::allocate(pool, associated_stream, num_bytes);
}

inline memory::pool_t device_t::default_memory_pool() const
{
	memory::pool::handle_t handle;
	auto status = cuDeviceGetDefaultMemPool(&handle, id_);
	throw_if_error_lazy(status, "Failed obtaining the default memory pool for " + device::detail_::identify(id_));
	return memory::pool::wrap(id_, handle, do_not_take_ownership);
}

#endif //  CUDA_VERSION >= 11020
} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_MEMORY_HPP_

