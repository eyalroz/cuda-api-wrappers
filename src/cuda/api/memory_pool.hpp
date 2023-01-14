/**
 * @file
 *
 * @brief The @ref cuda::memory::pool_t proxy class for memory pools, and related
 * code for creating, manipulating and allocating using memory pools.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_MEMORY_POOL_HPP_
#define CUDA_API_WRAPPERS_MEMORY_POOL_HPP_

#if CUDA_VERSION >= 11020

#include "memory.hpp"

namespace cuda {

namespace memory {

class pool_t;

namespace pool {

using handle_t = cudaMemPool_t;

namespace detail_ {

/**
 * Generate a degenerate form of one of the memory pool API arguments, for
 * a given device.
 *
 * @param device_id id of the device on which the allocation is to be made.
 */
inline CUmemLocation create_mem_location(cuda::device::id_t device_id) noexcept
{
	CUmemLocation result;
	result.id = device_id;
	result.type = CU_MEM_LOCATION_TYPE_DEVICE;
	return result;
}

#if CUDA_VERSION >= 11020
template<pool::shared_handle_kind_t SharedHandleKind = pool::shared_handle_kind_t::no_export>
#else
template<pool::shared_handle_kind_t SharedHandleKind>
#endif
CUmemPoolProps create_raw_properties(cuda::device::id_t device_id) noexcept
{
	CUmemPoolProps result;

	// We set the pool properties structure to 0, since it seems the CUDA driver
	// isn't too fond of arbitrary values, e.g. in the reserved fields
	::std::memset(&result, 0, sizeof(CUmemPoolProps));

	result.location = create_mem_location(device_id);
	result.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
	result.handleTypes = static_cast<CUmemAllocationHandleType>(SharedHandleKind);
	result.win32SecurityAttributes = nullptr; // TODO: What about the case of win32_handle ?
	return result;
}

inline ::std::string identify(pool::handle_t handle)
{
	return "memory pool at " + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(pool::handle_t handle, cuda::device::id_t device_id)
{
	return identify(handle) + " on " + cuda::device::detail_::identify(device_id);
}

::std::string identify(const pool_t &pool);



} // namespace detail_

using attribute_t = CUmemPool_attribute;

namespace detail_ {

template <attribute_t attribute> struct attribute_value {};

template <> struct attribute_value<CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES>    { using type = bool; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC>          { using type = bool; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES>  { using type = bool; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_RELEASE_THRESHOLD>                  { using type = size_t; };
#if CUDA_VERSION >= 11030
template <> struct attribute_value<CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT>               { using type = size_t; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH>                  { using type = size_t; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_USED_MEM_CURRENT>                   { using type = size_t; };
template <> struct attribute_value<CU_MEMPOOL_ATTR_USED_MEM_HIGH>                      { using type = size_t; };
#endif

template <typename T> struct attribute_value_inner_type { using type = T;  };
template <> struct attribute_value_inner_type<bool> { using type = int; };
template <> struct attribute_value_inner_type<size_t> { using type = cuuint64_t; };

template <typename T>
using attribute_value_inner_type_t = typename attribute_value_inner_type<T>::type;

} // namespace detail_


template <attribute_t attribute>
using attribute_value_t = typename detail_::attribute_value<attribute>::type;

namespace detail_ {

template<attribute_t attribute>
struct status_and_attribute_value {
	status_t status;
	attribute_value_t<attribute> value;
};

template<attribute_t attribute>
status_and_attribute_value<attribute> get_attribute_with_status(handle_t pool_handle)
{
	using outer_type = attribute_value_t <attribute>;
	using inner_type = attribute_value_inner_type_t<outer_type>;
	inner_type attribute_value;
	auto status = cuMemPoolGetAttribute(pool_handle, attribute, &attribute_value);
	return { status, static_cast<outer_type>(attribute_value) };
}

template<attribute_t attribute>
attribute_value_t<attribute> get_attribute(handle_t pool_handle)
{
	auto status_and_attribute_value = get_attribute_with_status<attribute>(pool_handle);
	throw_if_error_lazy(status_and_attribute_value.status,
		"Obtaining attribute " + ::std::to_string(static_cast<int>(attribute))
		+ " of " + detail_::identify(pool_handle));
	return status_and_attribute_value.value;
}

template<attribute_t attribute>
void set_attribute(handle_t pool_handle, attribute_value_t<attribute> value)
{
	using outer_type = attribute_value_t <attribute>;
	using inner_type = typename attribute_value_inner_type<outer_type>::type;
	inner_type value_ = static_cast<inner_type>(value);
	auto status = cuMemPoolSetAttribute(pool_handle, attribute, &value_);
	throw_if_error_lazy(status, "Setting attribute " + ::std::to_string(static_cast<int>(attribute))
		+ " of " + detail_::identify(pool_handle));
}

} // namespace detail_

/**
 * @brief Wrap an existing memory pool with a `memory::pool_t` wrapper
 *
 * @param device_id The device for which/on which the pool had been created
 *     (and on whose memory allocations from the pool are made)
 * @param handle A raw CUDA memory pool handle
 * @param owning true if the proxy object needs to destroy the pool at
 * the end of its lifetime
 */
pool_t wrap(cuda::device::id_t device_id, pool::handle_t handle, bool owning) noexcept;

} // namespace pool


namespace detail_ {

inline access_permissions_t access_permissions(cuda::device::id_t device_id, pool::handle_t pool_handle)
{
	CUmemAccess_flags access_flags;
	auto mem_location = pool::detail_::create_mem_location(device_id);
	auto status = cuMemPoolGetAccess(&access_flags, pool_handle, &mem_location);
	throw_if_error_lazy(status,
		"Determining access information for " + cuda::device::detail_::identify(device_id)
		+ " to " + pool::detail_::identify(pool_handle));
	return access_permissions_t::from_access_flags(access_flags);
}

inline void set_access_permissions(span<cuda::device::id_t> device_ids, pool::handle_t pool_handle, access_permissions_t permissions)
{
	if (permissions.write and not permissions.read) {
		throw ::std::invalid_argument("Memory pool access permissions cannot be write-only");
	}

	CUmemAccess_flags flags = permissions.read ?
	   (permissions.write ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE : CU_MEM_ACCESS_FLAGS_PROT_READ) :
	   CU_MEM_ACCESS_FLAGS_PROT_NONE;

	::std::vector<CUmemAccessDesc> descriptors;
	descriptors.reserve(device_ids.size());
	// TODO: This could use a zip iterator
	for(auto device_id : device_ids) {
		CUmemAccessDesc desc;
		desc.flags = flags;
		desc.location = pool::detail_::create_mem_location(device_id);
		descriptors.push_back(desc);
	}

	auto status = cuMemPoolSetAccess(pool_handle, descriptors.data(), descriptors.size());
	throw_if_error_lazy(status,
		"Setting access permissions for " + ::std::to_string(descriptors.size())
		+ " devices to " + pool::detail_::identify(pool_handle));
}

inline void set_access_permissions(cuda::device::id_t device_id, pool::handle_t pool_handle, access_permissions_t permissions)
{
	if (permissions.write and not permissions.read) {
		throw ::std::invalid_argument("Memory pool access permissions cannot be write-only");
	}

	CUmemAccessDesc desc;
	desc.flags = permissions.read ?
		(permissions.write ?
			CU_MEM_ACCESS_FLAGS_PROT_READWRITE :
			CU_MEM_ACCESS_FLAGS_PROT_READ) :
		CU_MEM_ACCESS_FLAGS_PROT_NONE;

	desc.location = pool::detail_::create_mem_location(device_id);
	auto status = cuMemPoolSetAccess(pool_handle, &desc, 1);
	throw_if_error_lazy(status,
		"Setting access permissions for " + cuda::device::detail_::identify(device_id)
		+ " to " + pool::detail_::identify(pool_handle));
}

} // namespace detail_

access_permissions_t access_permissions(const cuda::device_t& device, const pool_t& pool);
void set_access_permissions(const cuda::device_t& device, const pool_t& pool, access_permissions_t permissions);
template <typename DeviceRange>
void set_access_permissions(DeviceRange devices, const pool_t& pool_handle, access_permissions_t permissions);

namespace pool {

struct reuse_policy_t {
	/**
	 * Allow use by the pool of its memory such that the allocating is stream-ordering-dependent on a free
	 * action (ordering in the sense of event waits and interactions with the default stream).
	 */
	bool when_dependent_on_free;

	/**
	 * Allow use in allocation of pool memory that's already been been freed when the allocation
	 * is actually taking place, even if the allocation is not stream-order-dependent on the free action.
	 */
	bool independent_but_actually_freed;

	/**
	 * Lets the memory pool internally schedule (event-based-)waiting on relevant pool memory free actions,
	 * so that it can rely on being able to reuse this memory to satisfy allocation requirements, when it
	 * decides such reliance is necessary.
	 */
	bool allow_waiting_for_frees;
};

namespace ipc {

class imported_ptr_t;

} // namespace ipc

} // namespace pool


class pool_t {

public:
	region_t allocate(const stream_t& stream, size_t num_bytes) const;

	pool::ipc::imported_ptr_t import(const memory::pool::ipc::ptr_handle_t& exported_handle) const;

	void trim(size_t min_bytes_to_keep) const
	{
		auto status = cuMemPoolTrimTo(handle_, min_bytes_to_keep);
		throw_if_error_lazy(status, "Attempting to trim " + pool::detail_::identify(*this)
			+ " down to " + ::std::to_string(min_bytes_to_keep));
	}

	template<pool::attribute_t attribute>
	pool::attribute_value_t<attribute> get_attribute() const
	{
		auto attribute_with_status = pool::detail_::get_attribute_with_status<attribute>(handle_);
		throw_if_error_lazy(attribute_with_status.status, "Failed obtaining attribute "
			+ ::std::to_string(static_cast<int>(attribute)) + " of " + pool::detail_::identify(*this));
		return attribute_with_status.value;
	}

	template<pool::attribute_t attribute>
	void set_attribute(const pool::attribute_value_t<attribute>& value) const
	{
		using outer_type = pool::attribute_value_t <attribute>;
		using inner_type = typename pool::detail_::attribute_value_inner_type<outer_type>::type;
		auto inner_value = static_cast<inner_type>(value);
		auto status = cuMemPoolSetAttribute(handle_, attribute, &inner_value);
		throw_if_error_lazy(status, "Failed setting attribute " + ::std::to_string(static_cast<int>(attribute))
			+ " of " + pool::detail_::identify(*this));
	}

	size_t release_threshold() const
	{
		return static_cast<size_t>(get_attribute<CU_MEMPOOL_ATTR_RELEASE_THRESHOLD>());
	}

	void set_release_threshold(size_t threshold) const
	{
		set_attribute<CU_MEMPOOL_ATTR_RELEASE_THRESHOLD>(threshold);
	}

	access_permissions_t access_permissions(const cuda::device_t& device)
	{
		return memory::access_permissions(device, *this);
	}

	/**
	 * Set read and write permissions from a device to the allocations from
	 * this pool
	 *
	 * @param device the device the kernels running on which are governed by these permissions
	 * @param permissions new read and write permissions to use
	 *
	 * @note This affects both future _and past_ allocations from this pool.
	 */
	///@{
	/**
	 * @param device the device the kernels running on which are governed by this new setting
	 * @param permissions new read and write permissions to use
	 */
	void set_access_permissions(const cuda::device_t& device, access_permissions_t permissions)
	{
		return memory::set_access_permissions(device, *this, permissions);
	}

	/**
	 * @param device the device the kernels running on which are governed by this new setting
	 * @param read_permission true if kernels are allowed to read from memory allocated by this pool
	 * @param write_permission true if kernels are allowed to write to memory allocated by this pool
	 */
	void set_access_permissions(const cuda::device_t& device, bool read_permission, bool write_permission)
	{
		set_access_permissions(device, access_permissions_t{read_permission, write_permission});
	}

	/**
	 * @param device the devices the kernels running on which are governed by this new setting
	 * @param permissions new read and write permissions to use
	 */
	template <typename DeviceRange>
	void set_access_permissions(DeviceRange devices, access_permissions_t permissions)
	{
		return memory::set_access_permissions(devices, *this, permissions);
	}
	///@}

public: // non-field getters

	/**
	 * Get the user-controllable aspects of the pool's policy regarding reuse of memory
	 * and actions for enabling it.
	 */
	pool::reuse_policy_t reuse_policy() const
	{
		return {
			get_attribute<CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES>(),
			get_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC>(),
			get_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES>()
		};
	}

	/**
	 * Set the user-controllable aspects of the pool's policy regarding reuse of memory
	 * and actions for enabling it.
	 */
	void set_reuse_policy(pool::reuse_policy_t reuse_policy) const
	{
		set_attribute<CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES>(reuse_policy.when_dependent_on_free);
		set_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC>(reuse_policy.independent_but_actually_freed);
		set_attribute<CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES>(reuse_policy.allow_waiting_for_frees);
	}

#if CUDA_VERSION >= 11030
	/**
	 * Get the amount of memory the pool is currently using to guarantee it can satisfy all present
	 * and future-scheduled allocations.
	 */
	size_t backing_memory_size() const {
		return get_attribute<CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT>();
	}
#endif

public: // field getters
	/**
	 * Obtain the raw CUDA driver handle for this pool
	 */
	pool::handle_t handle() const noexcept { return handle_; }
	/**
	 * Obtain the id of the device in whose global memory this pool allocates
	 */
	cuda::device::id_t device_id() const noexcept { return device_id_; }
	/**
	 * Obtain the device in whose global memory this pool allocates
	 */
	cuda::device_t device() const noexcept;
	/**
	 * Determine whether this proxy object "owns" the pool, i.e. whether
	 * it is charged with destroying it at the end of its lifetime
	 */
	bool is_owning() const noexcept { return owning_; }


public: // construction & destruction
	friend pool_t pool::wrap(cuda::device::id_t device_id, pool::handle_t handle, bool owning) noexcept;

	pool_t(const pool_t& other) = delete;

	pool_t(pool_t&& other) noexcept : pool_t(other.device_id_, other.handle_, other.owning_)
	{
		other.owning_ = false;
	}

	~pool_t()
	{
		if (owning_) {
			cuMemPoolDestroy(handle_); // Note: Ignoring any potential exception
		}
	}

protected: // constructors
	pool_t(cuda::device::id_t device_id, pool::handle_t handle, bool owning) noexcept
	: device_id_(device_id), handle_(handle), owning_(owning)
	{ }

protected: // data members
	cuda::device::id_t device_id_;
	pool::handle_t handle_;
	bool owning_;
}; // class pool_t

inline bool operator==(const pool_t& lhs, const pool_t& rhs)
{
	// Note: Not comparing the ownership status
	return lhs.device_id() == rhs.device_id() and lhs.handle() == rhs.handle();
}

inline bool operator!=(const pool_t& lhs, const pool_t& rhs)
{
	return not (lhs == rhs);
}

namespace pool {

inline pool_t wrap(cuda::device::id_t device_id, pool::handle_t handle, bool owning) noexcept
{
	return { device_id, handle, owning };
}

namespace detail_ {

template<shared_handle_kind_t SharedHandleKind = shared_handle_kind_t::no_export>
pool_t create(cuda::device::id_t device_id)
{
	auto props = create_raw_properties<SharedHandleKind>(device_id);
	handle_t handle;
	auto status = cuMemPoolCreate(&handle, &props);
	throw_if_error_lazy(status, "Failed creating a memory pool on device " + cuda::device::detail_::identify(device_id));
	constexpr const bool is_owning { true };
	return wrap(device_id, handle, is_owning);
}

inline ::std::string identify(const pool_t& pool)
{
	return identify(pool.handle(), pool.device_id());
}

} // namespace detail_

template<shared_handle_kind_t SharedHandleKind>
pool_t create(const cuda::device_t& device);

} // namespace pool

} // namespace memory

} // namespace cuda

#endif // CUDA_VERSION >= 11020

#endif // CUDA_API_WRAPPERS_MEMORY_POOL_HPP_
