/**
 * @file
 */
#ifndef CUDA_API_WRAPPERS_VIRTUAL_MEMORY_HPP_
#define CUDA_API_WRAPPERS_VIRTUAL_MEMORY_HPP_

// We need this out of the #ifdef, as otherwise we don't know what
// the CUDA_VERSION is...
#include <cuda.h>

#if CUDA_VERSION >= 10020
#include "types.hpp"
#include "error.hpp"

namespace cuda {

///@cond
class device_t;
///@endcond

// TODO: Perhaps move this down into the device namespace ?
namespace memory {

///@cond
class physical_allocation_t;
///@endcond

namespace physical_allocation {

using handle_t = CUmemGenericAllocationHandle;

namespace detail_ {

physical_allocation_t wrap(handle_t handle, size_t size, bool holds_refcount_unit);

} // namespace detail_

namespace detail_ {
enum class granularity_kind_t : ::std::underlying_type<CUmemAllocationGranularity_flags_enum>::type {
	minimum_required = CU_MEM_ALLOC_GRANULARITY_MINIMUM,
	recommended_for_performance = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
};

} // namespace detail_

// Note: Not inheriting from CUmemAllocationProp_st, since
// that structure is a bit messed up
struct properties_t {
	// Note: Specifying a compression type is currently unsupported,
	// as the driver API does not document semantics for the relevant
	// properties field

public: // getters
	cuda::device_t device() const;

	// TODO: Is this only relevant to requests?
	shared_handle_kind_t requested_kind() const
	{
		return shared_handle_kind_t(raw.requestedHandleTypes);
	};

protected: // non-mutators
	size_t granularity(detail_::granularity_kind_t kind) const {
		size_t result;
		auto status = cuMemGetAllocationGranularity(&result, &raw,
			static_cast<CUmemAllocationGranularity_flags>(kind));
		throw_if_error_lazy(status, "Could not determine physical allocation granularity");
		return result;
	}

public: // non-mutators
	size_t minimum_granularity()     const { return granularity(detail_::granularity_kind_t::minimum_required); }
	size_t recommended_granularity() const { return granularity(detail_::granularity_kind_t::recommended_for_performance); }

public:
	properties_t(CUmemAllocationProp_st raw_properties) : raw(raw_properties)
	{
		if (raw.location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
			throw ::std::runtime_error("Unexpected physical_allocation type - we only know about devices!");
		}
	}

	properties_t(properties_t&&) = default;
	properties_t(const properties_t&) = default;

public:
	CUmemAllocationProp_st raw;

};

namespace detail_ {

template<physical_allocation::shared_handle_kind_t SharedHandleKind>
properties_t create_properties(cuda::device::id_t device_id)
{
	CUmemAllocationProp_st raw_props{};
	raw_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	raw_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	raw_props.location.id = static_cast<int>(device_id);
	raw_props.requestedHandleTypes = static_cast<CUmemAllocationHandleType>(SharedHandleKind);
	raw_props.win32HandleMetaData = nullptr;
	return properties_t{raw_props};
}

} // namespace detail_

template<physical_allocation::shared_handle_kind_t SharedHandleKind>
properties_t create_properties_for(const device_t& device);

} // namespace physical_allocation

namespace virtual_ {

class reserved_address_range_t;
class mapping_t;

namespace detail_ {

inline status_t cancel_reservation_nothrow(memory::region_t reserved) noexcept
{
	return cuMemAddressFree(memory::device::address(reserved.start()), reserved.size());
}

inline void cancel_reservation(memory::region_t reserved)
{
	auto status = cancel_reservation_nothrow(reserved);
	throw_if_error_lazy(status, "Failed freeing a reservation of " + memory::detail_::identify(reserved));
}

} // namespace detail_

using alignment_t = size_t;

enum alignment : alignment_t {
	default_ = 0,
	trivial = 1
};

namespace detail_ {

reserved_address_range_t wrap(region_t address_range, alignment_t alignment, bool take_ownership);

} // namespace detail_


class reserved_address_range_t {
protected:

	reserved_address_range_t(region_t region, alignment_t alignment, bool owning) noexcept
		: region_(region), alignment_(alignment), owning_(owning) { }

public:
	friend reserved_address_range_t detail_::wrap(region_t, alignment_t, bool);

	reserved_address_range_t(reserved_address_range_t&& other) noexcept
	: region_(other.region_), alignment_(other.alignment_), owning_(other.owning_)
	{
		other.owning_ = false;
	}

	~reserved_address_range_t() DESTRUCTOR_EXCEPTION_SPEC
	{
		if (not owning_) { return; }
#if THROW_IN_DESTRUCTORS
		detail_::cancel_reservation(region_);
#else
		detail_::cancel_reservation_nothrow(region_);
#endif
	}

public: // getters
	bool is_owning() const noexcept { return owning_; }
	region_t region() const noexcept{ return region_; }
	alignment_t alignment() const noexcept { return alignment_; }

protected: // data members
	const region_t     region_;
	const alignment_t  alignment_;
	bool               owning_;
};

namespace detail_ {

inline reserved_address_range_t wrap(region_t address_range, alignment_t alignment, bool take_ownership)
{
	return { address_range, alignment, take_ownership };
}

} // namespace detail_

inline reserved_address_range_t reserve(region_t requested_region, alignment_t alignment = alignment::default_)
{
	unsigned long flags { 0 };
	CUdeviceptr ptr;
	auto status = cuMemAddressReserve(&ptr, requested_region.size(), alignment, device::address(requested_region), flags);
	throw_if_error_lazy(status, "Failed making a reservation of " + cuda::memory::detail_::identify(requested_region)
		+ " with alignment value " + ::std::to_string(alignment));
	bool is_owning { true };
	return detail_::wrap(memory::region_t {as_pointer(ptr), requested_region.size() }, alignment, is_owning);
}

inline reserved_address_range_t reserve(size_t requested_size, alignment_t alignment = alignment::default_)
{
	return reserve(region_t{ nullptr, requested_size }, alignment);
}

} // namespace physical_allocation

class physical_allocation_t {
protected: // constructors
	physical_allocation_t(physical_allocation::handle_t handle, size_t size, bool holds_refcount_unit)
		: handle_(handle), size_(size), holds_refcount_unit_(holds_refcount_unit) { }

public: // constructors & destructor
	physical_allocation_t(const physical_allocation_t& other) noexcept : handle_(other.handle_), size_(other.size_), holds_refcount_unit_(false)
	{ }

	physical_allocation_t(physical_allocation_t&& other) noexcept  : handle_(other.handle_), size_(other.size_), holds_refcount_unit_(other.holds_refcount_unit_)
	{
		other.holds_refcount_unit_ = false;
	}

	~physical_allocation_t() DESTRUCTOR_EXCEPTION_SPEC
	{
		if (not holds_refcount_unit_) { return; }
		auto status = cuMemRelease(handle_);
#ifdef THROW_IN_DESTRUCTORS
		throw_if_error_lazy(status, "Failed making a virtual memory physical_allocation of size " + ::std::to_string(size_));
#else
		(void) status;
#endif
	}

public: // non-mutators
	friend physical_allocation_t physical_allocation::detail_::wrap(physical_allocation::handle_t handle, size_t size, bool holds_refcount_unit);

	size_t size() const noexcept { return size_; }
	physical_allocation::handle_t handle() const noexcept { return handle_; }
	bool holds_refcount_unit() const noexcept { return holds_refcount_unit_; }

	physical_allocation::properties_t properties() const {
		CUmemAllocationProp raw_properties;
		auto status = cuMemGetAllocationPropertiesFromHandle(&raw_properties, handle_);
		throw_if_error_lazy(status, "Obtaining the properties of a virtual memory physical_allocation with handle " + ::std::to_string(handle_));
		return { raw_properties };
	}

	template <physical_allocation::shared_handle_kind_t SharedHandleKind>
	physical_allocation::shared_handle_t<SharedHandleKind> sharing_handle() const
	{
		physical_allocation::shared_handle_t<SharedHandleKind> shared_handle_;
		static constexpr const unsigned long long flags { 0 };
		auto result = cuMemExportToShareableHandle(&shared_handle_, handle_, static_cast<CUmemAllocationHandleType>(SharedHandleKind), flags);
		throw_if_error_lazy(result, "Exporting a (generic CUDA) shared memory physical_allocation to a shared handle");
		return shared_handle_;
	}

protected: // data members
	const   physical_allocation::handle_t handle_;
	size_t  size_;
	bool    holds_refcount_unit_;
};

namespace physical_allocation {

inline physical_allocation_t create(size_t size, properties_t properties)
{
	static constexpr const unsigned long long flags { 0 };
	CUmemGenericAllocationHandle handle;
	auto result = cuMemCreate(&handle, size, &properties.raw, flags);
	throw_if_error_lazy(result, "Failed making a virtual memory physical_allocation of size " + ::std::to_string(size));
	static constexpr const bool is_owning { true };
	return detail_::wrap(handle, size, is_owning);
}

physical_allocation_t create(size_t size, device_t device);

namespace detail_ {

inline ::std::string identify(handle_t handle, size_t size) {
	return ::std::string("physical allocation with handle ") + ::std::to_string(handle)
		+ " of size " + ::std::to_string(size);
}

inline physical_allocation_t wrap(handle_t handle, size_t size, bool holds_refcount_unit)
{
	return { handle, size, holds_refcount_unit };
}

inline properties_t properties_of(handle_t handle)
{
	CUmemAllocationProp prop;
	auto result = cuMemGetAllocationPropertiesFromHandle (&prop, handle);
	throw_if_error_lazy(result, "Failed obtaining the properties of the virtual memory physical_allocation with handle "
	  + ::std::to_string(handle));
	return { prop };
}

} // namespace detail_

/**
 *
 * @note Unfortunately, importing a handle does not tell you how much memory is allocated
 *
 * @tparam SharedHandleKind In practice, a to choose between operating systems, as different
 * OSes would use different kinds of shared handles.
 * @param shared_handle a handle obtained from another process, where it had been
 * exported from a CUDA-specific physical_allocation handle.
 *
 * @return the
 */
template <physical_allocation::shared_handle_kind_t SharedHandleKind>
physical_allocation_t import(shared_handle_t<SharedHandleKind> shared_handle, size_t size, bool holds_refcount_unit = false)
{
	handle_t result_handle;
	auto result = cuMemImportFromShareableHandle(
		&result_handle, reinterpret_cast<void*>(shared_handle), CUmemAllocationHandleType(SharedHandleKind));
	throw_if_error_lazy(result, "Failed importing a virtual memory physical_allocation from a shared handle ");
	return physical_allocation::detail_::wrap(result_handle, size, holds_refcount_unit);
}

namespace detail_ {

inline ::std::string identify(physical_allocation_t physical_allocation) {
	return identify(physical_allocation.handle(), physical_allocation.size());
}

} // namespace detail_

} // namespace physical_allocation

/*
enum access_mode_t : ::std::underlying_type<CUmemAccess_flags>::type {
	no_access             = CU_MEM_ACCESS_FLAGS_PROT_NONE,
	read_access           = CU_MEM_ACCESS_FLAGS_PROT_READ,
	read_and_write_access = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
	rw_access             = read_and_write_access
};
*/

namespace virtual_ {
namespace mapping {
namespace detail_ {

inline mapping_t wrap(region_t address_range, bool owning = false);

inline ::std::string identify(region_t address_range) {
	return ::std::string("mapping of ") + memory::detail_::identify(address_range);
}

} // namespace detail_
} // namespace mapping

namespace detail_ {

inline permissions_t get_permissions(region_t fully_mapped_region, cuda::device::id_t device_id)
{
	CUmemLocation_st location { CU_MEM_LOCATION_TYPE_DEVICE, device_id };
	unsigned long long flags;
	auto result = cuMemGetAccess(&flags, &location, device::address(fully_mapped_region) );
	throw_if_error_lazy(result, "Failed determining the access mode for "
		+ cuda::device::detail_::identify(device_id)
		+ " to the virtual memory mapping to the range of size "
		+ ::std::to_string(fully_mapped_region.size()) + " bytes at " + cuda::detail_::ptr_as_hex(fully_mapped_region.data()));
	return permissions::detail_::from_flags(static_cast<CUmemAccess_flags>(flags)); // Does this actually work?
}

} // namespace detail_

/**
 * Determines what kind of access a device has to a mapped region in the (universal) address space
 *
 * @param fully_mapped_region a region in the universal (virtual) address space, which must be
 * covered entirely by virtual memory mappings.
 */
permissions_t get_access_mode(region_t fully_mapped_region, const device_t& device);

/**
 * Determines what kind of access a device has to a the region of memory mapped to a single
 * physical allocation.
 */
permissions_t get_access_mode(mapping_t mapping, const device_t& device);

/**
 * Set the access mode from a single device to a mapped region in the (universal) address space
 *
 * @param fully_mapped_region a region in the universal (virtual) address space, which must be
 * covered entirely by virtual memory mappings.
 */
void set_permissions(region_t fully_mapped_region, const device_t& device, permissions_t access_mode);

/**
 * Set the access mode from a single device to the region of memory mapped to a single
 * physical allocation.
 */
void set_permissions(mapping_t mapping, const device_t& device, permissions_t access_mode);
///@}

/**
 * Set the access mode from several devices to a mapped region in the (universal) address space
 *
 * @param fully_mapped_region a region in the universal (virtual) address space, which must be
 * covered entirely by virtual memory mappings.
 */
///@{
template <template <typename...> class ContiguousContainer>
void set_permissions(
	region_t fully_mapped_region,
	const ContiguousContainer<device_t>& devices,
	permissions_t access_mode);

template <template <typename...> class ContiguousContainer>
void set_permissions(
	region_t fully_mapped_region,
	ContiguousContainer<device_t>&& devices,
	permissions_t access_mode);
///@}

/**
 * Set the access mode from several devices to the region of memory mapped to a single
 * physical allocation.
 */
///@{
template <template <typename...> class ContiguousContainer>
inline void set_permissions(
	mapping_t mapping,
	const ContiguousContainer<device_t>& devices,
	permissions_t access_mode);

template <template <typename...> class ContiguousContainer>
inline void set_permissions(
	mapping_t mapping,
	ContiguousContainer<device_t>&& devices,
	permissions_t access_mode);
///@}


class mapping_t {
protected:  // constructors
	mapping_t(region_t region, bool owning) : address_range_(region), owning_(owning) { }

public: // constructors & destructors

	friend mapping_t mapping::detail_::wrap(region_t address_range, bool owning);

	mapping_t(const mapping_t& other) noexcept :
		address_range_(other.address_range()), owning_(false) { }

	mapping_t(mapping_t&& other) noexcept :
		address_range_(other.address_range()), owning_(other.owning_)
	{
		other.owning_ = false;
	}

	region_t address_range() const noexcept { return address_range_; }
	bool is_owning() const noexcept { return owning_; }

	permissions_t get_permissions(const device_t& device) const;
	void set_permissions(const device_t& device, permissions_t access_mode) const;

	template <template <typename...> class ContiguousContainer>
	inline void set_permissions(
		const ContiguousContainer<device_t>& devices,
		permissions_t access_mode) const;

	template <template <typename...> class ContiguousContainer>
	inline void set_permissions(
		ContiguousContainer<device_t>&& devices,
		permissions_t access_mode) const;

	~mapping_t() noexcept(false)
	{
		if (not owning_) { return; }
		auto result = cuMemUnmap(device::address(address_range_), address_range_.size());
		throw_if_error_lazy(result, "Failed unmapping " + mapping::detail_::identify(address_range_));
	}

public:
#if CUDA_VERSION >= 11000

	physical_allocation_t allocation() const
	{
		CUmemGenericAllocationHandle allocation_handle;
		auto status = cuMemRetainAllocationHandle(&allocation_handle, address_range_.data());
		throw_if_error_lazy(status, " Failed obtaining/retaining the physical_allocation handle for the virtual memory "
			"range mapped to " + cuda::detail_::ptr_as_hex(address_range_.data()) + " of size " +
				::std::to_string(address_range_.size()) + " bytes");
		constexpr const bool increase_refcount{false};
		return physical_allocation::detail_::wrap(allocation_handle, address_range_.size(), increase_refcount);
	}
#endif
protected:

	region_t address_range_;
	bool owning_;

};

namespace mapping {

namespace detail_ {

mapping_t wrap(region_t range, bool owning)
{
	return { range, owning };
}

inline ::std::string identify(mapping_t mapping)
{
	return mapping::detail_::identify(mapping.address_range());
}

} // namespace detail_

} // namespace mapping

inline mapping_t map(region_t region, physical_allocation_t physical_allocation)
{
	size_t offset_into_allocation { 0 }; // not yet supported, but in the API
	constexpr const unsigned long long flags { 0 };
	auto handle = physical_allocation.handle();
	auto status = cuMemMap(device::address(region), region.size(), offset_into_allocation, handle, flags);
	throw_if_error_lazy(status, "Failed making a virtual memory mapping of "
		+ physical_allocation::detail_::identify(physical_allocation)
		+ " to the range of size " + ::std::to_string(region.size()) + " bytes at " +
		cuda::detail_::ptr_as_hex(region.data()));
	constexpr const bool is_owning { true };
	return mapping::detail_::wrap(region, is_owning);
}

} // namespace virtual_
} // namespace memory
} // namespace cuda

#endif // CUDA_VERSION >= 10020
#endif // CUDA_API_WRAPPERS_VIRTUAL_MEMORY_HPP_
