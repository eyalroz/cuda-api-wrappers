/**
 * @file
 */
#ifndef CUDA_API_WRAPPERS_VIRTUAL_MEMORY_HPP_
#define CUDA_API_WRAPPERS_VIRTUAL_MEMORY_HPP_

// We need this out of the #ifdef, as otherwise we don't know what
// the CUDA_VERSION is...
#include <cuda.h>

#include "types.hpp"
#include "identify.hpp"

#if CUDA_VERSION >= 10020
#include "types.hpp"
#include "error.hpp"
#include "detail/handle_ownership.hpp"

namespace cuda_ {
///@cond
class device_t;
///@endcond

// TODO: Perhaps move this down into the device namespace ?
namespace memory {

///@cond
class physical_allocation_t;
///@endcond

namespace physical_allocation {

namespace detail {

physical_allocation_t wrap(handle_t handle, size_t size, bool holds_refcount_unit) noexcept;

} // namespace detail

namespace detail {
enum class granularity_kind_t : std::underlying_type<CUmemAllocationGranularity_flags_enum>::type {
	minimum_required = CU_MEM_ALLOC_GRANULARITY_MINIMUM,
	recommended_for_performance = CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
};

} // namespace detail

// Note: Not inheriting from CUmemAllocationProp_st, since
// that structure is a bit messed up
struct properties_t {
	// Note: Specifying a compression type is currently unsupported,
	// as the driver API does not document semantics for the relevant
	// properties field

public: // getters
	cuda_::device_t device() const;

	// TODO: Is this only relevant to requests?
	shared_handle_kind_t requested_kind() const
	{
		return shared_handle_kind_t(raw.requestedHandleTypes);
	};

protected: // non-mutators
	size_t granularity(detail::granularity_kind_t kind) const {
		size_t result;
		auto status = cuMemGetAllocationGranularity(&result, &raw,
			static_cast<CUmemAllocationGranularity_flags>(kind));
		throw_if_error_lazy(status, "Could not determine physical allocation granularity");
		return result;
	}

public: // non-mutators
	size_t minimum_granularity()     const { return granularity(detail::granularity_kind_t::minimum_required); }
	size_t recommended_granularity() const { return granularity(detail::granularity_kind_t::recommended_for_performance); }

public:
	properties_t(CUmemAllocationProp_st raw_properties) : raw(raw_properties)
	{
		if (raw.location.type != CU_MEM_LOCATION_TYPE_DEVICE) {
			throw std::runtime_error("Unexpected physical_allocation type - we only know about devices!");
		}
	}

	properties_t(properties_t&&) = default;
	properties_t(properties_t const&) = default;

public:
	CUmemAllocationProp_st raw;

};

namespace detail {

template<physical_allocation::shared_handle_kind_t SharedHandleKind>
properties_t create_properties(cuda_::device::id_t device_id)
{
	CUmemAllocationProp_st raw_props{};
	raw_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	raw_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	raw_props.location.id = static_cast<int>(device_id);
	raw_props.requestedHandleTypes = static_cast<CUmemAllocationHandleType>(SharedHandleKind);
	raw_props.win32HandleMetaData = nullptr;
	return properties_t{raw_props};
}

} // namespace detail

template<physical_allocation::shared_handle_kind_t SharedHandleKind>
properties_t create_properties_for(device_t const& device);

} // namespace physical_allocation

namespace virtual_ {

class reserved_address_range_t;
class mapping_t;

namespace detail {

inline status_t cancel_reservation_nothrow(memory::region_t reserved) noexcept
{
	return cuMemAddressFree(memory::device::address(reserved.start()), reserved.size());
}

inline void cancel_reservation(memory::region_t reserved)
{
	auto status = cancel_reservation_nothrow(reserved);
	throw_if_error_lazy(status, "Failed freeing a reservation of " + cuda_::detail::identify(reserved));
}

} // namespace detail

using alignment_t = size_t;

enum alignment : alignment_t {
	default_ = 0,
	trivial = 1
};

namespace detail {

reserved_address_range_t wrap(region_t address_range, alignment_t alignment, bool take_ownership) noexcept;

} // namespace detail


class reserved_address_range_t {
public: // types
	using handle_type = memory::region_t;

protected:

	reserved_address_range_t(region_t region, alignment_t alignment, bool owning) noexcept
		: region_(region), alignment_(alignment), ownership_(owning, { cuda_::context::detail::none, region })
	{ }

public:
	friend reserved_address_range_t detail::wrap(region_t, alignment_t, bool) noexcept;

	reserved_address_range_t(reserved_address_range_t const&) = delete;
	reserved_address_range_t(reserved_address_range_t&&) noexcept = default;
	reserved_address_range_t& operator=(reserved_address_range_t const&) = delete;
	reserved_address_range_t& operator=(reserved_address_range_t&&) noexcept = default;

public: // getters
	bool is_owning() const noexcept { return ownership_.has_token(); }
	region_t region() const noexcept{ return region_; }
	alignment_t alignment() const noexcept { return alignment_; }

protected: // data members
	region_t const     region_;
	alignment_t const  alignment_;
	cuda_::detail::handle_ownership_t<reserved_address_range_t> ownership_;

	CAW_DEFINE_HANDLE_RELEASE_MEMBERS(cuda_::memory::virtual_::detail::cancel_reservation_nothrow, cuMemAddressFree)

}; // reserved_address_range_t

namespace detail {

inline reserved_address_range_t wrap(region_t address_range, alignment_t alignment, bool take_ownership) noexcept
{
	return { address_range, alignment, take_ownership };
}

} // namespace detail

inline reserved_address_range_t reserve(region_t requested_region, alignment_t alignment = alignment::default_)
{
	unsigned long flags { 0 };
	CUdeviceptr ptr;
	auto status = cuMemAddressReserve(&ptr, requested_region.size(), alignment, device::address(requested_region), flags);
	throw_if_error_lazy(status, "Failed making a reservation of " + cuda_::detail::identify(requested_region)
		+ " with alignment value " + std::to_string(alignment));
	bool is_owning { true };
	return detail::wrap(memory::region_t {as_pointer(ptr), requested_region.size() }, alignment, is_owning);
}

inline reserved_address_range_t reserve(size_t requested_size, alignment_t alignment = alignment::default_)
{
	return reserve(region_t{ nullptr, requested_size }, alignment);
}

} // namespace virtual

namespace physical_allocation {
namespace detail {

struct release_helper {
	void operator()(handle_t handle) const CAW_DESTRUCTOR_EXCEPTION_SPEC {
		auto status = cuMemRelease(handle);
#ifdef CAW_THROW_IN_DESTRUCTORS
		throw_if_error_lazy(status, "Failed releasing a physical allocation for virtual memory");
#else
		(void) status;
#endif
	}
};

class refcount_unit_t : public cuda_::detail::token_holder<release_helper, handle_t> {
	using handle_type =  physical_allocation::handle_t;
	using parent_type = token_holder;
	using parent_type::token_holder;
};

} // namespace detail
} // namespace physical_allocation

class physical_allocation_t {
protected: // constructors
	physical_allocation_t(physical_allocation::handle_t handle, size_t size, bool holds_refcount_unit)
		: handle_(handle), size_(size), refcount_unit_({ holds_refcount_unit, handle }) { }

public: // constructors & destructor
	physical_allocation_t(physical_allocation_t const& other) = delete;
	physical_allocation_t(physical_allocation_t&& other) noexcept = default;

public: // non-mutators
	friend physical_allocation_t physical_allocation::detail::wrap(
		physical_allocation::handle_t handle, size_t size, bool holds_refcount_unit) noexcept;

	size_t size() const noexcept { return size_; }
	physical_allocation::handle_t handle() const noexcept { return handle_; }
	bool holds_refcount_unit() const noexcept { return refcount_unit_.has_token(); }

	physical_allocation::properties_t properties() const {
		CUmemAllocationProp raw_properties;
		auto status = cuMemGetAllocationPropertiesFromHandle(&raw_properties, handle_);
		throw_if_error_lazy(status, "Obtaining the properties of a virtual memory physical_allocation with handle " + std::to_string(handle_));
		return { raw_properties };
	}

	template <physical_allocation::shared_handle_kind_t SharedHandleKind>
	physical_allocation::shared_handle_t<SharedHandleKind> sharing_handle() const
	{
		physical_allocation::shared_handle_t<SharedHandleKind> shared_handle_;
		static constexpr unsigned long long flags { 0 };
		auto result = cuMemExportToShareableHandle(&shared_handle_, handle_, static_cast<CUmemAllocationHandleType>(SharedHandleKind), flags);
		throw_if_error_lazy(result, "Exporting a (generic CUDA) shared memory physical_allocation to a shared handle");
		return shared_handle_;
	}

protected: // data members
	const   physical_allocation::handle_t handle_;
	size_t  size_;
	physical_allocation::detail::refcount_unit_t refcount_unit_;
};

namespace physical_allocation {

inline physical_allocation_t create(size_t size, properties_t properties)
{
	static constexpr unsigned long long flags { 0 };
	CUmemGenericAllocationHandle handle;
	auto result = cuMemCreate(&handle, size, &properties.raw, flags);
	throw_if_error_lazy(result, "Failed making a virtual memory physical_allocation of size " + std::to_string(size));
	static constexpr bool is_owning { true };
	return detail::wrap(handle, size, is_owning);
}

physical_allocation_t create(size_t size, device_t device);

namespace detail {

inline physical_allocation_t wrap(handle_t handle, size_t size, bool holds_refcount_unit) noexcept
{
	return { handle, size, holds_refcount_unit };
}

inline properties_t properties_of(handle_t handle)
{
	CUmemAllocationProp prop;
	auto result = cuMemGetAllocationPropertiesFromHandle (&prop, handle);
	throw_if_error_lazy(result, "Failed obtaining the properties of the virtual memory physical_allocation with handle "
	  + std::to_string(handle));
	return { prop };
}

} // namespace detail

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
	return physical_allocation::detail::wrap(result_handle, size, holds_refcount_unit);
}

} // namespace physical_allocation

/*
enum access_mode_t : std::underlying_type<CUmemAccess_flags>::type {
	no_access             = CU_MEM_ACCESS_FLAGS_PROT_NONE,
	read_access           = CU_MEM_ACCESS_FLAGS_PROT_READ,
	read_and_write_access = CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
	rw_access             = read_and_write_access
};
*/

namespace virtual_ {
namespace mapping {
namespace detail {

inline mapping_t wrap(region_t address_range, bool owning = false) noexcept;

} // namespace detail
} // namespace mapping

namespace detail {

inline permissions_t get_permissions(region_t fully_mapped_region, cuda_::device::id_t device_id)
{
	CUmemLocation_st location { CU_MEM_LOCATION_TYPE_DEVICE, device_id };
	unsigned long long flags;
	auto result = cuMemGetAccess(&flags, &location, device::address(fully_mapped_region) );
	throw_if_error_lazy(result, "Failed determining the access mode for "
		+ cuda_::device::detail::identify(device_id)
		+ " to the virtual memory mapping to the range of size "
		+ std::to_string(fully_mapped_region.size()) + " bytes at " + cuda_::detail::ptr_as_hex(fully_mapped_region.data()));
	return permissions::detail::from_flags(static_cast<CUmemAccess_flags>(flags)); // Does this actually work?
}

} // namespace detail

/**
 * Determines what kind of access a device has to a mapped region in the (universal) address space
 *
 * @param fully_mapped_region a region in the universal (virtual) address space, which must be
 * covered entirely by virtual memory mappings.
 */
permissions_t get_access_mode(region_t fully_mapped_region, device_t const& device);

/**
 * Determines what kind of access a device has to a the region of memory mapped to a single
 * physical allocation.
 */
permissions_t get_access_mode(mapping_t mapping, device_t const& device);

/**
 * Set the access mode from a single device to a mapped region in the (universal) address space
 *
 * @param fully_mapped_region a region in the universal (virtual) address space, which must be
 * covered entirely by virtual memory mappings.
 */
void set_permissions(region_t fully_mapped_region, device_t const& device, permissions_t access_mode);

/**
 * Set the access mode from a single device to the region of memory mapped to a single
 * physical allocation.
 */
void set_permissions(mapping_t const& mapping, device_t const& device, permissions_t access_mode);
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
	ContiguousContainer<device_t> const& devices,
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
void set_permissions(
	mapping_t mapping,
	ContiguousContainer<device_t> const& devices,
	permissions_t access_mode);

template <template <typename...> class ContiguousContainer>
void set_permissions(
	mapping_t mapping,
	ContiguousContainer<device_t>&& devices,
	permissions_t access_mode);
///@}

namespace detail {

inline status_t unmap_nothrow(region_t address_range) noexcept
{
	return cuMemUnmap(device::address(address_range.start()), address_range.size());
}

inline void unmap_(region_t address_range)
{
	auto result = unmap_nothrow(address_range);
	throw_if_error_lazy(result, "Failed unmapping " + cuda_::detail::identify(address_range));
}

} // namespace detail

class mapping_t {
public: // types
	using handle_type = region_t;

protected:  // constructors
	mapping_t(region_t address_range, bool owning)
	: address_range_(address_range), ownership_(owning, { context::detail::none, address_range }) { }

public: // constructors & destructors
	mapping_t(mapping_t const&) = delete;
	mapping_t(mapping_t&&) noexcept = default;
	mapping_t& operator=(mapping_t const&) = delete;
	mapping_t& operator=(mapping_t&&) noexcept = default;

	friend mapping_t mapping::detail::wrap(region_t address_range, bool owning) noexcept;


	region_t address_range() const noexcept { return address_range_; }
	bool is_owning() const noexcept { return ownership_.has_token(); }

	permissions_t get_permissions(device_t const& device) const;
	void set_permissions(device_t const& device, permissions_t access_mode) const;

	template <template <typename...> class ContiguousContainer>
	inline void set_permissions(
		ContiguousContainer<device_t> const& devices,
		permissions_t access_mode) const;

	template <template <typename...> class ContiguousContainer>
	inline void set_permissions(
		ContiguousContainer<device_t>&& devices,
		permissions_t access_mode) const;

public:
#if CUDA_VERSION >= 11000

	physical_allocation_t allocation() const
	{
		CUmemGenericAllocationHandle allocation_handle;
		auto status = cuMemRetainAllocationHandle(&allocation_handle, address_range_.data());
		throw_if_error_lazy(status, " Failed obtaining/retaining the physical_allocation handle for the virtual memory "
			"range mapped to " + cuda_::detail::ptr_as_hex(address_range_.data()) + " of size " +
				std::to_string(address_range_.size()) + " bytes");
		constexpr bool increase_refcount{false};
		return physical_allocation::detail::wrap(allocation_handle, address_range_.size(), increase_refcount);
	}
#endif
protected:

	region_t address_range_;
	cuda_::detail::handle_ownership_t<mapping_t> ownership_;

	CAW_DEFINE_HANDLE_RELEASE_MEMBERS(memory::virtual_::detail::unmap_nothrow, cuMemUnmap)
}; // mapping_t

namespace mapping {

namespace detail {

mapping_t wrap(region_t address_range, bool owning) noexcept
{
	return { address_range, owning };
}

} // namespace detail

} // namespace mapping

inline mapping_t map(region_t region, physical_allocation_t const& physical_allocation)
{
	size_t offset_into_allocation { 0 }; // not yet supported, but in the API
	constexpr unsigned long long flags { 0 };
	auto handle = physical_allocation.handle();
	auto status = cuMemMap(device::address(region), region.size(), offset_into_allocation, handle, flags);
	throw_if_error_lazy(status, "Failed making a virtual memory mapping of "
		+ cuda_::detail::identify(physical_allocation)
		+ " to the range of size " + std::to_string(region.size()) + " bytes at " +
		cuda_::detail::ptr_as_hex(region.data()));
	constexpr bool is_owning { true };
	return mapping::detail::wrap(region, is_owning);
}

} // namespace virtual_
} // namespace memory

} // namespace cuda_

#endif // CUDA_VERSION >= 10020
#endif // CUDA_API_WRAPPERS_VIRTUAL_MEMORY_HPP_
