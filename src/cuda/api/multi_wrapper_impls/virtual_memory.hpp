/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * in the `cuda::memory::virtual_` namespace, and additional functionality related to
 * virtual memory.
 */
#ifndef CUDA_API_WRAPPERS_MULTI_WRAPPER_IMPLS_VIRTUAL_MEMORY_HPP_
#define CUDA_API_WRAPPERS_MULTI_WRAPPER_IMPLS_VIRTUAL_MEMORY_HPP_

#include "../device.hpp"
#include "../virtual_memory.hpp"

namespace cuda {

namespace memory {

#if CUDA_VERSION >= 10020
namespace physical_allocation {

inline device_t properties_t::device() const
{
	return cuda::device::wrap(raw.location.id);
}

template<shared_handle_kind_t SharedHandleKind>
properties_t create_properties_for(const device_t& device)
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

inline void set_permissions(
	region_t              fully_mapped_region,
	const device_t&       device,
	permissions_t         permissions)
{
	CUmemAccessDesc desc { { CU_MEM_LOCATION_TYPE_DEVICE, device.id() }, CUmemAccess_flags(permissions) };
	static constexpr const size_t count { 1 };
	auto result = cuMemSetAccess(device::address(fully_mapped_region), fully_mapped_region.size(), &desc, count);
	throw_if_error_lazy(result, "Failed setting the access mode to the virtual memory mapping to the range of size "
						   + ::std::to_string(fully_mapped_region.size()) + " bytes at " + cuda::detail_::ptr_as_hex(fully_mapped_region.data()));
}

inline void set_permissions(mapping_t mapping, const device_t& device, permissions_t permissions)
{
	set_permissions(mapping.address_range(), device, permissions);
}

template <template <typename...> class Container>
inline void set_permissions(
	region_t                     fully_mapped_region,
	const Container<device_t>&   devices,
	permissions_t                permissions)
{
	auto descriptors = ::std::unique_ptr<CUmemAccessDesc[]>(new CUmemAccessDesc[devices.size()]);
	for(::std::size_t i = 0; i < devices.size(); i++) {
		descriptors[i] = {{CU_MEM_LOCATION_TYPE_DEVICE, devices[i].id()}, CUmemAccess_flags(permissions)};
	}
	auto result = cuMemSetAccess(
		device::address(fully_mapped_region.start()), fully_mapped_region.size(), descriptors.get(), devices.size());
	throw_if_error_lazy(result, "Failed setting the access mode to the virtual memory mapping to the range of size "
						   + ::std::to_string(fully_mapped_region.size()) + " bytes at " + cuda::detail_::ptr_as_hex(fully_mapped_region.data()));
}

template <template <typename...> class Container>
inline void set_permissions(
	region_t              fully_mapped_region,
	Container<device_t>&& devices,
	permissions_t         permissions)
{
	return set_permissions(fully_mapped_region, devices, permissions);
}

template <template <typename...> class Container>
inline void set_permissions(
	mapping_t                    mapping,
	const Container<device_t>&&  devices,
	permissions_t                permissions)
{
	set_permissions(mapping.address_range(), devices, permissions);
}

template <template <typename...> class Container>
inline void set_permissions(
	mapping_t             mapping,
	Container<device_t>&& devices,
	permissions_t         permissions)
{
	set_permissions(mapping, devices, permissions);
}

inline permissions_t get_permissions(region_t fully_mapped_region, const device_t& device)
{
	return detail_::get_permissions(fully_mapped_region, device.id());
}

inline permissions_t get_permissions(const mapping_t& fully_mapped_region, const device_t& device)
{
	return get_permissions(fully_mapped_region.address_range(), device);
}

inline permissions_t mapping_t::get_permissions(const device_t& device) const
{
	return virtual_::get_permissions(*this, device);
}

inline void mapping_t::set_permissions(const device_t& device, permissions_t permissions) const
{
	virtual_::set_permissions(*this, device, permissions);
}

template <template <typename...> class ContiguousContainer>
void mapping_t::set_permissions(
	const ContiguousContainer<device_t>&  devices,
	permissions_t                         permissions) const
{
	virtual_::set_permissions(*this, devices, permissions);
}

template <template <typename...> class ContiguousContainer>
void mapping_t::set_permissions(
	ContiguousContainer<device_t>&&  devices,
	permissions_t                    permissions) const
{
	virtual_::set_permissions(*this, devices, permissions);
}

} // namespace virtual_

#endif // CUDA_VERSION >= 10020

} // namespace memory

} // namespace cuda

#endif //CUDA_API_WRAPPERS_MULTI_WRAPPER_IMPLS_VIRTUAL_MEMORY_HPP_
