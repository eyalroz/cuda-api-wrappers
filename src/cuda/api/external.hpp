/**
 * @file
 *
 * @brief Facilities for utilizing process-external resources: memory and semaphores,
 * in work with CUDA.
 *
 * @note mapping mipmapped arrays currently not supported.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_EXTERNAL_HPP_
#define CUDA_API_WRAPPERS_EXTERNAL_HPP_

#if CUDA_VERSION >= 10000

#include "memory.hpp"
#include "unique_region.hpp"
#include "detail/token_holder.hpp"

namespace cuda_ {

namespace memory {

/// Functionality regarding (process-)external memory resources and semaphores
namespace external {

enum kind_t : std::underlying_type<CUexternalMemoryHandleType_enum>::type {
	opaque_file_descriptor = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD,
	opaque_shared_windows_handle = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
	opaque_globally_shared_windows_handle = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT,
	direct3d_12_heap = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP,
	direct3d_12_committed_resource = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
#if CUDA_VERSION >= 10200
	direct3d_resource_shared_windows_handle = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE,
	direct3d_resource_globally_shared_handle = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT,
	nvscibuf_object = CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF
#endif // CUDA_VERSION >= 10200
};

namespace detail {

inline void destroy(handle_t handle)
{
	auto status = cuDestroyExternalMemory(handle);
	throw_if_error_lazy(status, std::string("Destroying a memory resource"));
}

inline handle_t import(const descriptor_t& descriptor)
{
	handle_t handle;
	auto status = cuImportExternalMemory(&handle, &descriptor);
	throw_if_error_lazy(status, "Failed importing " + identify(descriptor));
	return handle;
}

} // namespace detail

///@cond
class resource_t;
///@endcond

/// Construct an external memory resource class instance from its raw constituent fields
resource_t wrap(handle_t handle, descriptor_t descriptor, bool take_ownership = false) noexcept;

/**
 * A CUDA-recognized external memory resource - i.e. one that is not simply a region
 * in system memory.
 */
class resource_t {
public:
	using handle_type = handle_t;
	friend resource_t wrap(handle_t handle, descriptor_t descriptor, bool take_ownership) noexcept;

	handle_t handle() const noexcept { return handle_; }
	descriptor_t descriptor() const noexcept{ return descriptor_; }
	kind_t kind() const noexcept{ return static_cast<kind_t>(descriptor_.type); }
	size_t size() const noexcept { return descriptor_.size; }
	bool is_owning() const noexcept { return ownership_.has_token(); }

protected: // constructors
	resource_t(handle_t handle, descriptor_t descriptor, bool is_owning)
		: handle_(handle), descriptor_(std::move(descriptor)), ownership_(is_owning, { context::detail::none, handle } )
	{}

public: // constructors & operators
	resource_t(const resource_t&) = delete;
	resource_t(resource_t&&) noexcept = default;
	resource_t& operator=(const resource_t&) = delete;
	resource_t& operator=(resource_t&&) noexcept = default;

protected: // data members
	handle_t handle_;
	descriptor_t descriptor_;
	cuda_::detail::handle_ownership_t<resource_t> ownership_;
};

inline resource_t wrap(handle_t handle, descriptor_t descriptor, bool take_ownership) noexcept
{
	return { handle, std::move(descriptor), take_ownership };
}

/// Import an external memory resource to be recognized by CUDA
inline resource_t import(descriptor_t descriptor)
{
	handle_t handle = detail::import(descriptor);
	return wrap(handle, std::move(descriptor), do_take_ownership);
}

namespace detail {

inline region_t map(handle_t handle, subregion_spec_t subregion)
{
	device::address_t address;
	CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st buffer_desc;
	buffer_desc.flags = 0u;
	buffer_desc.offset = subregion.offset;
	buffer_desc.size = subregion.size;
	auto result = cuExternalMemoryGetMappedBuffer(&address, handle, &buffer_desc);
	throw_if_error_lazy(result, "Failed mapping " + detail::identify(subregion)
								+ " within " + cuda_::detail::identify(handle) + " to a device buffer");
	return region_t{as_pointer(address), subregion.size};
}

} // namespace detail

/// A mapped external memory region is just like a 'regular' region of device-global
/// memory, and can be handled, owned, and freed the same way
using unique_region = memory::unique_region<device::detail::deleter>;

/// Construct a unique_region of already-mapped external memory
inline unique_region wrap(region_t mapped_region)
{
	return unique_region{ mapped_region };
}

/// Map a sub-region of a memory resource into the CUDA-accessible address space
inline unique_region map(const resource_t& resource, subregion_spec_t subregion_to_map)
{
	auto mapped_region = detail::map(resource.handle(), subregion_to_map);
	return wrap(mapped_region);
}

/// Map an external memory resource into the CUDA-accessible address space
inline unique_region map(const resource_t& resource)
{
	auto subregion_spec = subregion_spec_t { 0u, resource.size() };
	return map(resource, subregion_spec);
}

} // namespace external
} // namespace memory

CAW_DEFINE_HANDLE_TRAITS(memory::external::resource_t, isnt_contextual, cuDestroyExternalMemory, cuDestroyExternalMemory)

} // namespace cuda_

#endif // CUDA_VERSION >= 10000

#endif // CUDA_API_WRAPPERS_EXTERNAL_HPP_
