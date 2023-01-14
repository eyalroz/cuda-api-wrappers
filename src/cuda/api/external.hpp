/**
 * @file
 *
 * @brief CUDA wrappers for "CUDA-external resources": memory and semaphores.
 *
 * @note This is a rudiem
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_EXTERNAL_HPP_
#define CUDA_API_WRAPPERS_EXTERNAL_HPP_

#if CUDA_VERSION >= 10000

#include "memory.hpp"

namespace cuda {
namespace memory {
namespace external {

enum kind_t : ::std::underlying_type<CUexternalMemoryHandleType_enum>::type {
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

using descriptor_t = CUDA_EXTERNAL_MEMORY_HANDLE_DESC;

namespace detail_ {

inline void destroy(handle_t handle, const descriptor_t &)
{
	auto status = cuDestroyExternalMemory(handle);
	throw_if_error_lazy(status, ::std::string("Destroying a memory resource"));
}

inline ::std::string identify(subregion_spec_t subregion_spec)
{
	return "subregion of size " + ::std::to_string(subregion_spec.size)
		   + " at offset " + ::std::to_string(subregion_spec.offset);
}

inline ::std::string identify(handle_t handle)
{
	return "external memory resource at " + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(descriptor_t descriptor)
{
	return "external memory resource of kind " + ::std::to_string(descriptor.type);
}

inline ::std::string identify(handle_t handle, descriptor_t descriptor)
{
	return "external memory resource of kind " + ::std::to_string(descriptor.type)
		   + " at " + cuda::detail_::ptr_as_hex(handle);
}

inline handle_t import(const descriptor_t& descriptor)
{
	handle_t handle;
	auto status = cuImportExternalMemory(&handle, &descriptor);
	throw_if_error_lazy(status, "Failed importing " + identify(descriptor));
	return handle;
}

} // namespace detail_

class resource_t;

resource_t wrap(handle_t handle, descriptor_t descriptor, bool take_ownership = false);
class mapped_region_t;

/**
 * A CUDA-recognized external memory resource - i.e. one that is not simply a region
 * in system memory.
 */
class resource_t {
public:
	friend resource_t wrap(handle_t handle, descriptor_t descriptor, bool take_ownership);

	handle_t handle() const { return handle_; }
	descriptor_t descriptor() const { return descriptor_; }
	kind_t kind() const { return static_cast<kind_t>(descriptor_.type); }
	size_t size() const { return descriptor_.size; }

protected:

	resource_t(handle_t handle, descriptor_t descriptor, bool is_owning)
		: handle_(handle), descriptor_(descriptor), owning_(is_owning)
	{}

public:
	resource_t(const resource_t& other) = delete;

	resource_t(resource_t&& other) noexcept : resource_t(
		other.handle_, other.descriptor_, other.owning_)
	{
		other.owning_ = false;
	};

	~resource_t()
	{
		if (owning_) {
#ifdef NDEBUG
			cuDestroyExternalMemory(handle_);
				// Note: "Swallowing" any potential error to avoid ::std::terminate()
#else
			detail_::destroy(handle_, descriptor_);
#endif
		}
	}

protected: // data members
	handle_t handle_;
	descriptor_t descriptor_;
	bool owning_;
};

inline resource_t wrap(handle_t handle, descriptor_t descriptor, bool take_ownership)
{
	return { handle, ::std::move(descriptor), take_ownership };
}

/**
 * Import an external memory resource to be recognized by CUDA
 */
inline resource_t import(descriptor_t descriptor)
{
	handle_t handle = detail_::import(descriptor);
	return wrap(handle, descriptor, do_take_ownership);
}

namespace detail_ {

inline region_t map(handle_t handle, subregion_spec_t subregion)
{
	device::address_t address;
	CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st buffer_desc;
	buffer_desc.flags = 0u;
	buffer_desc.offset = subregion.offset;
	buffer_desc.size = subregion.size;
	auto result = cuExternalMemoryGetMappedBuffer(&address, handle, &buffer_desc);
	throw_if_error_lazy(result, "Failed mapping " + detail_::identify(subregion)
								+ " within " + detail_::identify(handle) + " to a device buffer");
	return region_t{as_pointer(address), subregion.size};
}

} // namespace detail_

mapped_region_t wrap(region_t mapped_region);

/**
 * TODO: Rebase this on unique_region
 *
 * Essentially, a unique_ptr-like class, with the allocator and deleter being
 * mapping and unmapping parts of an external memory resource, and - a size.
 */
class mapped_region_t {
public:
	friend mapped_region_t wrap(region_t mapped_region);

	operator region_t() const { return region_; }
	region_t raw_region() const { return region_; }

	mapped_region_t(const mapped_region_t& other) = delete;
	mapped_region_t(mapped_region_t&& other) : mapped_region_t(other.region_) { }
	~mapped_region_t() { memory::device::free(region_); }

protected:
	mapped_region_t(region_t region) : region_(region) { }

	region_t region_;
};

inline mapped_region_t wrap(region_t mapped_region)
{
	return { mapped_region };
}

/**
 * Map a sub-region of a memory resource into the CUDA-accessible address space
 */
inline mapped_region_t map(const resource_t& resource, subregion_spec_t subregion_to_map)
{
	auto mapped_region = detail_::map(resource.handle(), subregion_to_map);
	return wrap(mapped_region);
}

/**
 * Map an external memory resource into the CUDA-accessible address space
 */
inline mapped_region_t map(const resource_t& resource)
{
	auto subregion_spec = subregion_spec_t { 0u, resource.size() };
	return map(resource, subregion_spec);
}

// Note: mapping mipmapped arrays currently not supported

} // namespace external
} // namespace memory
} // namespace cuda

#endif // CUDA_VERSION >= 10000

#endif // CUDA_API_WRAPPERS_EXTERNAL_HPP_
