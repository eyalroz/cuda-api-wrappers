#pragma once
#ifndef CUDA_API_WRAPPERS_MEMORY_HPP_
#define CUDA_API_WRAPPERS_MEMORY_HPP_

#include "cuda/api/error.hpp"
#include "cuda/api/constants.h"

#include "cuda_runtime.h" // needed, rather than cuda_runtime_api.h, e.g. for cudaMalloc

#include <memory>
#include <cstring> // for std::memset

namespace cuda {
namespace memory {

namespace mapped {

// This namespace regards memory appearing both on the device
// and on the host, with the regions mapped to each other.

// TODO: Perhaps make this an array of size 2 and use aspects to index it?
// Or maybe inherit a pair?
struct region_pair {

	struct allocation_options {
		bool portable_across_cuda_contexts;
		bool cpu_write_combining;
	};

	void* host_side;
	void* device_side;
	// size_t size_in_bytes; // common to both sides
	// allocation_options properties;

	// operator std::pair<void*, void*>() { return make_pair(host_side, device_side); }
	// size_t size() { return size_in_bytes; }
};

namespace detail {

inline unsigned make_cuda_host_alloc_flags(region_pair::allocation_options options) {
	return cudaHostAllocMapped &
		(options.portable_across_cuda_contexts ? cudaHostAllocPortable : 0) &
		(options.cpu_write_combining ? cudaHostAllocWriteCombined: 0);
}

} // namespace detail

} // namespace mapped
} // namespace memory


namespace memory {

namespace device {

namespace detail {

template <typename T = void>
T* malloc(size_t num_bytes)
{
	T* allocated = nullptr;
	// Note: the typed cudaMalloc also takes its size in bytes, apparently,
	// not in number of elements
	auto status = cudaMalloc<T>(&allocated, num_bytes);
	if (is_success(status) && allocated == nullptr) {
		// Can this even happen? hopefully not
		status = cudaErrorUnknown;
	}
	throw_if_error(status,
		"Failed allocating " + std::to_string(num_bytes) + " bytes of global memory on CUDA device");
	return allocated;
}

} // namespace detail

inline void free(void* ptr)
{
	auto result = cudaFree(ptr);
	throw_if_error(result, "Freeing device memory at 0x" + cuda::detail::ptr_as_hex(ptr));
}

namespace detail {
struct allocator {
	// Allocates on the current device!
	void* operator()(size_t num_bytes) const { return detail::malloc(num_bytes); }
};
struct deleter {
	void operator()(void* ptr) const { cuda::memory::device::free(ptr); }
};
} // namespace detail

inline void set(void* buffer_start, int byte_value, size_t num_bytes)
{
	auto result = cudaMemset(buffer_start, byte_value, num_bytes);
	throw_if_error(result, "memsetting an on-device buffer");
}

inline void zero(void* buffer_start, size_t num_bytes)
{
	return set(buffer_start, 0, num_bytes);
}

} // namespace device

/**
 * Copies data between memory spaces or within a memory space.
 *
 * @note Since we assume Compute Capability >= 2.0, all devices support the
 * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
 * where the data is located, and one does not have to specify this.
 *
 * @param dst A pointer to a num_bytes-long buffer, either in main memory or on any CUDA device's global memory
 * @param src A pointer to a num_bytes-long buffer, either in main memory or on any CUDA device's global memory
 * @param num_bytes The number of bytes to copy from @ref src to @ref dst
 */
inline void copy(void *destination, const void *source, size_t num_bytes)
{
	auto result = cudaMemcpy(destination, source, num_bytes, cudaMemcpyDefault);
	if (is_failure(result)) {
		std::string error_message("Synchronously copying data");
		// TODO: Determine whether it was from host to device, device to host etc and
		// add this information to the error string
		throw_if_error(result, error_message);
	}
}

template <typename T>
inline void copy_single(T& destination, const T& source)
{
	copy(&destination, &source, sizeof(T));
}

namespace async {

inline void copy(void *destination, const void *source, size_t num_bytes, stream::id_t stream_id)
{
	auto result = cudaMemcpyAsync(destination, source, num_bytes, cudaMemcpyDefault, stream_id);
	if (is_failure(result)) {
		std::string error_message("Scheduling a memory copy on stream " + cuda::detail::ptr_as_hex(stream_id));
		// TODO: Determine whether it was from host to device, device to host etc and
		// add this information to the error string
		throw_if_error(result, error_message);
	}
}

template <typename T>
inline void copy_single(T& destination, const T& source, stream::id_t stream_id)
{
	copy(&destination, &source, sizeof(T), stream_id);
}

inline void set(void* buffer_start, int byte_value, size_t num_bytes, stream::id_t stream_id)
{
	auto result = cudaMemsetAsync(buffer_start, byte_value, num_bytes, stream_id);
	throw_if_error(result, "memsetting an on-device buffer");
}

inline void zero(void* buffer_start, size_t num_bytes, stream::id_t stream_id = stream::default_stream_id)
{
	return set(buffer_start, 0, num_bytes, stream_id);
}


} // namespace async

namespace host {

// TODO: Consider a variant of this supporting the cudaHostAlloc flags
template <typename T>
inline T* allocate(size_t size_in_bytes /* write me:, bool recognized_by_all_contexts */)
{
	T* allocated = nullptr;
	// Note: the typed cudaMallocHost also takes its size in bytes, apparently, not in number of elements
	auto result = cudaMallocHost<T>(&allocated, size_in_bytes);
	if (is_success(result) && allocated == nullptr) {
		// Can this even happen? hopefully not
		result = cudaErrorUnknown;
	}
	throw_if_error(result, "Failed allocating " + std::to_string(size_in_bytes) + " bytes of host memory");
	return allocated;
}

/**
 * Allocates 'pinned' host memory, for which the CUDA driver knows the TLB entries
 * (and which is faster for I/O transactions with the GPU)
 *
 * @param size_in_bytes number of bytes to allocate
 * @result host pointer to the allocated memory region; this must be freed with ::cuda::host::free()
 * rather than C++'s delete() .
 */
inline void* allocate(size_t size_in_bytes)
{
	return (void*) allocate<char>(size_in_bytes);
}

inline void free(void* host_ptr)
{
	auto result = cudaFreeHost(host_ptr);
	throw_if_error(result, "Freeing pinned host memory at 0x" + cuda::detail::ptr_as_hex(host_ptr));
}

namespace detail {
struct allocator {
	void* operator()(size_t num_bytes) const { return cuda::memory::host::allocate(num_bytes); }
};
struct deleter {
	void operator()(void* ptr) const { cuda::memory::host::free(ptr); }
};

inline void register_(void *ptr, size_t size, unsigned flags)
{
	auto result = cudaHostRegister(ptr, size, flags);
	throw_if_error(result,
		"Could not register a region of page-locked host memory");
}

} // namespace detail

enum mapped_io_space : bool {
	is_mapped_io_space               = true,
	is_not_mapped_io_space           = false
};

enum mapped_into_device_memory : bool {
	is_mapped_into_device_memory     = true,
	is_not_mapped_into_device_memory = false
};

enum accessibility_on_all_devices : bool {
	is_accessible_on_all_devices     = true,
	is_not_accessible_on_all_devices = false
};


// Can't use register(), since that's a reserved word
inline void register_(void *ptr, size_t size,
	bool register_mapped_io_space,
	bool map_into_device_space,
	bool make_device_side_accesible_to_all)
{
	return detail::register_(
		ptr, size,
		  register_mapped_io_space ? cudaHostRegisterIoMemory : 0
		| map_into_device_space ? cudaHostRegisterMapped : 0
		| make_device_side_accesible_to_all ? cudaHostRegisterPortable : 0
	);
}

inline void register_(void *ptr, size_t size)
{
	detail::register_(ptr, size, cudaHostRegisterDefault);
}

// the CUDA API calls this "unregister", but that's semantically
// inaccurate. The registration is not undone, rolled back, it's
// just ended
inline void deregister(void *ptr)
{
	auto result = cudaHostUnregister(ptr);
	throw_if_error(result,
		"Could not unregister the memory segment starting at address *a");
}

inline void set(void* buffer_start, int byte_value, size_t num_bytes)
{
	std::memset(buffer_start, byte_value, num_bytes);
	// TODO: Error handling?
}

inline void zero(void* buffer_start, size_t num_bytes)
{
	return set(buffer_start, 0, num_bytes);
	// TODO: Error handling?
}


} // namespace host

namespace managed {

enum class initial_visibility_t {
	to_all_devices,
	to_supporters_of_concurrent_managed_access,
};

namespace detail {

template <typename T = void>
T* malloc(
	size_t num_bytes,
	initial_visibility_t initial_visibility = initial_visibility_t::to_all_devices)
{
	T* allocated = nullptr;
	auto flags = (initial_visibility == initial_visibility_t::to_all_devices) ?
		cudaMemAttachGlobal : cudaMemAttachHost;
	// Note: Despite the templating by T, the size is still in bytes,
	// not in number of T's
	auto status = cudaMallocManaged<T>(&allocated, num_bytes, flags);
	if (is_success(status) && allocated == nullptr) {
		// Can this even happen? hopefully not
		status = (status_t) cuda::error::unknown;
	}
	throw_if_error(status,
		"Failed allocating " + std::to_string(num_bytes) + " bytes of managed CUDA memory");
	return allocated;
}

inline void free(void* ptr)
{
	auto result = cudaFree(ptr);
	throw_if_error(result, "Freeing managed memory at 0x" + cuda::detail::ptr_as_hex(ptr));
}

template <initial_visibility_t InitialVisibility = initial_visibility_t::to_all_devices>
struct allocator {
	// Allocates on the current device!
	void* operator()(size_t num_bytes) const
	{
		return detail::malloc(num_bytes, InitialVisibility);
	}
};
struct deleter {
	void operator()(void* ptr) const { cuda::memory::device::free(ptr); }
};

} // namespace detail
} // namespace managed

namespace mapped {

inline void free(region_pair pair)
{
	auto result = cudaFreeHost(pair.host_side);
	throw_if_error(result,
		"Could not free the (supposed) region pair passed.");
}

inline void free_region_pair_of(void* ptr)
{
	cudaPointerAttributes attributes;
	auto result = cudaPointerGetAttributes(&attributes, ptr);
	throw_if_error(result,
		"Could not obtain the properties for the pointer"
		", being necessary for freeing the region pair it's (supposedly) "
		"associated with.");
	cudaFreeHost(attributes.hostPointer);
}

// Mostly for debugging purposes
inline bool is_part_of_a_region_pair(void* ptr)
{
	cudaPointerAttributes attributes;
	auto result = cudaPointerGetAttributes(&attributes, ptr);
	throw_if_error(result, "Could not obtain device pointer attributes");
#ifdef DEBUG
	auto self_copy = (attributes.memoryType == cudaMemoryTypeHost) ?
		attributes.hostPointer : attributes.devicePointer ;
	if (self_copy != ptr) {
		throw runtime_error(cudaErrorUnknown, "Inconsistent data obtained from the CUDA runtime API");
	}
#endif
	auto corresponding_buffer_ptr =
		(attributes.memoryType == cudaMemoryTypeHost) ?
		attributes.devicePointer : attributes.hostPointer;
	return (corresponding_buffer_ptr != nullptr);
}

} // namespace mapped

} // namespace memory

} // namespace cuda

#endif /* CUDA_API_WRAPPERS_MEMORY_HPP_ */
