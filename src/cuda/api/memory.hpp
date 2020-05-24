/**
 * @file memory.hpp
 *
 * @brief freestanding wrapper functions for working with CUDA's various
 * kinds of memory spaces, arranged into a relevant namespace hierarchy.
 *
 * @note Some of the CUDA API for allocating and copying memory involves
 * the concept of "pitch" and "pitched pointers". To better understand
 * what that means, consider the following two-dimensional representation
 * of an array (which is in fact embedded in linear memory):
 *
 *	 X X X X * * *
 *	 X X X X * * *
 *	 X X X X * * *
 *
 *	 * = padding element
 *	 X = actually used element of the array
 *
 *	 The pitch in the example above is 7 * sizeof(T)
 *	 The width is 4 * sizeof(T)
 *   The height is 3
 *
 * See also https://stackoverflow.com/questions/16119943/how-and-when-should-i-use-pitched-pointer-with-the-cuda-api
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_MEMORY_HPP_
#define CUDA_API_WRAPPERS_MEMORY_HPP_

#include <cuda/api/error.hpp>
#include <cuda/api/constants.hpp>
#include <cuda/api/current_device.hpp>
#include <cuda/api/pointer.hpp>
#include <cuda/api/array.hpp>

#include <cuda_runtime.h> // needed, rather than cuda_runtime_api.h, e.g. for cudaMalloc

#include <memory>
#include <cstring> // for std::memset

namespace cuda {

///@cond
class device_t;
class stream_t;
///@endcond

/**
 * @namespace memory
 * Representation, allocation and manipulation of CUDA-related memory, with
 * its various namespaces and kinds of memory regions.
 */
namespace memory {

/**
 * @namespace mapped
 * Memory regions appearing in both on the host-side and device-side address
 * spaces with the regions in both spaces mapped to each other (i.e. guaranteed
 * to have the same contents on access up to synchronization details). See @url
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory
 * for more details.
 */
namespace mapped {

// TODO: Perhaps make this an array of size 2 and use aspects to index it?
// Or maybe inherit a pair?

/**
 * @brief A pair of memory regions, one in system (=host) memory and one on a
 * CUDA device's memory - mapped to each other
 *
 * @note this is the mapped-pair equivalent of a `void *`; it is not a
 * proper memory region abstraction, i.e. it has no size information
 */
struct region_pair {

	enum : bool {
		is_portable_across_cuda_contexts   = true,
		isnt_portable_across_cuda_contexts = false
	};

	enum : bool {
		with_cpu_write_combining    = true,
		without_cpu_write_combining = false
	};

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

inline unsigned make_cuda_host_alloc_flags(
	region_pair::allocation_options options) noexcept
{
	return cudaHostAllocMapped &
		(options.portable_across_cuda_contexts ? cudaHostAllocPortable : 0) &
		(options.cpu_write_combining ? cudaHostAllocWriteCombined: 0);
}

} // namespace detail

} // namespace mapped

} // namespace memory


namespace memory {

/**
 * @namespace device
 * CUDA-Device-global memory on a single device (not accessible from the host)
 */
namespace device {

namespace detail {

/**
 * Allocate memory on current device
 *
 * @param num_bytes amount of memory to allocate in bytes
 */
inline void* allocate(size_t num_bytes)
{
	void* allocated = nullptr;
	// Note: the typed cudaMalloc also takes its size in bytes, apparently,
	// not in number of elements
	auto status = cudaMalloc(&allocated, num_bytes);
	if (is_success(status) && allocated == nullptr) {
		// Can this even happen? hopefully not
		status = cudaErrorUnknown;
	}
	throw_if_error(status,
		"Failed allocating " + std::to_string(num_bytes) +
		" bytes of global memory on CUDA device " +
		std::to_string(cuda::device::current::detail::get_id()));
	return allocated;
}

/**
 * Allocate device-side memory on a CUDA device.
 *
 * @note The CUDA memory allocator guarantees alignment "suitabl[e] for any kind of variable"
 * (CUDA 9.0 Runtime API documentation), so probably at least 128 bytes.
 *
 * @throws cuda::runtime_error if allocation fails for any reason
 *
 * @param size_in_bytes the amount of memory to allocate
 * @return a pointer to the allocated stretch of memory (on the CUDA device)
 */
inline void* allocate(cuda::device::id_t device_id, size_t size_in_bytes)
{
	cuda::device::current::detail::scoped_override_t<> set_device_for_this_scope(device_id);
	return memory::device::detail::allocate(size_in_bytes);
}

} // namespace detail

/**
 * Free a region of device-side memory which was allocated with @ref allocate.
 */
inline void free(void* ptr)
{
	auto result = cudaFree(ptr);
	throw_if_error(result, "Freeing device memory at 0x" + cuda::detail::ptr_as_hex(ptr));
}

inline void* allocate(cuda::device_t device, size_t size_in_bytes);

namespace detail {
struct allocator {
	// Allocates on the current device!
	void* operator()(size_t num_bytes) const { return detail::allocate(num_bytes); }
};
struct deleter {
	void operator()(void* ptr) const { cuda::memory::device::free(ptr); }
};
} // namespace detail

/**
 * @brief Sets all bytes in a region of memory to a fixed value
 *
 * @note The equivalent of @ref std::memset for CUDA device-side memory
 *
 * @param start starting address of the memory region to set, in a CUDA
 * device's global memory
 * @param byte_value value to set the memory region to
 * @param num_bytes size of the memory region in bytes
 */
inline void set(void* start, int byte_value, size_t num_bytes)
{
	auto result = cudaMemset(start, byte_value, num_bytes);
	throw_if_error(result, "memsetting an on-device buffer");
}

/**
 * @brief Sets all bytes in a region of memory to 0 (zero)
 *
 * @param start starting address of the memory region to zero-out,
 * in a CUDA device's global memory
 * @param num_bytes size of the memory region in bytes
 */
inline void zero(void* start, size_t num_bytes)
{
	set(start, 0, num_bytes);
}

/**
 * @brief Sets all bytes of a single pointed-to value to 0
 *
 * @param ptr pointer to a value of a certain type, in a CUDA device's
 * global memory
 */
template <typename T>
inline void zero(T* ptr)
{
	zero(ptr, sizeof(T));
}

} // namespace device

/**
 * Synchronously copies data between memory spaces or within a memory space.
 *
 * @note Since we assume Compute Capability >= 2.0, all devices support the
 * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
 * where the data is located, and one does not have to specify this.
 *
 * @param destination A pointer to a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param source A pointer to a a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param num_bytes The number of bytes to copy from @p source to @p destination
 */
inline void copy(void *destination, const void *source, size_t num_bytes)
{
	auto result = cudaMemcpy(destination, source, num_bytes, cudaMemcpyDefault);
	// TODO: Determine whether it was from host to device, device to host etc and
	// add this information to the error string
	throw_if_error(result, "Synchronously copying data");
}

/**
 * @brief Sets all bytes in a region of memory to a fixed value
 *
 * @note The equivalent of @ref std::memset - for any and all CUDA-related
 * memory spaces
 *
 * @param start starting address of the memory region to set;
 * may be in host-side memory, global CUDA-device-side memory or
 * CUDA-managed memory.
 * @param byte_value value to set the memory region to
 * @param num_bytes size of the memory region in bytes
 */
inline void set(void* start, int byte_value, size_t num_bytes)
{
	pointer_t<void> pointer { start };
	switch ( pointer.attributes().memory_type() ) {
	case device_memory:
	case managed_memory:
		memory::device::set(start, byte_value, num_bytes); break;
	case unregistered_memory:
	case host_memory:
		std::memset(start, byte_value, num_bytes); break;
	default:
		throw runtime_error(
			cuda::status::invalid_value,
			"CUDA returned an invalid memory type for the pointer 0x" + detail::ptr_as_hex(start)
		);
	}
}

/**
 * @brief Sets all bytes in a region of memory to 0 (zero)
 *
 * @param start starting address of the memory region to zero-out;
 * may be in host-side memory, global CUDA-device-side memory or
 * CUDA-managed memory.
 * @param num_bytes size of the memory region in bytes
 */
inline void zero(void* start, size_t num_bytes)
{
	return set(start, 0, num_bytes);
}

/**
 * @brief Sets all bytes of a single pointed-to value to 0
 *
 * @param ptr pointer to a single element of a certain type, which may
 * be in host-side memory, global CUDA-device-side memory or CUDA-managed
 * memory
 */
template <typename T>
inline void zero(T* ptr)
{
	zero(ptr, sizeof(T));
}

namespace detail {

/**
 * @note When constructing this class - destination first, source second
 * (otherwise you're implying the opposite direction of transfer).
 */
struct copy_params_t : cudaMemcpy3DParms {
	struct tag { };
protected:
	template <typename T>
	copy_params_t(tag, const void *ptr, const array_t<T, 3>& array) :
		cudaMemcpy3DParms { 0 },
		pitch(sizeof(T) * array.dimensions().width),
		pitched_ptr(make_cudaPitchedPtr(
			const_cast<void*>(ptr),
			pitch,
			array.dimensions().width,
			array.dimensions().height))
	{
		kind = cudaMemcpyDefault;
		extent = array.dimensions();
	}

public:
	template <typename T>
	copy_params_t(const array_t<T, 3>& destination, const void *source) :
		copy_params_t(tag{}, source, destination)
	{
		srcPtr = pitched_ptr;
		dstArray = destination.get();
	}

	template <typename T>
	copy_params_t(const void * destination, const array_t<T, 3>& source) :
		copy_params_t(tag{}, destination, source)
	{
		srcArray = source.get();
		dstPtr = pitched_ptr;
	}

	size_t pitch;
	cudaPitchedPtr pitched_ptr;
};

} // namespace detail

/**
 * Synchronously copies data from a 3-dimensional CUDA array into
 * non-array memory, on the device or on the host.
 *
 * @param destination A 3-dimensional CUDA array
 * @param source A pointer to a a memory region containing as many
 * elements as would fill the @source array, _contiguously_
 */
template<typename T>
void copy(array_t<T, 3>& destination, const void *source)
{
	const auto copy_params = detail::copy_params_t(destination, source);
	auto result = cudaMemcpy3D(&copy_params);
	throw_if_error(result, "Synchronously copying into a 3-dimensional CUDA array");
}

/**
 * Synchronously copies data from a 3-dimensional CUDA array into
 * non-array memory, on the device or on the host.
 *
 * @param destination A pointer to a a memory region large enough to fit
 * all elements of the @source array, contiguously
 * @param source A 3-dimensional CUDA array
 */
template<typename T>
void copy(void *destination, const array_t<T, 3>& source)
{
	const auto copy_params = detail::copy_params_t(destination, source);
	auto result = cudaMemcpy3D(&copy_params);
	throw_if_error(result, "Synchronously copying from a 3-dimensional CUDA array");
}

/**
 * Synchronously copies data from memory spaces into CUDA arrays.
 *
 * @param destination A CUDA array @ref cuda::array_t
 * @param source A pointer to a a memory region of size `destination.size() * sizeof(T)`
 */
template<typename T>
void copy(array_t<T, 2>& destination, const void *source)
{
	const auto dimensions = destination.dimensions();
	const auto width_in_bytes = sizeof(T) * dimensions.width;
	const auto source_pitch = width_in_bytes; // i.e. no padding
	const array::dimensions_t<2> offsets { 0, 0 };
	auto result = cudaMemcpy2DToArray(
		destination.get(),
		offsets.width,
		offsets.height,
		source,
		source_pitch,
		width_in_bytes,
		dimensions.height,
		cudaMemcpyDefault);
	throw_if_error(result, "Synchronously copying into a 2D CUDA array");
}

/**
 * Synchronously copies data from a 2-dimensional CUDA array into
 * non-array memory, on the device or on the host.
 *
 * @param destination A pointer to a a memory region large enough to fit
 * all elements of the @p source array, contiguously
 * @param source A 2-dimensional CUDA array */
template<typename T>
void copy(void* destination, const array_t<T, 2>& source)
{
	const auto dimensions = source.dimensions();
	const auto width_in_bytes = sizeof(T) * dimensions.width;
	const auto destination_pitch = width_in_bytes; // i.e. no padding
	const array::dimensions_t<2> offsets { 0, 0 };
	auto result = cudaMemcpy2DFromArrayAsync(
		destination,
		destination_pitch,
		source.get(),
		offsets.width,
		offsets.height,
		width_in_bytes,
		dimensions.height,
		cudaMemcpyDefault);
	throw_if_error(result, "Synchronously copying out of a 2D CUDA array");
}

/**
 * Synchronously copies a single (typed) value between memory spaces or within a memory space.
 *
 * @param destination a value residing either in host memory or on any CUDA
 * device's global memory
 * @param source a value residing either in host memory or on any CUDA
 * device's global memory
 */
template <typename T>
inline void copy_single(T* destination, const T* source)
{
	copy(destination, source, sizeof(T));
}

namespace async {

namespace detail {

/**
 * Asynchronously copies data between memory spaces or within a memory space.
 *
 * @note Since we assume Compute Capability >= 2.0, all devices support the
 * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
 * where the data is located, and one does not have to specify this.
 *
 * @note asynchronous version of @ref memory::copy
 *
 * @param destination A pointer to a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param source A pointer to a a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param num_bytes The number of bytes to copy from @p source to @p destination
 * @param stream_id A stream on which to enqueue the copy operation
 */
inline void copy(void *destination, const void *source, size_t num_bytes, stream::id_t stream_id)
{
	auto result = cudaMemcpyAsync(destination, source, num_bytes, cudaMemcpyDefault, stream_id);

	// TODO: Determine whether it was from host to device, device to host etc and
	// add this information to the error string
	throw_if_error(result, "Scheduling a memory copy on stream " + cuda::detail::ptr_as_hex(stream_id));
}

template<typename T>
void copy(array_t<T, 3>& destination, const void *source, stream::id_t stream_id)
{
	const auto copy_params = memory::detail::copy_params_t(destination, source);
	auto result = cudaMemcpy3DAsync(&copy_params, stream_id);
	throw_if_error(result, "Scheduling a memory copy into a 3D CUDA array on stream " + cuda::detail::ptr_as_hex(stream_id));
}

template<typename T>
void copy(void* destination, const array_t<T, 3>& source, stream::id_t stream_id)
{
	const auto copy_params = memory::detail::copy_params_t(destination, source);
	auto result = cudaMemcpy3DAsync(&copy_params, stream_id);
	throw_if_error(result, "Scheduling a memory copy out of a 3D CUDA array on stream " + cuda::detail::ptr_as_hex(stream_id));
}

template<typename T>
void copy(array_t<T, 2>& destination, const void *source, stream::id_t stream_id)
{
	const auto dimensions = destination.dimensions();
	const auto width_in_bytes = sizeof(T) * dimensions.width;
	const auto source_pitch = width_in_bytes; // i.e. no padding
	const array::dimensions_t<2> offsets { 0, 0 };
	auto result = cudaMemcpy2DToArrayAsync(
		destination.get(),
		offsets.width,
		offsets.height,
		source,
		source_pitch,
		width_in_bytes,
		dimensions.height,
		cudaMemcpyDefault,
		stream_id);
	throw_if_error(result, "Scheduling a memory copy into a 2D CUDA array on stream " + cuda::detail::ptr_as_hex(stream_id));
}

template<typename T>
void copy(void* destination, const array_t<T, 2>& source, cuda::stream::id_t stream_id)
{
	const auto dimensions = source.dimensions();
	const auto width_in_bytes = sizeof(T) * dimensions.width;
	const auto destination_pitch = width_in_bytes; // i.e. no padding
	const array::dimensions_t<2> offsets { 0, 0 };
	auto result = cudaMemcpy2DFromArrayAsync(
		destination,
		destination_pitch,
		source.get(),
		offsets.width,
		offsets.height,
		width_in_bytes,
		dimensions.height,
		cudaMemcpyDefault,
		stream_id);
	throw_if_error(result, "Scheduling a memory copy out of a 3D CUDA array on stream " + cuda::detail::ptr_as_hex(stream_id));
}

/**
 * Synchronously copies a single (typed) value between memory spaces or within a memory space.
 *
 * @note asynchronous version of @ref memory::copy_single
 *
 * @param destination a value residing either in host memory or on any CUDA
 * device's global memory
 * @param source a value residing either in host memory or on any CUDA
 * device's global memory
 * @param stream_id A stream on which to enqueue the copy operation
 */
template <typename T>
inline void copy_single(T& destination, const T& source, stream::id_t stream_id)
{
	copy(&destination, &source, sizeof(T), stream_id);
}

} // namespace detail

/**
 * Asynchronously copies data between memory spaces or within a memory space.
 *
 * @note Since we assume Compute Capability >= 2.0, all devices support the
 * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
 * where the data is located, and one does not have to specify this.
 *
 * @note asynchronous version of @ref memory::copy
 *
 * @param destination A pointer to a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param source A pointer to a a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param num_bytes The number of bytes to copy from @p source to @p destination
 * @param stream A stream on which to enqueue the copy operation
 */
inline void copy(void *destination, const void *source, size_t num_bytes, stream_t& stream);

/**
 * Asynchronously copies data from memory spaces into CUDA arrays.
 *
 * @note asynchronous version of @ref memory::copy
 *
 * @param destination A CUDA array @ref cuda::array_t
 * @param source A pointer to a a memory region of size `destination.size() * sizeof(T)`
 * @param stream schedule the copy operation into this CUDA stream
 */
template <typename T, size_t NumDimensions>
inline void copy(array_t<T, NumDimensions>& destination, const void *source, stream_t& stream);

/**
 * Asynchronously copies data from CUDA arrays into memory spaces.
 *
 * @note asynchronous version of @ref memory::copy
 *
 * @param destination A pointer to a a memory region of size `source.size() * sizeof(T)`
 * @param source A CUDA array @ref cuda::array_t
 * @param stream schedule the copy operation into this CUDA stream
 */
template <typename T, size_t NumDimensions>
inline void copy(void* destination, const array_t<T, NumDimensions>& source, stream_t& stream);

/**
 * Synchronously copies a single (typed) value between memory spaces or within a memory space.
 *
 * @note asynchronous version of @ref memory::copy_single
 *
 * @param destination a value residing either in host memory or on any CUDA
 * device's global memory
 * @param source a value residing either in host memory or on any CUDA
 * device's global memory
 */
template <typename T>
inline void copy_single(T& destination, const T& source, stream_t& stream);

} // namespace async

namespace device {

namespace async {

namespace detail {

inline void set(void* start, int byte_value, size_t num_bytes, stream::id_t stream_id)
{
	// TODO: Double-check that this call doesn't require setting the current device
	auto result = cudaMemsetAsync(start, byte_value, num_bytes, stream_id);
	throw_if_error(result, "asynchronously memsetting an on-device buffer");
}

inline void zero(void* start, size_t num_bytes, stream::id_t stream_id)
{
	set(start, 0, num_bytes, stream_id);
}

} // namespace detail

/**
 * Asynchronously sets all bytes in a stretch of memory to a single value
 *
 * @note asynchronous version of @ref memory::zero
 *
 * @param start starting address of the memory region to set,
 * in a CUDA device's global memory
 * @param byte_value value to set the memory region to
 * @param num_bytes size of the memory region in bytes
 * @param stream stream on which to schedule this action
 */
inline void set(void* start, int byte_value, size_t num_bytes, stream_t& stream);

/**
 * Similar to @ref set(), but sets the memory to zero rather than an arbitrary value
 */
inline void zero(void* start, size_t num_bytes, stream_t& stream);

/**
 * @brief Asynchronously sets all bytes of a single pointed-to value
 * to 0 (zero).
 *
 * @note asynchronous version of @ref memory::zero
 *
 * @param start starting address of the memory region to zero-out,
 * in a CUDA device's global memory
 * @param num_bytes size of the memory region in bytes
 * @param stream stream on which to schedule this action
 */
template <typename T>
inline void zero(T* ptr, stream_t& stream)
{
	zero(ptr, sizeof(T), stream);
}

} // namespace async

} // namespace device

/**
 * @namespace host
 * Host-side (= system) memory which is "pinned", i.e. resides in
 * a fixed physical location - and allocated by the CUDA driver.
 */
namespace host {

/**
 * Allocate pinned host memory
 *
 * @note "Pinned" memory is excepted from virtual memory swapping-out,
 * and is allocated in contiguous physical RAM addresses, making it
 * possible to copy to and from it to the the GPU using DMA without
 * assistance from the GPU. Typically for PCIe 3.0, the effective
 * bandwidth is twice as fast as copying from or to naively-allocated
 * host memory.
 *
 * @throws cuda::runtime_error if allocation fails for any reason
 *
 * @todo Consider a variant of this supporting the cudaHostAlloc flags
 *
 * @param size_in_bytes the amount of memory to allocate, in bytes
 * @return a pointer to the allocated stretch of memory
 */
inline void* allocate(size_t size_in_bytes /* write me:, bool recognized_by_all_contexts */)
{
	void* allocated = nullptr;
	// Note: the typed cudaMallocHost also takes its size in bytes, apparently, not in number of elements
	auto result = cudaMallocHost(&allocated, size_in_bytes);
	if (is_success(result) && allocated == nullptr) {
		// Can this even happen? hopefully not
		result = cudaErrorUnknown;
	}
	throw_if_error(result, "Failed allocating " + std::to_string(size_in_bytes) + " bytes of host memory");
	return allocated;
}

/**
 * Free a region of pinned host memory which was allocated with @ref allocate.
 */
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


/**
 * @brief Makes a preallocated memory region behave as though it were allocated with @ref host::allocate.
 *
 * Page-locks the memory range specified by ptr and size and maps it for the device(s) as specified by
 * flags. This memory range also is added to the same tracking mechanism as cudaHostAlloc() to
 * automatically accelerate calls to functions such as cudaMemcpy().
 *
 * @param ptr A pre-allocated stretch of host memory
 * @param size the size in bytes the memory region to register/pin
 * @param flags
 */
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
	detail::register_(
		ptr, size,
		  (register_mapped_io_space ? cudaHostRegisterIoMemory : 0)
		| (map_into_device_space ? cudaHostRegisterMapped : 0)
		| (make_device_side_accesible_to_all ? cudaHostRegisterPortable : 0)
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

/**
 * @brief Sets all bytes in a stretch of host-side memory to a single value
 *
 * @note a wrapper for @ref std::memset
 *
 * @param start starting address of the memory region to set,
 * in host memory; can be either CUDA-allocated or otherwise.
 * @param byte_value value to set the memory region to
 * @param num_bytes size of the memory region in bytes
 */
inline void set(void* start, int byte_value, size_t num_bytes)
{
	std::memset(start, byte_value, num_bytes);
	// TODO: Error handling?
}

inline void zero(void* start, size_t num_bytes)
{
	set(start, 0, num_bytes);
}

template <typename T>
inline void zero(T* ptr)
{
	zero(ptr, sizeof(T));
}


} // namespace host

/**
 * @namespace managed
 * This type of memory, also known as _unified_ memory, appears within
 * a unified, all-system address space - and is used with the same
 * address range on the host and on all relevant CUDA devices on a
 * system. It is paged, so that it may exceed the physical size of
 * a CUDA device's global memory. The CUDA driver takes care of
 * "swapping" pages "out" from a device to host memory or "swapping"
 * them back "in", as well as of propagation of changes between
 * devices and host-memory.
 *
 * @note For more details, see
 * <a href="https://devblogs.nvidia.com/parallelforall/unified-memory-cuda-beginners/">
 * Unified Memory for CUDA Beginners</a> on the
 * <a href="https://devblogs.nvidia.com/parallelforall">Parallel4All blog</a>.
 *
 */
namespace managed {

enum class initial_visibility_t {
	to_all_devices,
	to_supporters_of_concurrent_managed_access,
};

namespace detail {

inline void* allocate(
	size_t                num_bytes,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices)
{
	void* allocated = nullptr;
	auto flags = (initial_visibility == initial_visibility_t::to_all_devices) ?
		cudaMemAttachGlobal : cudaMemAttachHost;
	// Note: Despite the templating by T, the size is still in bytes,
	// not in number of T's
	auto status = cudaMallocManaged(&allocated, num_bytes, flags);
	if (is_success(status) && allocated == nullptr) {
		// Can this even happen? hopefully not
		status = (status_t) status::unknown;
	}
	throw_if_error(status,
		"Failed allocating " + std::to_string(num_bytes) + " bytes of managed CUDA memory");
	return allocated;
}

/**
 * Free a region of pinned host memory which was allocated with @ref allocate.
 */
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
		return detail::allocate(num_bytes, InitialVisibility);
	}
};
struct deleter {
	void operator()(void* ptr) const { cuda::memory::device::free(ptr); }
};

inline void* allocate(
	cuda::device::id_t    device_id,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices)
{
	cuda::device::current::detail::scoped_override_t<> set_device_for_this_scope(device_id);
	return detail::allocate(num_bytes, initial_visibility);
}

} // namespace detail

/**
 * @brief Allocate a a region of managed memory, accessible with the same
 * address on the host and on CUDA devices
 *
 * @param device_id the initial device which is likely to access the managed
 * memory region (and which will certainly have actually allocated for it)
 * @param num_bytes size of each of the regions of memory to allocate
 * @param initial_visibility will the allocated region be visible, using the
 * common address, to all CUDA device (= more overhead, more work for the CUDA
 * runtime) or just to those devices with some hardware features to assist in
 * this task (= less overhead)?
 */
inline void* allocate(
	cuda::device_t        device,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices
);

/**
 * Free a managed memory region (host-side and device-side regions on all devices
 * where it was allocated, all with the same address) which was allocated with
 * @ref allocate.
 */
inline void free(void* managed_ptr)
{
	auto result = cudaFree(managed_ptr);
	throw_if_error(result,
		"Freeing managed memory (host and device regions) at address 0x"
		+ cuda::detail::ptr_as_hex(managed_ptr));
}

namespace async {

namespace detail {

inline void prefetch(
	const void*         managed_ptr,
	size_t              num_bytes,
	cuda::device::id_t  destination,
	stream::id_t        stream_id)
{
	auto result = cudaMemPrefetchAsync(managed_ptr, num_bytes, destination, stream_id);
	throw_if_error(result,
		"Prefetching " + std::to_string(num_bytes) + " bytes of managed memory at address "
		 + cuda::detail::ptr_as_hex(managed_ptr) + " to device " + std::to_string(destination));
}

} // namespace detail

/**
 * @brief Prefetches a region of managed memory to a specific device, so
 * it can later be used there without waiting for I/O from the host or other
 * devices.
 */
inline void prefetch(
	const void*      managed_ptr,
	size_t           num_bytes,
	cuda::device_t   destination,
	cuda::stream_t&  stream);

/**
 * @brief Prefetches a region of managed memory into host memory. It can
 * later be used there without waiting for I/O from any of the CUDA devices.
 */
inline void prefetch_to_host(
	const void*      managed_ptr,
	size_t           num_bytes)
{
	auto result = cudaMemPrefetchAsync(managed_ptr, num_bytes, cudaCpuDeviceId, stream::default_stream_id);
		// The stream ID will be ignored by the CUDA runtime API when this pseudo
		// device indicator is used.
	throw_if_error(result,
		"Prefetching " + std::to_string(num_bytes) + " bytes of managed memory at address "
		 + cuda::detail::ptr_as_hex(managed_ptr) + " into host memory");
}

} // namespace async

} // namespace managed

namespace mapped {

namespace detail {

/**
 * Allocates a mapped pair of memory regions - on the current device
 * and in host memory.
 *
 * @param size_in_bytes size of each of the two regions, in bytes.
 * @param options indication of how the CUDA driver will manage
 * the region pair
 * @return the allocated pair (with both regions being non-null)
 */
inline region_pair allocate(
	size_t                           size_in_bytes,
	region_pair::allocation_options  options)
{
	region_pair allocated;
	auto flags = cuda::memory::mapped::detail::make_cuda_host_alloc_flags(options);
	// Note: the typed cudaHostAlloc also takes its size in bytes, apparently,
	// not in number of elements
	auto status = cudaHostAlloc(&allocated.host_side, size_in_bytes, flags);
	if (is_success(status) && (allocated.host_side == nullptr)) {
		// Can this even happen? hopefully not
		status = cudaErrorUnknown;
	}
	if (is_success(status)) {
		auto get_device_pointer_flags = 0u; // see the CUDA runtime documentation
		status = cudaHostGetDevicePointer(
			&allocated.device_side,
			allocated.host_side,
			get_device_pointer_flags);
	}
	throw_if_error(status,
		"Failed allocating a mapped pair of memory regions of size " + std::to_string(size_in_bytes)
			+ " bytes of global memory on device " + std::to_string(cuda::device::current::detail::get_id()));
	return allocated;
}

/**
 * Allocates a mapped pair of memory regions - on a CUDA device
 * and in host memory.
 *
 * @param device_id The device on which to allocate the device-side region
 * @param size_in_bytes size of each of the two regions, in bytes.
 * @param options indication of how the CUDA driver will manage
 * the region pair
 * @return the allocated pair (with both regions being non-null)
 */

inline region_pair allocate(
	cuda::device::id_t               device_id,
	size_t                           size_in_bytes,
	region_pair::allocation_options  options)
{
	cuda::device::current::detail::scoped_override_t<> set_device_for_this_scope(device_id);
	return detail::allocate(size_in_bytes, options);
}

} // namespace detail


inline region_pair allocate(
	cuda::device_t                   device,
	size_t                           size_in_bytes,
	region_pair::allocation_options  options = {
		region_pair::isnt_portable_across_cuda_contexts,
		region_pair::without_cpu_write_combining }
	);

/**
 * Free a pair of mapped memory regions
 *
 * @param pair a pair of regions allocated with @ref allocate (or with
 * the C-style CUDA runtime API directly)
 */
inline void free(region_pair pair)
{
	auto result = cudaFreeHost(pair.host_side);
	throw_if_error(result, "Could not free mapped memory region pair.");
}

/**
 * Free a pair of mapped memory regions using just one of them
 *
 * @param ptr a pointer to one of the mapped regions (can be either
 * the device-side or the host-side)
 */
inline void free_region_pair_of(void* ptr)
{
	auto wrapped_ptr = pointer_t<void> { ptr };
	auto result = cudaFreeHost(wrapped_ptr.get_for_host());
	throw_if_error(result, "Could not free mapped memory region pair.");
}

/**
 * Determine whether a given stretch of memory was allocated as part of
 * a mapped pair of host and device memory regions
 *
 * @param ptr the beginning of a memory region - in either host or device
 * memory - to check
 * @return `true` iff the region was allocated as one side of a mapped
 * memory region pair
 */
inline bool is_part_of_a_region_pair(void* ptr)
{
	auto wrapped_ptr = pointer_t<void> { ptr };
	return wrapped_ptr.other_side_of_region_pair().get() != nullptr;
}

} // namespace mapped

} // namespace memory

} // namespace cuda

#endif // CUDA_API_WRAPPERS_MEMORY_HPP_
