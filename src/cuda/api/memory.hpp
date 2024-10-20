/**
 * @file
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

#include "copy_parameters.hpp"
#include "array.hpp"
#include "constants.hpp"
#include "current_device.hpp"
#include "error.hpp"
#include "pointer.hpp"
#include "current_context.hpp"
#include "detail/unique_span.hpp"

// The following is needed for cudaGetSymbolAddress, cudaGetSymbolSize
#include <cuda_runtime.h>

#include <memory>
#include <cstring> // for ::std::memset
#include <vector>
#include <utility>

namespace cuda {

///@cond
class device_t;
class context_t;
class stream_t;
class module_t;
///@endcond

namespace memory {

/**
 * A memory allocation setting: Can the allocated memory be used in other
 * CUDA driver contexts (in addition to the implicit default context we
 * have with the Runtime API).
 */
enum class portability_across_contexts : bool {
	isnt_portable = false,
	is_portable   = true,
};

/**
 * A memory allocation setting: Should the allocated memory be configured
 * as _write-combined_, i.e. a write may not be immediately applied to the
 * allocated region and propagated (e.g. to caches, over the PCIe bus).
 * Instead, writes will be applied as convenient, possibly in batch.
 *
 * Write-combining memory frees up the host's L1 and L2 cache resources,
 * making more cache available to the rest of the application. In addition,
 * write-combining memory is not snooped during transfers across the PCI
 * Express bus, which can improve transfer performance.
 *
 * Reading from write-combining memory from the host is prohibitively slow,
 * so write-combining memory should in general be used for memory that the
 * host only writes to.
 */
enum cpu_write_combining : bool {
	without_wc = false,
	with_wc    = true,
};

/**
 * options accepted by CUDA's allocator of memory with a host-side aspect
 * (host-only or managed memory).
 */
struct allocation_options {
	/// whether or not the allocated region can be used in different CUDA contexts.
	portability_across_contexts  portability;

	/// whether or not the GPU can batch multiple writes to this area and propagate them at its convenience.
	cpu_write_combining          write_combining;
};

namespace detail_ {

template <typename T, bool CheckConstructibility = false>
inline void check_allocation_type() noexcept
{
	static_assert(::std::is_trivially_constructible<T>::value,
		"Attempt to create a typed buffer of a non-trivially-constructive type");
	static_assert(not CheckConstructibility or ::std::is_trivially_destructible<T>::value,
		"Attempt to create a typed buffer of a non-trivially-destructible type "
		"without allowing for its destruction");
	static_assert(::std::is_trivially_copyable<T>::value,
		"Attempt to create a typed buffer of a non-trivially-copyable type");
}

inline unsigned make_cuda_host_alloc_flags(allocation_options options)
{
	return
		(options.portability     == portability_across_contexts::is_portable ? CU_MEMHOSTALLOC_PORTABLE      : 0) |
		(options.write_combining == cpu_write_combining::with_wc             ? CU_MEMHOSTALLOC_WRITECOMBINED : 0);
}

} // namespace detail_

/**
 * Memory regions appearing in both on the host-side and device-side address
 * spaces with the regions in both spaces mapped to each other (i.e. guaranteed
 * to have the same contents on access up to synchronization details). Consult the
 * <a href="http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory">
 * CUDA C programming guide section on mapped memory</a> for more details.
 */
namespace mapped {

// TODO: Perhaps make this an array of size 2 and use aspects to index it?

/**
 * A pair of memory spans, one in device-global memory and one in host/system memory,
 * mapped to it.
 *
 * @note This can be thought of as a type-imbued @ref region_pair_t
 */
template <typename T>
struct span_pair_t {
	/// The two regions mapped to each other by the CUDA driver; they must be
	/// identical in size.
	span<T> host_side, device_side;

	///@cond
	constexpr operator ::std::pair<span<T>, span<T>>() const { return { host_side, device_side }; }
	constexpr operator ::std::pair<region_t, region_t>() const { return { host_side, device_side }; }
	///@endcond
};

/**
 * A pair of memory regions, one in system (=host) memory and one on a
 * CUDA device's memory - mapped to each other
 *
 * @note this is the mapped-pair equivalent of a `void *`; it is not a
 * proper memory region abstraction, i.e. it has no size information
 */
struct region_pair_t {
	/// The two regions mapped to each other by the CUDA driver; they must be
	/// identical in size.
	memory::region_t host_side, device_side;

	/// @returns two spans, one for the each of the host-side and device-side regions
	template <typename T>
	constexpr span_pair_t<T> as_spans() const
	{
		return { host_side.as_span<T>(), device_side.as_span<T>() };
	}
};

} // namespace mapped

///CUDA-Device-global memory on a single device (not accessible from the host)
namespace device {

namespace detail_ {

/**
 * Allocate memory on current device
 *
 * @param num_bytes amount of memory to allocate in bytes
 */
inline cuda::memory::region_t allocate_in_current_context(size_t num_bytes)
{
	device::address_t allocated = 0;
	// Note: the typed cudaMalloc also takes its size in bytes, apparently,
	// not in number of elements
	auto status = cuMemAlloc(&allocated, num_bytes);
	if (is_success(status) && allocated == 0) {
		// Can this even happen? hopefully not
		status = static_cast<status_t>(status::unknown);
	}
	throw_if_error_lazy(status, "Failed allocating " + ::std::to_string(num_bytes) +
		" bytes of global memory on the current CUDA device");
	return {as_pointer(allocated), num_bytes};
}

inline region_t allocate(context::handle_t context_handle, size_t size_in_bytes)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	return allocate_in_current_context(size_in_bytes);
}

} // namespace detail_

#if CUDA_VERSION >= 11020
namespace async {

namespace detail_ {

/// Allocate memory asynchronously on a specified stream.
inline region_t allocate(
	context::handle_t  context_handle,
	stream::handle_t   stream_handle,
	size_t             num_bytes)
{
	device::address_t allocated = 0;
	// Note: the typed cudaMalloc also takes its size in bytes, apparently,
	// not in number of elements
	auto status = cuMemAllocAsync(&allocated, num_bytes, stream_handle);
	if (is_success(status) && allocated == 0) {
		// Can this even happen? hopefully not
		status = static_cast<decltype(status)>(status::unknown);
	}
	throw_if_error_lazy(status,
		"Failed scheduling an asynchronous allocation of " + ::std::to_string(num_bytes) +
		" bytes of global memory on " + stream::detail_::identify(stream_handle, context_handle) );
	return {as_pointer(allocated), num_bytes};
}

} // namespace detail_

/**
 * Schedule an allocation of device-side memory on a CUDA stream.
 *
 * @note The CUDA memory allocator guarantees alignment "suitabl[e] for any kind of variable"
 * (CUDA 9.0 Runtime API documentation), so probably at least 128 bytes.
 *
 * @throws cuda::runtime_error if scheduling fails for any reason
 *
 * @param stream the stream on which to register the allocation
 * @param size_in_bytes the amount of memory to allocate
 * @return a pointer to the region of memory which will become allocated once the stream
 * completes all previous tasks and proceeds to also complete the allocation.
 */
region_t allocate(const stream_t& stream, size_t size_in_bytes);

} // namespace async
#endif

/// Free a region of device-side memory (regardless of how it was allocated)
inline void free(void* ptr)
{
	auto result = cuMemFree(address(ptr));
#ifdef CAW_THROW_ON_FREE_IN_DESTROYED_CONTEXT
	if (result == status::success) { return; }
#else
	if (result == status::success or result == status::context_is_destroyed) { return; }
#endif
	throw runtime_error(result, "Freeing device memory at " + cuda::detail_::ptr_as_hex(ptr));
}

/// @copydoc free(void*)
inline void free(region_t region) { free(region.start()); }

#if CUDA_VERSION >= 11020
namespace async {

namespace detail_ {

inline void free(
	context::handle_t  context_handle,
	stream::handle_t   stream_handle,
	void*              allocated_region_start)
{
	auto status = cuMemFreeAsync(device::address(allocated_region_start), stream_handle);
	throw_if_error_lazy(status,
		"Failed scheduling an asynchronous freeing of the global memory region starting at "
		+ cuda::detail_::ptr_as_hex(allocated_region_start) + " on "
		+ stream::detail_::identify(stream_handle, context_handle) );
}

} // namespace detail_

/**
 * Schedule a de-allocation of device-side memory on a CUDA stream.
 *
 * @throws cuda::runtime_error if freeing fails
 *
 * @param stream the stream on which to register the allocation
 */
 ///@{
void free(const stream_t& stream, void* region_start);

inline void free(const stream_t& stream, region_t region)
{
	free(stream, region.data());
}
///@}

} // namespace async
#endif

/**
 * Allocate device-side memory on a CUDA device context.
 *
 * @note The CUDA memory allocator guarantees alignment "suitabl[e] for any kind of variable"
 * (CUDA 9.0 Runtime API documentation), and the CUDA programming guide guarantees
 * since at least version 5.0 that the minimum allocation is 256 bytes.
 *
 * @throws cuda::runtime_error if allocation fails for any reason
 *
 * @param context the context in which to allocate memory
 * @param size_in_bytes the amount of global device memory to allocate
 * @return a pointer to the allocated stretch of memory (only usable within @p context)
 */
inline region_t allocate(const context_t& context, size_t size_in_bytes);

/**
 * Allocate device-side memory on a CUDA device.
 *
 * @note The CUDA memory allocator guarantees alignment "suitabl[e] for any kind of variable"
 * (CUDA 9.0 Runtime API documentation), and the CUDA programming guide guarantees
 * since at least version 5.0 that the minimum allocation is 256 bytes.
 *
 * @throws cuda::runtime_error if allocation fails for any reason
 *
 * @param device the device on which to allocate memory
 * @param size_in_bytes the amount of global device memory to allocate
 * @return a pointer to the allocated stretch of memory (only usable on @p device)
 */
inline region_t allocate(const device_t& device, size_t size_in_bytes);

namespace detail_ {

// Note: Allocates _in the current context_! No current context => failure!
struct allocator {
	void* operator()(size_t num_bytes) const { return detail_::allocate_in_current_context(num_bytes).start(); }
};

struct deleter {
	void operator()(void* ptr) const { cuda::memory::device::free(ptr); }
};

} // namespace detail_

/**
 * Sets consecutive elements of a region of memory to a fixed
 * value of some width
 *
 * @note A generalization of `set()`, for different-size units.
 *
 * @tparam T An unsigned integer type of size 1, 2, 4 or 8
 * @param start The first location to set to @p value ; must be properly aligned.
 * @param value A (properly aligned) value to set T-elements to.
 * @param num_elements The number of type-T elements (i.e. _not_ necessarily the number of bytes).
 */
template <typename T>
void typed_set(T* start, const T& value, size_t num_elements);

/**
 * Sets all bytes in a region of memory to a fixed value
 *
 * @note The equivalent of @ref ::std::memset for CUDA device-side memory
 *
 * @param byte_value value to set the memory region to
 * @param start starting address of the memory region to set, in a CUDA
 * device's global memory
 * @param num_bytes size of the memory region in bytes
 */
inline void set(void* start, int byte_value, size_t num_bytes)
{
	return typed_set<unsigned char>(static_cast<unsigned char*>(start), static_cast<unsigned char>(byte_value), num_bytes);
}

/**
 * Sets all bytes in a region of memory to a fixed value
 *
 * @note The equivalent of @ref ::std::memset for CUDA device-side memory
 *
 * @param byte_value value to set the memory region to
 * @param region a region to zero-out, in a CUDA device's global memory
 */
inline void set(region_t region, int byte_value)
{
	set(region.start(), byte_value, region.size());
}

/**
 * Sets all bytes in a region of memory to 0 (zero)
 *
 * @param start the beginning of a region of memory to zero-out, accessible
 *     within a CUDA device's global memory
 * @param num_bytes the size in bytes of the region of memory to zero-out
 */
inline void zero(void* start, size_t num_bytes)
{
	set(start, 0, num_bytes);
}

/**
 * Sets all bytes in a region of memory to 0 (zero)
 *
 * @param region the memory region to zero-out, accessible as a part of a
 * CUDA device's global memory
 */
inline void zero(region_t region)
{
	zero(region.start(), region.size());
}


/**
 * Sets all bytes of a single pointed-to value to 0
 *
 * @param ptr pointer to a value of a certain type, accessible within
 *     in a CUDA device's global memory
 */
template <typename T>
inline void zero(T* ptr)
{
	zero(ptr, sizeof(T));
}

} // namespace device

/**
 * @note Since we assume Compute Capability >= 2.0, all devices support the
 * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
 * used in a copy function, where the data is located, and one does not have to specify this.
 *
 * @note the sources and destinations may all be in any memory space addressable
 * in the the unified virtual address space, which could be host-side memory,
 * device global memory, device constant memory etc.
 *
 */
///@{

/**
 * Synchronously copy data between different locations in memory
 *
 * @param source A pointer to a a memory region of size @p num_bytes.
 * @param num_bytes The number of bytes to copy from @p source to @p destination
 */
void copy(void *destination, const void *source, size_t num_bytes);

/**
 * @param destination A memory region of the same size as @p source.
 * @param source A region whose contents is to be copied.
 */
inline void copy(void* destination, const_region_t source)
{
	return copy(destination, source.start(), source.size());
}

/**
 * @param destination A region of memory to which to copy the data in @p source, of
 *     size at least that of @p source , either in host memory or on any CUDA
 *     device's global memory.
 * @param source A region whose contents is to be copied, either in host memory
 *     or on any CUDA device's global memory
 */
inline void copy(region_t destination, const_region_t source)
{
#ifndef NDEBUG
	if (destination.size() < source.size()) {
		throw ::std::logic_error("Can't copy a large region into a smaller one");
	}
#endif
	return copy(destination.start(), source);
}

/**
 * @param destination A region of memory to which to copy the data in @p source,
 *     of size at least that of @p source.
 * @param source A plain array whose contents is to be copied.
 */
template <typename T, size_t N>
inline void copy(region_t destination, const T(&source)[N])
{
#ifndef NDEBUG
	if (destination.size() < N * sizeof(T)) {
		throw ::std::logic_error("Source size exceeds destination size");
	}
#endif
	return copy(destination.start(), source, sizeof(T) * N);
}

/**
 * Copy the contents of a C-style array into a span of same-type elements
 *
 * @param destination A span of elements to overwrite with the array contents.
 * @param source A fixed-size C-style array from which copy data into
 *     @p destination,. As this is taken by reference rather than by address
 *     of the first element, there is no array-decay.
 */
template <typename T, size_t N>
inline void copy(span<T> destination, const T(&source)[N])
{
#ifndef NDEBUG
	if (destination.size() < N) {
		throw ::std::logic_error("Source size exceeds destination size");
	}
#endif
	return copy(destination.data(), source, sizeof(T) * N);
}

/**
 * Copy the contents of memory region into a C-style array, interpreting the memory
 * as a sequence of elements of the array's element type
 *
 * @param destination A region of memory to which to copy the data in @p source,
 *     of size at least that of @p source.
 * @param source A region of at least `sizeof(T)*N` bytes with whose data to fill
 *     the @p destination array.
 */
template <typename T, size_t N>
inline void copy(T(&destination)[N], const_region_t source)
{
#ifndef NDEBUG
	size_t required_size = N * sizeof(T);
	if (source.size() != required_size) {
		throw ::std::invalid_argument(
			"Attempt to copy a region of " + ::std::to_string(source.size()) +
				" bytes into an array of size " + ::std::to_string(required_size) + " bytes");
	}
#endif
	return copy(destination, source.start(), sizeof(T) * N);
}

/**
 * Copy the contents of a span into a C-style array
 *
 * @param destination A fixed-size C-style array, to which to copy the data in
 *     @p source,of size at least that of @p source.; as it is taken by reference
 *     rather than by address of the first element, there is no array-decay.
 * @param source A span of the same element type as the destination array,
 *     containing the data to be copied
 */
template <typename T, size_t N>
inline void copy(T(&destination)[N], span<T const> source)
{
#ifndef NDEBUG
	if (source.size() > N) {
		throw ::std::invalid_argument(
			"Attempt to copy a span of " + ::std::to_string(source.size()) +
			" elements into an array of " + ::std::to_string(N) + " elements");
	}
#endif
	return copy(destination, source.start(), sizeof(T) * N);
}

/**
 * Copy the contents of a C-style array to another location in memory
 *
 * @param destination The starting address of a sequence of @tparam N values
 *     of type @tparam T to overwrite with the array contents.
 * @param source A fixed-size C-style array from which copy data into
 *     @p destination,. As this is taken by reference rather than by address
 *     of the first element, there is no array-decay.
 */
template <typename T, size_t N>
inline void copy(void* destination, T (&source)[N])
{
	return copy(destination, source, sizeof(T) * N);
}

/**
 * Copy memory into a C-style array
 *
 * @param destination A fixed-size C-style array, to which to copy the data in
 *     @p source,of size at least that of @p source.; as it is taken by reference
 *     rather than by address of the first element, there is no array-decay.
 * @param source The starting address of a sequence of @tparam N elements to copy
 */
template <typename T, size_t N>
inline void copy(T(&destination)[N], T* source)
{
	return copy(destination, source, sizeof(T) * N);
}

/**
 * Copy one region of memory into another
 *
 * @param destination A region of memory to which to copy the data in @p source,
 *     of size at least that of @p source.
 * @param source A pointer to a a memory region of size @p num_bytes.
 * @param num_bytes The number of bytes to copy from @p source to @p destination
 */
inline void copy(region_t destination, void* source, size_t num_bytes)
{
#ifndef NDEBUG
	if (destination.size() < num_bytes) {
		throw ::std::logic_error("Number of bytes to copy exceeds destination size");
	}
#endif
	return copy(destination.start(), source, num_bytes);
}

/**
 * Copy one region of memory to another location
 *
 * @param destination The beginning of a target region of memory (of size at least
 *     @p num_bytes) into which to copy
 * @param source A region of memory from which to copy, of size at least @p num_bytes
 * @param num_bytes The number of bytes to copy from @p source to @p destination
 */
inline void copy(void* destination, const_region_t source, size_t num_bytes)
{
#ifndef NDEBUG
	if (source.size() < num_bytes) {
		throw ::std::logic_error("Number of bytes to copy exceeds source size");
	}
#endif
	return copy(destination, source.start(), num_bytes);
}

/**
 * Copy memory between memory regions
 *
 * @param destination A target region of memory into which to copy; enough memory will
 *     be copied to fill this region
 * @param source The beginning of a region of memory from which to copy
 */
inline void copy(region_t destination, void* source)
{
	return copy(destination, source, destination.size());
}
///@}

/**
 * Sets a number of bytes in memory to a fixed value
 *
 * @note The equivalent of @ref ::std::memset - for any and all CUDA-related
 * memory spaces
 *
 * @param ptr Address of the first byte in memory to set. May be in host-side
 *     memory, global CUDA-device-side memory or CUDA-managed memory.
 * @param byte_value value to set the memory region to
 * @param num_bytes The amount of memory to set to @p byte_value
 */
void set(void* ptr, int byte_value, size_t num_bytes);

/**
 * Sets all bytes in a region of memory to a fixed value
 *
 * @note The equivalent of @ref ::std::memset - for any and all CUDA-related
 * memory spaces
 *
 * @param region the memory region to set; may be in host-side memory,
 * global CUDA-device-side memory or CUDA-managed memory.
 * @param byte_value value to set the memory region to
 */
inline void set(region_t region, int byte_value)
{
	return set(region.start(), byte_value, region.size());
}

/**
 * Sets all bytes in a region of memory to 0 (zero)
 *
 * @param region the memory region to zero-out; may be in host-side memory,
 * global CUDA-device-side memory or CUDA-managed memory.
 */
inline void zero(region_t region)
{
	return set(region, 0);
}

/**
 * Zero-out a region of memory
 *
 * @param ptr the beginning of a region of memory to zero-out; may be in host-side
 *     memory, global CUDA-device-side memory or CUDA-managed memory.
 * @param num_bytes the size in bytes of the region of memory to zero-out
 */
inline void zero(void* ptr, size_t num_bytes)
{
	return set(ptr, 0, num_bytes);
}

/**
 * Sets all bytes of a single pointed-to value to 0
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

namespace detail_ {

inline status_t multidim_copy(::std::integral_constant<dimensionality_t, 2>, copy_parameters_t<2> params)
{
	// TODO: Move this logic into the scoped ensurer class
	auto context_handle = context::current::detail_::get_handle();
	if  (context_handle != context::detail_::none) {
		return cuMemcpy2D(&params);
	}
	auto current_device_id = cuda::device::current::detail_::get_id();
	context_handle = cuda::device::primary_context::detail_::obtain_and_increase_refcount(current_device_id);
	context::current::detail_::push(context_handle);
	// Note this _must_ be an intra-context copy, as inter-context is not supported
	// and there's no indication of context in the relevant data structures
	auto status = cuMemcpy2D(&params);
	context::current::detail_::pop();
	cuda::device::primary_context::detail_::decrease_refcount(current_device_id);
	return status;
}

inline status_t multidim_copy(::std::integral_constant<dimensionality_t, 3>, copy_parameters_t<3> params)
{
	if (params.srcContext == params.dstContext) {
		context::current::detail_::scoped_ensurer_t ensure_context_for_this_scope{params.srcContext};
		auto *intra_context_params = reinterpret_cast<base_copy_params<3>::intra_context_type *>(&params);
		return cuMemcpy3D(intra_context_params);
	}
	return cuMemcpy3DPeer(&params);
}

template<dimensionality_t NumDimensions>
status_t multidim_copy(copy_parameters_t<NumDimensions> params)
{
	return multidim_copy(::std::integral_constant<dimensionality_t, NumDimensions>{}, params);
}


} // namespace detail_

/**
 * An almost-generalized-case memory copy, taking a rather complex structure of
 * copy parameters - wrapping the CUDA driver's own most-generalized-case copy
 *
 * @tparam NumDimensions The number of dimensions of the parameter structure.
 * @param params A parameter structure with details regarding the copy source
 * and destination, including CUDA context specifications, which must have been
 * set in advance. This function will _not_ verify its validity, but rather
 * merely pass it on to the CUDA driver
 */
template<dimensionality_t NumDimensions>
void copy(copy_parameters_t<NumDimensions> params)
{
	status_t status = detail_::multidim_copy(params);
	throw_if_error_lazy(status, "Copying using a general copy parameters structure");
}

/**
 * Synchronously copies data from a CUDA array into non-array memory.
 *
 * @tparam NumDimensions the number of array dimensions; only 2 and 3 are supported values
 * @tparam T array element type
 *
 * @param destination A {@tparam NumDimensions}-dimensional CUDA array, including a specification
 *     of the context in which the array is defined.
 * @param source A pointer to a region of contiguous memory holding `destination.size()` values
 *     of type @tparam T. The memory may be located either on a CUDA device or in host memory.
 * @param context The context in which the source memory was allocated - possibly different than
 *     the target array context
 */
template<typename T, dimensionality_t NumDimensions>
void copy(const array_t<T, NumDimensions>& destination, const context_t& source_context, const T *source)
{
	auto dims = destination.dimensions();
	auto params = copy_parameters_t<NumDimensions> {};
	params.clear_offsets();
	params.template set_extent<T>(dims);
	params.set_endpoint(endpoint_t::source, source_context.handle(), const_cast<T*>(source), dims);
	params.set_endpoint(endpoint_t::destination, destination);
	params.clear_rest();
	copy(params);
}

/**
 * Synchronously copies data from a CUDA array into non-array memory.
 *
 * @tparam NumDimensions the number of array dimensions; only 2 and 3 are supported values
 * @tparam T array element type
 *
 * @param destination A {@tparam NumDimensions}-dimensional CUDA array
 * @param source A pointer to a region of contiguous memory holding `destination.size()` values
 * of type @tparam T. The memory may be located either on a CUDA device or in host memory.
 */
template<typename T, dimensionality_t NumDimensions>
void copy(const array_t<T, NumDimensions>& destination, const T *source)
{
	copy(destination, context_of(source), source);
}

/**
 * Copies a contiguous sequence of elements in memory into a CUDA array
 *
 * @tparam T a trivially-copy-constructible, trivially-copy-destructible type of array elements
 *
 * @note only as many elements as fit in the array are copied, and any extra elements
 * in the source span are ignored
 */
template<typename T, dimensionality_t NumDimensions>
void copy(const array_t<T, NumDimensions>& destination, span<T const> source)
{
#ifndef NDEBUG
	if (destination.size() < source.size()) {
		throw ::std::invalid_argument(
			"Attempt to copy a span of " + ::std::to_string(source.size()) +
			" elements into a CUDA array of " + ::std::to_string(destination.size()) + " elements");
	}
#endif
	copy(destination, source.data());
}

/**
 * Synchronously copies data into a CUDA array from non-array memory.
 *
 * @tparam NumDimensions the number of array dimensions; only 2 and 3 are supported values
 * @tparam T array element type
 *
 * @param destination A pointer to a region of contiguous memory holding `destination.size()` values
 * of type @tparam T. The memory may be located either on a CUDA device or in host memory.
 * @param source A {@tparam NumDimensions}-dimensional CUDA array
 */
template <typename T, dimensionality_t NumDimensions>
void copy(const context_t& context, T *destination, const array_t<T, NumDimensions>& source)
{
	auto dims = source.dimensions();
	auto params = copy_parameters_t<NumDimensions> {};
	params.clear_offset(endpoint_t::source);
	params.clear_offset(endpoint_t::destination);
	params.template set_extent<T>(dims);
	params.set_endpoint(endpoint_t::source, source);
	params.template set_endpoint<T>(endpoint_t::destination, context.handle(), destination, dims);
	params.set_default_pitches();
	params.clear_rest();
	copy(params);
}

/**
 * Synchronously copies data into a CUDA array from non-array memory.
 *
 * @tparam NumDimensions the number of array dimensions; only 2 and 3 are supported values
 * @tparam T array element type
 *
 * @param destination A pointer to a region of contiguous memory holding `destination.size()` values
 * of type @tparam T. The memory may be located either on a CUDA device or in host memory.
 * @param source A {@tparam NumDimensions}-dimensional CUDA array
 */
template <typename T, dimensionality_t NumDimensions>
void copy(T *destination, const array_t<T, NumDimensions>& source)
{
	copy(context_of(destination), destination, source);
}

/**
 * Copies the contents of a CUDA array into a sequence of contiguous elements in memory
 *
 * @tparam T a trivially-copy-constructible, trivially-destructible, type of array elements
 *
 * @note The @p destination span must be at least as larger as the volume of the array.
 */
template <typename T, dimensionality_t NumDimensions>
void copy(span<T> destination, const array_t<T, NumDimensions>& source)
{
#ifndef NDEBUG
	if (destination.size() < source.size()) {
		throw ::std::invalid_argument(
			"Attempt to copy a CUDA array of " + ::std::to_string(source.size()) +
			" elements into a span of " + ::std::to_string(destination.size()) + " elements");
	}
#endif
	copy(destination.data(), source);
}

/**
 * Copies the contents of one CUDA array to another
 *
 * @tparam T a trivially-copy-constructible type of array elements
 *
 * @note The destination array must be at least as large in each dimension as the source array.
 */
template <typename T, dimensionality_t NumDimensions>
void copy(const array_t<T, NumDimensions>& destination, const array_t<T, NumDimensions>& source)
{
	auto dims = source.dimensions();
	auto params = copy_parameters_t<NumDimensions> {};
	params.clear_offset(endpoint_t::source);
	params.clear_offset(endpoint_t::destination);
	params.template set_extent<T>(dims);
	params.set_endpoint(endpoint_t::source, source);
	params.set_endpoint(endpoint_t::destination, destination);
	params.set_default_pitches();
	params.clear_rest();;
	auto status = //(source.context() == destination.context()) ?
		detail_::multidim_copy<NumDimensions>(source.context_handle(), params);
	throw_if_error_lazy(status, "Copying from a CUDA array into a regular memory region");
}

/**
 * Copies the contents of a CUDA array into a region of memory
 *
 * @tparam T a trivially-copy-constructible type of array elements
 *
 * @note the @p destination region must be large enough to hold all elements of the array,
 * and may also be larger.
 */
template <typename T, dimensionality_t NumDimensions>
void copy(region_t destination, const array_t<T, NumDimensions>& source)
{
	if (destination.size() < source.size_bytes()) {
		throw ::std::logic_error("Attempt to copy an array into a memory region too small to hold the copy");
	}
	copy(destination.start(), source);
}

/**
 * Copies the contents of a region of memory into a CUDA array
 *
 * @tparam T a trivially-copy-constructible type of array elements
 *
 * @note only as many elements as fit in the array are copied, while the source region may
 * be larger than what they take up.
 */
template <typename T, dimensionality_t NumDimensions>
void copy(const array_t<T, NumDimensions>& destination, const_region_t source)
{
	if (destination.size_bytes() < source.size()) {
		throw ::std::logic_error("Attempt to copy into an array from a source region larger than the array's size");
	}
	copy(destination, static_cast<T const*>(source.start()));
}

/**
 * Synchronously copies a single (typed) value between two memory locations.
 *
 * @param destination a value residing either in host memory or on any CUDA
 * device's global memory
 * @param source a value residing either in host memory or on any CUDA
 * device's global memory
 */
template <typename T>
void copy_single(T* destination, const T* source)
{
	copy(destination, source, sizeof(T));
}

/// Asynchronous memory operations
namespace async {

namespace detail_ {

/**
 * Asynchronous versions of @ref memory::copy functions.
 *
 *
 * @note Since we assume Compute Capability >= 2.0, all devices support the
 * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
 * where the data is located, and one does not have to specify this.
 */

///@{

/**
 * Asynchronously copies data between memory spaces or within a memory space, but
 * within a single CUDA context.
 *
 * @param destination A pointer to a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param source A pointer to a memory region of size at least @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param num_bytes number of bytes to copy from @p source
 * @param stream_handle The handle of a stream on which to schedule the copy operation
*/
inline void copy(void* destination, const void* source, size_t num_bytes, stream::handle_t stream_handle)
{
	auto result = cuMemcpyAsync(device::address(destination), device::address(source), num_bytes, stream_handle);

	// TODO: Determine whether it was from host to device, device to host etc and
	// add this information to the error string
	throw_if_error_lazy(result, "Scheduling a memory copy on " + stream::detail_::identify(stream_handle));
}

/**
 * @param destination a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param source a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param stream_handle The handle of a stream on which to schedule the copy operation
 */
inline void copy(region_t destination, const_region_t source, stream::handle_t stream_handle)
{
#ifndef NDEBUG
	if (destination.size() < source.size()) {
		throw ::std::logic_error("Source size exceeds destination size");
	}
#endif
	copy(destination.start(), source.start(), source.size(), stream_handle);
}
///@}

using memory::copy_parameters_t;

inline status_t multidim_copy_in_current_context(
	::std::integral_constant<dimensionality_t, 2>,
	copy_parameters_t<2> params,
	stream::handle_t stream_handle)
{
	// Must be an intra-context copy, because CUDA does not support 2D inter-context copies and the copy parameters
	// structure holds no information about contexts.
	return cuMemcpy2DAsync(&params, stream_handle);
}

inline status_t multidim_copy_in_current_context(
	::std::integral_constant<dimensionality_t, 3>,
	copy_parameters_t<3> params,
	stream::handle_t stream_handle)
{
	if (params.srcContext == params.dstContext) {
		using intra_context_type = memory::detail_::base_copy_params<3>::intra_context_type;
		auto* intra_context_params = reinterpret_cast<intra_context_type *>(&params);
		return cuMemcpy3DAsync(intra_context_params, stream_handle);
	}
	return cuMemcpy3DPeerAsync(&params, stream_handle);

}

template<dimensionality_t NumDimensions>
status_t multidim_copy_in_current_context(copy_parameters_t<NumDimensions> params, stream::handle_t stream_handle) {
	return multidim_copy_in_current_context(::std::integral_constant<dimensionality_t, NumDimensions>{}, params, stream_handle);
}

// Note: Assumes the stream handle is for a stream in the current context
template<dimensionality_t NumDimensions>
status_t multidim_copy(
	context::handle_t                 context_handle,
	copy_parameters_t<NumDimensions>  params,
	stream::handle_t                  stream_handle)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	return multidim_copy_in_current_context(::std::integral_constant<dimensionality_t, NumDimensions>{}, params, stream_handle);
}

// Assumes the array and the stream share the same context, and that the destination is
// accessible from that context (e.g. allocated within it, or being managed memory, etc.)
template <typename T, dimensionality_t NumDimensions>
void copy(T *destination, const array_t<T, NumDimensions>& source, stream::handle_t stream_handle)
{
	using  memory::endpoint_t;
	auto dims = source.dimensions();
	//auto params = make_multidim_copy_params(destination, const_cast<T*>(source), destination.dimensions());
	auto params = copy_parameters_t<NumDimensions> {};
	params.clear_offset(endpoint_t::source);
	params.clear_offset(endpoint_t::destination);
	params.template set_extent<T>(dims);
	params.set_endpoint(endpoint_t::source, source);
	params.set_endpoint(endpoint_t::destination, const_cast<T*>(destination), dims);
	params.set_default_pitches();
	params.clear_rest();
	auto status = multidim_copy_in_current_context<NumDimensions>(params, stream_handle);
	throw_if_error(status, "Scheduling an asynchronous copy from an array into a regular memory region");
}


template <typename T, dimensionality_t NumDimensions>
void copy(const array_t<T, NumDimensions>&  destination, const T* source, stream::handle_t stream_handle)
{
	using memory::endpoint_t;
	auto dims = destination.dimensions();
	//auto params = make_multidim_copy_params(destination, const_cast<T*>(source), destination.dimensions());
	auto params = copy_parameters_t<NumDimensions>{};
	params.clear_offset(endpoint_t::source);
	params.clear_offset(endpoint_t::destination);
	params.template set_extent<T>(dims);
	params.set_endpoint(endpoint_t::source, const_cast<T*>(source), dims);
	params.set_endpoint(endpoint_t::destination, destination);
	params.set_default_pitches();
	params.clear_rest();
	auto status = multidim_copy_in_current_context<NumDimensions>(params, stream_handle);
	throw_if_error(status, "Scheduling an asynchronous copy from regular memory into an array");
}

/**
 * Synchronously copies a single (typed) value between memory spaces or within a memory space.
 *
 * @note asynchronous version of @ref memory::copy_single
 *
 * @note assumes the source and destination are all valid in the same context as that of the
 * context handle
 *
 * @param destination a value residing either in host memory or on any CUDA device's
 *     global memory
 * @param source a value residing either in host memory or on any CUDA device's global
 *     memory
 * @param stream_handle A stream on which to enqueue the copy operation
 */
template <typename T>
void copy_single(T* destination, const T* source, stream::handle_t stream_handle)
{
	copy(destination, source, sizeof(T), stream_handle);
}

} // namespace detail_

/**
 * Asynchronously copies data between memory spaces or within a memory space.
 *
 * @note Since we assume Compute Capability >= 2.0, all devices support the
 * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
 * where the data is located, and one does not have to specify this.
 *
 * @note asynchronous version of {@ref memory::copy(void*, void const*, size_t)}
 *
 * @param destination A pointer to a memory region of size @p num_bytes, either in host
 *     memory or on any CUDA device's global memory. Must be defined in the same context
 *     as the stream.
 * @param source A pointer to a memory region of size @p num_bytes, either in host
 *     memory or on any CUDA device's global memory. Must be defined in the same context
 *     as the stream.
 * @param num_bytes The number of bytes to copy from @p source to @p destination
 * @param stream A stream on which to enqueue the copy operation
 */
void copy(void* destination, void const* source, size_t num_bytes, const stream_t& stream);

/**
 * Asynchronously copies data between memory regions
 *
 * @param destination The beginning of a memory region of size @p num_bytes, either in host
 *     memory or on any CUDA device's global memory. Must be registered with, or visible in,
 *     in the same context as @p stream.
 * @param source A memory region of size @p num_bytes, either in host memory or on any
 *     CUDA device's global memory. Must be defined in the same context as the stream.
 * @param num_bytes The number of bytes to copy from @p source to @p destination
 * @param stream A stream on which to enqueue the copy operation
 */
inline void copy(void* destination, const_region_t source, size_t num_bytes, const stream_t& stream)
{
#ifndef NDEBUG
	if (source.size() < num_bytes) {
		throw ::std::logic_error("Attempt to copy more than the source region's size");
	}
#endif
	copy(destination, source.start(), num_bytes, stream);
}

/**
 * Asynchronously copies data between memory spaces or within a memory space.
 *
 * @param destination A memory region of size no less than @p num_bytes, either in host
 *     memory or on any CUDA device's global memory. Must be registered with, or visible
 *     in, in the same context as @p stream.
 * @param source A memory region of size @p num_bytes, either in host memory or on any
 *     CUDA device's global memory. Must be defined in the same contextas the stream.
 * @param num_bytes The number of bytes to copy from @p source to @p destination
 * @param stream A stream on which to enqueue the copy operation
 */
inline void copy(region_t destination, const_region_t source, size_t num_bytes, const stream_t& stream)
{
#ifndef NDEBUG
	if (destination.size() < num_bytes) {
		throw ::std::logic_error("Attempt to copy beyond the end of the destination region");
	}
#endif
	copy(destination.start(), source.start(), num_bytes, stream);
}

/**
 * Asynchronously copies data between memory regions
 *
 * @param destination Beginning of a memory region into which to copy data, either in host
 *      memory or on any CUDA device's global memory. The memory must be registered in,
 *      or visible within, the same context as {@p stream}.
 * @param source A memory region of size @p num_bytes, either in host memory or on any CUDA
 *     device's global memory. Must be defined in the same context as the stream.
 * @param stream A stream on which to enqueue the copy operation
 */
inline void copy(void* destination, const_region_t source, const stream_t& stream)
{
	copy(destination, source, source.size(), stream);
}

/**
 * Asynchronously copies data between memory regions
 *
 * @param destination A region of memory, either in host memory or on any CUDA device's
 *     global memory. Must be defined in the same context as the stream.
 * @param source A region of memory, either in host memory or on any CUDA device's
 *     global memory. Must be defined in the same context as the stream.
 * @param stream A stream on which to enqueue the copy operation
 */
inline void copy(region_t destination, const_region_t source, const stream_t& stream)
{
	copy(destination, source, source.size(), stream);
}

/**
 * Asynchronously copies data between memory regions
 *
 * @param destination A region of memory, either in host memory or on any CUDA device's
 *     global memory. Must be defined in the same context as the stream.
 * @param source A pointer to region of memory, of size like that of @p destination,
 *     either in host memory or on any CUDA device's global memory. Must be defined
 *     in the same context as the stream.
 * @param stream A stream on which to enqueue the copy operation
 */
inline void copy(region_t destination, void* source, const stream_t& stream)
{
	return copy(destination.start(), source, destination.size(), stream);
}

/**
 * Asynchronously copies data from an array into a memory region
 *
 * @param destination A region of memory, either in host memory or on any CUDA device's
 *     global memory. Must be defined in the same context as the stream.
 * @param source An array, either in host memory or on any CUDA device's global memory.
 * @param stream A stream on which to enqueue the copy operation
 */
template <typename T, size_t N>
inline void copy(region_t destination, const T(&source)[N], const stream_t& stream)
{
#ifndef NDEBUG
	if (destination.size() < N) {
		throw ::std::logic_error("Source size exceeds destination size");
	}
#endif
	return copy(destination.start(), source, sizeof(T) * N, stream);
}

/**
 * Asynchronously copies data from one region of memory to another
 *
 * @param destination A region of memory, either in host memory or on any CUDA device's
 *     global memory. Must be defined in the same context as the stream.
 * @param source Beginning of the region of memory to copy
 * @param num_bytes Amount of memory to copy
 * @param stream A stream on which to enqueue the copy operation
 */
inline void copy(region_t destination, void* source, size_t num_bytes, const stream_t& stream)
{
#ifndef NDEBUG
	if (destination.size() < num_bytes) {
		throw ::std::logic_error("Number of bytes to copy exceeds destination size");
	}
#endif
	return copy(destination.start(), source, num_bytes, stream);
}

/**
 * Asynchronously copies data into a CUDA array.
 *
 * @note asynchronous version of @ref memory::copy<T>(array_t<T, NumDimensions>&, const T*)
 *
 * @param destination A CUDA array to copy data into
 * @param source A pointer to a a memory region of size `destination.size() * sizeof(T)`
 * @param stream schedule the copy operation into this CUDA stream
 */
template <typename T, dimensionality_t NumDimensions>
void copy(array_t<T, NumDimensions>& destination, const T* source, const stream_t& stream);

/**
 * Asynchronously copies data into a CUDA array.
 *
 * @note asynchronous version of @ref memory::copy<T>(array_t<T, NumDimensions>&, const T*)
 *
 * @param destination A CUDA array to copy data into
 * @param source A memory region of size `destination.size() * sizeof(T)`
 * @param stream schedule the copy operation into this CUDA stream
 */
template <typename T, dimensionality_t NumDimensions>
void copy(array_t<T, NumDimensions>& destination, const_region_t source, const stream_t& stream)
{
#ifndef NDEBUG
	size_t required_size = destination.size() * sizeof(T);
	if (source.size() != required_size) {
		throw ::std::invalid_argument(
			"Attempt to copy a region of " + ::std::to_string(source.size()) +
			" bytes into an array of size " + ::std::to_string(required_size) + " bytes");
	}
#endif
	copy(destination, static_cast<T const*>(source.start()), stream);
}

/**
 * Asynchronously copies data from a CUDA array elsewhere
 *
 * @note asynchronous version of @ref memory::copy
 *
 * @param destination A pointer to a a memory region of size `source.size() * sizeof(T)`
 * @param source A CUDA array @ref cuda::array_t
 * @param stream schedule the copy operation into this CUDA stream
 */
template <typename T, dimensionality_t NumDimensions>
void copy(T* destination, const array_t<T, NumDimensions>& source, const stream_t& stream);

/**
 * Asynchronously copies data from a CUDA array elsewhere
 *
 * @note asynchronous version of @ref memory::copy
 *
 * @param destination A memory region of size `source.size() * sizeof(T)`
 * @param source A CUDA array @ref cuda::array_t
 * @param stream schedule the copy operation in this CUDA stream
 */
template <typename T, dimensionality_t NumDimensions>
void copy(region_t destination, const array_t<T, NumDimensions>& source, const stream_t& stream)
{
#ifndef NDEBUG
	size_t required_size = source.size() * sizeof(T);
	if (destination.size() < required_size) {
		throw ::std::invalid_argument(
			"Attempt to copy " + ::std::to_string(required_size) + " bytes from an array into a "
			"region of smaller size (" + ::std::to_string(destination.size()) + " bytes)");
	}
#endif
	copy(destination.start(), source, stream);
}

/**
 * Asynchronously copies data from a memory region into a C-style array
 *
 * @param destination A fixed-size C-style array, to which to copy the data in
 *     @p source,of size at least that of @p source.; as it is taken by reference
 *     rather than by address of the first element, there is no array-decay.
 * @param source The starting address of a sequence of @tparam N elements to copy
 * @param stream schedule the copy operation in this CUDA stream
 */
template <typename T, size_t N>
inline void copy(T(&destination)[N], T* source, const stream_t& stream)
{
	return copy(destination, source, sizeof(T) * N, stream);
}

/**
 * Asynchronously copies data from a memory region into a C-style array
 *
 * @param destination A fixed-size C-style array, to which to copy the data in
 *     @p source,of size at least that of @p source.; as it is taken by reference
 *     rather than by address of the first element, there is no array-decay.
 * @param source A region of at least `sizeof(T)*N` bytes with whose data to fill
 *     the @p destination array.
 * @param stream schedule the copy operation in this CUDA stream
 */
template <typename T, size_t N>
inline void copy(T(&destination)[N], const_region_t source, const stream_t& stream)
{
#ifndef NDEBUG
	size_t required_size = N * sizeof(T);
	if (source.size() != required_size) {
		throw ::std::invalid_argument(
			"Attempt to copy a region of " + ::std::to_string(source.size()) +
				" bytes into an array of size " + ::std::to_string(required_size) + " bytes");
	}
#endif
	return copy(destination, source.start(), sizeof(T) * N, stream);
}

/**
 * Copy a single (typed) value between memory locations
 *
 * @note asynchronous version of @ref memory::copy_single<T>(T&, const T&)
 *
 * @param destination a value residing either in host memory or on any CUDA device's global memory
 * @param source a value residing either in host memory or on any CUDA device's global memory
 * @param stream The CUDA command queue on which this copying will be enqueued
 */
template <typename T>
void copy_single(T* destination, const T* source, const stream_t& stream);

} // namespace async

namespace device {

namespace async {

namespace detail_ {

inline void set(void* start, int byte_value, size_t num_bytes, stream::handle_t stream_handle)
{
	// TODO: Double-check that this call doesn't require setting the current device
	auto result = cuMemsetD8Async(address(start), static_cast<unsigned char>(byte_value), num_bytes, stream_handle);
	throw_if_error_lazy(result, "asynchronously memsetting an on-device buffer");
}

inline void set(region_t region, int byte_value, stream::handle_t stream_handle)
{
	set(region.start(), byte_value, region.size(), stream_handle);
}

inline void zero(void* start, size_t num_bytes, stream::handle_t stream_handle)
{
	set(start, 0, num_bytes, stream_handle);
}

inline void zero(region_t region, stream::handle_t stream_handle)
{
	zero(region.start(), region.size(), stream_handle);
}

// TODO: Drop this in favor of <algorithm>-like functions under `cuda::`.
template <typename T>
inline void typed_set(T* start, const T& value, size_t num_elements, stream::handle_t stream_handle)
{
	static_assert(::std::is_trivially_copyable<T>::value, "Non-trivially-copyable types cannot be used for setting memory");
	static_assert(
		sizeof(T) == 1 or sizeof(T) == 2 or
		sizeof(T) == 4 or sizeof(T) == 8,
		"Unsupported type size - only sizes 1, 2 and 4 are supported");
	// TODO: Consider checking for alignment when compiling without NDEBUG
	status_t result = static_cast<status_t>(cuda::status::success);
	switch(sizeof(T)) {
		case(1): result = cuMemsetD8Async (address(start), reinterpret_cast<const ::std::uint8_t& >(value), num_elements, stream_handle); break;
		case(2): result = cuMemsetD16Async(address(start), reinterpret_cast<const ::std::uint16_t&>(value), num_elements, stream_handle); break;
		case(4): result = cuMemsetD32Async(address(start), reinterpret_cast<const ::std::uint32_t&>(value), num_elements, stream_handle); break;
	}
	throw_if_error_lazy(result, "Setting global device memory bytes");
}

} // namespace detail_


/**
 * Sets consecutive elements of a region of memory to a fixed value of some width
 *
 * @note A generalization of `async::set()`, for different-size units.
 *
 * @tparam T An unsigned integer type of size 1, 2, 4 or 8
 * @param start The first location to set to @p value ; must be properly aligned.
 * @param value A (properly aligned) value to set T-elements to.
 * @param num_elements The number of type-T elements (i.e. _not_ necessarily the number of bytes).
 * @param stream The stream on which to enqueue the operation.
 */
template <typename T>
void typed_set(T* start, const T& value, size_t num_elements, const stream_t& stream);

/**
 * Asynchronously sets all bytes in a stretch of memory to a single value
 *
 * @note asynchronous version of @ref memory::set(void*, int, size_t)
 *
 * @param start starting address of the memory region to set,
 * in a CUDA device's global memory
 * @param byte_value value to set the memory region to
 * @param num_bytes size of the memory region in bytes
 * @param stream stream on which to schedule this action
 */
inline void set(void* start, int byte_value, size_t num_bytes, const stream_t& stream)
{
	return typed_set<unsigned char>(
		static_cast<unsigned char*>(start),
		static_cast<unsigned char>(byte_value),
		num_bytes,
		stream);
}

/**
 * Asynchronously sets all bytes in a stretch of memory to 0.
 *
 * @note asynchronous version of @ref memory::zero(void*, size_t)
 *
 * @param start starting address of the memory region to set,
 * in a CUDA device's global memory
 * @param num_bytes size of the memory region in bytes
 * @param stream stream on which to schedule this action
 */
void zero(void* start, size_t num_bytes, const stream_t& stream);

/**
 * Asynchronously sets all bytes of a single pointed-to value
 * to 0 (zero).
 *
 * @note asynchronous version of @ref memory::zero(T*)
 *
 * @param ptr a pointer to the value to be to zero; must be valid in the
 * CUDA context of @p stream
 * @param stream stream on which to schedule this action
 */
template <typename T>
inline void zero(T* ptr, const stream_t& stream)
{
	zero(ptr, sizeof(T), stream);
}

} // namespace async

} // namespace device

namespace inter_context {

namespace detail_ {

inline void copy(
	void *             destination_address,
	context::handle_t  destination_context,
	const void *       source_address,
	context::handle_t  source_context,
	size_t             num_bytes)
{
	auto status = cuMemcpyPeer(
		reinterpret_cast<device::address_t>(destination_address),
		destination_context,
		reinterpret_cast<device::address_t>(source_address),
		source_context, num_bytes);
	throw_if_error_lazy(status,
		::std::string("Failed copying data between devices: From address ")
			+ cuda::detail_::ptr_as_hex(source_address) + " in "
			+ context::detail_::identify(source_context) + " to address "
			+ cuda::detail_::ptr_as_hex(destination_address) + " in "
			+ context::detail_::identify(destination_context) );
}

} // namespace detail_

void copy(
	void *             destination,
	const context_t&   destination_context,
	const void *       source_address,
	const context_t&   source_context,
	size_t             num_bytes);

inline void copy(
	void *             destination,
	const context_t&   destination_context,
	const_region_t     source,
	const context_t&   source_context)
{
	copy(destination, destination_context, source.start(), source_context, source.size());
}

inline void copy(
	region_t           destination,
	const context_t&   destination_context,
	const_region_t     source,
	const context_t&   source_context)
{
#ifndef NDEBUG
	if (destination.size() < destination.size()) {
		throw ::std::invalid_argument(
			"Attempt to copy a region of " + ::std::to_string(source.size()) +
				" bytes into a region of size " + ::std::to_string(destination.size()) + " bytes");
	}
#endif
	copy(destination.start(), destination_context, source, source_context);
}

template <typename T, dimensionality_t NumDimensions>
inline void copy(
	array_t<T, NumDimensions>  destination,
	array_t<T, NumDimensions>  source)
{
	// for arrays, a single mechanism handles both intra- and inter-context copying
	return memory::copy(destination, source);
}

namespace async {

namespace detail_ {

inline void copy(
	void *destination,
	context::handle_t destination_context_handle,
	const void *source,
	context::handle_t source_context_handle,
	size_t num_bytes,
	stream::handle_t stream_handle)
{
	auto result = cuMemcpyPeerAsync(
		device::address(destination),
		destination_context_handle,
		device::address(source),
		source_context_handle,
		num_bytes, stream_handle);

	// TODO: Determine whether it was from host to device, device to host etc and
	// add this information to the error string
	throw_if_error_lazy(result, "Scheduling an inter-context memory copy from "
		+ context::detail_::identify(source_context_handle) + " to "
		+ context::detail_::identify(destination_context_handle) + " on "
		+ stream::detail_::identify(stream_handle));
}

/**
 * @param destination a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param source a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param stream_handle The handle of a stream on which to schedule the copy operation
 */
inline void copy(
	region_t destination,
	context::handle_t destination_context_handle,
	const_region_t source,
	context::handle_t source_context_handle,
	stream::handle_t stream_handle)
{
#ifndef NDEBUG
	if (destination.size() < source.size()) {
		throw ::std::logic_error("Can't copy a large region into a smaller one");
	}
#endif
	copy(destination.start(), destination_context_handle, source.start(), source_context_handle, source.size(),
		stream_handle);
}

} // namespace detail_

/// Asynchronously copy a region of memory defined in one context into a region defined in another
void copy(
	void *           destination_address,
	context_t        destination_context,
	const void *     source_address,
	context_t        source_context,
	size_t           num_bytes,
	const stream_t&  stream);

/// Asynchronously copy a region of memory defined in one context into a region defined in another
void copy(
	void *           destination,
	context_t        destination_context,
	const_region_t   source,
	context_t        source_context,
	const stream_t&  stream);

/// Asynchronously copy a region of memory defined in one context into a region defined in another
inline void copy(
	region_t        destination,
	context_t        destination_context,
	const_region_t   source,
	context_t        source_context,
	const stream_t&  stream);

/// Asynchronously copy a CUDA array defined in one context into a CUDA array defined in another
template <typename T, dimensionality_t NumDimensions>
inline void copy(
	array_t<T, NumDimensions>  destination,
	array_t<T, NumDimensions>  source,
	const stream_t&            stream)
{
	// for arrays, a single mechanism handles both intra- and inter-context copying
	return memory::async::copy(destination, source, stream);
}

} // namespace async

} // namespace inter_context

/// Host-side (= system) memory which is "pinned", i.e. resides in
/// a fixed physical location - and allocated by the CUDA driver.
namespace host {

namespace detail_ {

// Even though the pinned memory should not in principle be associated in principle with a context or a device, in
// practice it needs to be registered somewhere - and that somewhere is a context. Passing a context does not mean
// the allocation will have special affinity to the device terms of better performance etc.
inline region_t allocate(
	const context::handle_t  context_handle,
	size_t                   size_in_bytes,
	allocation_options       options);

} // namespace detail_

/**
 * Allocates pinned host memory
 *
 * @note "pinned" memory is allocated in contiguous physical ram
 * addresses, making it possible to copy to and from it to the the
 * gpu using dma without assistance from the gpu. this improves
 * the copying bandwidth significantly over naively-allocated
 * host memory, and reduces overhead for the cpu.
 *
 * @throws cuda::runtime_error if allocation fails for any reason
 *
 * @param size_in_bytes the amount of memory to allocate, in bytes
 * @param options
 *     options to pass to the cuda host-side memory allocator; see
 *     {@ref memory::allocation_options}.
 * @return a pointer to the allocated stretch of memory
 */
region_t allocate(size_t size_in_bytes, allocation_options options);

/**
 * @copydoc allocate(size_t, allocation_options)
 *
 * @param portability
 *     whether or not the allocated region can be used in different
 *     CUDA contexts.
 * @param cpu_wc
 *     whether or not the GPU can batch multiple writes to this area
 *     and propagate them at its convenience.
 *
 */
inline region_t allocate(
	size_t                       size_in_bytes,
	portability_across_contexts  portability = portability_across_contexts(false),
	cpu_write_combining          cpu_wc = cpu_write_combining(false))
{
	return allocate(size_in_bytes, allocation_options{ portability, cpu_wc } );
}

/// @copydoc allocate(size_t, portability_across_contexts, cpu_write_combining)
inline region_t allocate(size_t size_in_bytes, cpu_write_combining cpu_wc)
{
	return allocate(size_in_bytes, allocation_options{ portability_across_contexts(false), cpu_write_combining(cpu_wc)} );
}

/**
 * Frees a region of pinned host memory which was allocated with one of the pinned host
 * memory allocation functions.
 *
 * @note The address provided must be the _beginning_ of the region of allocated memory;
 * and the entire region is freed (i.e. the region size is known to/determined by the driver)
 */
inline void free(void* host_ptr)
{
	auto result = cuMemFreeHost(host_ptr);
#ifdef CAW_THROW_ON_FREE_IN_DESTROYED_CONTEXT
	if (result == status::success) { return; }
#else
	if (result == status::success or result == status::context_is_destroyed) { return; }
#endif
	throw runtime_error(result, "Freeing pinned host memory at " + cuda::detail_::ptr_as_hex(host_ptr));
}

/**
 * @copybrief free(void*)
 *
 * @param region The region of memory to free
 */
inline void free(region_t region) {	return free(region.data()); }

namespace detail_ {

struct allocator {
	void* operator()(size_t num_bytes) const { return cuda::memory::host::allocate(num_bytes).data(); }
};
struct deleter {
	void operator()(void* ptr) const { cuda::memory::host::free(ptr); }
};

/**
 * Makes a pre-allocated memory region behave as though it were allocated with @ref host::allocate.
 *
 * Page-locks the memory range specified by ptr and size and maps it for the device(s) as specified by
 * flags. This memory range also is added to the same tracking mechanism as cuMemAllocHost() to
 * automatically accelerate calls to functions such as cuMemcpy().
 *
 * @param ptr A pre-allocated stretch of host memory
 * @param size the size in bytes the memory region to register/pin
 * @param flags
 */
inline void register_(const void *ptr, size_t size, unsigned flags)
{
	auto result = cuMemHostRegister(const_cast<void *>(ptr), size, flags);
	throw_if_error_lazy(result,
		"Could not register and page-lock the region of " + ::std::to_string(size) +
		" bytes of host memory at " + cuda::detail_::ptr_as_hex(ptr) +
		" with flags " + cuda::detail_::as_hex(flags));
}

inline void register_(const_region_t region, unsigned flags)
{
	register_(region.start(), region.size(), flags);
}

} // namespace detail_

/**
 * Whether or not the registration of the host-side pointer should map
 * it into the CUDA address space for access on the device. When true,
 * one can then obtain the device-space pointer using
 * @ref mapped:device_side_pointer_for<T>(T *)
 */
enum mapped_io_space : bool {
	is_mapped_io_space               = true,
	is_not_mapped_io_space           = false
};

/**
 * Whether or not the registration of the host-side pointer should map
 * it into the CUDA address space for access on the device. When true,
 * one can then obtain the device-space pointer using
 * @ref mapped:device_side_pointer_for()
 */
enum map_into_device_memory : bool {
	map_into_device_memory           = true,
	do_not_map_into_device_memory    = false
};

/**
 * Whether the allocated host-side memory should be recognized as pinned memory by
 * all CUDA contexts, not just the (implicit Runtime API) context that performed the
 * allocation.
 */
enum accessibility_on_all_devices : bool {
	is_accessible_on_all_devices     = true,//!< is_accessible_on_all_devices
	is_not_accessible_on_all_devices = false//!< is_not_accessible_on_all_devices
};

/**
 * Register a memory region with the CUDA driver
 *
 * Page-locks the memory range specified by ptr and size and maps it for the device(s) as specified by
 * flags. This memory range also is added to the same tracking mechanism as cuMemAllocHost() to
 * automatically accelerate calls to functions such as cuMemcpy().
 *
 * @TODO Currently works within the current context
 *
 * @note we can't use the name `register`, since that's a reserved word
 *
 * @param ptr The beginning of a pre-allocated region of host memory
 * @param size the size in bytes the memory region to register
 * @param register_mapped_io_space region will be treated as being some memory-mapped
 *     I/O space, e.g. belonging to a third-party PCIe device. See
 *     @ref CU_MEMHOSTREGISTER_IOMEMORY for more details.
 * @param map_into_device_space If true, map the region to a region of addresses
 *     accessible from the (current context's) device; in practice, and with modern
 *     GPUs, this means the region itself will be accessible from the device. See
 *     @ref CU_MEMHOSTREGISTER_DEVICEMAP for more details.
 * @param make_device_side_accessible_to_all Make the region accessible in all
 *     CUDA contexts.
 * @param  considered_read_only_by_device Device-side code will consider this region
 *     (or rather the region it is mapped to and accessible from the device) as
 *     read-only; see @ref CU_MEMHOSTREGISTER_READ_ONLY for more details.
 */
inline void register_(const void *ptr, size_t size,
	bool register_mapped_io_space,
	bool map_into_device_space,
	bool make_device_side_accessible_to_all
#if CUDA_VERSION >= 11010
	, bool considered_read_only_by_device
#endif // CUDA_VERSION >= 11010
	)
{
	detail_::register_(
		ptr, size,
		(register_mapped_io_space ? CU_MEMHOSTREGISTER_IOMEMORY : 0)
		| (map_into_device_space ? CU_MEMHOSTREGISTER_DEVICEMAP : 0)
		| (make_device_side_accessible_to_all ? CU_MEMHOSTREGISTER_PORTABLE : 0)
#if CUDA_VERSION >= 11010
		| (considered_read_only_by_device ? CU_MEMHOSTREGISTER_READ_ONLY : 0)
#endif // CUDA_VERSION >= 11010
	);
}

/**
 * Register a memory region with the CUDA driver
 *
 * Page-locks the memory range specified by ptr and size and maps it for the device(s) as specified by
 * flags. This memory range also is added to the same tracking mechanism as cuMemAllocHost() to
 * automatically accelerate calls to functions such as cuMemcpy().
 *
 * @TODO Currently works within the current context
 *
 * @note we can't use the name `register`, since that's a reserved word
 *
 * @param region The region to register
 * @param register_mapped_io_space region will be treated as being some memory-mapped
 *     I/O space, e.g. belonging to a third-party PCIe device. See
 *     @ref CU_MEMHOSTREGISTER_IOMEMORY for more details.
 * @param map_into_device_space If true, map the region to a region of addresses
 *     accessible from the (current context's) device; in practice, and with modern
 *     GPUs, this means the region itself will be accessible from the device. See
 *     @ref CU_MEMHOSTREGISTER_DEVICEMAP for more details.
 * @param make_device_side_accessible_to_all Make the region accessible in all
 *     CUDA contexts.
 * @param  considered_read_only_by_device Device-side code will consider this region
 *     (or rather the region it is mapped to and accessible from the device) as
 *     read-only; see @ref CU_MEMHOSTREGISTER_READ_ONLY for more details.
 */
inline void register_(
	const_region_t region,
	bool register_mapped_io_space,
	bool map_into_device_space,
	bool make_device_side_accessible_to_all
#if CUDA_VERSION >= 11010
	, bool considered_read_only_by_device
#endif // CUDA_VERSION >= 11010
	)
{
	register_(
		region.start(),
		region.size(),
		register_mapped_io_space,
		map_into_device_space,
		make_device_side_accessible_to_all
#if CUDA_VERSION >= 11010
		, considered_read_only_by_device
#endif // CUDA_VERSION >= 11010
		);
}

/**
 * Register a memory region with the CUDA driver
 *
 * Page-locks the memory range specified by ptr and size and maps it for the device(s) as specified by
 * flags. This memory range also is added to the same tracking mechanism as cuMemAllocHost() to
 * automatically accelerate calls to functions such as cuMemcpy().
 *
 * @TODO Currently works within the current context
 *
 * @note we can't use the name `register`, since that's a reserved word
 *
 * @param ptr The beginning of a pre-allocated region of host memory
 * @param size the size in bytes the memory region to register
 */
inline void register_(void const *ptr, size_t size)
{
	unsigned no_flags_set { 0 };
	detail_::register_(ptr, size, no_flags_set);
}

/**
 * Register a memory region with the CUDA driver
 *
 * Page-locks the memory range specified by ptr and size and maps it for the device(s) as specified by
 * flags. This memory range also is added to the same tracking mechanism as cuMemAllocHost() to
 * automatically accelerate calls to functions such as cuMemcpy().
 *
 * @TODO Currently works within the current context
 *
 * @note we can't use the name `register`, since that's a reserved word
 *
 * @param region The region to register
 */
inline void register_(const_region_t region)
{
	register_(region.start(), region.size());
}

/**
 * Have the CUDA driver "forget" about a region of memory which was previously registered
 * with it, and page-unlock it
 *
 * @note the CUDA API calls this action "unregister", but that's semantically inaccurate. The
 * registration is not undone, rolled back, it's just ended
 */
inline void deregister(const void *ptr)
{
	auto result = cuMemHostUnregister(const_cast<void *>(ptr));
	throw_if_error_lazy(result,
		"Could not unregister the memory segment starting at address *a");
}

/// @copydoc deregister(const void *)
inline void deregister(const_region_t region)
{
	deregister(region.start());
}

/**
 * Sets all bytes in a stretch of host-side memory to a single value
 *
 * @note a wrapper for @ref ::std::memset
 *
 * @param start starting address of the memory region to set,
 * in host memory; can be either CUDA-allocated or otherwise.
 * @param byte_value value to set the memory region to
 * @param num_bytes size of the memory region in bytes
 */
inline void set(void* start, int byte_value, size_t num_bytes)
{
	::std::memset(start, byte_value, num_bytes);
	// TODO: Error handling?
}

/**
 * Zero-out a region of host memory
 *
 * @param ptr the beginning of a region of host memory to zero-out
 * @param num_bytes the size in bytes of the region of memory to zero-out
 */
inline void zero(void* start, size_t num_bytes)
{
	set(start, 0, num_bytes);
}

/**
 * Zero-out a region of host memory
 *
 * @param region the region of host-side memory to zero-out
 */
inline void zero(region_t region)
{
	set(region, 0);
}

/**
 * Asynchronously sets all bytes of a single pointed-to value
 * to 0 (zero).
 *
 * @param ptr a pointer to the value to be to zero, in host memory
 */
template <typename T>
inline void zero(T* ptr)
{
	zero(ptr, sizeof(T));
}


} // namespace host

namespace managed {

namespace range {

namespace detail_ {

using attribute_t = CUmem_range_attribute;
using advice_t = CUmem_advise;

template <typename T>
inline T get_scalar_attribute(const_region_t region, attribute_t attribute)
{
	uint32_t attribute_value { 0 };
	auto result = cuMemRangeGetAttribute(
		&attribute_value, sizeof(attribute_value), attribute, device::address(region.start()), region.size());
	throw_if_error_lazy(result,
		"Obtaining an attribute for a managed memory range at " + cuda::detail_::ptr_as_hex(region.start()));
	return static_cast<T>(attribute_value);
}

// CUDA's range "advice" is simply a way to set the attributes of a range; unfortunately that's
// not called cuMemRangeSetAttribute, and uses a different enum.
inline void advise(const_region_t region, advice_t advice, cuda::device::id_t device_id)
{
	auto result = cuMemAdvise(device::address(region.start()), region.size(), advice, device_id);
	throw_if_error_lazy(result, "Setting an attribute for a managed memory range at "
		+ cuda::detail_::ptr_as_hex(region.start()));
}

inline advice_t as_advice(attribute_t attribute, bool set)
{
	switch (attribute) {
	case CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY:
		return set ? CU_MEM_ADVISE_SET_READ_MOSTLY : CU_MEM_ADVISE_UNSET_READ_MOSTLY;
	case CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION:
		return set ? CU_MEM_ADVISE_SET_PREFERRED_LOCATION : CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION;
	case CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY:
		return set ? CU_MEM_ADVISE_SET_ACCESSED_BY : CU_MEM_ADVISE_UNSET_ACCESSED_BY;
	default:
		throw ::std::invalid_argument(
			"CUDA memory range attribute does not correspond to any range advice value");
	}
}

inline void set_attribute(const_region_t region, attribute_t settable_attribute, cuda::device::id_t device_id)
{
	static constexpr const bool set { true };
	advise(region, as_advice(settable_attribute, set), device_id);
}

inline void set_attribute(const_region_t region, attribute_t settable_attribute)
{
	static constexpr const bool set { true };
	static constexpr const cuda::device::id_t dummy_device_id { 0 };
	advise(region, as_advice(settable_attribute, set), dummy_device_id);
}

inline void unset_attribute(const_region_t region, attribute_t settable_attribute)
{
	static constexpr const bool unset { false };
	static constexpr const cuda::device::id_t dummy_device_id { 0 };
	advise(region, as_advice(settable_attribute, unset), dummy_device_id);
}

} // namespace detail_

} // namespace range

namespace detail_ {

template <typename GenericRegion>
struct region_helper : public GenericRegion {
	using GenericRegion::GenericRegion;

	bool is_read_mostly() const
	{
		return range::detail_::get_scalar_attribute<bool>(*this, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY);
	}

	void designate_read_mostly() const
	{
		range::detail_::set_attribute(*this, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY);
	}

	void undesignate_read_mostly() const
	{
		range::detail_::unset_attribute(*this, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY);
	}

	device_t preferred_location() const;
	void set_preferred_location(device_t& device) const;
	void clear_preferred_location() const;
};

} // namespace detail_

/// A child class of the generic @ref region_t with some managed-memory-specific functionality
using region_t = detail_::region_helper<memory::region_t>;
/// A child class of the generic @ref const_region_t with some managed-memory-specific functionality
using const_region_t = detail_::region_helper<memory::const_region_t>;

/// Advice the CUDA driver that @p device is expected to access @p region
void advise_expected_access_by(const_region_t region, device_t& device);

/// Advice the CUDA driver that @p device is not expected to access @p region
void advise_no_access_expected_by(const_region_t region, device_t& device);

/// @return the devices which are marked by attribute as being the accessors of a specified memory region
template <typename Allocator = ::std::allocator<cuda::device_t> >
typename ::std::vector<device_t, Allocator> expected_accessors(const_region_t region, const Allocator& allocator = Allocator() );

/// Kinds of managed memory region attachments
enum class attachment_t : unsigned {
	global        = CU_MEM_ATTACH_GLOBAL,
	host          = CU_MEM_ATTACH_HOST,
	single_stream = CU_MEM_ATTACH_SINGLE,
	};

namespace detail_ {

inline managed::region_t allocate_in_current_context(
	size_t                num_bytes,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices)
{
	device::address_t allocated = 0;
	auto flags = (initial_visibility == initial_visibility_t::to_all_devices) ?
		attachment_t::global : attachment_t::host;
	// This is necessary because managed allocation requires at least one (primary)
	// context to have been constructed. We could theoretically check what our current
	// context is etc., but that would be brittle, since someone can managed-allocate,
	// then change contexts, then de-allocate, and we can't be certain that whoever
	// called us will call free
	cuda::device::primary_context::detail_::increase_refcount(cuda::device::default_device_id);

	// Note: Despite the templating by T, the size is still in bytes,
	// not in number of T's
	auto status = cuMemAllocManaged(&allocated, num_bytes, static_cast<unsigned>(flags));
	if (is_success(status) && allocated == 0) {
		// Can this even happen? hopefully not
		status = static_cast<status_t>(status::unknown);
	}
	throw_if_error_lazy(status, "Failed allocating "
		+ ::std::to_string(num_bytes) + " bytes of managed CUDA memory");
	return {as_pointer(allocated), num_bytes};
}

/**
 * Free a region of managed memory which was allocated with @ref allocate_in_current_context.
 *
 * @note You can't just use @ref cuMemFree - or you'll leak a primary context reference unit.
 */
inline void free(void* ptr)
{
	auto result = cuMemFree(device::address(ptr));
	cuda::device::primary_context::detail_::decrease_refcount(cuda::device::default_device_id);
	throw_if_error_lazy(result, "Freeing managed memory at " + cuda::detail_::ptr_as_hex(ptr));
}

/// @copydoc free(void*)
inline void free(managed::region_t region)
{
	free(region.start());
}

template <initial_visibility_t InitialVisibility = initial_visibility_t::to_all_devices>
struct allocator {
	// Allocates in the current context!
	void* operator()(size_t num_bytes) const
	{
		return detail_::allocate_in_current_context(num_bytes, InitialVisibility).start();
	}
};

struct deleter {
	void operator()(void* ptr) const { detail_::free(ptr); }
};

inline managed::region_t allocate(
	context::handle_t     context_handle,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	return allocate_in_current_context(num_bytes, initial_visibility);
}

} // namespace detail_

/**
 * Allocate a a region of managed memory, accessible with the same
 * address on the host and on CUDA devices.
 *
 * @param context the initial context which is likely to access the managed
 * memory region (and which will certainly have the region actually allocated
 * for it)
 * @param num_bytes size of each of the regions of memory to allocate
 * @param initial_visibility will the allocated region be visible, using the
 * common address, to all CUDA device (= more overhead, more work for the CUDA
 * runtime) or just to those devices with some hardware features to assist in
 * this task (= less overhead)?
 */
inline region_t allocate(
	const context_t&      context,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices);

/**
 * Allocate a a region of managed memory, accessible with the same
 * address on the host and on CUDA devices
 *
 * @param device the initial device which is likely to access the managed
 * memory region (and which will certainly have the region actually allocated
 * for it)
 * @param num_bytes size of each of the regions of memory to allocate
 * @param initial_visibility will the allocated region be visible, using the
 * common address, to all CUDA device (= more overhead, more work for the CUDA
 * runtime) or just to those devices with some hardware features to assist in
 * this task (= less overhead)?
 */
inline region_t allocate(
	const device_t&       device,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices);

/**
 * Allocate a a region of managed memory, accessible with the same
 * address on the host and on all CUDA devices.
 *
 * @note While the allocated memory should be available universally, the
 * allocation itself does require some GPU context. This will be the current
 * context, if one exists, or the primary context on the runtime-defined current
 * device.
 */
region_t allocate(size_t num_bytes);

/**
 * Free a managed memory region (host-side and device-side regions on all devices
 * where it was allocated, all with the same address) which was allocated with
 * @ref allocate.
 */
inline void free(void* managed_ptr)
{
	auto result = cuMemFree(device::address(managed_ptr));
	throw_if_error_lazy(result,
		"Freeing managed memory (host and device regions) at address "
		+ cuda::detail_::ptr_as_hex(managed_ptr));
}

/// @copydoc free(void*)
inline void free(region_t region)
{
	free(region.start());
}

namespace async {

namespace detail_ {

inline void prefetch(
	const_region_t      region,
	cuda::device::id_t  destination,
	stream::handle_t    source_stream_handle)
{
	auto result = cuMemPrefetchAsync(device::address(region.start()), region.size(), destination, source_stream_handle);
	throw_if_error_lazy(result,
		"Prefetching " + ::std::to_string(region.size()) + " bytes of managed memory at address "
		 + cuda::detail_::ptr_as_hex(region.start()) + " to " + (
		 	(destination == CU_DEVICE_CPU) ? "the host" : cuda::device::detail_::identify(destination))  );
}

} // namespace detail_

/**
 * Prefetches a region of managed memory to a specific device, so
 * it can later be used there without waiting for I/O from the host or other
 * devices.
 */
void prefetch(
	const_region_t         region,
	const cuda::device_t&  destination,
	const stream_t&        stream);

/**
 * Prefetches a region of managed memory into host memory. It can
 * later be used there without waiting for I/O from any of the CUDA devices.
 */
void prefetch_to_host(
	const_region_t   region,
	const stream_t&  stream);

} // namespace async

} // namespace managed

namespace mapped {

/**
 * Obtain a pointer in the device-side memory space (= address range) given
 * given a host-side pointer mapped to it.
 *
 * @param[in] host_memory_ptr a pointer to host-side memory which has been allocated
 *    as one side of a CUDA mapped memory region pair
 */
template <typename T>
inline T* device_side_pointer_for(T* host_memory_ptr)
{
	auto unconsted_host_mem_ptr = const_cast<typename ::std::remove_const<T>::type *>(host_memory_ptr);
	device::address_t device_side_ptr;
	auto get_device_pointer_flags = 0u; // see the CUDA runtime documentation
	auto status = cuMemHostGetDevicePointer(
		&device_side_ptr,
		unconsted_host_mem_ptr,
		get_device_pointer_flags);
	throw_if_error_lazy(status,
		"Failed obtaining the device-side pointer for host-memory pointer "
		+ cuda::detail_::ptr_as_hex(host_memory_ptr) + " supposedly mapped to device memory");
	return as_pointer(device_side_ptr);
}

/**
 * Get the memory region mapped to a given host-side region
 *
 * @note if the input is already a device-side region, this function is idempotent
 */
inline region_t device_side_region_for(region_t region)
{
	return { device_side_pointer_for(region.start()), region.size() };
}

/// @copydoc device_side_region_for(region_t)
inline const_region_t device_side_region_for(const_region_t region)
{
	return { device_side_pointer_for(region.start()), region.size() };
}

namespace detail_ {

/**
 * Allocates a mapped pair of memory regions - in the current
 * context and in host and device memory.
 *
 * @param size_in_bytes size of each of the two regions, in bytes.
 * @param options indication of how the CUDA driver will manage
 * the region pair
 * @return the allocated pair (with both regions being non-null)
 */
inline region_pair_t allocate_in_current_context(
	context::handle_t   current_context_handle,
	size_t              size_in_bytes,
	allocation_options  options)
{
	region_pair_t allocated {};
	// The default initialization is unnecessary, but let's play it safe
	auto flags = cuda::memory::detail_::make_cuda_host_alloc_flags(options);
	void* allocated_ptr;
	auto status = cuMemHostAlloc(&allocated_ptr, size_in_bytes, flags);
	if (is_success(status) && (allocated_ptr == nullptr)) {
		// Can this even happen? hopefully not
		status = static_cast<status_t>(status::named_t::unknown);
	}
	throw_if_error_lazy(status,
		"Failed allocating a mapped pair of memory regions of size " + ::std::to_string(size_in_bytes)
		+ " bytes of global memory in " + context::detail_::identify(current_context_handle));
	allocated.host_side = { allocated_ptr, size_in_bytes };
	allocated.device_side = device_side_region_for(allocated.host_side);
	return allocated;
}

inline region_pair_t allocate(
	context::handle_t   context_handle,
	size_t              size_in_bytes,
	allocation_options  options)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	return detail_::allocate_in_current_context(context_handle, size_in_bytes, options);
}

inline void free(void* host_side_pair)
{
	auto result = cuMemFreeHost(host_side_pair);
	throw_if_error_lazy(result, "Freeing a mapped memory region pair with host-side address "
		+ cuda::detail_::ptr_as_hex(host_side_pair));
}

} // namespace detail_

/**
 * Allocate a memory region on the host, which is also mapped to a memory region in
 * a context of some CUDA device - so that changes to one will be reflected in the other.
 *
 * @param context The device context in which the device-side region in the pair will be
 *     allocated.
 * @param size_in_bytes amount of memory to allocate (in each of the regions)
 * @param options see @ref memory::allocation_options
 */
region_pair_t allocate(
	cuda::context_t&    context,
	size_t              size_in_bytes,
	allocation_options  options);

/**
 * Allocate a memory region on the host, which is also mapped to a memory region in
 * the global memory of a CUDA device - so that changes to one will be reflected in the other.
 *
 * @param device The device on which the device-side region in the pair will be allocated
 * @param size_in_bytes amount of memory to allocate (in each of the regions)
 * @param options see @ref memory::allocation_options
 */
region_pair_t allocate(
	cuda::device_t&     device,
	size_t              size_in_bytes,
	allocation_options  options = allocation_options{});


/**
 * Free a pair of mapped memory regions
 *
 * @param pair a pair of mapped host- and device-side regions
 */
inline void free(region_pair_t pair)
{
	detail_::free(pair.host_side.data());
}

/**
 * Free a pair of mapped memory regions using just one of them
 *
 * @param ptr a pointer to one of the mapped regions (can be either
 * the device-side or the host-side)
 */
inline void free_region_pair_of(void* ptr)
{
	// TODO: What if the pointer is not part of a mapped region pair?
	// We could check this...
	void* host_side_ptr;
	auto status = cuPointerGetAttribute (&host_side_ptr, CU_POINTER_ATTRIBUTE_HOST_POINTER, memory::device::address(ptr));
	throw_if_error_lazy(status, "Failed obtaining the host-side address of supposedly-device-side pointer "
		+ cuda::detail_::ptr_as_hex(ptr));
	detail_::free(host_side_ptr);
}

/**
 * Determine whether a given stretch of memory was allocated as part of
 * a mapped pair of host and device memory regions
 *
 * @todo What if it's a managed pointer?
 *
 * @param ptr the beginning of a memory region - in either host or device
 * memory - to check
 * @return `true` iff the region was allocated as one side of a mapped
 * memory region pair
 */
inline bool is_part_of_a_region_pair(const void* ptr)
{
	auto wrapped_ptr = pointer_t<const void> { ptr };
	return wrapped_ptr.other_side_of_region_pair().get() != nullptr;
}

} // namespace mapped

namespace detail_ {
/**
 * Create a unique_span without default construction, using raw-memory allocator
 * and deleter gadgets.
 *
 * @note We allow this only for "convenient" types; see @ref detail_check_allocation_type
 *
 * @tparam T Element of the created unique_span
 * @tparam UntypedAllocator can allocate untyped memory given a size
 * @tparam UntypedDeleter can delete memory given a pointer (disregarding the type)
 *
 * @param size number of elements in the unique_span to be created
 * @param raw_allocator a gadget for allocating untyped memory
 * @param raw_deleter a gadget which can de-allocate/delete allocations by @p raw_allocator
 * @return the newly-created unique_span
 */
template <typename T, typename RawDeleter, typename RegionAllocator>
unique_span<T> make_convenient_type_unique_span(size_t size, RegionAllocator allocator)
{
	memory::detail_::check_allocation_type<T>();
	auto deleter = [](span<T> sp) {
		return RawDeleter{}(sp.data());
	};
	region_t allocated_region = allocator(size * sizeof(T));
	return unique_span<T>(
		allocated_region.as_span<T>(), // no constructor calls - trivial construction
		deleter // no destructor calls - trivial destruction
	);
}

} // namespace detail_


namespace device {

namespace detail_ {

template <typename T>
unique_span<T> make_unique_span(const context::handle_t context_handle, size_t size)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	return memory::detail_::make_convenient_type_unique_span<T, detail_::deleter>(size, allocate_in_current_context);
}

} // namespace detail_

/**
 * Allocate memory for a consecutive sequence of typed elements in device-global memory.
 *
 * @tparam T  type of the individual elements in the allocated sequence
 *
 * @param context The CUDA device context in which to make the allocation.
 * @param size the number of elements to allocate
 * @return A @ref unique_span which owns the allocated memory (and will release said
 *
 * @note This function is somewhat similar to ::std:: make_unique_for_overwrite(), except
 * that the returned value is not "just" a unique pointer, but also has a size. It is also
 * similar to {@ref cuda::device::make_unique_region}, except that the allocation is
 * conceived as typed elements.
 *
 * @note Typically, this is used for trivially-constructible elements, for which reason the
 * non-construction of individual elements should not pose a problem. But - let the user beware.
 */
template <typename T>
unique_span<T> make_unique_span(const context_t& context, size_t size);

/**
 * @copydoc make_unique_span(const context_t&, size_t)
 *
 * @param device The CUDA device in whose primary context to make the allocation.
 */
template <typename T>
unique_span<T> make_unique_span(const device_t& device, size_t size);

/**
 * @copydoc make_unique_span(const context_t&, size_t)
 *
 * @note The current device's primary context will be used (_not_ the
 * current context).
 */
template <typename T>
unique_span<T> make_unique_span(size_t size);

} // namespace device

/// See @ref `device::make_unique_span(const context_t& context, size_t size)`
template <typename T>
inline unique_span<T> make_unique_span(const context_t& context, size_t size)
{
	return device::make_unique_span<T>(context, size);
}

/// See @ref `device::make_unique_span(const context_t& context, size_t num_elements)`
template <typename T>
inline unique_span<T> make_unique_span(const device_t& device, size_t size)
{
	return device::make_unique_span<T>(device, size);
}

namespace host {

/**
 * Allocate memory for a consecutive sequence of typed elements in system
 * (host-side) memory.
 *
 * @tparam T  type of the individual elements in the allocated sequence
 *
 * @param size the number of elements to allocate
 * @return A @ref unique_span which owns the allocated memory (and will release said
 * memory upon destruction)
 *
 * @note This function is somewhat similar to ::std:: make_unique_for_overwrite(), except
 * that the returned value is not "just" a unique pointer, but also has a size. It is also
 * similar to {@ref cuda::device::make_unique_region}, except that the allocation is
 * conceived as typed elements.
 *
 * @note We assume this memory is used for copying to or from device-side memory; hence,
 * we constrain the type to be trivially constructible, destructible and copyable
 *
 * @note ignoring alignment
 */
template <typename T>
unique_span<T> make_unique_span(size_t size)
{
	// Need this because of allocate takes more arguments and has default ones
	auto allocator = [](size_t size) { return allocate(size); };
	return memory::detail_::make_convenient_type_unique_span<T, detail_::deleter>(size, allocator);
}

} // namespace host

namespace managed {

namespace detail_ {

template <typename T, initial_visibility_t InitialVisibility = initial_visibility_t::to_all_devices>
unique_span<T> make_unique_span(
	const context::handle_t  context_handle,
	size_t                   size)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	auto allocator = [](size_t size) {
		return allocate_in_current_context(size, InitialVisibility);
	};
	return memory::detail_::make_convenient_type_unique_span<T, detail_::deleter>(size, allocator);
}

} // namespace detail_

/**
 * Allocate memory for a consecutive sequence of typed elements in system
 * (host-side) memory.
 *
 * @tparam T  type of the individual elements in the allocated sequence
 *
 * @param context The CUDA device context in which to register the allocation
 * @param size the number of elements to allocate
 * @param initial_visibility Choices of which category of CUDA devices must the managed
 *     region be guaranteed to be visible to
 * @return A @ref unique_span which owns the allocated memory (and will release said
 * memory upon destruction)
 *
 * @note This function is somewhat similar to ::std:: make_unique_for_overwrite(), except
 * that the returned value is not "just" a unique pointer, but also has a size. It is also
 * similar to {@ref cuda::device::make_unique_region}, except that the allocation is
 * conceived as typed elements.
 *
 * @note Typically, this is used for trivially-constructible elements, for which reason the
 * non-construction of individual elements should not pose a problem. But - let the user
 * beware, especially since this is accessible in host-side code.
 */
template <typename T>
unique_span<T> make_unique_span(
	const context_t&      context,
	size_t                size,
    initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices);

/**
 * @copydoc make_unique_span(const context_t&, size_t)
 *
 * @param device The CUDA device in whose primary context to make the allocation.
 */
template <typename T>
unique_span<T> make_unique_span(
	const device_t&       device,
	size_t                size,
    initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices);

/**
 * @copydoc make_unique_span(const context_t&, size_t)
 *
 * @note The current device's primary context will be used (_not_ the
 * current context).
 */
template <typename T>
unique_span<T> make_unique_span(
    size_t size,
    initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices);

} // namespace managed

} // namespace memory

namespace symbol {

/**
 * Locates a CUDA symbol in global or constant device memory
 *
 * @note `symbol_t` symbols are associated with the primary context
 *
 * @return The region of memory CUDA associates with the symbol
 */
template <typename T>
memory::region_t locate(T&& symbol)
{
	void *start;
	size_t symbol_size;
	auto api_call_result = cudaGetSymbolAddress(&start, ::std::forward<T>(symbol));
	throw_if_error_lazy(api_call_result, "Could not locate the device memory address for a symbol");
	api_call_result = cudaGetSymbolSize(&symbol_size, ::std::forward<T>(symbol));
	throw_if_error_lazy(api_call_result, "Could not locate the device memory address for the symbol at address"
		+ cuda::detail_::ptr_as_hex(start));
	return { start, symbol_size };
}

} // namespace symbol

} // namespace cuda

#endif // CUDA_API_WRAPPERS_MEMORY_HPP_
