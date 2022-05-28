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

#include <cuda/api/array.hpp>
#include <cuda/api/constants.hpp>
#include <cuda/api/current_device.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/pointer.hpp>
#include <cuda/api/current_context.hpp>

#include <cuda_runtime.h> // needed, rather than cuda_runtime_api.h, e.g. for cudaMalloc
#include <cuda.h>

#include <memory>
#include <cstring> // for ::std::memset
#include <vector>

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
	is_portable   = true,//!< is_portable
	isnt_portable = false//!< isnt_portable
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
	with_wc    = true,
	without_wc = false
};

/**
 * @brief options accepted by CUDA's allocator of memory with a host-side aspect
 * (host-only or managed memory).
 */
struct allocation_options {
	portability_across_contexts  portability;
	cpu_write_combining          write_combining;
};

namespace detail_ {

inline unsigned make_cuda_host_alloc_flags(allocation_options options)
{
	return
		(options.portability     == portability_across_contexts::is_portable ? CU_MEMHOSTALLOC_PORTABLE      : 0) &
		(options.write_combining == cpu_write_combining::with_wc             ? CU_MEMHOSTALLOC_WRITECOMBINED : 0);
}

} // namespace detail_

/**
 * @namespace mapped
 * Memory regions appearing in both on the host-side and device-side address
 * spaces with the regions in both spaces mapped to each other (i.e. guaranteed
 * to have the same contents on access up to synchronization details). Consult the
 * <a href="http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mapped-memory">
 * CUDA C programming guide section on mapped memory</a> for more details.
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
	void* host_side;
	void* device_side;
	size_t size_in_bytes; /// the allocated number of bytes, common to both sides
};

} // namespace mapped

/**
 * @brief CUDA-Device-global memory on a single device (not accessible from the host)
 */
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
		status = (status_t) status::unknown;
	}
	throw_if_error(status, "Failed allocating " + ::std::to_string(num_bytes) +
		" bytes of global memory on the current CUDA device");
	return {as_pointer(allocated), num_bytes};
}

inline region_t allocate(context::handle_t context_handle, size_t size_in_bytes)
{
	context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle);
	return allocate_in_current_context(size_in_bytes);
}

} // namespace detail_

namespace async {

namespace detail_ {

/**
 * Allocate memory asynchronously on a specified stream.
 */
inline region_t allocate(
	context::handle_t  context_handle,
	stream::handle_t   stream_handle,
	size_t             num_bytes)
{
#if CUDA_VERSION >= 11020
	device::address_t allocated = 0;
	// Note: the typed cudaMalloc also takes its size in bytes, apparently,
	// not in number of elements
	auto status = cuMemAllocAsync(&allocated, num_bytes, stream_handle);
	if (is_success(status) && allocated == 0) {
		// Can this even happen? hopefully not
		status = static_cast<decltype(status)>(status::unknown);
	}
	throw_if_error(status,
		"Failed scheduling an asynchronous allocation of " + ::std::to_string(num_bytes) +
		" bytes of global memory on " + stream::detail_::identify(stream_handle, context_handle) );
	return {as_pointer(allocated), num_bytes};
#else
	(void) context_handle;
	(void) stream_handle;
	(void) num_bytes;
	throw cuda::runtime_error(cuda::status::not_yet_implemented, "Asynchronous memory allocation is not supported with CUDA versions below 11.2");
#endif
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
inline region_t allocate(const stream_t& stream, size_t size_in_bytes);

} // namespace async


/**
 * Free a region of device-side memory (regardless of how it was allocated)
 */
///@{
inline void free(void* ptr)
{
	auto result = cuMemFree(address(ptr));
#if CAW_THROW_ON_FREE_IN_DESTROYED_CONTEXT
	if (result == status::success) { return; }
#else
	if (result == status::success or result == status::context_is_destroyed) { return; }
#endif
	throw runtime_error(result, "Freeing device memory at " + cuda::detail_::ptr_as_hex(ptr));
}
inline void free(region_t region) { free(region.start()); }
///@}

/**
 * Allocate device-side memory on a CUDA device context.
 *
 * @note The CUDA memory allocator guarantees alignment "suitabl[e] for any kind of variable"
 * (CUDA 9.0 Runtime API documentation), and the CUDA programming guide guarantees
 * since at least version 5.0 that the minimum allocation is 256 bytes.
 *
 * @throws cuda::runtime_error if allocation fails for any reason
 *
 * @param device the context in which to allocate memory
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
 * @brief Sets consecutive elements of a region of memory to a fixed
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
inline void typed_set(T* start, const T& value, size_t num_elements);

/**
 * @brief Sets all bytes in a region of memory to a fixed value
 *
 * @note The equivalent of @ref ::std::memset for CUDA device-side memory
 *
 * @param byte_value value to set the memory region to
 */
///@{
/**
 * @param start starting address of the memory region to set, in a CUDA
 * device's global memory
 * @param num_bytes size of the memory region in bytes
 */
inline void set(void* start, int byte_value, size_t num_bytes)
{
	return typed_set<unsigned char>(static_cast<unsigned char*>(start), (unsigned char) byte_value, num_bytes);
}

/**
 * @param region a region to zero-out, in a CUDA device's global memory
 */
inline void set(region_t region, int byte_value)
{
	set(region.start(), byte_value, region.size());
}
///@}

/**
 * @brief Sets all bytes in a region of memory to 0 (zero)
 */
///@{
/**
 * @param region a region to zero-out, in a CUDA device's global memory
 */
inline void zero(void* start, size_t num_bytes)
{
	set(start, 0, num_bytes);
}

/**
 * @param start starting address of the memory region to zero-out,
 * in a CUDA device's global memory
 * @param num_bytes size of the memory region in bytes
 */
inline void zero(region_t region)
{
	zero(region.start(), region.size());
}
///@}

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
 * @note the sources and destinations may all be in any memory space addressable
 * in the the unified virtual address space, which could be host-side memory,
 * device global memory, device constant memory etc.
 *
 * @param destination A pointer to a memory region of size @p num_bytes.
 */
///@{
/**
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
 * @param destination A region of memory to which to copy the data in@source, of
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
	if (destination.size() < N) {
		throw ::std::logic_error("Source size exceeds destination size");
	}
#endif
	return copy(destination.start(), source, sizeof(T) * N);
}

/**
 * @param destination A region of memory to which to copy the data in @p source,
 *     of size at least that of @p source.
 * @param source A region of at least `sizeof(T)*N` bytes with whose data to fill @destination
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

template <typename T, size_t N>
inline void copy(void* destination, T (&source)[N])
{
	return copy(destination, source, sizeof(T) * N);
}

/**
 * @param destination A region of memory to which to copy the data in @p source,
 *     of size at least that of @p source.
 * @param source The starting address of @tparam N elements to copy
 */
template <typename T, size_t N>
inline void copy(T(&destination)[N], T* source)
{
	return copy(destination, source, sizeof(T) * N);
}

/**
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

inline void copy(region_t destination, void* source)
{
	return copy(destination, source, destination.size());
}
///@}



/**
 * @brief Sets a number of bytes in memory to a fixed value
 *
 * @note The equivalent of @ref ::std::memset - for any and all CUDA-related
 * memory spaces
 *
 * @param ptr Address of the first byte in memory to set. May be in host-side
 *     memory, global CUDA-device-side memory or CUDA-managed memory.
 * @param byte_value value to set the memory region to
 * @param num_bytes The amount of memory to set to @p byte_value
 */
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


/**
 * @brief Sets all bytes in a region of memory to a fixed value
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
 * @brief Sets all bytes in a region of memory to 0 (zero)
 *
 * @param region the memory region to zero-out; may be in host-side memory,
 * global CUDA-device-side memory or CUDA-managed memory.
 */
inline void zero(region_t region)
{
	return set(region, 0);
}

/**
 * @brief Sets a number of bytes starting in at a given address of memory to 0 (zero)
 *
 * @param region the memory region to zero-out; may be in host-side memory,
 * global CUDA-device-side memory or CUDA-managed memory.
 */
inline void zero(void* ptr, size_t num_bytes)
{
	return set(ptr, 0, num_bytes);
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

namespace detail_ {

template<dimensionality_t NumDimensions>
struct base_copy_params;

template<>
struct base_copy_params<2> {
	using intra_context_type = CUDA_MEMCPY2D;
	using type = intra_context_type; // Why is there no inter-context type, CUDA_MEMCPY2D_PEER ?
};

template<>
struct base_copy_params<3> {
	using type = CUDA_MEMCPY3D_PEER;
	using intra_context_type = CUDA_MEMCPY3D;
};

// Note these, by default, support inter-context
template<dimensionality_t NumDimensions>
using base_copy_params_t = typename base_copy_params<NumDimensions>::type;


enum class endpoint_t {
	source, destination
};

template<dimensionality_t NumDimensions>
struct copy_parameters_t : base_copy_params_t<NumDimensions> {
	// TODO: Perhaps use proxies?

	using intra_context_type = typename base_copy_params<NumDimensions>::intra_context_type;

	using dimensions_type = array::dimensions_t<NumDimensions>;

	template<typename T>
	void set_endpoint(endpoint_t endpoint, const cuda::array_t<T, NumDimensions> &array);

	template<typename T>
	void set_endpoint(endpoint_t endpoint, T *ptr, array::dimensions_t<NumDimensions> dimensions);

	template<typename T>
	void set_endpoint(endpoint_t endpoint, context::handle_t context_handle, T *ptr,
		array::dimensions_t<NumDimensions> dimensions);

	// TODO: Perhaps we should have an dimensioned offset type?
	template<typename T>
	void set_offset(endpoint_t endpoint, dimensions_type offset);

	template<typename T>
	void clear_offset(endpoint_t endpoint)
	{ set_offset<T>(endpoint, dimensions_type::zero()); }

	template<typename T>
	void set_extent(dimensions_type extent);
	// Sets how much is being copies, as opposed to the sizes of the endpoints which may be larger

	void clear_rest();
	// Clear any dummy fields which are required to be set to 0. Note that important fields,
	// which you have not set explicitly, will _not_ be cleared by this method.

};

template<>
template<typename T>
void copy_parameters_t<2>::set_endpoint(endpoint_t endpoint, const cuda::array_t<T, 2> &array)
{
	(endpoint == endpoint_t::source ? srcMemoryType : dstMemoryType) = CU_MEMORYTYPE_ARRAY;
	(endpoint == endpoint_t::source ? srcArray : dstArray) = array.get();
	// Can't set the endpoint context - the basic data structure doesn't support that!
}

template<>
template<typename T>
void copy_parameters_t<3>::set_endpoint(endpoint_t endpoint, const cuda::array_t<T, 3> &array)
{
	(endpoint == endpoint_t::source ? srcMemoryType : dstMemoryType) = CU_MEMORYTYPE_ARRAY;
	(endpoint == endpoint_t::source ? srcArray : dstArray) = array.get();
	(endpoint == endpoint_t::source ? srcContext : dstContext) = array.context_handle();
}

template<>
template<typename T>
inline void copy_parameters_t<2>::set_endpoint(endpoint_t endpoint, context::handle_t context_handle, T *ptr,
	array::dimensions_t<2> dimensions)
{
	if (context_handle != context::detail_::none) {
		throw cuda::runtime_error(
			cuda::status::named_t::not_supported,
			"Inter-context copying of 2D arrays is not supported by the CUDA driver");
	}
	set_endpoint<2>(endpoint, ptr, dimensions);
}

template<>
template<typename T>
inline void copy_parameters_t<2>::set_endpoint(endpoint_t endpoint, T *ptr, array::dimensions_t<2> dimensions)
{
	auto memory_type = memory::type_of(ptr);
	if (memory_type == memory::type_t::unified_ or memory_type == type_t::device_) {
		(endpoint == endpoint_t::source ? srcDevice : dstDevice) = device::address(ptr);
	} else {
		if (endpoint == endpoint_t::source) { srcHost = ptr; }
		else { dstHost = ptr; }
	}
	(endpoint == endpoint_t::source ? srcPitch : dstPitch) = dimensions.width * sizeof(T);
	(endpoint == endpoint_t::source ? srcMemoryType : dstMemoryType) = (CUmemorytype) memory_type;
	// Can't set the endpoint context - the basic data structure doesn't support that!
}

template<>
template<typename T>
inline void copy_parameters_t<3>::set_endpoint(endpoint_t endpoint, context::handle_t context_handle, T *ptr,
	array::dimensions_t<3> dimensions)
{
	cuda::memory::pointer_t<void> wrapped{ptr};
	auto memory_type = memory::type_of(ptr);
	if (memory_type == memory::type_t::unified_ or memory_type == type_t::device_) {
		(endpoint == endpoint_t::source ? srcDevice : dstDevice) = device::address(ptr);
	} else {
		if (endpoint == endpoint_t::source) { srcHost = ptr; }
		else { dstHost = ptr; }
	}
	(endpoint == endpoint_t::source ? srcPitch : dstPitch) = dimensions.width * sizeof(T);
	(endpoint == endpoint_t::source ? srcHeight : dstHeight) = dimensions.height;
	(endpoint == endpoint_t::source ? srcMemoryType : dstMemoryType) = (CUmemorytype) memory_type;
	(endpoint == endpoint_t::source ? srcContext : dstContext) = context_handle;
}

template<>
template<typename T>
inline void copy_parameters_t<3>::set_endpoint(endpoint_t endpoint, T *ptr, array::dimensions_t<3> dimensions)
{
	set_endpoint<T>(endpoint, context::detail_::none, ptr, dimensions);
}

template<>
inline void copy_parameters_t<2>::clear_rest()
{}

template<>
inline void copy_parameters_t<3>::clear_rest()
{
	srcLOD = 0;
	dstLOD = 0;
}

template<>
template<typename T>
inline void copy_parameters_t<2>::set_extent(dimensions_type extent)
{
	WidthInBytes = extent.width * sizeof(T);
	Height = extent.height;
}

template<>
template<typename T>
void copy_parameters_t<3>::set_extent(dimensions_type extent)
{
	WidthInBytes = extent.width * sizeof(T);
	Height = extent.height;
	Depth = extent.depth;
}

template<>
template<typename T>
void copy_parameters_t<3>::set_offset(endpoint_t endpoint, dimensions_type offset)
{
	(endpoint == endpoint_t::source ? srcXInBytes : dstXInBytes) = offset.width * sizeof(T);
	(endpoint == endpoint_t::source ? srcY : dstY) = offset.height;
	(endpoint == endpoint_t::source ? srcZ : dstZ) = offset.depth;
}

template<>
template<typename T>
void copy_parameters_t<2>::set_offset(endpoint_t endpoint, dimensions_type offset)
{
	(endpoint == endpoint_t::source ? srcXInBytes : dstXInBytes) = offset.width * sizeof(T);
	(endpoint == endpoint_t::source ? srcY : dstY) = offset.height;
}

void set_endpoint(endpoint_t endpoint, void *src);

inline status_t multidim_copy(::std::integral_constant<dimensionality_t, 2>, copy_parameters_t<2> params)
{
	// Note this _must_ be an intra-context copy, as inter-context is not supported
	// and there's no indication of context in the relevant data structures
	return cuMemcpy2D(&params);
}

inline status_t multidim_copy(::std::integral_constant<dimensionality_t, 3>, copy_parameters_t<3> params)
{
	if (params.srcContext == params.dstContext) {
		auto *intra_context_params = reinterpret_cast<base_copy_params<3>::intra_context_type *>(&params);
		return cuMemcpy3D(intra_context_params);
	}
	return cuMemcpy3DPeer(&params);
}

template<dimensionality_t NumDimensions>
status_t multidim_copy(context::handle_t context_handle, copy_parameters_t<NumDimensions> params)
{
	context::current::detail_::scoped_ensurer_t ensure_context_for_this_scope{context_handle};
	return multidim_copy(::std::integral_constant<dimensionality_t, NumDimensions>{}, params);
}

} // namespace detail

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
	detail_::copy_parameters_t<NumDimensions> params{};
	auto dims = destination.dimensions();
	params.template clear_offset<T>(detail_::endpoint_t::source);
    params.template clear_offset<T>(detail_::endpoint_t::destination);
	params.template set_extent<T>(dims);
	params.clear_rest();
	params.set_endpoint(detail_::endpoint_t::source, const_cast<T*>(source), dims);
	params.set_endpoint(detail_::endpoint_t::destination, destination);
	auto status = detail_::multidim_copy<NumDimensions>(destination.context_handle(), params);
    throw_if_error(status, "Copying from a regular memory region into a CUDA array");
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
	detail_::copy_parameters_t<NumDimensions> params{};
	auto dims = source.dimensions();
	params.template clear_offset<T>(detail_::endpoint_t::source);
	params.template clear_offset<T>(detail_::endpoint_t::destination);
	params.template set_extent<T>(source.dimensions());
	params.clear_rest();
	params.set_endpoint(detail_::endpoint_t::source, source);
	params.template set_endpoint<T>(detail_::endpoint_t::destination, destination, dims);
    params.dstPitch = params.srcPitch = dims.width * sizeof(T);
    auto status = detail_::multidim_copy<NumDimensions>(source.context_handle(), params);
    throw_if_error(status, "Copying from a CUDA array into a regular memory region");
}

template <typename T, dimensionality_t NumDimensions>
void copy(array_t<T, NumDimensions> destination, array_t<T, NumDimensions> source)
{
	detail_::copy_parameters_t<NumDimensions> params{};
	auto dims = source.dimensions();
	params.template clear_offset<T>(detail_::endpoint_t::source);
	params.template clear_offset<T>(detail_::endpoint_t::destination);
	params.template set_extent<T>(source.dimensions());
	params.clear_rest();
	params.set_endpoint(detail_::endpoint_t::source, source);
	params.set_endpoint(detail_::endpoint_t::destination, destination);
	params.dstPitch = params.srcPitch = dims.width * sizeof(T);
	auto status = //(source.context() == destination.context()) ?
		detail_::multidim_copy<NumDimensions>(source.context_handle(), params);
	throw_if_error(status, "Copying from a CUDA array into a regular memory region");
}


template <typename T, dimensionality_t NumDimensions>
void copy(region_t destination, const array_t<T, NumDimensions>& source)
{
	if (source.size_bytes() < destination.size()) {
		throw ::std::logic_error("Attempt to copy an array into a memory region too small to hold the copy");
	}
	copy(destination.start(), source);
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

namespace async {

namespace detail_ {

/**
 * Asynchronously copies data between memory spaces or within a memory space, but
 * within a single CUDA context.
 *
 * @note Since we assume Compute Capability >= 2.0, all devices support the
 * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
 * where the data is located, and one does not have to specify this.
 *
 * @note asynchronous version of @ref memory::copy
 *
 * @param stream_handle A stream on which to enqueue the copy operation
 */

///@{
/**
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
	throw_if_error(result, "Scheduling a memory copy on " + stream::detail_::identify(stream_handle));
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

using memory::detail_::copy_parameters_t;

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
	context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle);
	return multidim_copy_in_current_context(::std::integral_constant<dimensionality_t, NumDimensions>{}, params, stream_handle);
}


// Assumes the array and the stream share the same context, and that the destination is
// accessible from that context (e.g. allocated within it, or being managed memory, etc.)
template <typename T, dimensionality_t NumDimensions>
void copy(T *destination, const array_t<T, NumDimensions>& source, stream::handle_t stream_handle)
{
	using  memory::detail_::endpoint_t;
	auto dims = source.dimensions();
	//auto params = make_multidim_copy_params(destination, const_cast<T*>(source), destination.dimensions());
	detail_::copy_parameters_t<NumDimensions> params{};
	params.template clear_offset<T>(endpoint_t::source);
	params.template clear_offset<T>(endpoint_t::destination);
	params.template set_extent<T>(dims);
	params.clear_rest();
	params.set_endpoint(endpoint_t::source, source);
	params.set_endpoint(endpoint_t::destination, const_cast<T*>(destination), dims);
    params.dstPitch = dims.width * sizeof(T);
    auto status = multidim_copy_in_current_context<NumDimensions>(params, stream_handle);
    throw_if_error(status, "Scheduling an asynchronous copy from an array into a regular memory region");
}


template <typename T, dimensionality_t NumDimensions>
void copy(const array_t<T, NumDimensions>&  destination, const T* source, stream::handle_t stream_handle)
{
	using  memory::detail_::endpoint_t;
	auto dims = destination.dimensions();
	//auto params = make_multidim_copy_params(destination, const_cast<T*>(source), destination.dimensions());
	detail_::copy_parameters_t<NumDimensions> params{};
	params.template clear_offset<T>(endpoint_t::source);
	params.template clear_offset<T>(endpoint_t::destination);
	params.template set_extent<T>(destination.dimensions());
    params.srcPitch = dims.width * sizeof(T);
	params.clear_rest();
	params.set_endpoint(endpoint_t::source, const_cast<T*>(source), dims);
	params.set_endpoint(endpoint_t::destination, destination);
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
 * @param destination a value residing either in host memory or on any CUDA
 * device's global memory
 * @param source a value residing either in host memory or on any CUDA
 * device's global memory
 * @param stream_handle A stream on which to enqueue the copy operation
 */
template <typename T>
void copy_single(T& destination, const T& source, stream::handle_t stream_handle)
{
	copy(&destination, &source, sizeof(T), stream_handle);
}

} // namespace detail_

/**
 * Asynchronously copies data between memory spaces or within a memory space.
 *
 * @note Since we assume Compute Capability >= 2.0, all devices support the
 * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
 * where the data is located, and one does not have to specify this.
 *
 * @note asynchronous version of @ref memory::copy
 *
 * @param destination A (pointer to) a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory. Must be defined in the same context
 * as the stream.
 * @param source A (pointer to) a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory. Must be defined in the same context
 * as the stream
 * @param num_bytes The number of bytes to copy from @p source to @p destination
 * @param stream A stream on which to enqueue the copy operation
 */
///@{
void copy(void* destination, void const* source, size_t num_bytes, const stream_t& stream);

inline void copy(void* destination, const_region_t source, size_t num_bytes, const stream_t& stream)
{
#ifndef NDEBUG
	if (source.size() < num_bytes) {
		throw ::std::logic_error("Attempt to copy more than the source region's size");
	}
#endif
	copy(destination, source.start(), num_bytes, stream);
}

inline void copy(region_t destination, const_region_t source, size_t num_bytes, const stream_t& stream)
{
#ifndef NDEBUG
	if (destination.size() < num_bytes) {
		throw ::std::logic_error("Attempt to copy beyond the end of the destination region");
	}
#endif
	copy(destination.start(), source.start(), num_bytes, stream);
}

inline void copy(void* destination, const_region_t source, const stream_t& stream)
{
	copy(destination, source, source.size(), stream);
}

inline void copy(region_t destination, const_region_t source, const stream_t& stream)
{
	copy(destination, source, source.size(), stream);
}

inline void copy(region_t destination, void* source, const stream_t& stream)
{
	return copy(destination.start(), source, destination.size(), stream);
}


/**
 * @param source A plain array whose contents is to be copied.
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

inline void copy(region_t destination, void* source, size_t num_bytes, const stream_t& stream)
{
#ifndef NDEBUG
	if (destination.size() < num_bytes) {
		throw ::std::logic_error("Number of bytes to copy exceeds destination size");
	}
#endif
	return copy(destination.start(), source, num_bytes, stream);
}
///@}

/**
 * Asynchronously copies data from memory spaces into CUDA arrays.
 *
 * @note asynchronous version of @ref memory::copy
 *
 * @param destination A CUDA array @ref cuda::array_t
 * @param source A pointer to a a memory region of size `destination.size() * sizeof(T)`
 * @param stream schedule the copy operation into this CUDA stream
 */
template <typename T, dimensionality_t NumDimensions>
void copy(array_t<T, NumDimensions>& destination, const T* source, const stream_t& stream);

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
	copy(destination, source.start(), stream);
}

/**
 * Asynchronously copies data from CUDA arrays into memory spaces.
 *
 * @note asynchronous version of @ref memory::copy
 *
 * @param destination A pointer to a a memory region of size `source.size() * sizeof(T)`
 * @param source A CUDA array @ref cuda::array_t
 * @param stream schedule the copy operation into this CUDA stream
 */
template <typename T, dimensionality_t NumDimensions>
void copy(T* destination, const array_t<T, NumDimensions>& source, const stream_t& stream);

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
 * @param destination A region of memory to which to copy the data in @p source,
 *     of size at least that of @p source.
 * @param source The starting address of @tparam N elements to copy
 */
template <typename T, size_t N>
inline void copy(T(&destination)[N], T* source, const stream_t& stream)
{
	return copy(destination, source, sizeof(T) * N, stream);
}

/**
 * @param destination A region of memory to which to copy the data in @p source,
 *     of size at least that of @p source.
 * @param source A region of at least `sizeof(T)*N` bytes with whose data to fill @destination
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
 * Synchronously copies a single (typed) value between memory spaces or within a memory space.
 *
 * @note asynchronous version of @ref memory::copy_single
 *
 * @param destination a value residing either in host memory or on any CUDA
 * device's global memory
 * @param source a value residing either in host memory or on any CUDA
 * device's global memory
 * @param stream The CUDA command queue on which this copyijg will be enqueued
 */
template <typename T>
void copy_single(T& destination, const T& source, const stream_t& stream);

} // namespace async

namespace device {

namespace async {

namespace detail_ {

inline void set(void* start, int byte_value, size_t num_bytes, stream::handle_t stream_handle)
{
	// TODO: Double-check that this call doesn't require setting the current device
	auto result = cuMemsetD8Async(address(start), (unsigned char) byte_value, num_bytes, stream_handle);
	throw_if_error(result, "asynchronously memsetting an on-device buffer");
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

// TODO: Drop this in favor of <algorithm>-like functions under `cuda::~.
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
	throw_if_error(result, "Setting global device memory bytes");
}

} // namespace detail_


/**
 * @brief Sets consecutive elements of a region of memory to a fixed
 * value of some width
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
 * @note asynchronous version of @ref memory::zero
 *
 * @param start starting address of the memory region to set,
 * in a CUDA device's global memory
 * @param byte_value value to set the memory region to
 * @param num_bytes size of the memory region in bytes
 * @param stream stream on which to schedule this action
 */
inline void set(void* start, int byte_value, size_t num_bytes, const stream_t& stream)
{
	return typed_set<unsigned char>(static_cast<unsigned char*>(start), (unsigned char) byte_value, num_bytes, stream);
}

/**
 * Similar to @ref set(), but sets the memory to zero rather than an arbitrary value
 */
inline void zero(void* start, size_t num_bytes, const stream_t& stream);

/**
 * @brief Asynchronously sets all bytes of a single pointed-to value
 * to 0 (zero).
 *
 * @note asynchronous version of @ref memory::zero
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
	throw_if_error(status,
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
	throw_if_error(result, "Scheduling an inter-context memory copy from "
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
///@}

} // namespace detail_

void copy(
	void *        destination_address,
	context_t     destination_context,
	const void *  source_address,
	context_t     source_context,
	size_t        num_bytes,
	stream_t      stream);

void copy(
	void *          destination,
	context_t       destination_context,
	const_region_t  source,
	context_t       source_context,
	stream_t        stream);

inline void copy(
	region_t        destination,
	context_t       destination_context,
	const_region_t  source,
	context_t       source_context,
	stream_t        stream);

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

/**
 * @namespace host
 * Host-side (= system) memory which is "pinned", i.e. resides in
 * a fixed physical location - and allocated by the CUDA driver.
 */
namespace host {

/**
 * Allocate pinned host memory
 *
 * @note This function will fail if
 *
 * @note "Pinned" memory is allocated in contiguous physical RAM
 * addresses, making it possible to copy to and from it to the the
 * GPU using DMA without assistance from the GPU. This improves
 * the copying bandwidth significantly over naively-allocated
 * host memory, and reduces overhead for the CPU.
 *
 * @throws cuda::runtime_error if allocation fails for any reason
 *
 * @todo Consider a variant of this supporting the cudaHostAlloc flags
 *
 * @param size_in_bytes the amount of memory to allocate, in bytes
 * @param options options to pass to the CUDA host-side memory allocator;
 * see {@ref memory::allocation_options}.
 *
 * @return a pointer to the allocated stretch of memory
 */
void* allocate(
	size_t              size_in_bytes,
	allocation_options  options);


inline void* allocate(
	size_t                       size_in_bytes,
	portability_across_contexts  portability = portability_across_contexts(false),
	cpu_write_combining          cpu_wc = cpu_write_combining(false))
{
	return allocate(size_in_bytes, allocation_options{ portability, cpu_wc } );
}

inline void* allocate(size_t size_in_bytes, cpu_write_combining cpu_wc)
{
	return allocate(size_in_bytes, allocation_options{ portability_across_contexts(false), cpu_write_combining(cpu_wc)} );
}

/**
 * Free a region of pinned host memory which was allocated with @ref allocate.
 *
 * @note You can't just use @ref cuMemFreeHost - or you'll leak a primary context reference unit.
 */
inline void free(void* host_ptr)
{
	auto result = cuMemFreeHost(host_ptr);
#if CAW_THROW_ON_FREE_IN_DESTROYED_CONTEXT
	if (result == status::success) { return; }
#else
	if (result == status::success or result == status::context_is_destroyed) { return; }
#endif
	throw runtime_error(result, "Freeing pinned host memory at " + cuda::detail_::ptr_as_hex(host_ptr));
}

inline void free(region_t region) {	return free(region.data()); }

namespace detail_ {

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
	throw_if_error(result,
		"Could not register and page-lock the region of " + ::std::to_string(size) +
		" bytes of host memory at " + cuda::detail_::ptr_as_hex(ptr));
}

inline void register_(const_region_t region, unsigned flags)
{
	register_(region.start(), region.size(), flags);
}

} // namespace detail_

/**
 * Whether or not the registration of the host-side pointer should map
 * it into the CUDA address space for access on the device. When true,
 * one can then obtain the device-space pointer using cudaHostGetDevicePointer().
 */
enum mapped_io_space : bool {
	is_mapped_io_space               = true,
	is_not_mapped_io_space           = false
};

/**
 * Whether or not the registration of the host-side pointer should map
 * it into the CUDA address space for access on the device. When true,
 * one can then obtain the device-space pointer using cudaHostGetDevicePointer().
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


// Can't use register(), since that's a reserved word
inline void register_(const void *ptr, size_t size,
	bool register_mapped_io_space,
	bool map_into_device_space,
	bool make_device_side_accesible_to_all)
{
	detail_::register_(
		ptr, size,
		(register_mapped_io_space ? CU_MEMHOSTREGISTER_IOMEMORY : 0)
		| (map_into_device_space ? CU_MEMHOSTREGISTER_DEVICEMAP : 0)
		| (make_device_side_accesible_to_all ? CU_MEMHOSTREGISTER_PORTABLE : 0)
	);
}

inline void register_(
	const_region_t region,
	bool register_mapped_io_space,
	bool map_into_device_space,
	bool make_device_side_accesible_to_all)
{
	register_(
		region.start(),
		region.size(),
		register_mapped_io_space,
		map_into_device_space,
		make_device_side_accesible_to_all);
}


inline void register_(void const *ptr, size_t size)
{
	unsigned no_flags_set { 0 };
	detail_::register_(ptr, size, no_flags_set);
}

inline void register_(const_region_t region)
{
	register_(region.start(), region.size());
}

// the CUDA API calls this "unregister", but that's semantically
// inaccurate. The registration is not undone, rolled back, it's
// just ended
inline void deregister(const void *ptr)
{
	auto result = cuMemHostUnregister(const_cast<void *>(ptr));
	throw_if_error(result,
		"Could not unregister the memory segment starting at address *a");
}

inline void deregister(const_region_t region)
{
	deregister(region.start());
}

/**
 * @brief Sets all bytes in a stretch of host-side memory to a single value
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

struct const_region_t;

namespace detail_ {

using advice_t = CUmem_advise;

template <typename T>
inline T get_scalar_range_attribute(managed::const_region_t region, range_attribute_t attribute);

inline void advise(managed::const_region_t region, advice_t advice, cuda::device::id_t device_id);
// inline void advise(managed::const_region_t region, advice_t attribute);

template <typename T>
struct base_region_t : public memory::detail_::base_region_t<T> {
	using parent = memory::detail_::base_region_t<T>;
	using parent::parent;

	bool is_read_mostly() const
	{
		return get_scalar_range_attribute<bool>(*this, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY);
	}

	void designate_read_mostly() const
	{
		set_range_attribute(*this, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY);
	}

	void undesignate_read_mostly() const
	{
		unset_range_attribute(*this, CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY);
	}

	device_t preferred_location() const;
	void set_preferred_location(device_t& device) const;
	void clear_preferred_location() const;

	// TODO: Consider using a field proxy
};

} // namespace detail_

struct region_t : public detail_::base_region_t<void> {
	using base_region_t<void>::base_region_t;
	operator memory::region_t() { return memory::region_t{ start(), size() }; }
};

struct const_region_t : public detail_::base_region_t<void const> {
	using base_region_t<void const>::base_region_t;
	const_region_t(const region_t& r) : detail_::base_region_t<void const>(r.start(), r.size()) {}
};

void advise_expected_access_by(managed::const_region_t region, device_t& device);
void advise_no_access_expected_by(managed::const_region_t region, device_t& device);

template <typename Allocator = ::std::allocator<cuda::device_t> >
	typename ::std::vector<device_t, Allocator> accessors(managed::const_region_t region, const Allocator& allocator = Allocator() );

namespace detail_ {

template <typename T>
inline T get_scalar_range_attribute(managed::const_region_t region, range_attribute_t attribute)
{
	uint32_t attribute_value { 0 };
	auto result = cuMemRangeGetAttribute(
		&attribute_value, sizeof(attribute_value), attribute, device::address(region.start()), region.size());
	throw_if_error(result,
		"Obtaining an attribute for a managed memory range at " + cuda::detail_::ptr_as_hex(region.start()));
	return static_cast<T>(attribute_value);
}

// CUDA's range "advice" is simply a way to set the attributes of a range; unfortunately that's
// not called cuMemRangeSetAttribute, and uses a different enum.
inline void advise(managed::const_region_t region, advice_t advice, cuda::device::id_t device_id)
{
	auto result = cuMemAdvise(device::address(region.start()), region.size(), advice, device_id);
	throw_if_error(result, "Setting an attribute for a managed memory range at "
	+ cuda::detail_::ptr_as_hex(region.start()));
}

// inline void set_range_attribute(managed::const_region_t region, range_attribute_t attribute, cuda::device::handle_t device_id)

inline advice_t as_advice(range_attribute_t attribute, bool set)
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

inline void set_range_attribute(managed::const_region_t region, range_attribute_t settable_attribute, cuda::device::id_t device_id)
{
	constexpr const bool set { true };
	advise(region, as_advice(settable_attribute, set), device_id);
}

inline void unset_range_attribute(managed::const_region_t region, range_attribute_t settable_attribute)
{
	constexpr const bool unset { false };
	constexpr const cuda::device::id_t dummy_device_id { 0 };
	advise(region, as_advice(settable_attribute, unset), dummy_device_id);
}

} // namespace detail_


enum class attachment_t : unsigned {
	global        = CU_MEM_ATTACH_GLOBAL,
	host          = CU_MEM_ATTACH_HOST,
	single_stream = CU_MEM_ATTACH_SINGLE,
	};


namespace detail_ {

inline region_t allocate_in_current_context(
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
	auto status = cuMemAllocManaged(&allocated, num_bytes, (unsigned) flags);
	if (is_success(status) && allocated == 0) {
		// Can this even happen? hopefully not
		status = (status_t) status::unknown;
	}
	throw_if_error(status, "Failed allocating "
		+ ::std::to_string(num_bytes) + " bytes of managed CUDA memory");
	return {as_pointer(allocated), num_bytes};
}

/**
 * Free a region of managed memory which was allocated with @ref allocate_in_current_context.
 *
 * @note You can't just use @ref cuMemFree - or you'll leak a primary context reference unit.
 */
///@{
inline void free(void* ptr)
{
	auto result = cuMemFree(device::address(ptr));
	cuda::device::primary_context::detail_::decrease_refcount(cuda::device::default_device_id);
	throw_if_error(result, "Freeing managed memory at " + cuda::detail_::ptr_as_hex(ptr));
}
inline void free(region_t region)
{
	free(region.start());
}
///@}

template <initial_visibility_t InitialVisibility = initial_visibility_t::to_all_devices>
struct allocator {
	// Allocates in the current context!
	void* operator()(size_t num_bytes) const
	{
		return detail_::allocate_in_current_context(num_bytes, InitialVisibility).start();
	}
};

struct deleter {
	void operator()(void* ptr) const { memory::device::free(ptr); }
};

inline region_t allocate(
	context::handle_t     context_handle,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices)
{
	context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle);
	return allocate_in_current_context(num_bytes, initial_visibility);
}

} // namespace detail_

/**
 * @brief Allocate a a region of managed memory, accessible with the same
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
 * @brief Allocate a a region of managed memory, accessible with the same
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
	device_t              device,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices);

/**
 * @brief Allocate a a region of managed memory, accessible with the same
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
	throw_if_error(result,
		"Freeing managed memory (host and device regions) at address "
		+ cuda::detail_::ptr_as_hex(managed_ptr));
}

inline void free(region_t region)
{
	free(region.start());
}

namespace advice {

enum kind_t {
	read_mostly = CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY,
	preferred_location = CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION,
	accessor = CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY,
	// Note: CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION is never set
};

namespace detail_ {

inline void set(const_region_t region, kind_t advice, cuda::device::id_t device_id)
{
	auto result = cuMemAdvise(device::address(region.start()), region.size(), (managed::detail_::advice_t) advice, device_id);
	throw_if_error(result, "Setting advice on a (managed) memory region at"
		+ cuda::detail_::ptr_as_hex(region.start()) + " w.r.t. " + cuda::device::detail_::identify(device_id));
}

} // namespace detail_

void set(const_region_t region, kind_t advice, const device_t& device);

} // namespace advice

namespace async {

namespace detail_ {

inline void prefetch(
	const_region_t      region,
	cuda::device::id_t  destination,
	stream::handle_t    source_stream_handle)
{
	auto result = cuMemPrefetchAsync(device::address(region.start()), region.size(), destination, source_stream_handle);
	throw_if_error(result,
		"Prefetching " + ::std::to_string(region.size()) + " bytes of managed memory at address "
		 + cuda::detail_::ptr_as_hex(region.start()) + " to " + (
		 	(destination == CU_DEVICE_CPU) ? "the host" : cuda::device::detail_::identify(destination))  );
}

} // namespace detail_

/**
 * @brief Prefetches a region of managed memory to a specific device, so
 * it can later be used there without waiting for I/O from the host or other
 * devices.
 */
void prefetch(
	const_region_t         region,
	const cuda::device_t&  destination,
	const stream_t&        stream);

/**
 * @brief Prefetches a region of managed memory into host memory. It can
 * later be used there without waiting for I/O from any of the CUDA devices.
 */
void prefetch_to_host(
	const_region_t   region,
	const stream_t&  stream);

} // namespace async

} // namespace managed

namespace mapped {

/**
 * Obtain a pointer in the device-side memory space (= address range)
 * for the device-side memory mapped to the host-side pointer @ref host_memory_ptr
 */
template <typename T>
inline T* device_side_pointer_for(T* host_memory_ptr)
{
	device::address_t device_side_ptr;
	auto get_device_pointer_flags = 0u; // see the CUDA runtime documentation
	auto status = cuMemHostGetDevicePointer(
		&device_side_ptr,
		host_memory_ptr,
		get_device_pointer_flags);
	throw_if_error(status,
		"Failed obtaining the device-side pointer for host-memory pointer "
		+ cuda::detail_::ptr_as_hex(host_memory_ptr) + " supposedly mapped to device memory");
	return as_pointer(device_side_ptr);
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
inline region_pair allocate_in_current_context(
	context::handle_t   current_context_handle,
	size_t              size_in_bytes,
	allocation_options  options)
{
	region_pair allocated {};
	// The default initialization is unnecessary, but let's play it safe
	allocated.size_in_bytes = size_in_bytes;
	auto flags = CU_MEMHOSTALLOC_DEVICEMAP &
		cuda::memory::detail_::make_cuda_host_alloc_flags(options);
	auto status = cuMemHostAlloc(&allocated.host_side, size_in_bytes, flags);
	if (is_success(status) && (allocated.host_side == nullptr)) {
		// Can this even happen? hopefully not
		status = (status_t) status::named_t::unknown;
	}
	throw_if_error(status,
		"Failed allocating a mapped pair of memory regions of size " + ::std::to_string(size_in_bytes)
		+ " bytes of global memory in " + context::detail_::identify(current_context_handle));
	allocated.device_side = device_side_pointer_for(allocated.host_side);
	return allocated;
}

inline region_pair allocate(
	context::handle_t   context_handle,
	size_t              size_in_bytes,
	allocation_options  options)
{
	context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle);
	return detail_::allocate_in_current_context(context_handle, size_in_bytes, options);
}

inline void free(void* host_side_pair)
{
	auto result = cuMemFreeHost(host_side_pair);
	throw_if_error(result, "Freeing a mapped memory region pair with host-side address "
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
 * @param options see @ref allocation_options
 */
region_pair allocate(
	cuda::context_t&    context,
	size_t              size_in_bytes,
	allocation_options  options);

/**
 * Allocate a memory region on the host, which is also mapped to a memory region in
 * the global memory of a CUDA device - so that changes to one will be reflected in the other.
 *
 * @param device The device on which the device-side region in the pair will be allocated
 * @param size_in_bytes amount of memory to allocate (in each of the regions)
 * @param options see @ref allocation_options
 */
region_pair allocate(
	cuda::device_t&     device,
	size_t              size_in_bytes,
	allocation_options  options = allocation_options{});


/**
 * Free a pair of mapped memory regions
 *
 * @param pair a pair of regions allocated with @ref allocate (or with
 * the C-style CUDA runtime API directly)
 */
inline void free(region_pair pair)
{
	detail_::free(pair.host_side);
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
	throw_if_error(status, "Failed obtaining the host-side address of supposedly-device-side pointer "
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
inline memory::region_t locate(T&& symbol)
{
	void *start;
	size_t symbol_size;
	auto api_call_result = cudaGetSymbolAddress(&start, ::std::forward<T>(symbol));
	throw_if_error(api_call_result, "Could not locate the device memory address for a symbol");
	api_call_result = cudaGetSymbolSize(&symbol_size, ::std::forward<T>(symbol));
	throw_if_error(api_call_result, "Could not locate the device memory address for the symbol at address"
		+ cuda::detail_::ptr_as_hex(start));
	return { start, symbol_size };
}

} // namespace symbol

} // namespace cuda

#endif // CUDA_API_WRAPPERS_MEMORY_HPP_
