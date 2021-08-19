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

#include <cuda/api/array.hpp>
#include <cuda/api/constants.hpp>
#include <cuda/api/current_device.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/pointer.hpp>
#include <cuda_runtime.h> // needed, rather than cuda_runtime_api.h, e.g. for cudaMalloc

#include <memory>
#include <cstring> // for ::std::memset
#include <vector>

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

namespace detail_ {

template <class T>
	class base_region_t {
	private:
		T* start_ = nullptr;
		size_t size_in_bytes_ = 0;
	public:
		base_region_t() = default;
		base_region_t(T* start, size_t size_in_bytes)
		: start_(start), size_in_bytes_(size_in_bytes) {}

		T*& start() { return start_; }
		size_t& size() { return size_in_bytes_; }

		size_t size() const { return size_in_bytes_; }
		T* start() const { return start_; }
		T* data() const { return start(); }
		T* get() const { return start(); }
	};

}  // namespace detail_

struct region_t : public detail_::base_region_t<void> {
	using base_region_t<void>::base_region_t;
};

struct const_region_t : public detail_::base_region_t<void const> {
	using base_region_t<void const>::base_region_t;
	const_region_t(const region_t& r) : base_region_t(r.start(), r.size()) {}
};

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
		(options.portability     == portability_across_contexts::is_portable ? cudaHostAllocPortable      : 0) &
		(options.write_combining == cpu_write_combining::with_wc             ? cudaHostAllocWriteCombined : 0);
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

} // namespace memory


namespace memory {

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
inline region_t allocate(size_t num_bytes)
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
		"Failed allocating " + ::std::to_string(num_bytes) +
		" bytes of global memory on CUDA device " +
		::std::to_string(cuda::device::current::detail_::get_id()));
	return {allocated, num_bytes};
}

inline region_t allocate(cuda::device::id_t device_id, size_t size_in_bytes)
{
	cuda::device::current::detail_::scoped_override_t set_device_for_this_scope(device_id);
	return memory::device::detail_::allocate(size_in_bytes);
}

} // namespace detail_


namespace async {

namespace detail_ {

/**
 * Allocate memory asynchronously on a specified stream.
 */
inline region_t allocate(
	cuda::device::id_t  device_id,
	cuda::stream::id_t  stream_id,
	size_t              num_bytes)
{
#if CUDART_VERSION >= 11020
	void* allocated = nullptr;
	// Note: the typed cudaMalloc also takes its size in bytes, apparently,
	// not in number of elements
	auto status = cudaMallocAsync(&allocated, num_bytes, stream_id);
	if (is_success(status) && allocated == nullptr) {
		// Can this even happen? hopefully not
		status = static_cast<decltype(status)>(cuda::status::unknown);
	}
	throw_if_error(status,
		"Failed scheduling an asynchronous allocation of " + ::std::to_string(num_bytes) +
		" bytes of global memory "
		+ " on stream " + cuda::detail_::ptr_as_hex(stream_id)
		+ " on CUDA device " + ::std::to_string(device_id));
	return {allocated, num_bytes};
#else
	(void) device_id;
	(void) stream_id;
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
inline region_t allocate(const cuda::stream_t& stream, size_t size_in_bytes);


} // namespace async


/**
 * Free a region of device-side memory (regardless of how it was allocated)
 */
///@{
inline void free(void* ptr)
{
	auto result = cudaFree(ptr);
	throw_if_error(result, "Freeing device memory at 0x" + cuda::detail_::ptr_as_hex(ptr));
}
inline void free(region_t region) { free(region.start()); }
///@}

/**
 * Allocate device-side memory on a CUDA device.
 *
 * @note The CUDA memory allocator guarantees alignment "suitabl[e] for any kind of variable"
 * (CUDA 9.0 Runtime API documentation), so probably at least 128 bytes.
 *
 * @throws cuda::runtime_error if allocation fails for any reason
 *
 * @param device the device on which to allocate memory
 * @param size_in_bytes the amount of memory to allocate
 * @return a pointer to the allocated stretch of memory (only usable on the CUDA device)
 */
inline region_t allocate(cuda::device_t device, size_t size_in_bytes);

namespace detail_ {
struct allocator {
	// Allocates on the current device!
	void* operator()(size_t num_bytes) const { return detail_::allocate(num_bytes).start(); }
};
struct deleter {
	void operator()(void* ptr) const { cuda::memory::device::free(ptr); }
};
} // namespace detail_

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
	auto result = cudaMemset(start, byte_value, num_bytes);
	throw_if_error(result, "memsetting an on-device buffer");
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
 */
///@{
/**
 *  @param destination A pointer to a memory region of size @p num_bytes, either in
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
 * @param destination A memory region of size @p num_bytes, either in
 *     host memory or on any CUDA device's global memory
 * @param source A region whose contents is to be copied,  either in host memory
 *     or on any CUDA device's global memory
 */
inline void copy(void* destination, const_region_t source)
{
	return copy(destination, source.start(), source.size());
}

/**
 * @param destination A region of memory to which to copy the data in@source, of
 *     size @p num_bytes at least, either in host memory or on any CUDA device's
 *     global memory.
 * @param source A region whose contents is to be copied,  either in host memory
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
///@}

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
	pointer_t<void> pointer { region.start() };
	switch ( pointer.attributes().memory_type() ) {
	case device_memory:
	case managed_memory:
		memory::device::set(region, byte_value); break;
	case unregistered_memory:
	case host_memory:
		::std::memset(region.start(), byte_value, region.size()); break;
	default:
		throw runtime_error(
			cuda::status::invalid_value,
			"CUDA returned an invalid memory type for the pointer 0x" + cuda::detail_::ptr_as_hex(region.start())
		);
	}
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
	copy_params_t(const T* destination, const array_t<T, 3>& source) :
		copy_params_t(tag{}, destination, source)
	{
		srcArray = source.get();
		dstPtr = pitched_ptr;
	}

	size_t pitch;
	cudaPitchedPtr pitched_ptr;
};

template<typename T>
inline void copy(array_t<T, 2>& destination, const T *source)
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

template <typename T>
inline void copy(array_t<T, 3>& destination, const T *source)
{
	const auto copy_params = detail_::copy_params_t(destination, source);
	auto result = cudaMemcpy3D(&copy_params);
	throw_if_error(result, "Synchronously copying into a 3-dimensional CUDA array");
}

template <typename T>
inline void copy(T *destination, const array_t<T, 2>& source)
{
	const auto dimensions = source.dimensions();
	const auto width_in_bytes = sizeof(T) * dimensions.width;
	const auto destination_pitch = width_in_bytes; // i.e. no padding
	const array::dimensions_t<2> offsets { 0, 0 };
	auto result = cudaMemcpy2DFromArray(
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

template <typename T>
inline void copy(T* destination, const array_t<T, 3>& source)
{
	const auto copy_params = detail_::copy_params_t(destination, source);
	auto result = cudaMemcpy3D(&copy_params);
	throw_if_error(result, "Synchronously copying from a 3-dimensional CUDA array");
}

} // namespace detail_


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
template <typename T, dimensionality_t NumDimensions>
inline void copy(array_t<T, NumDimensions>& destination, const T* source)
{
	detail_::copy(destination, source);
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
inline void copy(T* destination, const array_t<T, NumDimensions>& source)
{
	detail_::copy(destination, source);
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
inline void copy_single(T* destination, const T* source)
{
	copy(destination, source, sizeof(T));
}

namespace async {

namespace detail_ {

/**
 * Asynchronously copies data between memory spaces or within a memory space.
 *
 * @note Since we assume Compute Capability >= 2.0, all devices support the
 * Unified Virtual Address Space, so the CUDA driver can determine, for each pointer,
 * where the data is located, and one does not have to specify this.
 *
 * @note asynchronous version of @ref memory::copy
 *
 * @param stream_id A stream on which to enqueue the copy operation
 */

/**
* @param destination A pointer to a memory region of size @p num_bytes, either in
* host memory or on any CUDA device's global memory
* @param source A pointer to a memory region of size at least @p num_bytes, either in
* host memory or on any CUDA device's global memory
* @param num_bytes number of bytes to copy from @p source
* @parem stream_id The stream on which to schedule the copying
*/
inline void copy(void* destination, const void* source, size_t num_bytes, stream::id_t stream_id)
{
	auto result = cudaMemcpyAsync(destination, source, num_bytes, cudaMemcpyDefault, stream_id);

	// TODO: Determine whether it was from host to device, device to host etc and
	// add this information to the error string
	throw_if_error(result, "Scheduling a memory copy on stream " + cuda::detail_::ptr_as_hex(stream_id));
}

template<typename T>
void copy(array_t<T, 3>& destination, const T* source, stream::id_t stream_id)
{
	const auto copy_params = memory::detail_::copy_params_t(destination, source);
	auto result = cudaMemcpy3DAsync(&copy_params, stream_id);
	throw_if_error(result, "Scheduling a memory copy into a 3D CUDA array on stream " + cuda::detail_::ptr_as_hex(stream_id));
}

template<typename T>
void copy(T* destination, const array_t<T, 3>& source, stream::id_t stream_id)
{
	const auto copy_params = memory::detail_::copy_params_t(destination, source);
	auto result = cudaMemcpy3DAsync(&copy_params, stream_id);
	throw_if_error(result, "Scheduling a memory copy out of a 3D CUDA array on stream " + cuda::detail_::ptr_as_hex(stream_id));
}

template<typename T>
void copy(array_t<T, 2>& destination, const T* source, stream::id_t stream_id)
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
	throw_if_error(result, "Scheduling a memory copy into a 2D CUDA array on stream " + cuda::detail_::ptr_as_hex(stream_id));
}

template<typename T>
void copy(T* destination, const array_t<T, 2>& source, cuda::stream::id_t stream_id)
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
	throw_if_error(result, "Scheduling a memory copy out of a 3D CUDA array on stream " + cuda::detail_::ptr_as_hex(stream_id));
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
 * @param destination A pointer to a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param source A pointer to a a memory region of size @p num_bytes, either in
 * host memory or on any CUDA device's global memory
 * @param num_bytes The number of bytes to copy from @p source to @p destination
 * @param stream A stream on which to enqueue the copy operation
 */
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
inline void copy(array_t<T, NumDimensions>& destination, const T* source, const stream_t& stream);

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
	size_t required_size = destination.size() * sizeof(T);
	if (destination.size() < required_size) {
		throw ::std::invalid_argument(
			"Attempt to copy " + ::std::to_string(required_size) + " bytes from an array into a "
			"region of smaller size (" + ::std::to_string(destination.size()) + " bytes)");
	}
#endif
	copy(destination.start(), source, stream);
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

inline void set(void* start, int byte_value, size_t num_bytes, stream::id_t stream_id)
{
	// TODO: Double-check that this call doesn't require setting the current device
	auto result = cudaMemsetAsync(start, byte_value, num_bytes, stream_id);
	throw_if_error(result, "asynchronously memsetting an on-device buffer");
}

inline void set(region_t region, int byte_value, stream::id_t stream_id)
{
	set(region.start(), byte_value, region.size(), stream_id);
}


inline void zero(void* start, size_t num_bytes, stream::id_t stream_id)
{
	set(start, 0, num_bytes, stream_id);
}

inline void zero(region_t region, stream::id_t stream_id)
{
	zero(region.start(), region.size(), stream_id);
}

} // namespace detail_

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
inline void set(void* start, int byte_value, size_t num_bytes, const stream_t& stream);

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
 * @param ptr a pointer to the value to be to zero
 * @param stream stream on which to schedule this action
 */
template <typename T>
inline void zero(T* ptr, const stream_t& stream)
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
 * @param options options to pass to the CUDA host-side memory allocator;
 * see {@ref memory::allocation_options}.
 *
 * @return a pointer to the allocated stretch of memory
 */
inline void* allocate(
	size_t              size_in_bytes,
	allocation_options  options)
{
	void* allocated = nullptr;
	auto flags = cuda::memory::detail_::make_cuda_host_alloc_flags(options);
	auto result = cudaHostAlloc(&allocated, size_in_bytes, flags);
	if (is_success(result) && allocated == nullptr) {
		// Can this even happen? hopefully not
		result = cudaErrorUnknown;
	}
	throw_if_error(result, "Failed allocating " + ::std::to_string(size_in_bytes) + " bytes of host memory");
	return allocated;
}

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
 */
inline void free(void* host_ptr)
{
	auto result = cudaFreeHost(host_ptr);
	throw_if_error(result, "Freeing pinned host memory at 0x" + cuda::detail_::ptr_as_hex(host_ptr));
}

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
 * flags. This memory range also is added to the same tracking mechanism as cudaHostAlloc() to
 * automatically accelerate calls to functions such as cudaMemcpy().
 *
 * @param ptr A pre-allocated stretch of host memory
 * @param size the size in bytes the memory region to register/pin
 * @param flags
 */
inline void register_(const void *ptr, size_t size, unsigned flags)
{
	auto result = cudaHostRegister(const_cast<void *>(ptr), size, flags);
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
		  (register_mapped_io_space ? cudaHostRegisterIoMemory : 0)
		| (map_into_device_space ? cudaHostRegisterMapped : 0)
		| (make_device_side_accesible_to_all ? cudaHostRegisterPortable : 0)
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
	detail_::register_(ptr, size, cudaHostRegisterDefault);
}

inline void register_(const_region_t region)
{
	register_(region.start(), region.size());
}

// the CUDA API calls this "unregister", but that's semantically
// inaccurate. The registration is not undone, rolled back, it's
// just ended
inline void deregister(void const *ptr)
{
	auto result = cudaHostUnregister(const_cast<void *>(ptr));
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

class const_region_t;

namespace detail_ {

template <typename T>
inline T get_scalar_range_attribute(managed::const_region_t region, cudaMemRangeAttribute attribute);

inline void set_scalar_range_attribute(managed::const_region_t region, cudaMemoryAdvise advice, cuda::device::id_t device_id);
inline void set_scalar_range_attribute(managed::const_region_t region, cudaMemoryAdvise attribute);

template <typename T>
struct base_region_t : public memory::detail_::base_region_t<T> {
	using parent = memory::detail_::base_region_t<T>;
	using parent::parent;

	bool is_read_mostly() const
	{
		return get_scalar_range_attribute<bool>(*this, cudaMemRangeAttributeReadMostly);
	}

	void designate_read_mostly() const
	{
		set_scalar_range_attribute(*this, cudaMemAdviseSetReadMostly);
	}

	void undesignate_read_mostly() const
	{
		detail_::set_scalar_range_attribute(*this, cudaMemAdviseUnsetReadMostly);
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
inline T get_scalar_range_attribute(managed::const_region_t region, cudaMemRangeAttribute attribute)
{
	uint32_t attribute_value { 0 };
	auto result = cudaMemRangeGetAttribute(
		&attribute_value, sizeof(attribute_value), attribute, region.start(), region.size());
	throw_if_error(result,
		"Obtaining an attribute for a managed memory range at " + cuda::detail_::ptr_as_hex(region.start()));
	return static_cast<T>(attribute_value);
}

inline void set_scalar_range_attribute(managed::const_region_t region, cudaMemoryAdvise advice, cuda::device::id_t device_id)
{
	auto result = cudaMemAdvise(region.start(), region.size(), advice, device_id);
	throw_if_error(result,
		"Setting an attribute for a managed memory range at " + cuda::detail_::ptr_as_hex(region.start()));
}

inline void set_scalar_range_attribute(managed::const_region_t region, cudaMemoryAdvise attribute)
{
	cuda::device::id_t ignored_device_index{};
	set_scalar_range_attribute(region, attribute, ignored_device_index);
}

} // namespace detail_


enum class initial_visibility_t {
	to_all_devices,
	to_supporters_of_concurrent_managed_access,
};


enum class attachment_t {
	global        = cudaMemAttachGlobal,
	host          = cudaMemAttachHost,
	single_stream = cudaMemAttachSingle,
};


namespace detail_ {

inline region_t allocate(
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
		"Failed allocating " + ::std::to_string(num_bytes) + " bytes of managed CUDA memory");
	return {allocated, num_bytes};
}

/**
 * Free a region of pinned host memory which was allocated with @ref allocate.
 */
///@{
inline void free(void* ptr)
{
	auto result = cudaFree(ptr);
	throw_if_error(result, "Freeing managed memory at 0x" + cuda::detail_::ptr_as_hex(ptr));
}
inline void free(region_t region)
{
	free(region.start());
}
///@}

template <initial_visibility_t InitialVisibility = initial_visibility_t::to_all_devices>
struct allocator {
	// Allocates on the current device!
	void* operator()(size_t num_bytes) const
	{
		return detail_::allocate(num_bytes, InitialVisibility).start();
	}
};
struct deleter {
	void operator()(void* ptr) const { cuda::memory::device::free(ptr); }
};

inline region_t allocate(
	cuda::device::id_t    device_id,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices)
{
	cuda::device::current::detail_::scoped_override_t set_device_for_this_scope(device_id);
	return detail_::allocate(num_bytes, initial_visibility);
}

} // namespace detail_

/**
 * @brief Allocate a a region of managed memory, accessible with the same
 * address on the host and on CUDA devices
 *
 * @param device the initial device which is likely to access the managed
 * memory region (and which will certainly have actually allocated for it)
 * @param num_bytes size of each of the regions of memory to allocate
 * @param initial_visibility will the allocated region be visible, using the
 * common address, to all CUDA device (= more overhead, more work for the CUDA
 * runtime) or just to those devices with some hardware features to assist in
 * this task (= less overhead)?
 */
region_t allocate(
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
		+ cuda::detail_::ptr_as_hex(managed_ptr));
}

inline void free(region_t region)
{
	free(region.start());
}

namespace advice {

enum device_inspecific_kind_t {
	read_mostly = cudaMemAdviseSetReadMostly,
};

enum device_specific_kind_t {
	preferred_location,
	accessor,
};

inline void set(const_region_t region, device_inspecific_kind_t advice)
{
	cuda::device::id_t ignored_device_index{};
	auto result = cudaMemAdvise(region.start(), region.size(), (cudaMemoryAdvise) advice, ignored_device_index);
	throw_if_error(result,
		"Setting advice on a (managed) memory region at" + cuda::detail_::ptr_as_hex(region.start()));
}

} // namespace advice

namespace async {

namespace detail_ {

inline void prefetch(
	const_region_t      region,
	cuda::device::id_t  destination,
	stream::id_t        stream_id)
{
	auto result = cudaMemPrefetchAsync(region.start(), region.size(), destination, stream_id);
	throw_if_error(result,
		"Prefetching " + ::std::to_string(region.size()) + " bytes of managed memory at address "
		 + cuda::detail_::ptr_as_hex(region.start()) + " to device " + ::std::to_string(destination));
}

} // namespace detail_

/**
 * @brief Prefetches a region of managed memory to a specific device, so
 * it can later be used there without waiting for I/O from the host or other
 * devices.
 */
void prefetch(
	const_region_t   region,
	cuda::device_t   destination,
	const stream_t&  stream);

/**
 * @brief Prefetches a region of managed memory into host memory. It can
 * later be used there without waiting for I/O from any of the CUDA devices.
 */
inline void prefetch_to_host(const_region_t managed_region)
{
	auto result = cudaMemPrefetchAsync(
		managed_region.start(),
		managed_region.size(),
		cudaCpuDeviceId,
		stream::default_stream_id);
		// The stream ID will be ignored by the CUDA runtime API when this pseudo
		// device indicator is used.
	throw_if_error(result,
		"Prefetching " + ::std::to_string(managed_region.size()) + " bytes of managed memory at address "
		 + cuda::detail_::ptr_as_hex(managed_region.start()) + " into host memory");
}

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
	T* device_side_ptr;
	auto get_device_pointer_flags = 0u; // see the CUDA runtime documentation
	auto status = cudaHostGetDevicePointer(
		&device_side_ptr,
		host_memory_ptr,
		get_device_pointer_flags);
	throw_if_error(status,
		"Failed obtaining the device-side pointer for host-memory pointer "
		+ cuda::detail_::ptr_as_hex(host_memory_ptr) + " supposedly mapped to device memory");
	return device_side_ptr;
}

namespace detail_ {

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
	size_t              size_in_bytes,
	allocation_options  options)
{
	region_pair allocated;
	allocated.size_in_bytes = size_in_bytes;
	auto flags = cudaHostAllocMapped &
		cuda::memory::detail_::make_cuda_host_alloc_flags(options);
	// Note: the typed cudaHostAlloc also takes its size in bytes, apparently,
	// not in number of elements
	auto status = cudaHostAlloc(&allocated.host_side, size_in_bytes, flags);
	if (is_success(status) && (allocated.host_side == nullptr)) {
		// Can this even happen? hopefully not
		status = cudaErrorUnknown;
	}
	throw_if_error(status,
		"Failed allocating a mapped pair of memory regions of size " + ::std::to_string(size_in_bytes)
			+ " bytes of global memory on device " + ::std::to_string(cuda::device::current::detail_::get_id()));
	allocated.device_side = device_side_pointer_for(allocated.host_side);
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
	cuda::device::id_t  device_id,
	size_t              size_in_bytes,
	allocation_options  options)
{
	cuda::device::current::detail_::scoped_override_t set_device_for_this_scope(device_id);
	return detail_::allocate(size_in_bytes, options);
}

} // namespace detail_

/**
 * Allocate a pair of memory regions, on the host and on the device, mapped to each other so
 * that changes to one will be reflected in the other.
 *
 * @param device The device on which the device-side region in the pair will be allocated
 * @param size_in_bytes amount of memory to allocate (in each of the regions)
 * @param options see @ref allocation_options
 */
region_pair allocate(
	cuda::device_t&     device,
	size_t              size_in_bytes,
	allocation_options  options);

/**
 * @brief A variant of @ref allocate facilitating only specifying some of the allocation options
 */
inline region_pair allocate(
	cuda::device_t&              device,
	size_t                       size_in_bytes,
	portability_across_contexts  portability = portability_across_contexts(false),
	cpu_write_combining          cpu_wc = cpu_write_combining(false))
{
	return allocate(device, size_in_bytes, allocation_options{ portability, cpu_wc } );
}

/**
 * @brief A variant of @ref allocate facilitating only specifying some of the allocation options
 */
inline region_pair allocate(
	cuda::device_t&     device,
	size_t              size_in_bytes,
	cpu_write_combining cpu_wc)
{
	return allocate(device, size_in_bytes, allocation_options{ portability_across_contexts(false), cpu_write_combining(cpu_wc)} );
}


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
inline bool is_part_of_a_region_pair(const void* ptr)
{
	auto wrapped_ptr = pointer_t<void const> { ptr };
	return wrapped_ptr.other_side_of_region_pair().get() != nullptr;
}

} // namespace mapped

/**
 * Locates a CUDA symbol in global or constant device memory
 *
 * @return The region of memory CUDA associates with the symbol
 */
inline region_t locate(symbol_t symbol)
{
	void* start;
	size_t symbol_size;
	auto api_call_result = cudaGetSymbolAddress(&start, symbol.handle);
	throw_if_error(api_call_result, "Could not locate the device memory address for symbol " + cuda::detail_::ptr_as_hex(symbol.handle));
	api_call_result = cudaGetSymbolSize(&symbol_size, symbol.handle);
	throw_if_error(api_call_result, "Could not locate the device memory address for symbol " + cuda::detail_::ptr_as_hex(symbol.handle));
	return {start, symbol_size};
}

} // namespace memory

} // namespace cuda

#endif // CUDA_API_WRAPPERS_MEMORY_HPP_
