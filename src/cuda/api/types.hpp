/**
 * @file
 *
 * @brief Fundamental CUDA-related type definitions.
 *
 * This is a common file for all definitions of fundamental CUDA-related types,
 * some shared by different APIs.
 *
 * @note In this file you'll find several numeric or opaque handle types, e.g.
 * for devices, streams and events. These are mostly to be ignored; they
 * appear here to make interaction with the unwrapped API easier and to break
 * dependencies in the code. Instead, this library offers wrapper classes for
 * them, in separate header files. For example: `stream.hpp` contains a stream_t
 * class with its unique stream handle. Those are the ones you will want
 * to use - they are more convenient and safer.
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_COMMON_TYPES_HPP_
#define CUDA_API_WRAPPERS_COMMON_TYPES_HPP_

#if (__cplusplus < 201103L && (!defined(_MSVC_LANG) || _MSVC_LANG < 201103L))
#error "The CUDA API headers can only be compiled with C++11 or a later version of the C++ language standard"
#endif

#include "detail/optional.hpp"
#include "detail/span.hpp"
#include "detail/region.hpp"
#include "detail/type_traits.hpp"

#ifndef __CUDACC__
#include <builtin_types.h>
#endif
#include <cuda.h>

#include <type_traits>
#include <utility>
#include <cassert>
#include <cstddef> // for ::std::size_t
#include <cstdint>
#include <vector>
#ifndef NDEBUG
#include <stdexcept>
#endif

#ifndef __CUDACC__
#ifndef __device__
#define __device__
#define __host__
#endif
#endif


/// @brief Definitions and functionality wrapping CUDA APIs.
namespace cuda {

/**
 * Indicates either the result (success or error index) of a CUDA Runtime or Driver API call,
 * or the overall status of the API (which is typically the last triggered error).
 *
 * @note This single type really needs to double as both CUresult for driver API calls and
 * cudaError_t for runtime API calls. These aren't actually the same type - but they are both enums,
 * sharing most of the defined values. See also @ref error.hpp where we unify the set of errors.
 */
using status_t = CUresult;

/// A size type for use throughout the wrappers library (except when specific API functions
/// limit the size further)
using size_t = ::std::size_t;

/// The index or number of dimensions of an entity (as opposed to the extent in any
/// dimension) - typically just 0, 1, 2 or 3.
using dimensionality_t = size_t;

/**
 * CUDA facilities for interpolating access to multidimensional array objects,
 * in particular via the @ref array_t class.
 */
namespace array {

/// An individual dimension extent for an array
using dimension_t = size_t;

/**
 * CUDA's array memory-objects are multi-dimensional; but their dimensions,
 * or extents, are not the same as @ref cuda::grid::dimensions_t ; they may be
 * much larger in each axis.
 *
 * @note See also <a href="https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaExtent.html">
 * the description of `struct cudaExtent`</a> in the CUDA Runtime API documentation.
 */
template<dimensionality_t NumDimensions>
struct dimensions_t;

/// Dimensions for 3D CUDA arrays
template<>
struct dimensions_t<3> // this almost-inherits cudaExtent
{
	/// The three constituent individual dimensions, named
	dimension_t width, height, depth;

	constexpr __host__ __device__ dimensions_t(dimension_t width_, dimension_t height_, dimension_t depth_)
		: width(width_), height(height_), depth(depth_) { }
	constexpr __host__ __device__ dimensions_t(cudaExtent e)
		: dimensions_t(e.width, e.height, e.depth) { }
	constexpr __host__ __device__ dimensions_t(const dimensions_t& other)
		: dimensions_t(other.width, other.height, other.depth) { }
	constexpr __host__ __device__ dimensions_t(dimensions_t&& other)
		: dimensions_t(other.width, other.height, other.depth) { }
	constexpr __host__ __device__ dimensions_t(dimension_t linear_size)
		: dimensions_t(linear_size, 1, 1) { }

	CPP14_CONSTEXPR dimensions_t& operator=(const dimensions_t& other) = default;
	CPP14_CONSTEXPR dimensions_t& operator=(dimensions_t&& other) = default;

	constexpr __host__ __device__ operator cudaExtent() const
	{
		return { width, height, depth };
			// Note: We're not using make_cudaExtent here because:
			// 1. It's not constexpr and
			// 2. It doesn't do anything except construct the plain struct - as of CUDA 10 at least
	}

	/// The total number of elements in a 3D entity with these dimensions
	constexpr __host__ __device__ size_t volume() const { return width * height * depth; }

	/// @copydoc volume
	constexpr __host__ __device__ size_t size() const { return volume(); }

	/// The number of non-trivial dimensions in this object (axes in which an object with
	/// these dimensions is not "flat")
	constexpr __host__ __device__ dimensionality_t dimensionality() const
	{
		return ((width > 1) + (height> 1) + (depth > 1));
	}

	/// Named constructor idiom: Dimensions for an equi-lateral cube
	static constexpr __host__ __device__ dimensions_t cube(dimension_t x)   { return dimensions_t{ x, x, x }; }

	///Named constructor idiom: Dimensions for a one-element object: "Flat" in all individual
	// dimensions
	static constexpr __host__ __device__ dimensions_t zero() { return cube(0); }
};

/// Dimensions for 2D CUDA arrays
template<>
struct dimensions_t<2>
{
	/// The two constituent individual dimensions, named; no "depth" for the 2D case.
	dimension_t width, height;

	constexpr __host__ __device__ dimensions_t(dimension_t width_, dimension_t height_)
		: width(width_), height(height_) { }
	constexpr __host__ __device__ dimensions_t(const dimensions_t& other)
		: dimensions_t(other.width, other.height) { }
	constexpr __host__ __device__ dimensions_t(dimensions_t&& other)
		: dimensions_t(other.width, other.height) { }
	constexpr __host__ __device__ dimensions_t(dimension_t linear_size)
		: dimensions_t(linear_size, 1) { }

	CPP14_CONSTEXPR __host__ __device__ dimensions_t& operator=(const dimensions_t& other)
	{
		width = other.width; height = other.height;
		return *this;

	}
	CPP14_CONSTEXPR __host__ __device__ dimensions_t& operator=(dimensions_t&& other)
	{
		width = other.width; height = other.height;
		return *this;
	}

	/// The total number of elements in a 2D entity with these dimensions
	constexpr __host__ __device__ size_t area() const { return width * height; }

	/// @copydoc area
	constexpr __host__ __device__ size_t size() const { return area(); }

	/// The number of non-trivial dimensions in this object (axes in which an object with
	/// these dimensions is not "flat")
	constexpr __host__ __device__ dimensionality_t dimensionality() const
	{
		return ((width > 1) + (height> 1));
	}

	// Named constructor idioms

	/// Named constructor idiom: Dimensions for an equi-lateral cube
	static constexpr __host__ __device__ dimensions_t square(dimension_t x)   { return dimensions_t{ x, x }; }

	///Named constructor idiom: Dimensions for a one-element object: "Flat" in all individual
	// dimensions
	static constexpr __host__ __device__ dimensions_t zero() { return square(0); }
};

} // namespace array

/**
 * CUDA timing functionality, via events and their related code (not including
 * the event wrapper type @ref event_t itself)
 */
namespace event {

/// The CUDA driver's raw handle for events
using handle_t = CUevent;

/**
 * Definitions and functionality related to CUDA events (not
 * including the event wrapper type @ref event_t itself)
 */
namespace ipc {

/// The CUDA driver's raw handle for events passed between processes
using handle_t = CUipcEventHandle;

} // namespace ipc

} // namespace event

/**
 * Definitions and functionality related to CUDA streams (not
 * including the device wrapper type @ref stream_t itself)
 */
namespace stream {

/// The CUDA driver's raw handle for streams
using handle_t = CUstream;

/**
 * CUDA streams have a scheduling priority, with lower values meaning higher priority.
 * The types represents a larger range of values than those actually used; they can
 * be obtained by @ref device_t::stream_priority_range() .
 */
using priority_t       = int;
enum : priority_t {
	/// the scheduling priority of a stream created without specifying any other priority value
	default_priority   = 0
};

namespace detail_ {

/// The CUDA driver's raw handle for a host-side callback function
#if CUDA_VERSION >= 10000
using callback_t = CUhostFn;
#else
using callback_t = CUstreamCallback;
#endif

} // namespace detail_

} // namespace stream

namespace grid {

/**
 * CUDA kernels are launched in grids of blocks of threads, in 3 dimensions.
 * In each of these, the numbers of blocks per grid is specified in this type.
 *
 * @note Theoretically, CUDA could split the type for blocks per grid and
 * threads per block, but for now they're the same.
 */
using dimension_t        = decltype(dim3::x);

/**
 * CUDA kernels are launched in grids of blocks of threads, in 3 dimensions.
 * In each of these, the number of threads per block is specified in this type.
 *
 * @note Theoretically, CUDA could split the type for blocks per grid and
 * threads per block, but for now they're the same.
 *
 * @note At the time of writing, a grid dimension value cannot exceed 2^31
 * on any axis (even lower on the y and z axes), so signed 32-bit integers
 * can may be safely narrowing-cast into from this type.
 */
using block_dimension_t  = dimension_t;


/**
 * A richer (kind-of-a-)wrapper for CUDA's `dim3` class, used
 * to specify dimensions for blocks (in terms of threads) and of
 * grids(in terms of blocks, or overall).
 *
 * @note Unfortunately, `dim3` does not have constexpr methods -
 * preventing us from having constexpr methods here.
 *
 * @note Unlike 3D dimensions in general, grid dimensions cannot actually
 * be empty: A grid must have some threads. Thus, the value in each
 * axis must be positive.
 */
struct dimensions_t // this almost-inherits dim3
{
	dimension_t x, y, z;
	constexpr __host__ __device__ dimensions_t(dimension_t x_ = 1, dimension_t y_ = 1, dimension_t z_ = 1)
	: x(x_), y(y_), z(z_) { }

	constexpr __host__ __device__ dimensions_t(const uint3& v) : dimensions_t(v.x, v.y, v.z) { }
	constexpr __host__ __device__ dimensions_t(const dim3& dims) : dimensions_t(dims.x, dims.y, dims.z) { }
	constexpr __host__ __device__ dimensions_t(dim3&& dims) : dimensions_t(dims.x, dims.y, dims.z) { }

	constexpr __host__ __device__ operator uint3(void) const { return { x, y, z }; }

	// This _should_ have been constexpr, but nVIDIA have not marked the dim3 constructors
	// as constexpr, so it isn't
	__host__ __device__ operator dim3(void) const { return { x, y, z }; }

	/// The number of total elements in a 3D object with these dimensions
	constexpr __host__ __device__ size_t volume() const { return static_cast<size_t>(x) * y * z; }

	/// Number of dimensions in which this dimension structure is non-trivial, i.e. coordinates can
	/// have more than a single value
	constexpr __host__ __device__ dimensionality_t dimensionality() const
	{
		return ((z > 1) + (y > 1) + (x > 1));
	}

	// Named constructor idioms

	/// Dimensions of an equi-lateral 3D cube
	static constexpr __host__ __device__ dimensions_t cube(dimension_t x)   { return dimensions_t{ x, x, x }; }

	/// Dimensions of an equi-lateral 2D square, with a trivial third dimension
	static constexpr __host__ __device__ dimensions_t square(dimension_t x) { return dimensions_t{ x, x, 1 }; }

	/// Dimensions of a 1D line, with the last two dimensions trivial
	static constexpr __host__ __device__ dimensions_t line(dimension_t x)   { return dimensions_t{ x, 1, 1 }; }

	/// Dimensions of a single point - trivial in in all axes
	static constexpr __host__ __device__ dimensions_t point()               { return dimensions_t{ 1, 1, 1 }; }

	/// @returns true if the dimensions on the left-hand side divide, elementwise, those
	/// on the right-hand side
	static bool divides(dimensions_t lhs, dimensions_t rhs)
	{
		return
			(rhs.x % lhs.x == 0) and
			(rhs.y % lhs.y == 0) and
			(rhs.z % lhs.z == 0);
	}
};

///@cond
constexpr inline bool operator==(const dim3& lhs, const dim3& rhs) noexcept
{
	return lhs.x == rhs.x and lhs.y == rhs.y and lhs.z == rhs.z;
}
constexpr inline bool operator!=(const dim3& lhs, const dim3& rhs) noexcept
{
	return not (lhs == rhs);
}
constexpr inline bool operator==(const dimensions_t& lhs, const dimensions_t& rhs) noexcept
{
	return lhs.x == rhs.x and lhs.y == rhs.y and lhs.z == rhs.z;
}
constexpr inline bool operator!=(const dimensions_t& lhs, const dimensions_t& rhs) noexcept
{
	return not (lhs == rhs);
}
///@endcond


/**
 * CUDA kernels are launched in grids of blocks of threads. This expresses the
 * dimensions of a block within such a grid, in terms of threads.
 *
 * @todo Consider having both grid and block dims inhert from the same dimensions_t
 * structure, but be incompatible, to prevent mis-casting one as the other.
 */
using block_dimensions_t = dimensions_t;

/**
 * Dimension of a grid in threads along one axis, i.e. a multiplication
 * of a grid's block dimension and the grid's dimension in blocks, on
 * some axis.
 */
using overall_dimension_t = size_t;

/**
 * Dimensions of a grid in threads, i.e. the axis-wise multiplication of
 * block and grid dimensions of a grid.
 */
struct overall_dimensions_t
{
	using dimension_type = overall_dimension_t;
	dimension_type x, y, z;

	constexpr __host__ __device__ overall_dimensions_t(
		dimension_type width_, dimension_type height_, dimension_type depth_) noexcept
		: x(width_), y(height_), z(depth_) { }

	constexpr __host__ __device__ overall_dimensions_t(const dim3& dims) noexcept
		: x(dims.x), y(dims.y), z(dims.z) { }

	constexpr __host__ __device__ overall_dimensions_t(dim3&& dims) noexcept
		: x(dims.x), y(dims.y), z(dims.z) { }

	constexpr __host__ __device__ overall_dimensions_t(const overall_dimensions_t& other) noexcept
		: overall_dimensions_t(other.x, other.y, other.z) { }

	constexpr __host__ __device__ overall_dimensions_t(overall_dimensions_t&& other) noexcept
		: overall_dimensions_t(other.x, other.y, other.z) { }

	explicit constexpr __host__ __device__ overall_dimensions_t(dimensions_t dims) noexcept
		: overall_dimensions_t(dims.x, dims.y, dims.z) { }

	CPP14_CONSTEXPR overall_dimensions_t& operator=(const overall_dimensions_t& other) noexcept = default;
	CPP14_CONSTEXPR overall_dimensions_t& operator=(overall_dimensions_t&& other) noexcept = default;

	constexpr __host__ __device__ size_t volume() const noexcept { return x * y * z; }
	constexpr __host__ __device__ size_t size() const noexcept { return volume(); }
	constexpr __host__ __device__ dimensionality_t dimensionality() const noexcept
	{
		return ((x > 1) + (y > 1) + (z > 1));
	}
};

///@cond
constexpr bool operator==(overall_dimensions_t lhs, overall_dimensions_t rhs) noexcept
{
	return (lhs.x == rhs.x) and (lhs.y == rhs.y) and (lhs.z == rhs.z);
}

constexpr bool operator!=(overall_dimensions_t lhs, overall_dimensions_t rhs) noexcept
{
	return not (lhs == rhs);
}

constexpr overall_dimensions_t operator*(dimensions_t grid_dims, block_dimensions_t block_dims) noexcept
{
	return overall_dimensions_t {
		grid_dims.x * overall_dimension_t { block_dims.x },
		grid_dims.y * overall_dimension_t { block_dims.y },
		grid_dims.z * overall_dimension_t { block_dims.z },
	};
}
///@endcond

/**
 * Composite dimensions for a grid - in terms of blocks, then also down
 * into the block dimensions completing the information to the thread level.
 */
struct composite_dimensions_t {
	grid::dimensions_t       grid;
	grid::block_dimensions_t block;

	/// @returns The overall dimensions of the entire grid as a single 3D entity
	constexpr overall_dimensions_t flatten() const { return grid * block; }

	/// @returns The total number of threads over all blocks of the grid
	constexpr size_t volume() const { return flatten().volume(); }

	/// @returns the number of axes in which the grid overall has non-trivial dimension
	constexpr size_t dimensionality() const { return flatten().dimensionality(); }

	/// A named constructor idiom for the composite dimensions of a single-block grid
	/// with a single-thread block
	static constexpr composite_dimensions_t point()
	{
		return { dimensions_t::point(), block_dimensions_t::point() };
	}
};

///@cond
constexpr bool operator==(composite_dimensions_t lhs, composite_dimensions_t rhs) noexcept
{
	return (lhs.grid == rhs.grid) and (lhs.block == rhs.block);
}

constexpr bool operator!=(composite_dimensions_t lhs, composite_dimensions_t rhs) noexcept
{
	return not (lhs == rhs);
}
///@endcond

} // namespace grid

/// Representation, allocation and manipulation of CUDA-related memory, of different
namespace memory {

#if CUDA_VERSION >= 10020
/// Named boolean constants used in memory read and write access control
enum : bool {
	read_enabled = true,
	read_disabled = false,
	write_enabled = true,
	write_disabled = false
};

/**
 * An access permission specification structure for physical allocations
 * performed by / registered with the CUDA driver
 *
 * @note defined for each of use instead of the raw CUDA driver's @ref CUmemAccess_flags
 * type.
 */
struct access_permissions_t {
	bool read : 1;
	bool write : 1;

	/// This allows passing this access permissions structure to CUDA driver memory
	/// access control API functions
	operator CUmemAccess_flags() const noexcept
	{
		return read ?
			   (write ? CU_MEM_ACCESS_FLAGS_PROT_READWRITE : CU_MEM_ACCESS_FLAGS_PROT_READ) :
			   CU_MEM_ACCESS_FLAGS_PROT_NONE;
	}

	/// A named constructor idiom for access permissions
	static access_permissions_t from_access_flags(CUmemAccess_flags access_flags)
	{
		access_permissions_t result;
		result.read = (access_flags & CU_MEM_ACCESS_FLAGS_PROT_READ);
		result.write = (access_flags & CU_MEM_ACCESS_FLAGS_PROT_READWRITE);
		return result;
	}

	/// A named constructor idiom for allowing both reading and writing
	static constexpr access_permissions_t read_and_write()
	{
		return access_permissions_t{ read_enabled, write_enabled };
	}
};

namespace physical_allocation {

/// The different kinds of memory handles the CUDA driver recognizes, and can possibly
/// utilize
enum class shared_handle_kind_t : ::std::underlying_type<CUmemAllocationHandleType>::type {
#if CUDA_VERSION >= 11020
	no_export             = CU_MEM_HANDLE_TYPE_NONE,
#endif
	posix_file_descriptor = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
	win32_handle          = CU_MEM_HANDLE_TYPE_WIN32,
	win32_kmt             = CU_MEM_HANDLE_TYPE_WIN32_KMT,
};

namespace detail_ {

template<shared_handle_kind_t SharedHandleKind> struct shared_handle_type_helper;

template <> struct shared_handle_type_helper<shared_handle_kind_t::posix_file_descriptor> { using type = int; };
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
template <> struct shared_handle_type_helper<shared_handle_kind_t::win32_handle> { using type = void *; };
#endif
// TODO: What about WIN32_KMT?

} // namespace detail_

/// The raw handle for different kinds of shared memory the CUDA driver recognizes
template<shared_handle_kind_t SharedHandleKind>
using shared_handle_t = typename detail_::shared_handle_type_helper<SharedHandleKind>::type;

} // namespace physical_allocation
#endif // CUDA_VERSION >= 10020
#if CUDA_VERSION >= 11020

/// pool-based allocation functionality
namespace pool {

/// Raw CUDA driver handle for a memory pool; avoid using these and prefer @ref memory::pool_t}.
using handle_t = CUmemoryPool;

/// The different kinds of memory handles the CUDA driver recognizes, and can
/// possibly be used with pool allocation
using shared_handle_kind_t = physical_allocation::shared_handle_kind_t;

/// The raw handle for different kinds of shared memory the CUDA driver recognizes, for
/// possible use with memory pools
using physical_allocation::shared_handle_t;

namespace ipc {

/// The concrete value passed between processes, used to tell the CUDA Runtime API which
/// pool-allocated memory area is desired.
using ptr_handle_t = CUmemPoolPtrExportData;

} // namespace ipc

} // namespace pool
#endif // CUDA_VERSION >= 11020

namespace pointer {

/// Raw CUDA driver choice type for attributes of pointers
using attribute_t = CUpointer_attribute;

} // namespace pointer

namespace device {

/// The numeric type which can represent the range of memory addresses on a CUDA device.
/// As these addresses are typically just part of the single, unified all-system memory
/// space, this should be the same type as a system memory address' numeric equivalent.

using address_t = CUdeviceptr;

static_assert(sizeof(void *) == sizeof(device::address_t), "Unexpected address size");

/**
 * @returns a cast of a proper pointer into a numeric address in device memory space
 * (which is usually just a part of the unified all-system memory space)
 *
 * @note Typically, this is just a reinterpretation of the same value.
 */
inline address_t address(const void* device_ptr) noexcept
{
	static_assert(sizeof(void*) == sizeof(address_t), "Incompatible sizes for a void pointer and memory::device::address_t");
	return reinterpret_cast<address_t>(device_ptr);
}

/**
 * @returns The numeric address of the beginning of a memory region
 *
 * @note Typically, this is just a reinterpretation of the same value.
 */
inline address_t address(memory::const_region_t region) noexcept { return address(region.start()); }

} // namespace device

/// @returns a cast of a numeric address in device memory space (which, in recent CUDA
/// versions, is just a part of the unified all-system memory space) into a proper
/// pointer.
inline void* as_pointer(device::address_t address) noexcept
{
	static_assert(sizeof(void*) == sizeof(device::address_t), "Incompatible sizes for a void pointer and memory::device::address_t");
	return reinterpret_cast<void*>(address);
}

/**
 * A memory space whose contents is shared by all threads in a CUDA kernel block,
 * but specific to each kernel block separately. Shared memory is stored in a special
 * area in each of a GPU's individual physical cores (symmetric multiprocessors, or
 * SM's, in NVIDIA parlance) - which is also the area used for L1 cache. One may
 * therefore think of it as L1-cache-like memory which holds arbitrary data rather
 * than a cached copy of any global memory locations. It is only usable in
 * device-side (= kernel) code, but control and inspection of its size is part of
 * the CUDA API functionality.
 */
namespace shared {

/**
 * Each physical core ("Symmetric Multiprocessor") on an nVIDIA GPU has a space
 * of shared memory (see
 * <a href="https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/">this blog entry</a>
 * ). This type is large enough to hold its size.
 *
 * @note actually, uint16_t is usually large enough to hold the shared memory
 * size (as of Volta/Turing architectures), but there are exceptions to this rule,
 * so we have to go with the next smallest.
 *
 * @todo consider using uint32_t.
 */
using size_t = unsigned;

} // namespace shared

/**
 * Paged memory accessible in both device-side and host-side code by triggering transfers
 * of pages between physical system memory and physical device memory.
 *
 * This type of memory, also known as _unified_ memory, appears within a unified, all-system
 * address space - and is used with the same address range on the host and on all relevant
 * CUDA devices on a system. It is paged, so that it may exceed the physical size of a CUDA
 * device's global memory. The CUDA driver takes care of "swapping" pages "out" from a device
 * to host memory or "swapping" them back "in", as well as of propagation of changes between
 * devices and host-memory.
 *
 * @note For more details, see
 * <a href="https://devblogs.nvidia.com/parallelforall/unified-memory-cuda-beginners/">
 * Unified Memory for CUDA Beginners</a> on the
 * <a href="https://devblogs.nvidia.com/parallelforall">Parallel4All blog</a>.
 */
namespace managed {

/// The choices of which categories CUDA devices must a managed memory region be visible to
enum class initial_visibility_t {
	to_all_devices,
	to_supporters_of_concurrent_managed_access,
};

} // namespace managed

#if CUDA_VERSION >= 11070
/// Memory barriers can apply to different scops of scheduled work which must reach it before
/// continuing
enum class barrier_scope_t : typename ::std::underlying_type<CUstreamMemoryBarrier_flags>::type {
	/// All wortk on
	device = CU_STREAM_MEMORY_BARRIER_TYPE_GPU,
	system = CU_STREAM_MEMORY_BARRIER_TYPE_SYS
};
#endif // CUDA_VERSION >= 11700

#if CUDA_VERSION >= 10000
/// Representation of memory resources external to CUDA
namespace external {

/// Raw CUDA driver handle for an external memory resource represented for the driver
using handle_t = CUexternalMemory;

/**
 * Describes a subregion with the context of a larger (memory) region
 */
struct subregion_spec_t {
	size_t offset;
	size_t size;
};

} // namespace external

#endif // CUDA_VERSION >= 10000

} // namespace memory

/**
 * Holds the parameters necessary to "launch" a CUDA kernel (i.e. schedule it for
 * execution on some stream of some device).
 */
struct launch_configuration_t;

/**
 * @brief L1-vs-shared-memory balance option
 *
 * In some GPU micro-architectures, it's possible to have the multiprocessors
 * change the balance in the allocation of L1-cache-like resources between
 * actual L1 cache and shared memory; these are the possible choices.
 */
enum class multiprocessor_cache_preference_t : ::std::underlying_type<CUfunc_cache_enum>::type {
	/** No preference for more L1 cache or for more shared memory; the API can do as it please */
	no_preference                 = CU_FUNC_CACHE_PREFER_NONE,
	/** Divide the cache resources equally between actual L1 cache and shared memory */
	equal_l1_and_shared_memory    = CU_FUNC_CACHE_PREFER_EQUAL,
	/** Divide the cache resources to maximize available shared memory at the expense of L1 cache */
	prefer_shared_memory_over_l1  = CU_FUNC_CACHE_PREFER_SHARED,
	/** Divide the cache resources to maximize available L1 cache at the expense of shared memory */
	prefer_l1_over_shared_memory  = CU_FUNC_CACHE_PREFER_L1,
	// aliases
	none                          = no_preference,
	equal                         = equal_l1_and_shared_memory,
	prefer_shared                 = prefer_shared_memory_over_l1,
	prefer_l1                     = prefer_l1_over_shared_memory,
};

/**
 * A physical core (SM)'s shared memory has multiple "banks"; at most
 * one datum per bank may be accessed simultaneously, while data in
 * different banks can be accessed in parallel. The number of banks
 * and bank sizes differ for different GPU architecture generations;
 * but in some of them (e.g. Kepler), they are configurable - and you
 * can trade the number of banks for bank size, in case that makes
 * sense for your data access pattern - by using
 * @ref device_t::shared_memory_bank_size .
 */
enum multiprocessor_shared_memory_bank_size_option_t
	: ::std::underlying_type<CUsharedconfig>::type
{
	device_default       = CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE,
	four_bytes_per_bank  = CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE,
	eight_bytes_per_bank = CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE
};

/**
 * @brief Definitions and functionality related to CUDA devices (not
 * including the device wrapper type @ref cuda::device_t itself)
 */
namespace device {

/**
 * @brief Numeric ID of a CUDA device used by the CUDA Runtime API.
 *
 * @note at the time of writing and the foreseeable future, this
 * type should be an int.
 */
using id_t               = CUdevice;

/**
 * CUDA devices have both "attributes" and "properties". This is the
 * type for attribute identifiers/indices.
 */
using attribute_t        = CUdevice_attribute;
/**
 * All CUDA device attributes (@ref cuda::device::attribute_t) have a value of this type.
 */
using attribute_value_t  = int;

namespace peer_to_peer {

/**
 * While Individual CUDA devices have individual "attributes" (@ref attribute_t),
 * there are also attributes characterizing pairs; this type is used for
 * identifying/indexing them.
 */
using attribute_t = CUdevice_P2PAttribute;

} // namespace peer_to_peer

} // namespace device

namespace context {

/// Raw CUDA driver handle for a context; see {@ref context_t}.
using handle_t = CUcontext;

using flags_t = unsigned;

/// Scheduling policies the CUDA driver may use when the host-side thread it is
/// running in needs to wait for results from a certain device or context.
enum host_thread_sync_scheduling_policy_t : unsigned int {

	/**
	 * @brief Default behavior; yield or spin based on a heuristic.
	 *
	 * The default value if the flags parameter is zero, uses a heuristic
	 * based on the number of active CUDA contexts in the process C and
	 * the number of logical processors in the system P. If C > P, then
	 * CUDA will yield to other OS threads when waiting for the device,
	 * otherwise CUDA will not yield while waiting for results and
	 * actively spin on the processor.
	 */
	heuristic = CU_CTX_SCHED_AUTO,

	/**
	 * @brief Alias for the default behavior; see @ref heuristic .
	 */
	default_ = heuristic,

	/**
	 * @brief Keep control and spin-check for result availability
	 *
	 * Instruct CUDA to actively spin when waiting for results from the
	 * device. This can decrease latency when waiting for the device, but
	 * may lower the performance of CPU threads if they are performing
	 * work in parallel with the CUDA thread.
	 *
	 */
	spin      = CU_CTX_SCHED_SPIN,

	/**
	 * @brief Block the thread until results are available.
	 *
	 * Instruct CUDA to block the CPU thread on a synchronization
	 * primitive when waiting for the device to finish work.
	 */
	block     = CU_CTX_SCHED_BLOCKING_SYNC,

	/**
	 * @brief Yield control while waiting for results.
	 *
	 * Instruct CUDA to yield its thread when waiting for results from
	 * the device. This can increase latency when waiting for the
	 * device, but can increase the performance of CPU threads
	 * performing work in parallel with the device.
	 *
	 */
	yield     = CU_CTX_SCHED_YIELD,

	/** see @ref heuristic */
	automatic = heuristic,
};

} // namespace context

namespace device {

using flags_t = context::flags_t;

namespace primary_context {

/// Raw CUDA driver handle for a device's primary context
using handle_t = cuda::context::handle_t;

} // namespace primary_context

/// @copydoc context::host_thread_sync_scheduling_policy_t
using host_thread_sync_scheduling_policy_t = context::host_thread_sync_scheduling_policy_t;

} // namespace device

namespace detail_ {

template <typename T, typename U>
inline T identity_cast(U&& x)
{
	static_assert(::std::is_same<
			typename ::std::remove_reference<T>::type,
			typename ::std::remove_reference<U>::type
		>::value,
		"Casting to a different type - don't use identity_cast");
	return static_cast<T>(::std::forward<U>(x));
}

} // namespace detail_

/// The CUDA-driver-specific representation of a UUID value; see also {@ref device_t::uuid()}
using uuid_t = CUuuid;

namespace module {

/// Raw CUDA driver handle of a module of compiled code; see @ref module_t
using handle_t = CUmodule;

} // namespace module

namespace kernel {

/// Raw CUDA driver selector of a kernel attribute
using attribute_t = CUfunction_attribute;

/// The uniform type the CUDA driver uses for all kernel attributes; it is typically more
/// appropriate to use @ref cuda::kernel_t methods, which also employ more specific,
/// appropriate types.
using attribute_value_t = int;

// A raw CUDA driver handle for a kernel; prefer using the @ref cuda::kernel_t type.
using handle_t = CUfunction;

} // namespace kernel

} // namespace cuda

#endif // CUDA_API_WRAPPERS_COMMON_TYPES_HPP_
