/**
 * @file types.hpp
 *
 * @brief Fundamental CUDA-related type definitions.

 * This is a common file for all definitions of fundamental CUDA-related types,
 * some shared by different APIs.
 *
 * @note Most types here are defined using "Runtime API terminology", but this is
 * inconsequential, as the corresponding Driver API types are merely aliases of
 * them. For example, in CUDA's own header files, we have:
 *
 *   typedef CUevent_st * CUevent
 *   typedef CUevent_st * cudaEvent_t
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_COMMON_TYPES_HPP_
#define CUDA_API_WRAPPERS_COMMON_TYPES_HPP_

#if (__cplusplus < 201103L && (!defined(_MSVC_LANG) || _MSVC_LANG < 201103L))
#error "The CUDA Runtime API headers can only be compiled with C++11 or a later version of the C++ language standard"
#endif

#ifndef __CUDACC__
#include <builtin_types.h>
#endif

#include <type_traits>
#include <cassert>
#include <cstddef> // for ::std::size_t
#include <cstdint>

#ifndef __CUDACC__
#ifndef __device__
#define __device__
#define __host__
#endif
#endif

#ifndef CPP14_CONSTEXPR
#if __cplusplus >= 201402L
#define CPP14_CONSTEXPR constexpr
#else
#define CPP14_CONSTEXPR
#endif
#endif


#ifdef _MSC_VER
/*
 * Microsoft Visual C++ (upto v2017) does not support the C++
 * keywords `and`, `or` and `not`. Apparently, the following
 * include is a work-around.
 */
#include <ciso646>
#endif

/**
 * @brief All definitions and functionality wrapping the CUDA Runtime API.
 */
namespace cuda {

/*
 * The 3 ID types - for devices, streams and events - in this file are just
 * numeric identifiers (mostly useful for breaking dependencies and for
 * interaction with code using the original CUDA APIs); we also have wrapper
 * classes for them (each constructible by the corresponding numeric ID type)
 * which allow convenient access to their related functionality: these are
 * @ref cuda::device_t, @ref cuda::stream_t and @ref cuda::event_t.
 *
 * TODO: Perhaps we should call them cuda::device::type,
 * cuda::stream::type and cuda::event::type,
 */


/**
 * Indicates either the result (success or error index) of a CUDA Runtime API call,
 * or the overall status of the Runtime API (which is typically the last triggered
 * error).
 */
using status_t = cudaError_t;

using size_t = ::std::size_t;

/**
 * The index or number of dimensions of an entity (as opposed to the extent in any
 * dimension) - typically just 0, 1, 2 or 3.
 */
using dimensionality_t = unsigned;

namespace array {

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

/**
 * Dimensions for 3D CUDA arrays
 */
template<>
struct dimensions_t<3> // this almost-inherits cudaExtent
{
	dimension_t width, height, depth;

	constexpr __host__ __device__ dimensions_t(dimension_t width_, dimension_t height_, dimension_t depth_)
		: width(width_), height(height_), depth(depth_) { }
	constexpr __host__ __device__ dimensions_t(cudaExtent e)
		: dimensions_t(e.width, e.height, e.depth) { }
	constexpr __host__ __device__ dimensions_t(const dimensions_t& other)
		: dimensions_t(other.width, other.height, other.depth) { }
	constexpr __host__ __device__ dimensions_t(dimensions_t&& other)
		: dimensions_t(other.width, other.height, other.depth) { }

	CPP14_CONSTEXPR __host__ __device__ dimensions_t& operator=(const dimensions_t& other)
	{
		width = other.width; height = other.height; depth = other.depth;
		return *this;

	}
	CPP14_CONSTEXPR __host__ __device__ dimensions_t& operator=(dimensions_t&& other)
	{
		width = other.width; height = other.height; depth = other.depth;
		return *this;
	}

	constexpr __host__ __device__ operator cudaExtent(void) const
	{
		return { width, height, depth };
			// Note: We're not using make_cudaExtent here because:
			// 1. It's not constexpr and
			// 2. It doesn't do anything except construct the plain struct - as of CUDA 10 at least
	}

	constexpr __host__ __device__ size_t volume() const { return width * height * depth; }
	constexpr __host__ __device__ size_t size() const { return volume(); }
	constexpr __host__ __device__ dimensionality_t dimensionality() const
	{
		return ((width > 1) + (height> 1) + (depth > 1));
	}

	// Named constructor idioms

	static constexpr __host__ __device__ dimensions_t cube(dimension_t x)   { return dimensions_t{ x, x, x }; }
};

/**
 * Dimensions for 2D CUDA arrays
 */
template<>
struct dimensions_t<2>
{
	dimension_t width, height;

	constexpr __host__ __device__ dimensions_t(dimension_t width_, dimension_t height_)
		: width(width_), height(height_) { }
	constexpr __host__ __device__ dimensions_t(const dimensions_t& other)
		: dimensions_t(other.width, other.height) { }
	constexpr __host__ __device__ dimensions_t(dimensions_t&& other)
		: dimensions_t(other.width, other.height) { }

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

	constexpr __host__ __device__ size_t area() const { return width * height; }
	constexpr __host__ __device__ size_t size() const { return area(); }
	constexpr __host__ __device__ dimensionality_t dimensionality() const
	{
		return ((width > 1) + (height> 1));
	}

	// Named constructor idioms

	static constexpr __host__ __device__ dimensions_t square(dimension_t x)   { return dimensions_t{ x, x }; }
};

} // namespace array

/**
 * @brief Definitions and functionality related to CUDA events (not
 * including the event wrapper type @ref event_t itself)
 */
namespace event {

/**
 * The CUDA Runtime API's numeric handle for events
 */
using id_t              = cudaEvent_t;
} // namespace event

/**
 * @brief Definitions and functionality related to CUDA streams (not
 * including the device wrapper type @ref stream_t itself)
 */

namespace stream {

/**
 * The CUDA Runtime API's numeric handle for streams
 */
using id_t             = cudaStream_t;

/**
 * CUDA streams have a scheduling priority, with lower values meaning higher priority.
 * The types represents a larger range of values than those actually used; they can
 * be obtained by @ref device_t::stream_priority_range() .
 */
using priority_t       = int;
enum : priority_t {
	/**
	 * the scheduling priority of a stream created without specifying any other priority
	 * value
	 */
	default_priority   = 0
};

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
 * are "usable" even though this type is unsigned.
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

    constexpr __host__ __device__ size_t volume() const { return (size_t) x * y * z; }
    constexpr __host__ __device__ dimensionality_t dimensionality() const
	{
		return ((z > 1) + (y > 1) + (x > 1));
	}

	// Named constructor idioms

	static constexpr __host__ __device__ dimensions_t cube(dimension_t x)   { return dimensions_t{ x, x, x }; }
	static constexpr __host__ __device__ dimensions_t square(dimension_t x) { return dimensions_t{ x, x, 1 }; }
	static constexpr __host__ __device__ dimensions_t line(dimension_t x)   { return dimensions_t{ x, 1, 1 }; }
	static constexpr __host__ __device__ dimensions_t point()               { return dimensions_t{ 1, 1, 1 }; }
};

///@cond
constexpr inline bool operator==(const dim3& lhs, const dim3& rhs) noexcept
{
	return lhs.x == rhs.x and lhs.y == rhs.y and lhs.z == rhs.z;
}
constexpr inline bool operator==(const dimensions_t& lhs, const dimensions_t& rhs) noexcept
{
	return lhs.x == rhs.x and lhs.y == rhs.y and lhs.z == rhs.z;
}
///@endcond


/**
 * CUDA kernels are launched in grids of blocks of threads. This expresses the
 * dimensions of a block within such a grid, in terms of threads.
 */
using block_dimensions_t = dimensions_t;

} // namespace grid

/**
 * @brief Management and operations on memory in different CUDA-recognized
 * spaces.
 */
namespace memory {
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
} // namespace memory

/**
 * Holds the parameters necessary to "launch" a CUDA kernel (i.e. schedule it for
 * execution on some stream of some device).
 */
struct launch_configuration_t {
	grid::dimensions_t        grid_dimensions { 0 }; /// in blocks
	grid::block_dimensions_t  block_dimensions { 0 }; /// in threads
	memory::shared::size_t    dynamic_shared_memory_size { 0u };
		/// ... in bytes per block

	// In C++11, an inline initializer for a struct's field costs us a lot
	// of its defaulted constructors; but - we must initialize the shared
	// memory size to 0, as otherwise, people might be tempted to initialize
	// a launch configuration with { num_blocks, num_threads } - and get an
	// uninitialized shared memory size which they did not expect. So,
	// we do have the inline initializers above regardless of the language
	// standard version, and we just have to "pay the price" of spelling things out:
	launch_configuration_t() = delete;
	constexpr launch_configuration_t(const launch_configuration_t&) = default;
	constexpr launch_configuration_t(launch_configuration_t&&) = default;

	constexpr launch_configuration_t(
		grid::dimensions_t grid_dims,
		grid::dimensions_t block_dims,
		memory::shared::size_t dynamic_shared_mem = 0u
	) :
		grid_dimensions(grid_dims),
		block_dimensions(block_dims),
		dynamic_shared_memory_size(dynamic_shared_mem)
	{ }

	// A "convenience" delegating ctor to avoid narrowing-conversion warnings
	constexpr launch_configuration_t(
		int grid_dims,
		int block_dims,
		memory::shared::size_t dynamic_shared_mem = 0u
	) : launch_configuration_t(grid::dimensions_t(grid_dims), grid::dimensions_t(block_dims), dynamic_shared_mem)
	{ }

	/**
	 * @brief The overall dimensions, in thread, of the launch grid
	 */
	constexpr grid::dimensions_t combined_grid_dimensions() const {
		return {
			block_dimensions.x * grid_dimensions.x,
			block_dimensions.y * grid_dimensions.y,
			block_dimensions.z * grid_dimensions.z
		};
	}
};

/**
 * @brief a named constructor idiom for a @ref launch_config_t
 */
constexpr inline launch_configuration_t make_launch_config(
	grid::dimensions_t        grid_dimensions,
	grid::block_dimensions_t  block_dimensions,
	memory::shared::size_t    dynamic_shared_memory_size = 0u) noexcept
{
	return cuda::launch_configuration_t{ grid_dimensions, block_dimensions, dynamic_shared_memory_size };
}

constexpr inline bool operator==(const launch_configuration_t lhs, const launch_configuration_t& rhs) noexcept
{
	return
		lhs.grid_dimensions    == rhs.grid_dimensions    and
		lhs.block_dimensions   == rhs.block_dimensions   and
		lhs.dynamic_shared_memory_size == rhs.dynamic_shared_memory_size;
}

/**
 * @brief L1-vs-shared-memory balance option
 *
 * In some GPU micro-architectures, it's possible to have the multiprocessors
 * change the balance in the allocation of L1-cache-like resources between
 * actual L1 cache and shared memory; these are the possible choices.
 */
enum class multiprocessor_cache_preference_t {
	/** No preference for more L1 cache or for more shared memory; the API can do as it please */
	no_preference                 = cudaFuncCachePreferNone,
	/** Divide the cache resources equally between actual L1 cache and shared memory */
	equal_l1_and_shared_memory    = cudaFuncCachePreferEqual,
	/** Divide the cache resources to maximize available shared memory at the expense of L1 cache */
	prefer_shared_memory_over_l1  = cudaFuncCachePreferShared,
	/** Divide the cache resources to maximize available L1 cache at the expense of shared memory */
	prefer_l1_over_shared_memory  = cudaFuncCachePreferL1,
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
	: ::std::underlying_type<cudaSharedMemConfig>::type
{
	device_default       = cudaSharedMemBankSizeDefault,
	four_bytes_per_bank  = cudaSharedMemBankSizeFourByte,
	eight_bytes_per_bank = cudaSharedMemBankSizeEightByte
};

/**
 * @brief Definitions and functionality related to CUDA devices (not
 * including the device wrapper type @ref device_t itself)
 */
namespace device {

/**
 * @brief Numeric ID of a CUDA device used by the CUDA Runtime API.
 */
using id_t               = int;

/**
 * CUDA devices have both "attributes" and "properties". This is the
 * type for attribute identifiers/indices, aliasing @ref cudaDeviceAttr.
 */
using attribute_t        = cudaDeviceAttr;
/**
 * All CUDA device attributes (@ref cuda::device::attribute_t) have a value of this type.
 */
using attribute_value_t  = int;

/**
 * While Individual CUDA devices have individual "attributes" (@ref attribute_t),
 * there are also attributes characterizing pairs; this type is used for
 * identifying/indexing them, aliasing `cudaDeviceP2PAttr`.
 */
using pair_attribute_t   = cudaDeviceP2PAttr;

} // namespace device

namespace detail_ {

/**
 * @brief adapt a type to be usable as a kernel parameter.
 *
 * CUDA kernels don't accept just any parameter type a C++ function may accept.
 * Specifically: No references, arrays decay (IIANM) and functions pass by address.
 * However - not all "decaying" of `::std::decay` is necessary. Such transformation
 * can be effected by this type-trait struct.
 */
template<typename P>
struct kernel_parameter_decay {
private:
    typedef typename ::std::remove_reference<P>::type U;
public:
    typedef typename ::std::conditional<
        ::std::is_array<U>::value,
        typename ::std::remove_extent<U>::type*,
        typename ::std::conditional<
            ::std::is_function<U>::value,
            typename ::std::add_pointer<U>::type,
            U
        >::type
    >::type type;
};

template<typename P>
using kernel_parameter_decay_t = typename kernel_parameter_decay<P>::type;

} // namespace detail_

/**
 * Scheduling policies the Runtime API may use when the host-side
 * thread it is running in needs to wait for results from a certain
 * device
 */
enum host_thread_synch_scheduling_policy_t : unsigned int {
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
	heuristic = cudaDeviceScheduleAuto,
	/**
	 * @brief Keep control and spin-check for result availability
	 *
	 * Instruct CUDA to actively spin when waiting for results from the
	 * device. This can decrease latency when waiting for the device, but
	 * may lower the performance of CPU threads if they are performing
	 * work in parallel with the CUDA thread.
	 *
	 */
	spin      = cudaDeviceScheduleSpin,
	/**
	 * @brief Yield control while waiting for results.
	 *
	 * Instruct CUDA to yield its thread when waiting for results from
	 * the device. This can increase latency when waiting for the
	 * device, but can increase the performance of CPU threads
	 * performing work in parallel with the device.
	 *
	 */
	block     = cudaDeviceScheduleBlockingSync,
	/**
	 * @brief Block the thread until results are available.
	 *
	 * Instruct CUDA to block the CPU thread on a synchronization
	 * primitive when waiting for the device to finish work.
	 */
	yield     = cudaDeviceScheduleYield,
	/** see @ref heuristic */
	automatic = heuristic,
};

using native_word_t = unsigned;

/**
 * Object-code symbols
 */
struct symbol_t {
	const void* handle;
};

} // namespace cuda

#ifndef __CUDACC__
#ifndef __device__
#define __device__
#define __host__
#endif
#endif

#endif // CUDA_API_WRAPPERS_COMMON_TYPES_HPP_
