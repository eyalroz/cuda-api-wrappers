/**
 * @file types.h
 *
 * @brief Fundamental, plain-old-data, CUDA-related type definitions.

 * This is a common file for all definitions fundamental,
 * plain-old-data, CUDA-related types - those with no methods, or with
 * only constexpr methods (which, specifically, do not involve making
 * any Runtime API calls).
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_TYPES_H_
#define CUDA_API_WRAPPERS_TYPES_H_

#ifndef __CUDACC__
#include <builtin_types.h>
#include <driver_types.h>
#endif

#include <type_traits>

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
using status_t                = cudaError_t;


/**
 * CUDA kernels are launched in grids of blocks of threads, in 3 dimensions.
 * In each of these, the numbers of blocks per grid is specified in this type.
 *
 * @note Theoretically, CUDA could split the type for blocks per grid and
 * threads per block, but for now they're the same.
 */
using grid_dimension_t        = decltype(dim3::x);

/**
 * CUDA kernels are launched in grids of blocks of threads, in 3 dimensions.
 * In each of these, the number of threads per block is specified in this type.
 *
 * @note Theoretically, CUDA could split the type for blocks per grid and
 * threads per block, but for now they're the same.
 */
using grid_block_dimension_t  = grid_dimension_t;


namespace event {
using id_t              = cudaEvent_t;
} // namespace event

namespace stream {
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
	default_priority   = 0,
	unbounded_priority = -1 };

} // namespace stream

/**
 * A richer (kind-of-a-)wrapper for CUDA's @ref dim3 class, used
 * to specify dimensions for blocks and grid (up to 3 dimensions).
 *
 * @note Unfortunately, dim3 does not have constexpr methods -
 * preventing us from having constexpr methods here.
 */
struct dimensions_t // this almost-inherits dim3
{
	grid_dimension_t x, y, z;
    constexpr __host__ __device__ dimensions_t(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1)
    : x(x_), y(y_), z(z_) {}

    __host__ __device__ constexpr dimensions_t(const uint3& v) : dimensions_t(v.x, v.y, v.z) { }
    __host__ __device__ constexpr dimensions_t(const dim3& dims) : dimensions_t(dims.x, dims.y, dims.z) { }

    __host__ __device__ constexpr operator uint3(void) const { return { x, y, z }; }
    __host__ __device__ operator dim3(void) const { return { x, y, z }; }

    __host__ __device__ inline size_t volume() const { return (size_t) x * y * z; }
    __host__ __device__ inline bool empty() const {	return volume() == 0; }
    __host__ __device__ inline unsigned char dimensionality() const
	{
		return ((z > 1) + (y > 1) + (x > 1)) * (!empty());
	}
};

constexpr inline bool operator==(const dim3& lhs, const dim3& rhs)
{
	return lhs.x == rhs.x and lhs.y == rhs.y and lhs.z == rhs.z;
}
constexpr inline bool operator==(const dimensions_t& lhs, const dimensions_t& rhs)
{
	return lhs.x == rhs.x and lhs.y == rhs.y and lhs.z == rhs.z;
}

/**
 * CUDA kernels are launched in grids of blocks of threads. This expresses the
 * dimensions of a grid, in terms of blocks.
 */
using grid_dimensions_t       = dimensions_t;

/**
 * CUDA kernels are launched in grids of blocks of threads. This expresses the
 * dimensions of a block within such a grid, in terms of threads.
 */
using grid_block_dimensions_t = dimensions_t;


/**
 * Each physical core ("Symmetric Multiprocessor") on an nVIDIA GPU has a space
 * of shared memory (see
 * @link https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
 * ). This type is large enough to hold its size.
 */
using shared_memory_size_t    = unsigned short;
	// Like size_t, but for shared memory spaces, which, currently in nVIDIA (and AMD)
	// GPUs are sized at no more than 64 KiB. Note that using this for computations
	// might not be the best idea since there are penalties for sub-32-bit computation
	// sometimes

/**
 * Holds the parameters necessary to "launch" a CUDA kernel (i.e. schedule it for
 * execution on some stream of some device).
 */
typedef struct {
	grid_dimensions_t       grid_dimensions;
	grid_block_dimensions_t block_dimensions;
	shared_memory_size_t    dynamic_shared_memory_size; // in bytes
} launch_configuration_t;

inline launch_configuration_t make_launch_config(
	grid_dimensions_t       grid_dimensions,
	grid_block_dimensions_t block_dimensions,
	shared_memory_size_t    dynamic_shared_memory_size = 0)
{
	return { grid_dimensions, block_dimensions, dynamic_shared_memory_size };
}

inline bool operator==(const launch_configuration_t lhs, const launch_configuration_t& rhs)
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
	: std::underlying_type<cudaSharedMemConfig>::type
{
    device_default       = cudaSharedMemBankSizeDefault,
    four_bytes_per_bank  = cudaSharedMemBankSizeFourByte,
    eight_bytes_per_bank = cudaSharedMemBankSizeEightByte
};

namespace device {

/**
 * @brief Numeric ID of a CUDA device used by the CUDA Runtime API.
 */
using id_t               = int;

/**
 * CUDA devices have both "attributes" and "properties". This is the
 * type for attribute identifiers/indices, aliasing {@ref cudaDeviceAttr}.
 */
using attribute_t        = enum cudaDeviceAttr;
/**
 * All CUDA device attributes (@ref attribute_t) have a value of this type.
 */
using attribute_value_t  = int;

/**
 * While Individual CUDA devices have individual "attributes" (@ref attribute_t),
 * there are also attributes characterizing pairs; this type is used for
 * identifying/indexing them, aliasing {@ref cudaDeviceP2PAttr}.
 */
using pair_attribute_t   = cudaDeviceP2PAttr;

} // namespace device

namespace detail {

enum : bool {
	assume_device_is_current        = true,
	do_not_assume_device_is_current = false
};

} // namespace detail

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

} // namespace cuda

#endif /* CUDA_API_WRAPPERS_TYPES_H_ */
