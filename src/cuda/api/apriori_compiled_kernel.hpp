/**
 * @file
 *
 * @brief An implementation of a subclass of @ref `kernel_t` for kernels
 * compiled together with the host-side program.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_APRIORI_COMPILED_KERNEL_HPP_
#define CUDA_API_WRAPPERS_APRIORI_COMPILED_KERNEL_HPP_

#include "kernel.hpp"
#include "current_context.hpp"

// The following is needed for occupancy-related calculation convenience
// and kernel-attribute-related API functions
#include <cuda_runtime.h>

#include <type_traits>

namespace cuda {

///@cond
class device_t;
class apriori_compiled_kernel_t;
///@nocond

namespace kernel {

namespace detail_ {

#if CUDA_VERSION < 11000
inline handle_t get_handle(const void *, const char* = nullptr)
{
	throw cuda::runtime_error(status::not_supported,
		"Only CUDA versions 11.0 and later support obtaining CUDA driver handles "
		"for kernels compiled alongside the program source");
}
#else
inline handle_t get_handle(const void *kernel_function_ptr, const char* name = nullptr)
{
	handle_t handle;
	auto status = cudaGetFuncBySymbol(&handle, kernel_function_ptr);
	throw_if_error_lazy(status, "Failed obtaining a CUDA function handle for "
		+ ((name == nullptr) ? ::std::string("a kernel function") : ::std::string("kernel function ") + name)
		+ " at " + cuda::detail_::ptr_as_hex(kernel_function_ptr));
	return handle;
}
#endif

apriori_compiled_kernel_t wrap(
	device::id_t device_id,
	context::handle_t primary_context_handle,
	kernel::handle_t f,
	const void* ptr,
	bool hold_primary_context_refcount_unit = false);


} // namespace detail_

#if ! CAN_GET_APRIORI_KERNEL_HANDLE
/**
 * @brief a wrapper around `cudaFuncAttributes`, offering
 * a few convenience member functions.
 */
struct attributes_t : cudaFuncAttributes {

	cuda::device::compute_capability_t ptx_version() const noexcept {
		return device::compute_capability_t::from_combined_number(ptxVersion);
	}

	cuda::device::compute_capability_t binary_compilation_target_architecture() const noexcept {
		return device::compute_capability_t::from_combined_number(binaryVersion);
	}
};

#endif // CAN_GET_APRIORI_KERNEL_HANDLE

namespace occupancy {

namespace detail_ {

#if CUDA_VERSION < 11000

template<typename UnaryFunction, class T>
static __inline__ cudaError_t cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags_(
	int           *minGridSize,
	int           *blockSize,
	T              func,
	UnaryFunction  blockSizeToDynamicSMemSize,
	int            blockSizeLimit = 0,
	unsigned int   flags = 0)
{
	cudaError_t status;

	// Device and function properties
	int                       device;
	struct cudaFuncAttributes attr;

	// Limits
	int maxThreadsPerMultiProcessor;
	int warpSize;
	int devMaxThreadsPerBlock;
	int multiProcessorCount;
	int funcMaxThreadsPerBlock;
	int occupancyLimit;
	int granularity;

	// Recorded maximum
	int maxBlockSize = 0;
	int numBlocks    = 0;
	int maxOccupancy = 0;

	// Temporary
	int blockSizeToTryAligned;
	int blockSizeToTry;
	int blockSizeLimitAligned;
	int occupancyInBlocks;
	int occupancyInThreads;
	size_t dynamicSMemSize;

	///////////////////////////
	// Check user input
	///////////////////////////

	if (!minGridSize || !blockSize || !func) {
		return cudaErrorInvalidValue;
	}

	//////////////////////////////////////////////
	// Obtain device and function properties
	//////////////////////////////////////////////

	status = ::cudaGetDevice(&device);
	if (status != cudaSuccess) {
		return status;
	}

	status = cudaDeviceGetAttribute(
		&maxThreadsPerMultiProcessor,
		cudaDevAttrMaxThreadsPerMultiProcessor,
		device);
	if (status != cudaSuccess) {
		return status;
	}

	status = cudaDeviceGetAttribute(
		&warpSize,
		cudaDevAttrWarpSize,
		device);
	if (status != cudaSuccess) {
		return status;
	}

	status = cudaDeviceGetAttribute(
		&devMaxThreadsPerBlock,
		cudaDevAttrMaxThreadsPerBlock,
		device);
	if (status != cudaSuccess) {
		return status;
	}

	status = cudaDeviceGetAttribute(
		&multiProcessorCount,
		cudaDevAttrMultiProcessorCount,
		device);
	if (status != cudaSuccess) {
		return status;
	}

	status = cudaFuncGetAttributes(&attr, func);
	if (status != cudaSuccess) {
		return status;
	}

	funcMaxThreadsPerBlock = attr.maxThreadsPerBlock;

	/////////////////////////////////////////////////////////////////////////////////
	// Try each block size, and pick the block size with maximum occupancy
	/////////////////////////////////////////////////////////////////////////////////

	occupancyLimit = maxThreadsPerMultiProcessor;
	granularity    = warpSize;

	if (blockSizeLimit == 0) {
		blockSizeLimit = devMaxThreadsPerBlock;
	}

	if (devMaxThreadsPerBlock < blockSizeLimit) {
		blockSizeLimit = devMaxThreadsPerBlock;
	}

	if (funcMaxThreadsPerBlock < blockSizeLimit) {
		blockSizeLimit = funcMaxThreadsPerBlock;
	}

	blockSizeLimitAligned = ((blockSizeLimit + (granularity - 1)) / granularity) * granularity;

	for (blockSizeToTryAligned = blockSizeLimitAligned; blockSizeToTryAligned > 0; blockSizeToTryAligned -= granularity) {
		// This is needed for the first iteration, because
		// blockSizeLimitAligned could be greater than blockSizeLimit
		//
		if (blockSizeLimit < blockSizeToTryAligned) {
			blockSizeToTry = blockSizeLimit;
		} else {
			blockSizeToTry = blockSizeToTryAligned;
		}

		dynamicSMemSize = blockSizeToDynamicSMemSize(blockSizeToTry);

		status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
			&occupancyInBlocks,
			func,
			blockSizeToTry,
			dynamicSMemSize,
			flags);

		if (status != cudaSuccess) {
			return status;
		}

		occupancyInThreads = blockSizeToTry * occupancyInBlocks;

		if (occupancyInThreads > maxOccupancy) {
			maxBlockSize = blockSizeToTry;
			numBlocks    = occupancyInBlocks;
			maxOccupancy = occupancyInThreads;
		}

		// Early out if we have reached the maximum
		//
		if (occupancyLimit == maxOccupancy) {
			break;
		}
	}

	///////////////////////////
	// Return best available
	///////////////////////////

	// Suggested min grid size to achieve a full machine launch
	//
	*minGridSize = numBlocks * multiProcessorCount;
	*blockSize = maxBlockSize;

	return status;
}

#if CUDA_VERSION > 10000
// Note: If determine_shared_mem_by_block_size is not null, fixed_shared_mem_size is ignored;
// if block_size_limit is 0, it is ignored.
template <typename UnaryFunction>
inline grid::composite_dimensions_t min_grid_params_for_max_occupancy(
	const void*                    kernel_function_ptr,
	cuda::device::id_t             device_id,
	UnaryFunction                  determine_shared_mem_by_block_size,
	cuda::grid::block_dimension_t  block_size_limit,
	bool                           disable_caching_override)
{
	int min_grid_size_in_blocks { 0 };
	int block_size { 0 };
	// Note: only initializing the values her because of a
	// spurious (?) compiler warning about potential uninitialized use.

	unsigned flags = disable_caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault;
	auto result = (cuda::status_t) cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags_<UnaryFunction, const void*>(
		&min_grid_size_in_blocks,
		&block_size,
		kernel_function_ptr,
		determine_shared_mem_by_block_size,
		(int) block_size_limit,
		flags);

	throw_if_error(result,
		"Failed obtaining parameters for a minimum-size grid for " + kernel::detail_::identify(kernel_function_ptr, device_id)
		+ " with maximum occupancy given dynamic shared memory and block size data");
	return { (grid::dimension_t) min_grid_size_in_blocks, (grid::block_dimension_t) block_size };
}
#endif // CUDA_VERSION > 10000

inline grid::dimension_t max_active_blocks_per_multiprocessor(
	const void*              kernel_function_ptr,
	grid::block_dimension_t  block_size_in_threads,
	memory::shared::size_t   dynamic_shared_memory_per_block,
	bool                     disable_caching_override)
{
	// Assuming we don't need to set the current device here
	int result;
	cuda::status_t status = CUDA_SUCCESS;
	auto flags = (unsigned) disable_caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault;
	status = (cuda::status_t) cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
		&result, kernel_function_ptr, (int) block_size_in_threads, (int) dynamic_shared_memory_per_block, flags);
	throw_if_error(status,
		"Determining the maximum occupancy in blocks per multiprocessor, given the block size and the amount of dynamic memory per block");
	return result;
}

#endif

} // namespace detail_

} // namespace occupancy

} // namespace kernel

/**
 * @brief A subclass of the @ref `kernel_t` interface for kernels being
 * functions marked as __global__ in source files and compiled apriori.
 */
class apriori_compiled_kernel_t final : public kernel_t {
public: // getters
	const void *ptr() const noexcept { return ptr_; }
	const void *get() const noexcept { return ptr_; }

public: // type_conversions
	explicit operator const void *() noexcept { return ptr_; }

public: // non-mutators

#if ! CAN_GET_APRIORI_KERNEL_HANDLE
	kernel::attributes_t attributes() const;
	void set_cache_preference(multiprocessor_cache_preference_t preference) const override;
	void set_shared_memory_bank_size(multiprocessor_shared_memory_bank_size_option_t config) const override;

	cuda::device::compute_capability_t ptx_version() const override
	{
		return attributes().ptx_version();
	}

	cuda::device::compute_capability_t binary_compilation_target_architecture() const override
	{
		return attributes().binary_compilation_target_architecture();
	}


	grid::block_dimension_t maximum_threads_per_block() const override
	{
		return attributes().maxThreadsPerBlock;
	}

	void set_attribute(kernel::attribute_t attribute, kernel::attribute_value_t value) const override;

#if CUDA_VERSION > 10000
	grid::composite_dimensions_t min_grid_params_for_max_occupancy(
		memory::shared::size_t dynamic_shared_memory_size = no_dynamic_shared_memory,
		grid::block_dimension_t block_size_limit = 0,
		bool disable_caching_override = false) const override
	{
		auto shared_memory_size_determiner =
			[dynamic_shared_memory_size](int) -> size_t { return dynamic_shared_memory_size; };
		return kernel::occupancy::detail_::min_grid_params_for_max_occupancy(
			ptr(), device_id(),
			shared_memory_size_determiner,
			block_size_limit, disable_caching_override);
	}

	grid::composite_dimensions_t min_grid_params_for_max_occupancy(
		kernel::shared_memory_size_determiner_t shared_memory_size_determiner,
		grid::block_dimension_t block_size_limit = 0,
		bool disable_caching_override = false) const override
	{
		return kernel::occupancy::detail_::min_grid_params_for_max_occupancy(
			ptr(), device_id(),
			shared_memory_size_determiner,
			block_size_limit, disable_caching_override);
	}
#endif

	kernel::attribute_value_t get_attribute(kernel::attribute_t attribute) const override;

	/**
	 * @brief Calculates the number of grid blocks which may be "active" on a given GPU
	 * multiprocessor simultaneously (i.e. with warps from any of these block
	 * being schedulable concurrently)
	 *
	 * @param block_size_in_threads
	 * @param dynamic_shared_memory_per_block
	 * @param disable_caching_override On some GPUs, the choice of whether to
	 * cache memory reads affects occupancy. But what if this caching results in 0
	 * potential occupancy for a kernel? There are two options, controlled by this flag.
	 * When it is set to false - the calculator will assume caching is off for the
	 * purposes of its work; when set to true, it will return 0 for such device functions.
	 * See also the "Unified L1/Texture Cache" section of the
	 * <a href="http://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html">Maxwell
	 * tuning guide</a>.
	 */
	grid::dimension_t max_active_blocks_per_multiprocessor(
		grid::block_dimension_t block_size_in_threads,
		memory::shared::size_t dynamic_shared_memory_per_block,
		bool disable_caching_override = false) const override
	{
		return kernel::occupancy::detail_::max_active_blocks_per_multiprocessor(
			ptr(),
			block_size_in_threads,
			dynamic_shared_memory_per_block,
			disable_caching_override);
	}
#endif // ! CAN_GET_APRIORI_KERNEL_HANDLE

protected: // ctors & dtor
	apriori_compiled_kernel_t(device::id_t device_id, context::handle_t primary_context_handle,
		kernel::handle_t handle, const void *f, bool hold_pc_refcount_unit)
	: kernel_t(device_id, primary_context_handle, handle, hold_pc_refcount_unit), ptr_(f) {
		// TODO: Consider checking whether this actually is a device function, at all and in this context
#ifndef NDEBUG
		assert(f != nullptr && "Attempt to construct a kernel object for a nullptr kernel function pointer");
#endif
	}
	apriori_compiled_kernel_t(
		device::id_t device_id,
		context::handle_t primary_context_handle,
		const void *f,
		bool hold_primary_context_refcount_unit)
	: apriori_compiled_kernel_t(
		device_id,
		primary_context_handle,
		kernel::detail_::get_handle(f),
		f,
		hold_primary_context_refcount_unit)
	{ }

public: // ctors & dtor
	apriori_compiled_kernel_t(const apriori_compiled_kernel_t&) = default;
	apriori_compiled_kernel_t(apriori_compiled_kernel_t&&) = default;

public: // friends
	friend apriori_compiled_kernel_t kernel::detail_::wrap(device::id_t, context::handle_t, kernel::handle_t, const void*, bool);

protected: // data members
	const void *const ptr_;
};

namespace kernel {
namespace detail_ {

inline apriori_compiled_kernel_t wrap(
	device::id_t       device_id,
	context::handle_t  primary_context_handle,
	kernel::handle_t   f,
	const void *       ptr,
	bool               hold_primary_context_refcount_unit)
{
	return { device_id, primary_context_handle, f, ptr, hold_primary_context_refcount_unit };
}

#if ! CAN_GET_APRIORI_KERNEL_HANDLE
inline ::std::string identify(const apriori_compiled_kernel_t& kernel)
{
	return "apriori-compiled kernel " + cuda::detail_::ptr_as_hex(kernel.ptr())
		+ " in " + context::detail_::identify(kernel.context());
}
#endif // ! CAN_GET_APRIORI_KERNEL_HANDLE

} // namespace detail

#if CAN_GET_APRIORI_KERNEL_HANDLE
inline attribute_value_t get_attribute(const void* function_ptr, attribute_t attribute)
{
	auto handle = detail_::get_handle(function_ptr);
	return kernel::detail_::get_attribute_in_current_context(handle, attribute);
}

inline void set_attribute(const void* function_ptr, attribute_t attribute, attribute_value_t value)
{
	auto handle = detail_::get_handle(function_ptr);
	return detail_::set_attribute_in_current_context(handle, attribute, value);
}

inline attribute_value_t get_attribute(
	const context_t&  context,
	const void*       function_ptr,
	attribute_t       attribute)
{
	CAW_SET_SCOPE_CONTEXT(context.handle());
	return get_attribute(function_ptr, attribute);
}

inline void set_attribute(
	const context_t&   context,
	const void*        function_ptr,
	attribute_t        attribute,
	attribute_value_t  value)
{
	CAW_SET_SCOPE_CONTEXT(context.handle());
	return set_attribute(function_ptr, attribute, value);
}
#endif // CAN_GET_APRIORI_KERNEL_HANDLE

/**
 * @note The returned kernel proxy object will keep the device's primary
 * context active while the kernel exists.
 */
template<typename KernelFunctionPtr>
apriori_compiled_kernel_t get(const device_t& device, KernelFunctionPtr function_ptr);

template<typename KernelFunctionPtr>
apriori_compiled_kernel_t get(context_t context, KernelFunctionPtr function_ptr);

} // namespace kernel

} // namespace cuda

#endif // CUDA_API_WRAPPERS_APRIORI_COMPILED_KERNEL_HPP_
