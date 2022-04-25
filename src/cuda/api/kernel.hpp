/**
 * @file
 *
 * @brief Contains a base wrapper class for CUDA kernels - both statically and
 * dynamically compiled; and some related functionality.
 *
 * @note This file does _not_ define any kernels itself.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_KERNEL_HPP_
#define CUDA_API_WRAPPERS_KERNEL_HPP_

#include <cuda/api/types.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/current_context.hpp>
// #include <cuda/api/module.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

#if CUDA_VERSION < 11000
#define CAN_GET_APRIORI_KERNEL_HANDLE 0
#define VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE virtual
#else
#define CAN_GET_APRIORI_KERNEL_HANDLE 1
#define VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE
#endif

namespace cuda {

///@cond
class device_t;
class kernel_t;
///@nocond

namespace kernel {

using shared_memory_size_determiner_t = size_t (*)(int block_size);

/**
 * Obtain a proxy object for a CUDA kernel
 *
 * @note This is a named constructor idiom, existing of direct access to the ctor
 * of the same signature, to emphasize that a new kernel is _not_ somehow created.
 *
 * @param id Device on which the texture is located
 * @param context_handle Handle of the context in which the kernel was created or added
 * @param handle raw CUDA driver handle for the kernel
 * @return a wrapper object associated with the specified kernel
 */
kernel_t wrap(
	device::id_t       device_id,
	context::handle_t  context_id,
	kernel::handle_t   f);

namespace detail_ {

inline ::std::string identify(const kernel_t& kernel);

#ifndef NDEBUG
static const char* attribute_name(int attribute_index)
{
	// Note: These correspond to the values of enum CUfunction_attribute_enum
	static const char* names[] = {
		"Maximum number of threads per block",
		"Statically-allocated shared memory size in bytes",
		"Required constant memory size in bytes",
		"Required local memory size in bytes",
		"Number of registers used by each thread",
		"PTX virtual architecture version into which the kernel code was compiled",
		"Binary architecture version for which the function was compiled",
		"Indication whether the function was compiled with cache mode CA",
		"Maximum allowed size of dynamically-allocated shared memory use size bytes",
		"Preferred shared memory carve-out to actual shared memory"
	};
	return names[attribute_index];
}
#endif

inline attribute_value_t get_attribute_in_current_context(handle_t handle, attribute_t attribute)
{
	kernel::attribute_value_t attribute_value;
	auto result = cuFuncGetAttribute(&attribute_value,  attribute, handle);
	throw_if_error(result,
		::std::string("Failed obtaining attribute ") +
#ifdef NDEBUG
			::std::to_string(static_cast<::std::underlying_type<kernel::attribute_t>::type>(attribute))
#else
			attribute_name(attribute)
#endif
	);
	return attribute_value;
}

} // namespace detail_

} // namespace kernel

/**
 * A non-owning wrapper for CUDA kernels - whether they be `__global__` functions compiled
 * apriori, or the result of dynamic NVRTC compilation, or obtained in some other future
 * way.
 *
 * @note The association of a `kernel_t` with an individual device or context is somewhat
 * tenuous. That is, the same function could be used with any other compatible device;
 * However, many/most of the features, attributes and settings are context-specific
 * or device-specific.
 *
 * @note NVRTC-compiled kernels can only use this class, with apriori-compiled
 * kernels can use their own subclass.
 *
 * @todo Consider holding a module handle (possibly null/0/invalid), and a boolean
 * saying whether this kernel wrapper holds it. This would allow passing kernel_t's
 * without accompanying module_t's.
 */
class kernel_t {

public: // getters
	context_t context() const noexcept;
	device_t device() const noexcept;

	device::id_t      device_id() const noexcept { return device_id_; }
	context::handle_t context_handle() const noexcept { return context_handle_; }
#if ! CAN_GET_APRIORI_KERNEL_HANDLE
	kernel::handle_t  handle() const
	{
#ifndef NDEBUG
		if (handle_ == nullptr) {
			throw runtime_error(status::named_t::invalid_resource_handle,
			    "CUDA driver handle unavailable for kernel");
		}
#endif
		return handle_;
	}
#else
	kernel::handle_t  handle() const noexcept { return handle_; }
#endif

public: // non-mutators

	VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE
	kernel::attribute_value_t get_attribute(kernel::attribute_t attribute) const
	{
		context::current::detail_::scoped_override_t set_context_for_this_context(context_handle_);
		return kernel::detail_::get_attribute_in_current_context(handle(), attribute);
	}

	VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE
	cuda::device::compute_capability_t ptx_version() const
	{
		auto raw_attribute = get_attribute(CU_FUNC_ATTRIBUTE_PTX_VERSION);
		return device::compute_capability_t::from_combined_number(raw_attribute);
	}

	VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE
	cuda::device::compute_capability_t binary_compilation_target_architecture() const {
		auto raw_attribute = get_attribute(CU_FUNC_ATTRIBUTE_BINARY_VERSION);
		return device::compute_capability_t::from_combined_number(raw_attribute);
	}

	/**
	 * @return the maximum number of threads per block for which the GPU device can satisfy
	 * this kernel's hardware requirement - typically, the number of registers in use.
	 *
	 * @note the kernel may have other constraints, requiring a different number of threads
	 * per block; these cannot be determined using this method.
	 */
	VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE
	grid::block_dimension_t maximum_threads_per_block() const
	{
		return get_attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
	}

#if CUDA_VERSION >= 10000
	/**
	 * @brief obtain the dimensions of a minimum grid which is expected to
	 * achieve maximum occupancy on the GPU the kernel is associated with.
	 *
	 * @param dynamic_shared_memory_size
	 *     The amount of dynamic shared memory each grid block will need.
	 *     Either this value is specified, or @p kernel::shared_memory_size_determiner.
	 * @param shared_memory_size_determiner
	 *     A function which returns the desired amount of shared memory given the block
	 *     size in overall number of threads. Either this function is provided, or
	 *     @p dynamic_shared_memory_size is specified.
	 * @param block_size_limit
	 *     do not return a block size above this value; the default, 0, means no limit
	 *     on the returned block size.
	 * @param disable_caching_override
	 *     On platforms where global caching affects occupancy, and when enabling caching
	 *     would result in zero occupancy, the occupancy calculator will calculate the
	 *     occupancy as if caching is disabled. Setting this to true makes the occupancy
	 *     calculator return 0 in such cases. More information can be found about this
	 *     feature in the "Unified L1/Texture Cache" section of the
	 *     <a href="https://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html">Maxwell tuning guide</a>.
	 *
	 * @return A pair, with the second element being the maximum achievable block size
	 * (1-dimensional), and the first element being the minimum number of such blocks necessary
	 * for keeping the GPU "busy" (again, in a 1-dimensional grid).
	 */
	///@{
	VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE
	grid::composite_dimensions_t min_grid_params_for_max_occupancy(
		memory::shared::size_t dynamic_shared_memory_size = no_dynamic_shared_memory,
		grid::block_dimension_t block_size_limit = 0,
		bool disable_caching_override = false) const;

	VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE
	grid::composite_dimensions_t min_grid_params_for_max_occupancy(
		kernel::shared_memory_size_determiner_t  shared_memory_size_determiner,
		grid::block_dimension_t                  block_size_limit = 0,
		bool                                     disable_caching_override = false) const;
	///@}
#endif // CUDA_VERSION >= 10000

	/**
	 * @brief Calculates the number of grid blocks which may be "active" on a given GPU
	 * multiprocessor simultaneously (i.e. with warps from any of these block
	 * being schedulable concurrently)
	 *
	 * @param block_size_in_threads
	 * @param dynamic_shared_memory_per_block
	 * @param disable_caching_override
	 *     On some GPUs, the choice of whether to cache memory reads affects occupancy.
	 *     But what if this caching results in 0 potential occupancy for a kernel?
	 *     There are two options, controlled by this flag. When it is set to false -
	 *     the calculator will assume caching is off for the purposes of its work; when
	 *     set to true, it will return 0 for such device functions.
	 *
	 * @see The
	 *     "Unified L1/Texture Cache" section of the <a href="http://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html">Maxwell
	 * tuning guide</a> regarding caching override.
	 */
	VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE
	grid::dimension_t max_active_blocks_per_multiprocessor(
		grid::block_dimension_t block_size_in_threads,
		memory::shared::size_t  dynamic_shared_memory_per_block,
		bool                    disable_caching_override = false) const;



public: // methods mutating the kernel-in-context, but not this reference object

	VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE
	void set_attribute(kernel::attribute_t attribute, kernel::attribute_value_t value) const;

	/**
	 * @brief Change the hardware resource carve-out between L1 cache and shared memory
	 * for launches of the kernel to allow for at least the specified amount of
	 * shared memory.
	 *
	 * On several nVIDIA GPU micro-architectures, the L1 cache and the shared memory in each
	 * symmetric multiprocessor (=physical core) use the same hardware resources. The
	 * carve-out between the two uses has a device-wide value (which can be changed), but can
	 * also be set on the individual device-function level, by specifying the amount of shared
	 * memory the kernel may require.
	 */
	void set_maximum_dynamic_shared_memory_per_block(cuda::memory::shared::size_t amount_required_by_kernel) const
	{
		auto amount_required_by_kernel_ = (kernel::attribute_value_t) amount_required_by_kernel;
		if (amount_required_by_kernel != (cuda::memory::shared::size_t) amount_required_by_kernel_) {
			throw ::std::invalid_argument("Requested amount of maximum shared memory exceeds the "
				"representation range for kernel attribute values");
		}
		// TODO: Consider a check in debug mode for the value being within range
		set_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,amount_required_by_kernel_);
	}

	memory::shared::size_t get_maximum_dynamic_shared_memory_per_block() const
	{
		return get_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
	}

	/**
	 * @brief Indicate the desired carve-out between shared memory and L1 cache when launching
	 * this kernel - with coarse granularity.
	 *
	 * On several nVIDIA GPU micro-architectures, the L1 cache and the shared memory in each
	 * symmetric multiprocessor (=physical core) use the same hardware resources. The
	 * carve-out between the two uses has a device-wide value (which can be changed), but the
	 * driver can set another value for a specific function. This function doesn't make a demand
	 * from the CUDA runtime (as in @p opt_in_to_extra_dynamic_memory), but rather indicates
	 * what is the fraction of L1 to shared memory it would like the kernel scheduler to carve
	 * out.
	 *
	 * @param preference one of: as much shared memory as possible, as much
	 * L1 as possible, or no preference (i.e. using the device default).
	 *
	 * @note similar to @ref set_preferred_shared_mem_fraction() - but with coarser granularity.
	 */
	VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE
	void set_cache_preference(multiprocessor_cache_preference_t preference) const
	{
		context::current::detail_::scoped_override_t set_context_for_this_context(context_handle_);
		auto result = cuFuncSetCacheConfig(handle(), (CUfunc_cache) preference);
		throw_if_error(result,
			"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
			"CUDA device function");
	}

	/**
	 * @brief Sets a device function's preference of shared memory bank size
	 *
	 * @param config bank size setting to make
	 */
	VIRTUAL_UNLESS_CAN_GET_APRIORI_KERNEL_HANDLE
	void set_shared_memory_bank_size(multiprocessor_shared_memory_bank_size_option_t config) const
	{
		// TODO: Need to set a context, not a device
		context::current::detail_::scoped_override_t set_context_for_this_context(context_handle_);
		auto result = cuFuncSetSharedMemConfig(handle(), static_cast<CUsharedconfig>(config) );
		throw_if_error(result, "Failed setting the shared memory bank size");
	}

protected: // ctors & dtor
	kernel_t(device::id_t device_id, context::handle_t context_handle, kernel::handle_t handle)
		: device_id_(device_id), context_handle_(context_handle), handle_(handle) { }

public: // ctors & dtor
	friend kernel_t kernel::wrap(device::id_t, context::handle_t, kernel::handle_t);

	kernel_t(const kernel_t& other) = default; // Note: be careful with subclasses
	kernel_t(kernel_t&& other) = default; // Note: be careful with subclasses

public: // ctors & dtor
#if ! CAN_GET_APRIORI_KERNEL_HANDLE
	virtual ~kernel_t() = default;
#endif

protected: // data members
	device::id_t device_id_; // We don't _absolutely_ need the device ID, but - why not have it if we can?
	context::handle_t context_handle_;
	mutable kernel::handle_t handle_;
}; // kernel_t

namespace kernel {

inline kernel_t wrap(
	device::id_t       device_id,
	context::handle_t  context_id,
	kernel::handle_t   f)
{
	return kernel_t{ device_id, context_id, f };
}

namespace occupancy {

namespace detail_ {

inline grid::dimension_t max_active_blocks_per_multiprocessor(
	handle_t                handle,
	grid::block_dimension_t block_size_in_threads,
	memory::shared::size_t  dynamic_shared_memory_per_block,
	bool                    disable_caching_override)
{
	int result;
	cuda::status_t status = CUDA_SUCCESS;
		// We don't need the initialization, but NVCC backed by GCC 8 warns us about it.
	auto flags = (unsigned) disable_caching_override ? CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE : CU_OCCUPANCY_DEFAULT;
	status = cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
		&result, handle, (int) block_size_in_threads, dynamic_shared_memory_per_block, flags);
	throw_if_error(status,
		"Determining the maximum occupancy in blocks per multiprocessor, given the block size and the amount of dyanmic memory per block");
	return result;
}

#if CUDA_VERSION >= 10000
// Note: If determine_shared_mem_by_block_size is not null, fixed_shared_mem_size is ignored;
// if block_size_limit is 0, it is ignored.
inline grid::composite_dimensions_t min_grid_params_for_max_occupancy(
	CUfunction                     kernel_handle,
	cuda::device::id_t             device_id,
	CUoccupancyB2DSize             determine_shared_mem_by_block_size,
	cuda::memory::shared::size_t   fixed_shared_mem_size,
	cuda::grid::block_dimension_t  block_size_limit,
	bool                           disable_caching_override)
{
	int min_grid_size_in_blocks { 0 };
	int block_size { 0 };
	// Note: only initializing the values her because of a
	// spurious (?) compiler warning about potential uninitialized use.

	auto result =  cuOccupancyMaxPotentialBlockSizeWithFlags(
		&min_grid_size_in_blocks, &block_size,
		kernel_handle,
		determine_shared_mem_by_block_size,
		fixed_shared_mem_size,
		static_cast<int>(block_size_limit),
		disable_caching_override ? CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE : CU_OCCUPANCY_DEFAULT
	);

	throw_if_error(result,
		"Failed obtaining parameters for a minimum-size grid for " + kernel::detail_::identify(kernel_handle, device_id)
		+ " with maximum occupancy given dynamic shared memory and block size data");
	return { (grid::dimension_t) min_grid_size_in_blocks, (grid::block_dimension_t) block_size };
}
#endif // CUDA_VERSION >= 10000

} // namespace detail_


/**
* @brief See the Driver API documentation for @ref cuOccupancyAvailableDynamicSMemPerBlock
*/
#if CUDA_VERSION < 11000
inline memory::shared::size_t max_dynamic_shared_memory_per_block(
	const kernel_t &,
	grid::dimension_t,
	grid::block_dimension_t)
{
	throw cuda::runtime_error(status::not_supported,
		"cuOccupancyAvailableDynamicSMemPerBlock() requires CUDA 11.0 or later");
}
#else
inline memory::shared::size_t max_dynamic_shared_memory_per_block(
	const kernel_t &kernel,
	grid::dimension_t blocks_on_multiprocessor,
	grid::block_dimension_t block_size_in_threads)
{
	size_t result;
	auto status = cuOccupancyAvailableDynamicSMemPerBlock(
		&result, kernel.handle(), (int) blocks_on_multiprocessor, (int) block_size_in_threads);
	throw_if_error(status,
		"Determining the available dynamic memory per block, given the number of blocks on a multiprocessor and their size");
	return (memory::shared::size_t) result;
}
#endif // CUDA_VERSION < 11000

/**
* @brief See the Driver API documentation for @ref cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags
*/
inline grid::dimension_t max_active_blocks_per_multiprocessor(
	const kernel_t &kernel,
	grid::block_dimension_t block_size_in_threads,
	memory::shared::size_t dynamic_shared_memory_per_block,
	bool disable_caching_override = false);

} // namespace occupancy

namespace detail_ {

inline ::std::string identify(const kernel_t& kernel)
{
	return kernel::detail_::identify(kernel.handle()) + " in " + context::detail_::identify(kernel.context());
}

} // namespace detail_
} // namespace kernel

#if CUDA_VERSION >= 10000
inline grid::composite_dimensions_t kernel_t::min_grid_params_for_max_occupancy(
	memory::shared::size_t   dynamic_shared_memory_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override) const
{
	kernel::shared_memory_size_determiner_t no_shared_memory_size_determiner { nullptr };
	return kernel::occupancy::detail_::min_grid_params_for_max_occupancy(
		handle(), device_id(), no_shared_memory_size_determiner,
		dynamic_shared_memory_size, block_size_limit, disable_caching_override);
}

inline grid::composite_dimensions_t kernel_t::min_grid_params_for_max_occupancy(
	kernel::shared_memory_size_determiner_t  shared_memory_size_determiner,
	cuda::grid::block_dimension_t            block_size_limit,
	bool                                     disable_caching_override) const
{
	size_t no_fixed_dynamic_shared_memory_size { 0 };
	return kernel::occupancy::detail_::min_grid_params_for_max_occupancy(
		handle(), device_id(), shared_memory_size_determiner,
		no_fixed_dynamic_shared_memory_size, block_size_limit, disable_caching_override);
}
#endif // CUDA_VERSION >= 10000

inline grid::dimension_t kernel_t::max_active_blocks_per_multiprocessor(
	grid::block_dimension_t  block_size_in_threads,
	memory::shared::size_t   dynamic_shared_memory_per_block,
	bool                     disable_caching_override) const
{
	return kernel::occupancy::detail_::max_active_blocks_per_multiprocessor(
		handle(), block_size_in_threads,
		dynamic_shared_memory_per_block, disable_caching_override);
}

} // namespace cuda

#endif // CUDA_API_WRAPPERS_KERNEL_HPP_
