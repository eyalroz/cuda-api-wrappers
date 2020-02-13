/**
 * @file device_function.hpp
 *
 * @brief Functions for querying information and making settings
 * regarding device-side functions - kernels or otherwise.
 *
 * @note This file does _not_ have device-side functions itself,
 * nor is it about the device-side part of the runtime API (i.e.
 * API functions which may be called from the device).
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DEVICE_FUNCTION_HPP_
#define CUDA_API_WRAPPERS_DEVICE_FUNCTION_HPP_

#include <cuda/api/types.hpp>
#include <cuda/api/device_properties.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/current_device.hpp>

#include <cuda_runtime_api.h>

namespace cuda {

template<bool AssumedCurrent>
class device_t;

namespace device_function {

/**
 * @brief a wrapper around @ref cudaFuncAttributes, offering
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

/**
 * @brief Calculate the effective maximum size of allocatable (dynamic)
 * shared memory in a grid block
 *
 * @param attributes Attributes of the `__global__` kernel function
 * for which we wish to determine the allocation limit
 * @param compute_capability the GPU device's compute capability figure (e.g. 3.5
 * or 5.0), which fully determines the maximum allocation size
 */
inline memory::shared::size_t maximum_dynamic_shared_memory_per_block(
	attributes_t attributes, device::compute_capability_t compute_capability)
{
	auto available_without_static_allocation = compute_capability.max_shared_memory_per_block();
	auto statically_allocated_shared_mem = attributes.sharedSizeBytes;
	if (statically_allocated_shared_mem > available_without_static_allocation) {
		throw std::logic_error("More static shared memory has been allocated for a device function"
		" than seems to be available on devices with the specified compute capability.");
	}
	return available_without_static_allocation - statically_allocated_shared_mem;
}

} // namespace device_function

/**
 * A non-owning wrapper class for CUDA `__global__` functions
 *
 * @note despite the name, a `device_function_t` is not associated with a specific
 * device - it can't just be _executed_ on a device.
 */
class device_function_t {
public: // getters
	const void* ptr() const noexcept { return ptr_; }

public: // type_conversions
	operator const void*() noexcept { return ptr_; }

public: // non-mutators

	inline device_function::attributes_t attributes() const
	{
		device_function::attributes_t function_attributes;
		auto status = cudaFuncGetAttributes(&function_attributes, ptr_);
		throw_if_error(status, "Failed obtaining attributes for a CUDA device function");
		return function_attributes;
	}

/*
	// The following are commented out because there are no CUDA API calls for them!
	// You may uncomment them if you'd rather get an exception...

	**
	 * Obtains a device function's preference (on the current device probably) of
	 * either having more L1 cache or more shared memory space
	 *
	multiprocessor_cache_preference_t cache_preference() const
	{
		throw cuda_inspecific_runtime_error(
			"There's no CUDA runtime API call for obtaining the cache preference!");
	}


	multiprocessor_shared_memory_bank_size_option_t  shared_memory_bank_size() const
	{
		throw cuda_inspecific_runtime_error(
			"There's no CUDA runtime API call for obtaining the shared memory bank size!");
	}
*/


public: // mutators
	/**
	 * @brief Sets a device function's preference (on the current device probably) of
	 * either having more L1 cache or more shared memory space
	 */
	void cache_preference(multiprocessor_cache_preference_t preference)
	{
		auto result = cudaFuncSetCacheConfig(ptr_, (cudaFuncCache) preference);
		throw_if_error(result,
			"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
			"CUDA device function");
	}

	/**
	 * @brief Sets a device function's preference of either having more L1 cache or
	 * more shared memory space when executing on some device
     *
	 * @param device_id the CUDA device for execution on which the preference is set
	 * @param preference value to set for the device function (more cache, more L1 or make the equal)
	 */
	void cache_preference(
		device_t<detail::do_not_assume_device_is_current>& device,
		multiprocessor_cache_preference_t preference);

	void set_attribute(cudaFuncAttribute attribute, int value)
	{
		auto result = cudaFuncSetAttribute(ptr_, attribute, value);
		throw_if_error(result, "Setting CUDA device function attribute " + std::to_string(attribute) + " to value " + std::to_string(value));
	}

	void opt_in_to_extra_dynamic_memory(cuda::memory::shared::size_t maximum_shared_memory_required_by_kernel)
	{
#if CUDART_VERSION >= 9000
		auto result = cudaFuncSetAttribute(ptr_, cudaFuncAttributeMaxDynamicSharedMemorySize, maximum_shared_memory_required_by_kernel);
		throw_if_error(result, "Trying to opt-in to a (potentially) higher maximum amount of dynamic shared memory");
#else
		throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif
	}

	void opt_in_to_extra_dynamic_memory(
		cuda::memory::shared::size_t maximum_shared_memory_required_by_kernel,
		device_t<detail::do_not_assume_device_is_current>& device);

	void set_shared_mem_to_l1_cache_fraction(unsigned shared_mem_percentage)
	{
		if (shared_mem_percentage > 100) {
			throw std::invalid_argument("Invalid percentage");
		}
#if CUDART_VERSION >= 9000
		auto result = cudaFuncSetAttribute(ptr_, cudaFuncAttributePreferredSharedMemoryCarveout, shared_mem_percentage);
		throw_if_error(result, "Trying to set the carve-out of shared memory/L1 cache memory");
#else
		throw(cuda::runtime_error {cuda::status::not_yet_implemented});
#endif
	}

	void set_shared_mem_to_l1_cache_fraction(
		unsigned shared_mem_percentage,
		device_t<detail::do_not_assume_device_is_current>& device);


	/**
	 * @brief Sets a device function's preference of shared memory bank size preference
	 * (for the current device probably)
     *
	 * @param config bank size setting to make
	 */
	void shared_memory_bank_size(multiprocessor_shared_memory_bank_size_option_t config)
	{
		auto result = cudaFuncSetSharedMemConfig(ptr_, (cudaSharedMemConfig) config);
		if (result == cudaErrorInvalidDeviceFunction) { return; }
		throw_if_error(result);
	}

public: // ctors & dtor
	template <typename DeviceFunction>
	device_function_t(DeviceFunction f) : ptr_(reinterpret_cast<const void*>(f))
	{
		// TODO: Consider checking whether this actually is a device function
		// TODO: Consider performing a check for nullptr
	}
	~device_function_t() { };

public: // data members
	const void* const ptr_;

};

namespace device_function {

/**
 * @brief A 'version' of
 * @ref cuda::compute_capability_t::maximum_dynamic_shared_memory_per_block()
 * for use with a specific device function - which will take its use of
 * static shared memory into account.
 *
 * @param device_function The (`__global__` or `__device__`)
 * function for which to calculate
 * the effective available shared memory per block
 * @param compute_capability on which kind of device the kernel function is to
 * be launched;
 * @return the maximum amount of shared memory per block which a launch of the
 * specified function can require

 * @todo It's not clear whether this is actually necessary given the {@ref device_function_t}
 * pointer.
 *
 */
inline memory::shared::size_t maximum_dynamic_shared_memory_per_block(
	const device_function_t& device_function, device::compute_capability_t compute_capability)
{
	return device_function::maximum_dynamic_shared_memory_per_block(
		device_function.attributes(), compute_capability);
}

/**
 * @brief Calculates the number of grid blocks which may be "active" on a given GPU
 * multiprocessor simultaneously (i.e. with warps from any of these block
 * being schedulable concurrently)
 *
 * @param disable_caching_override On some GPUs, the choice of whether to
 * cache memory reads affects occupancy. But what if this caching results in 0
 * potential occupancy for a kernel? There are two options, controlled by this flag.
 * When it is set to false - the calculator will assume caching is off for the
 * purposes of its work; when set to true, it will return 0 for such device functions.
 * See also the "Unified L1/Texture Cache" section of the Maxwell tuning guide:
 * @url http://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html
 */
inline grid::dimension_t maximum_active_blocks_per_multiprocessor(
	device_t<detail::do_not_assume_device_is_current>
                              device,
	const device_function_t&  device_function,
	grid::block_dimension_t    num_threads_per_block,
	memory::shared::size_t     dynamic_shared_memory_per_block,
	bool                      disable_caching_override = false);


namespace detail {

//
//template<typename KernelFunction, typename... KernelParameters>
//using raw_device_function_t = void(*)(KernelParameters...);
//

template<typename... KernelParameters>
struct raw_device_function_typegen {
	using type = void(*)(KernelParameters...);
};

template<typename KernelFunction, typename... KernelParameters>
typename raw_device_function_typegen<KernelParameters...>::type unwrap_inner(std::true_type, device_function_t wrapped)
{
	using raw_device_function_t = typename raw_device_function_typegen<KernelParameters ...>::type;
	return reinterpret_cast<raw_device_function_t>(wrapped.ptr());
}

template<typename KernelFunction, typename... KernelParameters>
KernelFunction unwrap_inner(std::false_type, KernelFunction raw_function)
{
	static_assert(
		std::is_function<typename std::decay<KernelFunction>::type>::value or
		(std::is_pointer<KernelFunction>::value and  std::is_function<typename std::remove_pointer<KernelFunction>::type>::value)
		, "Invalid KernelFunction type - it must be either a function or a pointer-to-a-function");
	return raw_function;
}

} // namespace detail

template<typename KernelFunction, typename... KernelParameters>
auto unwrap(KernelFunction f) -> typename std::conditional<
	std::is_same<typename std::decay<KernelFunction>::type, device_function_t>::value,
	typename detail::raw_device_function_typegen<KernelParameters...>::type,
	KernelFunction>::type
{
	using got_a_device_function_t =
		std::integral_constant<bool, std::is_same<typename std::decay<KernelFunction>::type, device_function_t>::value>;
	return detail::unwrap_inner<KernelFunction, KernelParameters...>(got_a_device_function_t{}, f);
}

} // namespace device_function
} // namespace cuda

#endif // CUDA_API_WRAPPERS_DEVICE_FUNCTION_HPP_
