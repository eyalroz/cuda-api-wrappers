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

#include <cuda/api/types.h>
#include <cuda/api/device_properties.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/current_device.hpp>

#include <cuda_runtime_api.h>

namespace cuda {

namespace device_function {

struct attributes_t : cudaFuncAttributes {

	cuda::device::compute_capability_t ptx_version() const {
		return device::compute_capability_t::from_combined_number(ptxVersion);
	}

	cuda::device::compute_capability_t binary_compilation_target_architecture() const {
		return device::compute_capability_t::from_combined_number(binaryVersion);
	}
};

/**
 * @brief Calculate the effective maximum size of allocatable (dynamic)
 * shared memory in a grid block
 *
 * @param attributes Attributes of the {@code __global__} kernel function
 * for which we wish to determine the allocation limit
 * @param compute_capability the GPU device's compute capability figure (e.g. 3.5
 * or 5.0), which fully determines the maximum allocation size
 */
inline shared_memory_size_t maximum_dynamic_shared_memory_per_block(
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
 * A non-owning wrapper class for CUDA {@code __device__} functions
 */
class device_function_t {
public: // getters
	const void* ptr() const { return ptr_; }

public: // type_conversions
	operator const void*() { return ptr_; }

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
	 * Sets a device functions preference (on the current device probably) of
	 * either having more L1 cache or more shared memory space
	 */
	void cache_preference(multiprocessor_cache_preference_t preference)
	{
		auto result = cudaFuncSetCacheConfig(ptr_, (cudaFuncCache) preference);
		throw_if_error(result,
			"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
			"CUDA device function");
	}

	// Is the cache preference really per-device? ... well, it should be, right?
	void cache_preference(
		device::id_t  device_id, multiprocessor_cache_preference_t preference)
	{
		// Not using device::current::scoped_override_t here
		// to minimize dependecies
		device::id_t  old_device;
		auto result = cudaGetDevice(&old_device);
		throw_if_error(result, "Failed obtaining the current CUDA device ID");
		result = cudaSetDevice(device_id);
		throw_if_error(result, "Failed setting the current CUDA device ID to " + std::to_string(device_id));
		cache_preference(preference);
		result = cudaSetDevice(old_device);
		throw_if_error(result, "Failure setting device back to " + std::to_string(old_device));
	}

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
 * A 'version' of @ref cuda::compute_capability_t::maximum_dynamic_shared_memory_per_block() for
 * use with a specific device function - which will take its use of static shared memory into account
 *
 * @param device_function The ({@code __global__} or {@code __device__}) function for which to calculate
 * the effective available shared memory per block
 * @param compute_capability on which kind of device the kernel function is to be launched;
 * @return the maximum amount of shared memory per block which a launch of the specified
 * function can require

 * @todo It's not clear whether this is actually necessary given the {@ref device_function_t}
 * pointer.
 *
 */
inline shared_memory_size_t maximum_dynamic_shared_memory_per_block(
	const device_function_t& device_function, device::compute_capability_t compute_capability)
{
	return device_function::maximum_dynamic_shared_memory_per_block(
		device_function.attributes(), compute_capability);
}

inline grid_dimension_t maximum_active_blocks_per_multiprocessor(
	device::id_t              device_id,
	const device_function_t&  device_function,
	grid_block_dimension_t    num_threads_per_block,
	size_t                    dynamic_shared_memory_per_block,
	bool                      disable_caching_override = false)
{
	device::current::scoped_override_t<> set_device_for_this_context(device_id);
	int result;
	unsigned int flags = disable_caching_override ?
		cudaOccupancyDisableCachingOverride : cudaOccupancyDefault;
	auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
		&result, device_function.ptr(), num_threads_per_block,
		dynamic_shared_memory_per_block, flags);
	throw_if_error(status, "Failed calculating the maximum occupancy "
		"of device function blocks per multiprocessor");
	return result;
}

} // namespace device_function
} // namespace cuda

#endif /* CUDA_API_WRAPPERS_DEVICE_FUNCTION_HPP_ */
