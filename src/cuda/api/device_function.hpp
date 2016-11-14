#pragma once
#ifndef CUDA_API_WRAPPERS_DEVICE_FUNCTION_HPP_
#define CUDA_API_WRAPPERS_DEVICE_FUNCTION_HPP_

#include "cuda/api/types.h"
#include "cuda/api/device_properties.hpp"
#include "cuda/api/error.hpp"

#include <cuda_runtime_api.h>

namespace cuda {

/**
 * A non-owning wrapper class for CUDA __device__ functions
 */
class device_function_t {
public: // type definitions
	using attributes_t   = cudaFuncAttributes;

public: // statics

	static shared_memory_size_t maximum_dynamic_shared_memory_per_block(
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

public: // getters

	const void* ptr() { return ptr_; }

public: // type_conversions
	operator const void*() { return ptr_; }


public: // non-mutators

	// TODO: A getter for the shared mem / L1 cache config

	inline attributes_t attributes() const
	{
		attributes_t function_attributes;
		auto result = cudaFuncGetAttributes(&function_attributes, ptr_);
		throw_if_error(result, "Failed obtaining attributes for a CUDA device function");
		return function_attributes;
	}

	/**
	 * A 'version' of @ref compute_capability_t::effective_max_shared_memory_per_block for
	 * use with a specific device function - which will take its use of static shared memory into account
	 *
	 * @param device_function The (__global__ or __device__) function for which to calculate
	 * the effective available shared memory per block
	 * @param compute_capability on which kind of device the kernel function is to be launched;
	 * TODO: it's not clear whether this is actually necessary given the {@ref device_function}
	 * pointer
	 * @return the maximum amount of shared memory per block which a launch of the specified
	 * function can require
	 */
	shared_memory_size_t maximum_dynamic_shared_memory_per_block(
		device::compute_capability_t compute_capability) const
	{
		return maximum_dynamic_shared_memory_per_block(attributes(), compute_capability);
	}

	/**
	 * Obtains a device function's preference (on the current device probably) of
	 * either having more L1 cache or more shared memory space
	 */
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
	const void* ptr_;

};

} // namespace cuda

#endif /* CUDA_API_WRAPPERS_DEVICE_FUNCTION_HPP_ */
