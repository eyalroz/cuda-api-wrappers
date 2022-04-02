/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard apriori-compiled kernels. Specifically:
 *
 * 1. Functions in the `cuda::kernel` namespace.
 * 2. Methods of @ref `cuda::kernel_t` and possibly some relates classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_APRIORI_COMPILED_KERNEL_HPP_
#define MULTI_WRAPPER_IMPLS_APRIORI_COMPILED_KERNEL_HPP_

#include "../array.hpp"
#include "../device.hpp"
#include "../event.hpp"
#include "../kernel_launch.hpp"
#include "../pointer.hpp"
#include "../stream.hpp"
#include "../unique_ptr.hpp"
#include "../primary_context.hpp"
#include "../kernel.hpp"
#include "../apriori_compiled_kernel.hpp"
#include "../module.hpp"
#include "../virtual_memory.hpp"
#include "../current_context.hpp"
#include "../current_device.hpp"
#include "../texture_view.hpp"
#include "../peer_to_peer.hpp"

namespace cuda {

#if ! CAN_GET_APRIORI_KERNEL_HANDLE
#if defined(__CUDACC__)

// Unfortunately, the CUDA runtime API does not allow for computation of the grid parameters for maximum occupancy
// from code compiled with a host-side-only compiler! See cuda_runtime.h for details

inline kernel::attributes_t apriori_compiled_kernel_t::attributes() const
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device_id_);
	kernel::attributes_t function_attributes;
	auto status = cudaFuncGetAttributes(&function_attributes, ptr_);
	throw_if_error(status, "Failed obtaining attributes for a CUDA device function");
	return function_attributes;
}

inline void apriori_compiled_kernel_t::set_cache_preference(multiprocessor_cache_preference_t preference) const
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device_id_);
	auto result = cudaFuncSetCacheConfig(ptr_, (cudaFuncCache) preference);
	throw_if_error(result,
		"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
		"CUDA device function");
}

inline void apriori_compiled_kernel_t::set_shared_memory_bank_size(
	multiprocessor_shared_memory_bank_size_option_t  config) const
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device_id_);
	auto result = cudaFuncSetSharedMemConfig(ptr_, (cudaSharedMemConfig) config);
	throw_if_error(result);
}

inline void apriori_compiled_kernel_t::set_attribute(kernel::attribute_t attribute, kernel::attribute_value_t value) const
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device_id_);
	cudaFuncAttribute runtime_attribute = [attribute]() {
		switch (attribute) {
			case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
				return cudaFuncAttributeMaxDynamicSharedMemorySize;
			case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
				return cudaFuncAttributePreferredSharedMemoryCarveout;
			default:
				throw cuda::runtime_error(status::not_supported,
					"Kernel attribute " + std::to_string(attribute) + " not supported (with CUDA version "
					+ std::to_string(CUDA_VERSION));
		}
	}();
	auto result = cudaFuncSetAttribute(ptr_, runtime_attribute, value);
	throw_if_error(result, "Setting CUDA device function attribute " + ::std::to_string(attribute) + " to value " + ::std::to_string(value));
}

kernel::attribute_value_t apriori_compiled_kernel_t::get_attribute(kernel::attribute_t attribute) const
{
	kernel::attributes_t attrs = attributes();
	switch(attribute) {
		case CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK:
			return attrs.maxThreadsPerBlock;
		case CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES:
			return attrs.sharedSizeBytes;
		case CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES:
			return attrs.constSizeBytes;
		case CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES:
			return attrs.localSizeBytes;
		case CU_FUNC_ATTRIBUTE_NUM_REGS:
			return attrs.numRegs;
		case CU_FUNC_ATTRIBUTE_PTX_VERSION:
			return attrs.ptxVersion;
		case CU_FUNC_ATTRIBUTE_BINARY_VERSION:
			return attrs.binaryVersion;
		case CU_FUNC_ATTRIBUTE_CACHE_MODE_CA:
			return attrs.cacheModeCA;
		case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
			return attrs.maxDynamicSharedSizeBytes;
		case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
			return attrs.preferredShmemCarveout;
		default:
			throw cuda::runtime_error(status::not_supported,
				::std::string("Attribute ") +
#ifdef NDEBUG
					::std::to_string(static_cast<::std::underlying_type<kernel::attribute_t>::type>(attribute))
#else
					kernel::detail_::attribute_name(attribute)
#endif
				+ " cannot be obtained for apriori-compiled kernels before CUDA version 11.0"
			);
	}
}

#endif // defined(__CUDACC__)
#endif // ! CAN_GET_APRIORI_KERNEL_HANDLE

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_APRIORI_COMPILED_KERNEL_HPP_

