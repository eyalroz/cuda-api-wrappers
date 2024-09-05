/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard apriori-compiled kernels. Specifically:
 *
 * 1. Functions in the `cuda::kernel` namespace.
 * 2. Methods of @ref cuda::kernel_t and possibly some relates classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_APRIORI_COMPILED_KERNEL_HPP_
#define MULTI_WRAPPER_IMPLS_APRIORI_COMPILED_KERNEL_HPP_

#include "../kernels/apriori_compiled.hpp"
#include "device.hpp"
#include "kernel.hpp"

namespace cuda {

namespace kernel {

#if ! CAW_CAN_GET_APRIORI_KERNEL_HANDLE
#if defined(__CUDACC__)

// Unfortunately, the CUDA runtime API does not allow for computation of the grid parameters for maximum occupancy
// from code compiled with a host-side-only compiler! See cuda_runtime.h for details

inline apriori_compiled::attributes_t apriori_compiled_t::attributes() const
{
	// Note: assuming the primary context is active
	CAW_SET_SCOPE_CONTEXT(context_handle_);
	apriori_compiled::attributes_t function_attributes;
	auto status = cudaFuncGetAttributes(&function_attributes, ptr_);
	throw_if_error_lazy(status, "Failed obtaining attributes for a CUDA device function");
	return function_attributes;
}

inline void apriori_compiled_t::set_cache_preference(multiprocessor_cache_preference_t preference) const
{
	// Note: assuming the primary context is active
	CAW_SET_SCOPE_CONTEXT(context_handle_);
	auto result = cudaFuncSetCacheConfig(ptr_, (cudaFuncCache) preference);
	throw_if_error_lazy(result,
		"Setting the multiprocessor L1/Shared Memory cache distribution preference for a "
		"CUDA device function");
}

inline void apriori_compiled_t::set_shared_memory_bank_size(
	multiprocessor_shared_memory_bank_size_option_t  config) const
{
	// Note: assuming the primary context is active
	CAW_SET_SCOPE_CONTEXT(context_handle_);
	auto result = cudaFuncSetSharedMemConfig(ptr_, (cudaSharedMemConfig) config);
	throw_if_error_lazy(result, "Failed setting shared memory bank size to " + ::std::to_string(config));
}

inline void apriori_compiled_t::set_attribute(attribute_t attribute, attribute_value_t value) const
{
	// Note: assuming the primary context is active
	CAW_SET_SCOPE_CONTEXT(context_handle_);
	cudaFuncAttribute runtime_attribute = [attribute]() {
		switch (attribute) {
			case CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES:
				return cudaFuncAttributeMaxDynamicSharedMemorySize;
			case CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT:
				return cudaFuncAttributePreferredSharedMemoryCarveout;
			default:
				throw cuda::runtime_error(status::not_supported,
					"Kernel attribute " + ::std::to_string(attribute) + " not supported (with CUDA version "
					+ ::std::to_string(CUDA_VERSION));
		}
	}();
	auto result = cudaFuncSetAttribute(ptr_, runtime_attribute, value);
	throw_if_error_lazy(result, "Setting CUDA device function attribute " + ::std::to_string(attribute) + " to value " + ::std::to_string(value));
}

inline attribute_value_t apriori_compiled_t::get_attribute(attribute_t attribute) const
{
	apriori_compiled::attributes_t attrs = attributes();
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
					::std::to_string(static_cast<::std::underlying_type<attribute_t>::type>(attribute))
#else
					detail_::attribute_name(attribute)
#endif
				+ " cannot be obtained for apriori-compiled kernels before CUDA version 11.0"
			);
	}
}

#endif // defined(__CUDACC__)
#endif // ! CAW_CAN_GET_APRIORI_KERNEL_HANDLE

namespace apriori_compiled {

namespace detail_ {

template<typename KernelFunctionPtr>
apriori_compiled_t get(
	device::id_t         device_id,
	context::handle_t &  primary_context_handle,
	KernelFunctionPtr    function_ptr)
{
	static_assert(
		::std::is_pointer<KernelFunctionPtr>::value
		and ::std::is_function<typename ::std::remove_pointer<KernelFunctionPtr>::type>::value,
		"function must be a bona fide pointer to a kernel (__global__) function");

	auto ptr_ = reinterpret_cast<const void *>(function_ptr);
#if CAW_CAN_GET_APRIORI_KERNEL_HANDLE
	auto handle = detail_::get_handle(ptr_);
#else
	auto handle = nullptr;
#endif
	return wrap(device_id, primary_context_handle, handle, ptr_, do_hold_primary_context_refcount_unit);
}

} // namespace detail_

} // namespace apriori_compiled


/**
 * @brief Obtain a wrapped kernel object corresponding to a "raw" kernel function
 *
 * @note Kernel objects are device (and context) specific;' but kernels built
 * from functions in program sources are used (only?) with the primary context of a device
 *
 * @note The returned kernel proxy object will keep the device's primary
 * context active while the kernel exists.
 */
template<typename KernelFunctionPtr>
apriori_compiled_t get(const device_t &device, KernelFunctionPtr function_ptr)
{
	auto primary_context_handle = device::primary_context::detail_::obtain_and_increase_refcount(device.id());
	return apriori_compiled::detail_::get(device.id(), primary_context_handle, function_ptr);
}

} // namespace kernel

namespace detail_ {

template<>
inline ::cuda::device::primary_context_t
get_implicit_primary_context<kernel::apriori_compiled_t>(kernel::apriori_compiled_t kernel)
{
	const kernel_t &kernel_ = kernel;
	return get_implicit_primary_context(kernel_);
}

} // namespace detail_

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_APRIORI_COMPILED_KERNEL_HPP_

