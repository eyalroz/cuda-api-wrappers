/**
 * @file
 *
 * @brief Implementations of `cuda::memory::pointer_t` methods requiring the definitions
 * of multiple CUDA entity proxy classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_POINTER_HPP_
#define MULTI_WRAPPER_IMPLS_POINTER_HPP_

#include "../pointer.hpp"
#include "../device.hpp"
#include "../context.hpp"

namespace cuda {

namespace memory {

template <typename T>
inline device_t pointer_t<T>::device() const
{
	cuda::device::id_t device_id = get_attribute<CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>();
	return cuda::device::get(device_id);
}

template <typename T>
inline context_t pointer_t<T>::context() const
{
	pointer::attribute_t attributes[] = {
		CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
		CU_POINTER_ATTRIBUTE_CONTEXT
	};
	cuda::device::id_t device_id;
	context::handle_t context_handle;
	void* value_ptrs[] = {&device_id, &context_handle};
	pointer::detail_::get_attributes(2, attributes, value_ptrs, ptr_);
	return context::wrap(device_id, context_handle);
}

} // namespace memory

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_POINTER_HPP_

