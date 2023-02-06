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

namespace pointer {

namespace detail_ {

inline cuda::device::id_t device_id_of(const void *ptr)
{
#if CUDA_VERSION >= 9020
	return pointer::detail_::get_attribute<CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>(ptr);
#else
	auto context_handle = context_handle_of(ptr);
	return context::detail_::get_device_id(context_handle);
#endif
}

} // namespace detail_

} // namespace pointer


template <typename T>
inline device_t pointer_t<T>::device() const
{
	return cuda::device::get(pointer::detail_::device_id_of(ptr_));
}

template <typename T>
inline context_t pointer_t<T>::context() const
{
	return context_of(ptr_);
}

inline context_t context_of(const void* ptr)
{
#if CUDA_VERSION >= 9020
	pointer::attribute_t attributes[] = {
		CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
		CU_POINTER_ATTRIBUTE_CONTEXT
	};
	cuda::device::id_t device_id;
	context::handle_t context_handle;
	void* value_ptrs[] = {&device_id, &context_handle};
	pointer::detail_::get_attributes(2, attributes, value_ptrs, ptr);
#else
	auto context_handle = pointer::detail_::context_handle_of(ptr_);
	auto device_id = context::detail_::get_device_id(ptr_);
#endif
	return context::wrap(device_id, context_handle);
}

} // namespace memory

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_POINTER_HPP_

