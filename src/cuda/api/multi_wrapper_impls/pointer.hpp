/**
 * @file
 *
 * @brief Implementations of `cuda_::memory::pointer_t` methods requiring the definitions
 * of multiple CUDA entity proxy classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_POINTER_HPP_
#define MULTI_WRAPPER_IMPLS_POINTER_HPP_

#include "../pointer.hpp"
#include "../device.hpp"
#include "../context.hpp"

namespace cuda_ {

namespace memory {

namespace pointer {

namespace detail {

inline cuda_::device::id_t device_id_of(void const *ptr)
{
#if CUDA_VERSION >= 9020
	return pointer::detail::get_attribute<CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>(ptr);
#else
	auto context_handle = context_handle_of(ptr);
	return context::detail::get_device_id(context_handle);
#endif
}

} // namespace detail

} // namespace pointer


template <typename T>
device_t pointer_t<T>::device() const
{
	return cuda_::device::get(pointer::detail::device_id_of(ptr_));
}

template <typename T>
context_t pointer_t<T>::context() const
{
	return context_of(ptr_);
}

inline context_t context_of(void const* ptr)
{
#if CUDA_VERSION >= 9020
	pointer::attribute_t attributes[] = {
		CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL,
		CU_POINTER_ATTRIBUTE_CONTEXT
	};
	cuda_::device::id_t device_id;
	context::handle_t context_handle;
	void* value_ptrs[] = {&device_id, &context_handle};
	pointer::detail::get_attributes(2, attributes, value_ptrs, ptr);
#else
	auto context_handle = pointer::detail::context_handle_of(ptr);
	auto device_id = context::detail::get_device_id(context_handle);
#endif
	return context::wrap(device_id, context_handle);
}

} // namespace memory

} // namespace cuda_

#endif // MULTI_WRAPPER_IMPLS_POINTER_HPP_

