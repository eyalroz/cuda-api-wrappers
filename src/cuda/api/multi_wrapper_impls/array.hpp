/**
 * @file
 *
* @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard events. Specifically:
 *
 * 1. Functions in the `cuda::array` namespace.
 * 2. Methods of @ref `cuda::array_t` and @ref `cuda::texture_view_t`.
  */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_ARRAY_HPP_
#define MULTI_WRAPPER_IMPLS_ARRAY_HPP_

#include "../array.hpp"
#include "../device.hpp"
#include "../event.hpp"
#include "../primary_context.hpp"
#include "../current_context.hpp"
#include "../current_device.hpp"
#include "../texture_view.hpp"

namespace cuda {

namespace array {

template <typename T, dimensionality_t NumDimensions>
array_t<T,NumDimensions> create(
	const context_t&             context,
	dimensions_t<NumDimensions>  dimensions)
{
	handle_t handle = detail_::create<T, NumDimensions>(context.handle(), dimensions);
	return wrap<T, NumDimensions>(context.device_id(), context.handle(), handle, dimensions);
}

template <typename T, dimensionality_t NumDimensions>
array_t<T,NumDimensions> create(
	device_t                     device,
	dimensions_t<NumDimensions>  dimensions)
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device.id());
	auto context_handle =  set_device_for_this_scope.primary_context_handle;
	handle_t handle = detail_::create<T, NumDimensions>(context_handle, dimensions);
	return wrap<T, NumDimensions>(device.id(), context_handle, handle, dimensions);
}

} // namespace array

inline device_t texture_view::device() const
{
	return device::get(context::detail_::get_device_id(context_handle_));
}

inline context_t texture_view::context() const
{
	return context::detail_::from_handle(context_handle_);
}

template <typename T, dimensionality_t NumDimensions>
device_t array_t<T, NumDimensions>::device() const noexcept
{
	return device::get(device_id_);
}

template <typename T, dimensionality_t NumDimensions>
context_t array_t<T, NumDimensions>::context() const
{
	// TODO: Save the device id in the array_t as well.
	return context::detail_::from_handle(context_handle_);
}

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_ARRAY_HPP_

