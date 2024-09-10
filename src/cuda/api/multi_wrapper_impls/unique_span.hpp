/**
 * @file
 *
 * @brief Implementations of utility functions related to the @ref cuda::unique_span class
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_UNIQUE_SPAN_HPP_
#define MULTI_WRAPPER_IMPLS_UNIQUE_SPAN_HPP_

#include "../detail/unique_span.hpp"
#include "../current_device.hpp"
#include "../current_context.hpp"
#include "../primary_context.hpp"
#include "../memory.hpp"
#include "../types.hpp"
#include "../device.hpp"

namespace cuda {

namespace memory {

namespace device {

template <typename T>
unique_span<T> make_unique_span(const context_t& context, size_t num_elements)
{
	return detail_::make_unique_span<T>(context.handle(), num_elements);
}

/**
 * @brief Allocate (but do)
 * device-global memory
 *
 * @tparam T  an array type; _not_ the type of individual elements
 *
 * @param device        on which to construct the array of elements
 * @param num_elements  the number of elements to allocate
 * @return an ::std::unique_ptr pointing to the constructed T array
 */
template <typename T>
unique_span<T> make_unique_span(const device_t& device, size_t num_elements)
{
	auto pc = device.primary_context();
	CAW_SET_SCOPE_CONTEXT(pc.handle());
	return make_unique_span<T>(pc, num_elements);
}

/**
 * @brief Create a variant of ::std::unique_pointer for an array in
 * device-global memory on the current device.
 *
 * @note The allocation will be made in the device's primary context -
 * which will be created if it has not yet been.
 *
 * @tparam T  an array type; _not_ the type of individual elements
 *
 * @param num_elements  the number of elements to allocate
 *
 * @return an ::std::unique_ptr pointing to the constructed T array
 */
template <typename T>
unique_span<T> make_unique_span(size_t num_elements)
{
	auto current_device_id = cuda::device::current::detail_::get_id();
	auto pc = cuda::device::primary_context::detail_::leaky_get(current_device_id);
	return make_unique_span<T>(pc, num_elements);
}

} // namespace device

namespace managed {

template <typename T>
unique_span<T> make_unique_span(
	const context_t&      context,
	size_t                size,
    initial_visibility_t  initial_visibility)
{
	CAW_SET_SCOPE_CONTEXT(context.handle());
    switch (initial_visibility) {
    case initial_visibility_t::to_all_devices:
        return detail_::make_unique_span<T, initial_visibility_t::to_all_devices>(context.handle(), size);
    case initial_visibility_t::to_supporters_of_concurrent_managed_access:
        return detail_::make_unique_span<T, initial_visibility_t::to_supporters_of_concurrent_managed_access>(context.handle(), size);
    default:
        throw ::std::logic_error("Library not yet updated to support additional initial visibility values");
    }
}

template <typename T>
unique_span<T> make_unique_span(
	const device_t&       device,
	size_t                size,
    initial_visibility_t  initial_visibility)
{
	auto pc = device.primary_context();
	return make_unique_span<T>(pc, size, initial_visibility);
}

template <typename T>
unique_span<T> make_unique_span(
	size_t                size,
	initial_visibility_t  initial_visibility)
{
	auto current_device_id = cuda::device::current::detail_::get_id();
	auto pc = cuda::device::primary_context::detail_::leaky_get(current_device_id);
	return make_unique_span<T>(pc, size, initial_visibility);
}

} // namespace managed

} // namespace memory

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_UNIQUE_SPAN_HPP_

