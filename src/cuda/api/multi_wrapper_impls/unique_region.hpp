/**
 * @file
 *
 * @brief Implementations of `make_unique_region()` functions in different
 * memory spaces
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_UNIQUE_REGION_HPP_
#define MULTI_WRAPPER_IMPLS_UNIQUE_REGION_HPP_

#include "../unique_region.hpp"
#include "../types.hpp"
#include "../device.hpp"

namespace cuda {

namespace memory {

namespace device {

inline unique_region make_unique_region(const context_t& context, cuda::size_t num_elements)
{
	return detail_::make_unique_region(context.handle(), num_elements);
}

/**
 * @brief Create a variant of ::std::unique_pointer for an array in
 * device-global memory
 *
 * @tparam T  an array type; _not_ the type of individual elements
 *
 * @param device        on which to construct the array of elements
 * @param num_elements  the number of elements to allocate
 * @return an ::std::unique_ptr pointing to the constructed T array
 */
inline unique_region make_unique_region(const device_t& device, size_t num_elements)
{
	auto pc = device.primary_context();
	return make_unique_region(pc, num_elements);
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
inline unique_region make_unique_region(size_t num_elements)
{
	auto current_device_id = cuda::device::current::detail_::get_id();
	auto pc = cuda::device::primary_context::detail_::leaky_get(current_device_id);
	return make_unique_region(pc, num_elements);
}

} // namespace device

namespace host {

inline unique_region make_unique_region(size_t num_bytes)
{
	return unique_region { allocate(num_bytes) };
}

} // namespace host

namespace managed {

/**
 * @brief Allocate a region of managed memory, accessible both from CUDA devices
 * and from the CPU.
 *
 * @param context A context of possible single-device-visibility
 *
 * @returns An owning RAII/CADRe object for the allocated managed memory region
 */
inline unique_region make_unique_region(
	const context_t&      context,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility)
{
	CAW_SET_SCOPE_CONTEXT(context.handle());
	return unique_region { detail_::allocate_in_current_context(num_bytes, initial_visibility) };
}

/**
 * @brief Allocate a region of managed memory, accessible both from CUDA devices
 * and from the CPU.
 *
 * @param context A context of possible single-device-visibility
 *
 * @returns An owning RAII/CADRe object for the allocated managed memory region
 */
inline unique_region make_unique_region(
	const device_t&       device,
	size_t                num_bytes,
	initial_visibility_t  initial_visibility)
{
	auto pc = device.primary_context();
	return make_unique_region(pc, num_bytes, initial_visibility);
}

inline unique_region make_unique_region(
	size_t                num_bytes,
	initial_visibility_t  initial_visibility)
{
	auto current_device_id = cuda::device::current::detail_::get_id();
	auto pc = cuda::device::primary_context::detail_::leaky_get(current_device_id);
	return make_unique_region(pc, num_bytes, initial_visibility);
}

} // namespace managed

} // namespace memory

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_UNIQUE_REGION_HPP_

