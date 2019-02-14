/**
 * @file multi_wrapper_impls.hpp
 *
 * @brief Implementations of memory-related API wrapper functions or methods
 * requiring the definition of the device wrapper class.

 * individual proxy class files, with the other classes forward-declared.
 */
#pragma once
#ifndef MEMORY_DEVICE_IMPLS_HPP_
#define MEMORY_DEVICE_IMPLS_HPP_

#include  <cuda/api/memory.hpp>
#include  <cuda/api/device.hpp>

namespace cuda {
namespace memory {

namespace device {

template <bool AssumedCurrent>
inline void* allocate(cuda::device_t<AssumedCurrent>& device, size_t size_in_bytes)
{
	return memory::device::allocate(device.id(), size_in_bytes);
}

}

namespace managed {

template <bool AssumedCurrent>
inline void* allocate(
	cuda::device_t<AssumedCurrent>&  device,
	size_t                           num_bytes,
	initial_visibility_t             initial_visibility)
{
	return allocate(device.id(), num_bytes, initial_visibility);
}

} // namespace managed


namespace mapped {

template <bool AssumedCurrent>
inline region_pair allocate(
	cuda::device_t<AssumedCurrent>&  device,
	size_t                           size_in_bytes,
	region_pair::allocation_options  options)
{
	return cuda::memory::mapped::allocate(device.id(), size_in_bytes, options);
}

} // namespace mapped
} // namespace memory
} // namespace cuda


#endif // MEMORY_DEVICE_IMPLS_HPP_
