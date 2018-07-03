/**
 * @file pointer.hpp
 *
 * @brief A wrapper class for host and/or device pointers, allowing
 * easy access to CUDA's pointer attributes.
 *
 * @note at the moment, this class is not used by other sections of the API
 * wrappers; specifically, freestanding functions and methods returning
 * pointers return raw `T*`'s rather than `pointer_t<T>`'s.
 * This may change in the future.
 *
 * @todo Consider allowing for storing attributes within the class,
 * lazily (e.g. with an std::optional).
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_POINTER_HPP_
#define CUDA_API_WRAPPERS_POINTER_HPP_

#include <cuda/api/types.h>
#include <cuda/api/constants.h>
#include <cuda/api/error.hpp>

#include <cuda_runtime_api.h>

namespace cuda {
namespace memory {

namespace pointer {

struct attributes_t : cudaPointerAttributes {
	bool on_host()    const  { return memoryType == cudaMemoryTypeHost; }
	bool on_device()  const  { return memoryType == cudaMemoryTypeDevice; }
	bool is_managed() const  { return isManaged; }
};

} // namespace pointer

/**
 * A convenience wrapper around a raw pointer "known" to the CUDA runtime
 * and which thus has various kinds of associated information which this
 * wrapper allows access to.
 */
template <typename T>
class pointer_t {
public: // getters and operators
	T* get() const { return ptr_; }
	operator T*() const { return ptr_; }

public: // other non-mutators
	pointer::attributes_t attributes() const
	{
		pointer::attributes_t the_attributes;
		auto status = cudaPointerGetAttributes (&the_attributes, ptr_);
		throw_on_error(status, "Failed obtaining attributes of pointer " + cuda::detail::ptr_as_hex(ptr_));
		return the_attributes;
	}
	bool                is_on_host()     const  { return attributes().on_host();     }
	bool                is_on_device()   const  { return attributes().on_device();   }
	bool                is_managed()     const  { return attributes().is_managed();  }
	cuda::device::id_t  device_id()      const  { return attributes().device;        }
	T*                  get_for_device() const  { return attributes().hostPointer;   }
	T*                  get_for_host()   const  { return attributes().devicePointer; }

public: // constructors
	pointer_t(T* ptr) noexcept : ptr_(ptr) { }
	pointer_t(const pointer_t& other) noexcept = default;
	pointer_t(pointer_t&& other) noexcept = default;

protected: // data members
	T* const ptr_;
};

namespace pointer {

/**
 * Wraps an existing pointer in a @ref pointer_t wrapper
 *
 * @param ptr a pointer - into either device or host memory -
 * to be wrapped.
 */
template<typename T>
inline pointer_t<T> wrap(T* ptr) noexcept { return pointer_t<T>(ptr); }

} // namespace pointer
} // namespace memory
} // namespace cuda

#endif /* CUDA_API_WRAPPERS_POINTER_HPP_ */
