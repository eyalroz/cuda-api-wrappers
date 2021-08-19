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
 * lazily (e.g. with an ::std::optional).
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_POINTER_HPP_
#define CUDA_API_WRAPPERS_POINTER_HPP_

#include <cuda/api/constants.hpp>
#include <cuda/api/error.hpp>
#include <cuda/common/types.hpp>

#include <cuda_runtime_api.h>

#ifndef NDEBUG
#include <cassert>
#endif

namespace cuda {

///@cond
class device_t;
///@endcond

namespace memory {


/**
 * @brief see @ref memory::host, @ref memory::device, @ref memory::managed
 */
enum type_t : ::std::underlying_type<cudaMemoryType>::type {
    host_memory         = cudaMemoryTypeHost,
    device_memory       = cudaMemoryTypeDevice,
#if CUDART_VERSION >= 10000
    unregistered_memory = cudaMemoryTypeUnregistered,
    managed_memory      = cudaMemoryTypeManaged,
#else
    unregistered_memory,
    managed_memory,
#endif // CUDART_VERSION >= 10000
};


namespace pointer {

/**
 * Holds various CUDA-related attributes of a pointer.
 */
struct attributes_t : cudaPointerAttributes {

	/**
	 * @brief indicates a choice memory space, management and access type -
	 * from among the options in @ref type_t .
	 *
	 * @return A value whose semantics are those introduced in CUDA 10.0,
	 * rather than those of CUDA 9 and earlier, where `type_t::device`
	 * actually signifies either device-only or `type_t::managed`.
	 */
	type_t memory_type() const
	{
	    // TODO: For some strange reason, g++ 6.x claims that converting
	    // to the underlying type is a "narrowing conversion", and doesn't
		// like some other conversions I've tried - so let's cast
		// more violently instead
		// Note: In CUDA v10.0, the semantics changed to what we're supporting

#if CUDART_VERSION >= 10000
	    return (type_t)cudaPointerAttributes::type;
#else // CUDART_VERSION < 10000
		using utype = typename ::std::underlying_type<cudaMemoryType>::type;
		if ( ((utype) memoryType == utype {type_t::device_memory}) and cudaPointerAttributes::isManaged) 
		{
			return type_t::managed_memory;
		}
	    return (type_t)(memoryType);
#endif // CUDART_VERSION >= 10000
	}
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

	/**
	 * @return Address of the pointed-to memory, regardless of which memory
	 * space it's in and whether or not it is accessible from the host
	 */
	T* get() const { return ptr_; }
	operator T*() const { return ptr_; }

public: // other non-mutators
	pointer::attributes_t attributes() const
	{
		pointer::attributes_t the_attributes;
		auto status = cudaPointerGetAttributes (&the_attributes, ptr_);
		throw_if_error(status, "Failed obtaining attributes of pointer " + cuda::detail_::ptr_as_hex(ptr_));
		return the_attributes;
	}
	device_t device() const noexcept;

	/**
	 * @returns A pointer into device-accessible memory (not necessary on-device memory though).
	 * CUDA ensures that, for pointers to memory not accessible on the CUDA device, `nullptr`
	 * is returned.
	 */
	T* get_for_device() const { return attributes().devicePointer; }

	/**
	 * @returns A pointer into device-accessible memory (not necessary on-device memory though).
	 * CUDA ensures that, for pointers to memory not accessible on the CUDA device, `nullptr`
	 * is returned.
	 */
	T* get_for_host() const { return attributes().hostPointer; }

	/**
	 * @returns For a mapped-memory pointer, returns the other side of the mapping,
	 * i.e. if this is the device pointer, returns the host pointer, otherwise
	 * returns the device pointer. For a managed-memory pointer, returns the
	 * single pointer usable on both device and host. In other cases returns `nullptr`.
	 *
	 * @note this relies on either the device and host pointers being `nullptr` in
	 * the case of a non-mapped pointer; and on the device and host pointers being
	 * identical to ptr_ for managed-memory pointers.
	 */
	pointer_t other_side_of_region_pair() const {
	    auto attrs = attributes();
#ifndef NDEBUG
	    assert(attrs.devicePointer == ptr_ or attrs.hostPointer == ptr_);
#endif
	    return pointer_t { ptr_ == attrs.devicePointer ? attrs.hostPointer : ptr_ };
	}

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

#endif // CUDA_API_WRAPPERS_POINTER_HPP_
