/**
 * @file
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
#include <cuda/api/types.hpp>

#include <cuda_runtime_api.h>
#include <cuda.h>

#ifndef NDEBUG
#include <cassert>
#endif

namespace cuda {

///@cond
class device_t;
class context_t;
///@endcond

namespace memory {

/**
 * @brief see @ref memory::host, @ref memory::device, @ref memory::managed
 */
enum type_t : ::std::underlying_type<CUmemorytype>::type {
    host_         = CU_MEMORYTYPE_HOST,
    device_       = CU_MEMORYTYPE_DEVICE,
	array         = CU_MEMORYTYPE_ARRAY,
    unified_      = CU_MEMORYTYPE_UNIFIED,
	managed_      = CU_MEMORYTYPE_UNIFIED, // an alias (more like the runtime API name)
	non_cuda      = ~(::std::underlying_type<CUmemorytype>::type{0})
};

namespace pointer {

namespace detail_ {

// Note: We could theoretically template this, but - there don't seem to be a lot of "clients" for this
// function right now, and I would rather not drag in <tuple>
void get_attributes(unsigned num_attributes, pointer::attribute_t* attributes, void** value_ptrs, const void* ptr);

template <attribute_t attribute> struct attribute_value {};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_CONTEXT>                    { using type = context::handle_t;};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_MEMORY_TYPE>                { using type = memory::type_t;};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_DEVICE_POINTER>             { using type = void*;};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_HOST_POINTER>               { using type = void*;};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_P2P_TOKENS>                 { using type = struct CUDA_POINTER_ATTRIBUTE_P2P_TOKEN;};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_SYNC_MEMOPS>                { using type = int;};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_BUFFER_ID>                  { using type = unsigned long long;};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_IS_MANAGED>                 { using type = int;};
#if CUDA_VERSION >= 9020
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>             { using type = cuda::device::id_t;};
#if CUDA_VERSION >= 10020
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_RANGE_START_ADDR>           { using type = void*;};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_RANGE_SIZE>                 { using type = size_t;};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_MAPPED>                     { using type = int;};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE> { using type = int;};
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES>       { using type = uint64_t;};
#if CUDA_VERSION >= 11030
template <> struct attribute_value<CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE>             { using type = pool::handle_t;};
#endif // CUDA_VERSION >= 11030
#endif // CUDA_VERSION >= 10020
#endif // CUDA_VERSION >= 9020

template <CUpointer_attribute attribute>
using attribute_value_t = typename attribute_value<attribute>::type;

template<attribute_t attribute>
struct status_and_attribute_value {
	status_t status;
	attribute_value_t<attribute> value;
};

template<attribute_t attribute>
status_and_attribute_value<attribute> get_attribute_with_status(const void *ptr);

template <attribute_t attribute>
attribute_value_t<attribute> get_attribute(const void* ptr);

inline context::handle_t context_handle_of(const void* ptr)
{
	return pointer::detail_::get_attribute<CU_POINTER_ATTRIBUTE_CONTEXT>(ptr);
}

inline cuda::device::id_t device_id_of(const void* ptr);

} // namespace detail_

} // namespace pointer

inline memory::type_t type_of(const void* ptr)
{
	auto result = pointer::detail_::get_attribute_with_status<CU_POINTER_ATTRIBUTE_MEMORY_TYPE>(ptr);
	// Note: As of CUDA 12, CUDA treats passing a non-CUDA-allocated pointer to the memory type check
	// as an error, though it really should not be
	return (result.status == status::named_t::invalid_value) ?
		memory::type_t::non_cuda : result.value;
}

inline context_t context_of(const void* ptr);


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

protected:
	template <pointer::attribute_t attribute>
	pointer::detail_::attribute_value_t<attribute> get_attribute() const
	{
		return pointer::detail_::get_attribute<attribute>(ptr_);
	}

public: // other non-mutators

	/**
	 * Returns a proxy for the device into whose global memory the pointer points.
	 */
	device_t device() const;

	/**
	 * Returns a proxy for the context in which the memory area, into which the pointer
	 * points, was allocated.
	 */
	context_t context() const;

	/**
	 * @returns A pointer into device-accessible memory (not necessary on-device memory though).
	 * CUDA ensures that, for pointers to memory not accessible on the CUDA device, `nullptr`
	 * is returned.
	 *
	 * @note With unified memory, this will typically return the same value as the original pointer.
	 */
	T* get_for_device() const
	{
		return (T*) pointer::detail_::get_attribute<CU_POINTER_ATTRIBUTE_DEVICE_POINTER>(ptr_);
	}

	/**
	 * @returns A pointer into device-accessible memory (not necessary on-device memory though).
	 * CUDA ensures that, for pointers to memory not accessible on the CUDA device, `nullptr`
	 * is returned.
	 *
	 * @note With unified memory, this will typically return the same value as the original pointer.
	 */
	T* get_for_host() const
	{
		return (T*) pointer::detail_::get_attribute<CU_POINTER_ATTRIBUTE_HOST_POINTER>(ptr_);
	}

#if CUDA_VERSION >= 10020
	region_t containing_range() const
	{
		// TODO: Consider checking the alignment
		auto range_start = pointer::detail_::get_attribute<CU_POINTER_ATTRIBUTE_RANGE_START_ADDR>(ptr_);
		auto range_size = pointer::detail_::get_attribute<CU_POINTER_ATTRIBUTE_RANGE_SIZE>(ptr_);
		return { range_start, range_size};
	}
#endif

	/**
	 * @returns For a mapped-memory pointer, returns the other side of the mapping,
	 * i.e. if this is the device pointer, returns the host pointer, otherwise
	 * returns the device pointer. For a managed-memory pointer, or in a unified setting,
	 * returns the single pointer usable on both device and host. In other cases
	 * returns `nullptr`.
	 *
	 * @note this relies on either the device and host pointers being `nullptr` in
	 * the case of a non-mapped pointer; and on the device and host pointers being
	 * identical to ptr_ for managed-memory pointers.
	 */
	pointer_t other_side_of_region_pair() const
	{
		pointer::attribute_t attributes[] = {
		CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
		CU_POINTER_ATTRIBUTE_HOST_POINTER,
		CU_POINTER_ATTRIBUTE_DEVICE_POINTER
		};
		type_t memory_type;
		T* host_ptr;
		T* device_ptr;
		void* value_ptrs[] = { &memory_type, &host_ptr, &device_ptr };
		pointer::detail_::get_attributes(3, attributes, value_ptrs, ptr_);

#ifndef NDEBUG
		assert(host_ptr == ptr_ or device_ptr == ptr_);
#endif
		return { ptr_ == host_ptr ? device_ptr : host_ptr };
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
inline pointer_t<T> wrap(T* ptr) noexcept { return { ptr }; }

} // namespace pointer
} // namespace memory
} // namespace cuda

#endif // CUDA_API_WRAPPERS_POINTER_HPP_
