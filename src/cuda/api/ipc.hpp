/**
 * @file
 *
 * @brief wrappers for CUDA's facilities for sharing on-device
 * memory addresses and CUDA events between host processes (Inter-
 * Process Communication)
 *
 * CUDA addresses into device memory are not valid across different
 * host processes - somewhat, but not entirely, similarly to the
 * case of host memory addresses. Still, there is no reason why
 * different processes should not be able to interact with the same
 * on-device memory region. The same is also true for other entities,
 * such as streams and events.
 *
 * <p>CUDA provides several functions to enable different processes
 * to share at least memory addresses and events, which are wrapped
 * here. In addition to the free-standing functions, the class
 * @ref cuda::memory::ipc::imported_ptr_t is defined, usable by receiving
 * processes as an 'adapter' to incoming handles which may be passed
 * as-is to code requiring a proper pointer.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_IPC_HPP_
#define CUDA_API_WRAPPERS_IPC_HPP_

#include "context.hpp"
#include "types.hpp"
#include "error.hpp"

#include <string>

namespace cuda {

///@cond
class device_t;
class event_t;
///@endcond

namespace memory {

class pool_t;

namespace ipc {

/**
 * The concrete value passed between processes, used to tell
 * the CUDA Runtime API which memory area is desired.
 */
using ptr_handle_t = CUipcMemHandle;

class imported_ptr_t;
imported_ptr_t wrap(void * ptr,	bool owning) noexcept;

namespace detail_ {

/**
 * @brief Obtain a CUDA pointer from a handle passed
 * by inter-process communication
 *
 * @note the counterpart of @ref memory::ipc::unmap.
 *
 * @param handle the handle which allows us access to the on-device address
 * @return a pointer to the relevant address (which may not have the same value
 * as it would on a different processor.
 */
inline void* import(const ptr_handle_t& handle)
{
	CUdeviceptr device_ptr;
	auto status = cuIpcOpenMemHandle(&device_ptr, handle, CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS);
	throw_if_error_lazy(status, "Failed obtaining a device pointer from an IPC memory handle");
	return memory::as_pointer(device_ptr);
}

/**
 * @brief Unmap CUDA host-side memory shared by another process
 *
 * @param ipc_mapped_ptr pointer to the memory region to unmap
 */
inline void unmap(void* ipc_mapped_ptr)
{
	auto status = cuIpcCloseMemHandle(device::address(ipc_mapped_ptr));
	throw_if_error_lazy(status, "Failed unmapping IPC memory mapped to " + cuda::detail_::ptr_as_hex(ipc_mapped_ptr));
}

} // namespace detail_

/**
 * Obtain a handle for a region of on-device memory which can
 * be transmitted for use in another operating system process
 *
 * @note The name contains an underscore so as not to clash
 * with the C++ reserved word `export`
 *
 * @param device_ptr beginning of the region of memory
 * to be shared with other processes
 * @return a handle which another process can call @ref detail_::import()
 * on to obtain a device pointer it can use
 */
inline ptr_handle_t export_(void* device_ptr)
{
	ptr_handle_t handle;
	auto status = cuIpcGetMemHandle(&handle, device::address(device_ptr));
	throw_if_error_lazy(status, "Failed producing an IPC memory handle for device pointer "
		+ cuda::detail_::ptr_as_hex(device_ptr));
	return handle;
}

/**
 * @brief A smart-pointer-like class for memory obtained via inter-process communication.
 *
 * This RAII wrapper class maps memory in the current process' address space on
 * construction, and unmaps it on destruction, using a CUDA IPC handle.
 *
 * @tparam the element type in the stretch of IPC-shared memory
 */
class imported_ptr_t {
protected: // constructors & destructor
	imported_ptr_t(void* ptr, bool owning) : ptr_(ptr), owning_(owning)
	{
		if (ptr_ == nullptr) {
			throw ::std::logic_error("IPC memory handle yielded a null pointer");
		}
	}

public: // constructors & destructors
	friend imported_ptr_t wrap(void * ptr, bool owning) noexcept;

	~imported_ptr_t() noexcept(false)
	{
		if (owning_) { detail_::unmap(ptr_); }
	}

public: // operators

	imported_ptr_t(const imported_ptr_t& other) = delete;
	imported_ptr_t& operator=(const imported_ptr_t& other) = delete;
	imported_ptr_t& operator=(imported_ptr_t&& other) noexcept
	{
		::std::swap(ptr_, other.ptr_);
		::std::swap(owning_, other.owning_);
		return *this;
	}
	imported_ptr_t(imported_ptr_t&& other) noexcept = default;

public: // getters

	/// @return the unwrapped, raw, pointer to the imported memory
	template <typename T = void>
	T* get() const noexcept
	{
		// If you're wondering why this cast is necessary - some IDEs/compilers
		// have the notion that if the method is const, `ptr_` is a const void* within it
		return static_cast<T*>(const_cast<void*>(ptr_));
	}

	/// @return true if this object is charged with unmapping the imported memory upon destruction
	bool is_owning() const noexcept { return owning_; }

protected: // data members
	void*  ptr_;
	bool   owning_;
}; // class imported_ptr_t

/// Construct an instance of our wrapper class for IPC-imported memory from a raw pointer to the mapping
inline imported_ptr_t wrap(void * ptr, bool owning) noexcept
{
	return imported_ptr_t(ptr, owning);
}

/// Import memory from another process, given the appropriate handle
inline imported_ptr_t import(const ptr_handle_t& ptr_handle)
{
	auto raw_ptr = detail_::import(ptr_handle);
	return wrap(raw_ptr, do_take_ownership);
}

} // namespace ipc

#if CUDA_VERSION >= 11020
namespace pool {

namespace ipc {

using handle_t = void *;

template <shared_handle_kind_t Kind>
shared_handle_t<Kind> export_(const pool_t& pool);

namespace detail_ {

template <shared_handle_kind_t Kind>
pool::handle_t import(const shared_handle_t<Kind>& shared_pool_handle)
{
	memory::pool::handle_t result;
	static constexpr const unsigned long long flags { 0 };
	void * ptr_to_handle = static_cast<void*>(const_cast<shared_handle_t<Kind>*>(&shared_pool_handle));
	auto status = cuMemPoolImportFromShareableHandle(
		&result, ptr_to_handle, static_cast<CUmemAllocationHandleType>(Kind), flags);
	throw_if_error_lazy(status, "Importing an IPC-shared memory pool handle");
	return result;
}

} // namespace detail_

template <shared_handle_kind_t Kind>
pool_t import(const device_t& device, const shared_handle_t<Kind>& shared_pool_handle);

inline ptr_handle_t export_ptr(void* pool_allocated) {
	ptr_handle_t handle;
	auto status = cuMemPoolExportPointer(&handle, device::address(pool_allocated));
	throw_if_error_lazy(status,
		"Failed producing an IPC handle for memory-pool-allocated pointer "
		+ cuda::detail_::ptr_as_hex(pool_allocated));
	return handle;
}

namespace detail_ {

inline void* import_ptr(const pool::handle_t pool_handle, const ptr_handle_t& handle)
{
	CUdeviceptr imported;
	auto status = cuMemPoolImportPointer(&imported, pool_handle, const_cast<ptr_handle_t*>(&handle));
	throw_if_error_lazy(status, "Failed importing an IPC-exported a pool-allocated pointer");
	return as_pointer(imported);
}

} // namespace detail_

/**
 * @brief A smart-pointer-like class for memory obtained via IPC (inter-process communication),
 * allocated in a memory also shared over IPC.
 *
 * @note This class does not allocate any memory itself; and as for its "freeing" - it does
 * supposedly "free" the shared pointer - but that does not free it in the original pool;
 * it is merely a prerequisite for it being freed there.
 *
 * @note if a stream is provided upon construction, freeing will be scheduled on that stream;
 * otherwise, freeing will be streamless and may be synchronous/blocking.
 */
class imported_ptr_t;

imported_ptr_t import_ptr(const pool_t& shared_pool, const ptr_handle_t& ptr_handle);
imported_ptr_t import_ptr(const pool_t& shared_pool, const ptr_handle_t& ptr_handle, const stream_t& freeing_stream);

} // namespace ipc

} // namespace pool
#endif // CUDA_VERSION >= 11020

} // namespace memory

namespace event {
namespace ipc {

/**
 * The concrete value passed between processes, used to tell
 * the CUDA Runtime API which event is desired.
 */
using handle_t = CUipcEventHandle;

namespace detail_ {

inline handle_t export_(event::handle_t event_handle)
{
	handle_t ipc_handle;
	auto status = cuIpcGetEventHandle(&ipc_handle, event_handle);
	throw_if_error_lazy(status, "Failed obtaining an IPC event handle for " +
		event::detail_::identify(event_handle));
	return ipc_handle;
}

inline event::handle_t import(const handle_t& handle)
{
	event::handle_t event_handle;
	auto status = cuIpcOpenEventHandle(&event_handle, handle);
	throw_if_error_lazy(status, "Failed obtaining an event handle from an IPC event handle");
	return event_handle;
}

} // namespace detail_

/**
 * Enable use of an event which this process created by other processes
 *
 * @param event the event to share with other processes
 * @return the handle to pass directly to other processes with which they
 * may obtain a proper CUDA event
 *
 */
inline handle_t export_(const event_t& event);

/**
 * Obtain a proper CUDA event, corresponding to an event created by another
 * process, using a handle communicated via operating-system inter-process communications
 *
 * @note IMHO, the CUDA runtime API should allow for obtaining the device
 * from an event handle (or otherwise - have a handle provide both an event handle and
 * a device ID), but that is not currently the case.
 *
 * @param event_ipc_handle the handle obtained via inter-process communications
 */
///@{

 /**
  * @param device the device with which the imported event is associated
  */
inline event_t import(const device_t& device, const handle_t& event_ipc_handle);

/**
 * @param context the device-context with which the imported event is associated
 * @param event_ipc_handle The handle created by another process, to be imported
 * @return An event usable in the current process
 */
inline event_t import(const context_t& context, const handle_t& event_ipc_handle);
///@}

} // namespace ipc
} // namespace event
} // namespace cuda

#endif // CUDA_API_WRAPPERS_IPC_HPP_
