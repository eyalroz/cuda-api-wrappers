/**
 * @file ipc.hpp
 *
 * @brief wrappers for CUDA's facilities for sharing on-device
 * memory addresses and CUDA events between host processes
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
 * @ref cuda::memory::ipc::imported_t is defined, usable by receiving
 * processes as an 'adapter' to incoming handles which may be passed
 * as-is to code requiring a propoer pointer.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_IPC_HPP_
#define CUDA_API_WRAPPERS_IPC_HPP_

#include <cuda/api/device.hpp>
#include <cuda/api/error.hpp>
#include <cuda/common/types.hpp>

#include <cuda_runtime_api.h>

#include <string>

namespace cuda {

class device_t;
class event_t;

namespace memory {
namespace ipc {

/**
 * The concrete value passed between processes, used to tell
 * the CUDA Runtime API which memory area is desired.
 */
using handle_t = cudaIpcMemHandle_t;

/**
 * Obtain a handle for a region of on-device memory which can
 * be transmitted for use in another operating system process
 *
 * @note The name contains an underscore so as not to clash
 * with the C++ reserved word `export`
 *
 * @param device_ptr beginning of the region of memory
 * to be shared with other processes
 * @return a handle which another process can call @ref import()
 * on to obtain a device pointer it can use
 */
inline handle_t export_(void* device_ptr) {
	handle_t handle;
	auto status = cudaIpcGetMemHandle(&handle, device_ptr);
		cuda::throw_if_error(status,
			"Failed producing an IPC memory handle for device pointer " + cuda::detail_::ptr_as_hex(device_ptr));
	return handle;
}

/**
 * @brief Obtain a CUDA pointer from a handle passed
 * by inter-process communication
 *
 * @note the couterpart of @ref memory::ipc::unmap.
 *
 * @param handle the handle which allows us access to the on-device address
 * @return a pointer to the relevant address (which may not have the same value
 * as it would on a different processor.
 */
template <typename T = void>
inline T* import(const handle_t& handle)
{
	void* device_ptr;
	auto status = cudaIpcOpenMemHandle(&device_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
	cuda::throw_if_error(status,
		"Failed obtaining a device pointer from an IPC memory handle");
	return reinterpret_cast<T*>(device_ptr);
}

/**
 * @brief Unmap CUDA host-side memory shared by another process
 *
 * @param ipc_mapped_ptr pointer to the memory region to unmap
 */
inline void unmap(void* ipc_mapped_ptr)
{
	auto status = cudaIpcCloseMemHandle(ipc_mapped_ptr);
	cuda::throw_if_error(status,
		"Failed unmapping IPC memory mapped to " +
		cuda::detail_::ptr_as_hex(ipc_mapped_ptr));
}

/**
 * @brief A smart-pointer-like class for memory obtained via inter-process communication.
 *
 * This RAII wrapper class maps memory in the current process' address space on
 * construction, and unmaps it on destruction, using a CUDA IPC handle.
 *
 * @tparam the element type in the stretch of IPC-shared memory
 */
template <typename T = void>
class imported_t {
public: // constructors & destructor
	imported_t(const handle_t& handle) : ptr_(import<T>(handle))
	{
		if (ptr_ == nullptr) {
			throw ::std::logic_error("IPC memory handle yielded a null pointer");
		}
	}

	/**
	 * @note This may (?) throw! Be very careful.
	 */
	~imported_t() {
		if (ptr_ == nullptr) { return; }
		unmap(ptr_);
	}

public: // operators

	imported_t(const imported_t& other) = delete;
	imported_t& operator=(const imported_t& other) = delete;
	imported_t& operator=(imported_t&& other) = delete;
	imported_t(const imported_t&& other) = delete;

	operator T*() const { return ptr_; }

public: // getters

	T* get() const { return ptr_; }

protected: // data members
	/**
	 * Also used to indicate ownership of the handle; if it's nullptr,
	 * ownership has passed to another imported_t and we don't need
	 * to close the handle
	 */
	T*         ptr_;
}; // class imported_t

} // namespace ipc
} // namespace memory

namespace event {
namespace ipc {

/**
 * The concrete value passed between processes, used to tell
 * the CUDA Runtime API which event is desired.
 */
using handle_t = cudaIpcEventHandle_t;

namespace detail_ {

inline handle_t export_(id_t event_id)
{
	handle_t ipc_handle;
	auto status = cudaIpcGetEventHandle(&ipc_handle, event_id);
	cuda::throw_if_error(status,
		"Failed obtaining an IPC event handle for event " + cuda::detail_::ptr_as_hex(event_id));
	return ipc_handle;
}

inline event::id_t import(const handle_t& handle)
{
	event::id_t event_id;
	auto status = cudaIpcOpenEventHandle(&event_id, handle);
	cuda::throw_if_error(status,
		"Failed obtaining an event ID from an IPC event handle");
	return event_id;
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
inline handle_t export_(event_t& event);

/**
 * Obtain a proper CUDA event, corresponding to an event created by another
 * process, using a handle communicated via operating-system inter-process communications
 *
 * @note IMHO, the CUDA runtime API should allow for obtaining the device
 * from an event handle (or otherwise - have a handle provide both an event ID and
 * a device ID), but that is not currently the case.
 *
 * @param device the device to which the imported event corresponds
 * @param handle the handle obtained via inter-process communications
 */
inline event_t import(device_t& device, const handle_t& handle);

} // namespace ipc
} // namespace event
} // namespace cuda

#endif // CUDA_API_WRAPPERS_IPC_HPP_
