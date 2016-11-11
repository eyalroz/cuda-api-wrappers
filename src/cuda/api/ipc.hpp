#pragma once
#ifndef CUDA_API_WRAPPERS_IPC_HPP_
#define CUDA_API_WRAPPERS_IPC_HPP_

#include <cuda/api/error.hpp>
#include <cuda/api/types.h>

#include <cuda_runtime_api.h>

#include <string>

namespace cuda {
namespace memory {
namespace ipc {

cudaIpcMemHandle_t export_(void* device_ptr) {
	cudaIpcMemHandle_t handle;
	auto status = cudaIpcGetMemHandle(&handle, device_ptr);
		throw_if_error(status,
			"Failed producing an IPC memory handle for device pointer " + cuda::detail::as_hex((size_t)device_ptr));
	return handle;
}

template <typename T = void>
inline T* import(const cudaIpcMemHandle_t& handle)
{
	void* device_ptr;
	auto status = cudaIpcOpenMemHandle(&device_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
	throw_if_error(status,
		"Failed obtaining a device pointer from an IPC memory handle");
	return reinterpret_cast<T*>(device_ptr);
}

void unmap(void* ipc_mapped_ptr)
{
	auto status = cudaIpcCloseMemHandle(ipc_mapped_ptr);
	throw_if_error(status, "Failed unmapping IPC memory mapped to " + cuda::detail::as_hex((size_t) ipc_mapped_ptr));
}

template <typename T = void>
class imported_t {
public: // constructors & destructor
	imported_t(const cudaIpcMemHandle_t& handle) : ptr_(import<T>(handle))
	{
		if (ptr_ == nullptr) {
			throw std::logic_error("IPC memory handle yielded a null pointer");
		}
	}

	/**
	 * @note May throw! Be very careful.
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

/*
	imported_t& operator=(imported_t&& other)
	{
		if (this == &other) { return *this; }
		std::swap(ptr_, other.ptr_);
		return *this;
	}
	imported_t(const imported_t&& other) {
		std::swap(ptr_, other.ptr_);
	}
*/

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
};

} // namespace ipc
} // namespace memory

namespace event {
namespace ipc {

cudaIpcEventHandle_t export_(id_t event_id)
{
	cudaIpcEventHandle_t ipc_handle;
	auto status = cudaIpcGetEventHandle(&ipc_handle, event_id);
	throw_if_error(status,
		"Failed obtaining an IPC event handle for event " + cuda::detail::as_hex((size_t) event_id));
	return ipc_handle;
}

inline event::id_t import(const cudaIpcEventHandle_t& handle)
{
	event::id_t event_id;
	auto status = cudaIpcOpenEventHandle(&event_id, handle);
	throw_if_error(status,
		"Failed obtaining an event ID from an IPC event handle");
	return event_id;
}

} // namespace ipc
} // namespace event
} // namespace cuda

#endif /* CUDA_API_WRAPPERS_IPC_HPP_ */
