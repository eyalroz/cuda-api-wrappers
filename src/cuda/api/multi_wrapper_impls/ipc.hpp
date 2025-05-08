/**
 * @file
 *
 * @brief Implementations of inter-processing-communications related functions and
 * classes requiring the definitions of multiple CUDA entity proxy classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_IPC_HPP_
#define MULTI_WRAPPER_IMPLS_IPC_HPP_

#if CUDA_VERSION >= 11020

#include "../ipc.hpp"
#include "../stream.hpp"
#include "../memory_pool.hpp"

namespace cuda {

namespace memory {

namespace pool {

namespace ipc {

class imported_ptr_t;

imported_ptr_t wrap(
	cuda::device::id_t device_id,
	context::handle_t context_handle,
	pool::handle_t pool_handle,
	void * ptr,
	stream::handle_t stream_handle,
	bool free_using_stream,
	bool owning) noexcept;

class imported_ptr_t {
protected: // constructors & destructor
	imported_ptr_t(
		cuda::device::id_t device_id,
		context::handle_t context_handle,
		pool::handle_t pool_handle,
		void * ptr,
		stream::handle_t stream_handle,
		bool free_using_stream,
		bool owning) noexcept
   	:
		device_id_(device_id),
		context_handle_(context_handle),
		pool_handle_(pool_handle),
		ptr_(ptr),
		stream_handle_(stream_handle),
		free_using_stream_(free_using_stream),
		owning_(owning) { }

public: // constructors & destructor
	friend imported_ptr_t wrap(
		cuda::device::id_t device_id,
		context::handle_t context_handle,
		pool::handle_t pool_handle,
		void * ptr,
		stream::handle_t stream_handle,
		bool free_using_stream,
		bool owning) noexcept;

	~imported_ptr_t() noexcept(false)
	{
		if (not owning_) { return; }
#ifdef THROW_IN_DESTRUCTORS
		try
#endif
		{
			if (free_using_stream_) {
				stream().enqueue.free(ptr_);
			}
			else {
				device::free(ptr_);
			}
		}
#ifdef THROW_IN_DESTRUCTORS
		catch (...) {}
#endif
	}

public: // operators

	imported_ptr_t(const imported_ptr_t& other) = delete;
	imported_ptr_t& operator=(const imported_ptr_t& other) = delete;
	imported_ptr_t& operator=(imported_ptr_t&& other) noexcept
	{
		::std::swap(device_id_, other.device_id_);
		::std::swap(context_handle_, other.context_handle_);
		::std::swap(pool_handle_, other.pool_handle_);
		::std::swap(ptr_, other.ptr_);
		::std::swap(stream_handle_, other.stream_handle_);
		::std::swap(free_using_stream_, other.free_using_stream_);
		::std::swap(owning_, other.owning_);
		return *this;
	}
	imported_ptr_t(imported_ptr_t&& other) noexcept = default;

public: // getters

	template <typename T = void>
	T* get() const noexcept
	{
		// If you're wondering why this cast is necessary - some IDEs/compilers
		// have the notion that if the method is const, `ptr_` is a const void* within it
		return static_cast<T*>(const_cast<void*>(ptr_));
	}
	stream_t stream() const
	{
		if (not free_using_stream_) throw ::std::runtime_error(
			"Request of the freeing stream of an imported pointer"
			"which is not to be freed on a stream.");
		return stream::wrap(device_id_, context_handle_, stream_handle_);
	}
	pool_t pool() noexcept
	{
		static constexpr bool non_owning { false };
		return memory::pool::wrap(device_id_, pool_handle_, non_owning);
	}

protected: // data members
	cuda::device::id_t  device_id_;
	context::handle_t   context_handle_;
	pool::handle_t      pool_handle_;
	void*               ptr_;
	stream::handle_t    stream_handle_;
	bool                free_using_stream_;
	bool                owning_;
}; // class imported_ptr_t

inline imported_ptr_t wrap(
	cuda::device::id_t device_id,
	context::handle_t context_handle,
	pool::handle_t pool_handle,
	void * ptr,
	stream::handle_t stream_handle,
	bool free_using_stream,
	bool owning) noexcept
{
	return imported_ptr_t { device_id, context_handle, pool_handle, ptr, stream_handle, free_using_stream, owning };
}

inline imported_ptr_t import_ptr(const pool_t& shared_pool, const ptr_handle_t& ptr_handle, const stream_t& freeing_stream)
{
	constexpr const auto free_using_stream { true };
	assert(shared_pool.device_id() == freeing_stream.device_id());
	void* raw_ptr = detail_::import_ptr(shared_pool.handle(), ptr_handle);
	static constexpr const bool is_owning { true };
	return wrap(
		shared_pool.device_id(),
		freeing_stream.context_handle(),
		shared_pool.handle(),
		raw_ptr,
		freeing_stream.handle(),
		free_using_stream,
		is_owning);
}

inline imported_ptr_t import_ptr(const pool_t& shared_pool, const ptr_handle_t& ptr_handle)
{
	constexpr const auto free_using_stream { false };
	auto free_without_using_stream = static_cast<bool>(free_using_stream);
	void* raw_ptr = detail_::import_ptr(shared_pool.handle(), ptr_handle);
	static constexpr const bool is_owning { true };
	return wrap(
		shared_pool.device_id(),
		context::detail_::none,
		shared_pool.handle(),
		raw_ptr,
		stream::default_stream_handle,
		free_without_using_stream,
		is_owning);
}

} // namespace ipc

} // namespace pool

} // namespace memory

} // namespace cuda

#endif // CUDA_VERSION >= 11020

#endif //CUDA_API_WRAPPERS_IPC_HPP
