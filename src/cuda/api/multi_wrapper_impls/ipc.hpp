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

namespace cuda_ {

namespace memory {

namespace pool {

namespace ipc {

class imported_ptr_t;

imported_ptr_t wrap(
	cuda_::device::id_t device_id,
	context::handle_t context_handle,
	pool::handle_t pool_handle,
	void * ptr,
	stream::handle_t stream_handle,
	bool free_using_stream,
	bool owning) noexcept;

namespace detail {


// Note: We cannot use the vanilla handle ownership mechanism, because for this class,
// destruction is possible either immediately or on a stream.
//
// TODO: Consider splitting the class according to the destruction method

struct release_state_t {
	handle_t handle;
	optional<stream::handle_t> stream_handle;
};

struct releaser {
	void operator()(release_state_t const& release_state) const CAW_DESTRUCTOR_EXCEPTION_SPEC
	{
		// TODO: Consider creating nothrow and optional-stream-handle versions of the free functions,
		// to make our life here easier
#ifndef CAW_THROW_IN_DESTRUCTORS
		try
#endif
		{
			if (release_state.stream_handle) {
				memory::device::detail::free_on_stream(release_state.handle, *release_state.stream_handle);
			}
			else {
				memory::device::free(release_state.handle);
			}
		}
#ifndef CAW_THROW_IN_DESTRUCTORS
		catch (std::exception&) { }
#endif
	}
};

inline void release(handle_t handle, optional<stream::handle_t> const& stream_handle)
{
	releaser{}({ handle, stream_handle });
}

} // namespace detail

class imported_ptr_t {
protected: // constructors & destructor
	imported_ptr_t(
		cuda_::device::id_t device_id,
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
		ownership_(owning, { ptr, free_using_stream ? nullopt : make_optional(stream_handle) }) { }

public: // constructors & destructor
	friend imported_ptr_t wrap(
		cuda_::device::id_t device_id,
		context::handle_t context_handle,
		pool::handle_t pool_handle,
		void * ptr,
		stream::handle_t stream_handle,
		bool free_using_stream,
		bool owning) noexcept;

public: // operators

	imported_ptr_t(const imported_ptr_t& other) = delete;
	imported_ptr_t& operator=(const imported_ptr_t& other) = delete;
	imported_ptr_t& operator=(imported_ptr_t&& other) noexcept = default;
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
		if (not stream_handle_) throw std::runtime_error(
			"Request of the freeing stream of an imported pointer"
			"which is not to be freed on a stream.");
		return stream::wrap(device_id_, context_handle_, *stream_handle_);
	}
	pool_t pool() const noexcept
	{
		static constexpr bool non_owning { false };
		return memory::pool::wrap(device_id_, pool_handle_, non_owning);
	}

protected: // data members
	cuda_::device::id_t  device_id_;
	context::handle_t    context_handle_;
	pool::handle_t       pool_handle_;
	void*                ptr_;
	optional<stream::handle_t> stream_handle_;
	cuda_::detail::token_holder<detail::releaser, detail::release_state_t> ownership_;
}; // class imported_ptr_t

inline imported_ptr_t wrap(
	cuda_::device::id_t device_id,
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
	constexpr auto free_using_stream { true };
	assert(shared_pool.device_id() == freeing_stream.device_id());
	void* raw_ptr = detail::import_ptr(shared_pool.handle(), ptr_handle);
	static constexpr bool is_owning { true };
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
	constexpr auto free_using_stream { false };
	auto free_without_using_stream = static_cast<bool>(free_using_stream);
	void* raw_ptr = detail::import_ptr(shared_pool.handle(), ptr_handle);
	static constexpr bool is_owning { true };
	return wrap(
		shared_pool.device_id(),
		context::detail::none,
		shared_pool.handle(),
		raw_ptr,
		stream::default_stream_handle,
		free_without_using_stream,
		is_owning);
}

} // namespace ipc

} // namespace pool

} // namespace memory

} // namespace cuda_

#endif // CUDA_VERSION >= 11020

#endif //CUDA_API_WRAPPERS_IPC_HPP
