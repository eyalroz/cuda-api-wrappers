/**
 * @file
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_CURRENT_CONTEXT_HPP_
#define CUDA_API_WRAPPERS_CURRENT_CONTEXT_HPP_

#include <cuda/api/types.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/versions.hpp>

#include <cuda.h>

namespace cuda {

///@cond
class device_t;
class context_t;
///@endcond

namespace context {

namespace current {

namespace detail_ {

/**
 * Returns a raw handle for the current CUDA context
 *
 * @return the raw handle from the CUDA driver - if one exists; none
 * if no context is current/active.
 */
inline bool is_(handle_t handle)
{
	handle_t current_context_handle;
	auto status = cuCtxGetCurrent(&current_context_handle);
	switch(status) {
	case CUDA_ERROR_NOT_INITIALIZED:
	case CUDA_ERROR_INVALID_CONTEXT:
		return false;
	case CUDA_SUCCESS:
		return (handle == current_context_handle);
	default:
		throw cuda::runtime_error((status_t) status, "Failed determining whether there's a current context, or what it is");
	}
}

/**
 * Returns a raw handle for the current CUDA context
 *
 * @return the raw handle from the CUDA driver - if one exists; none
 * if no context is current/active.
 */
inline handle_t get_handle()
{
	handle_t handle;
	auto status = cuCtxGetCurrent(&handle);
	throw_if_error(status, "Failed obtaining the current context's handle");
	return handle;
}

// Note: not calling this get_ since flags are read-only anyway
inline context::flags_t get_flags()
{
	context::flags_t result;
	auto status = cuCtxGetFlags(&result);
	throw_if_error(status, "Failed obtaining the current context's flags");
	return result;
}

inline device::id_t get_device_id()
{
	device::id_t device_id;
	auto result = cuCtxGetDevice(&device_id);
	throw_if_error(result, "Failed obtaining the current context's device");
	return device_id;
}

} // namespace detail_

inline bool exists();
inline context_t get();
inline void set(const context_t& context);

namespace detail_ {

/**
 * Push a context handle onto the top of the context stack - if it is not already on the
 * top of the stack
 *
 * @param context_handle A context handle to push
 *
 * @note behavior undefined if you try to push @ref none
 */
inline void push(handle_t context_handle)
{
	auto status = cuCtxPushCurrent(context_handle);
	throw_if_error(status,
		"Failed pushing to the top of the context stack: " + context::detail_::identify(context_handle));
}

/**
 * Push a context handle onto the top of the context stack - if it is not already on the
 * top of the stack
 *
 * @param context_handle A context handle to push
 *
 * @return true if a push actually occurred
 *
 * @note behavior undefined if you try to push @ref none
 * @note The CUDA context stack is not a proper stack, in that it doesn't allow multiple
 * consecutive copes of the same context on the stack; hence there is no `push()` method.
 */
inline bool push_if_not_on_top(handle_t context_handle)
{
	if (detail_::get_handle() == context_handle) { return false; }
	push(context_handle); return true;
}

inline context::handle_t pop()
{
	handle_t popped_context_handle;
	auto status = cuCtxPopCurrent(&popped_context_handle);
	throw_if_error(status, "Failed popping the current CUDA context");
	return popped_context_handle;
}

inline void set(handle_t context_handle)
{
	// TODO: Would this help?
	// if (detail_::get_handle() == context_handle_) { return; }
	auto status = static_cast<status_t>(cuCtxSetCurrent(context_handle));
	throw_if_error(status,
	    "Failed setting the current context to " + context::detail_::identify(context_handle));
}

/**
 * @note See the out-of-`detail_::` version of this class.
 *
 */
class scoped_override_t {
public:
	bool hold_primary_context_ref_unit_;
	device::id_t device_id_or_0_;

	explicit scoped_override_t(handle_t context_handle)	: scoped_override_t(false, 0, context_handle) {}
	scoped_override_t(device::id_t device_for_which_context_is_primary, handle_t context_handle)
		: scoped_override_t(true, device_for_which_context_is_primary, context_handle) {}
	explicit scoped_override_t(bool hold_primary_context_ref_unit, device::id_t device_id, handle_t context_handle);
	~scoped_override_t();
};

/**
 * @note See also the more complex @ref cuda::context::current::scoped_existence_ensurer_t ,
 * which does _not_ take a fallback context handle, and rather obtains a reference to
 * a primary context on its own.
 */
class scoped_ensurer_t {
public:
	bool context_was_pushed_on_construction;

	explicit scoped_ensurer_t(bool force_push, handle_t fallback_context_handle)
		: context_was_pushed_on_construction(force_push)
	{
		if (force_push) { push(fallback_context_handle); }
	}

	explicit scoped_ensurer_t(handle_t fallback_context_handle)
		: scoped_ensurer_t(not exists(), fallback_context_handle)
	{}

	~scoped_ensurer_t() { if (context_was_pushed_on_construction) { pop(); } }
};

} // namespace detail_

/**
 * A RAII-based mechanism for pushing a context onto the context stack
 * for what remains of the current (C++ language) scope - making it the
 * current context - then popping it back when exiting the scope -
 * restoring the stack and the current context to what they had been
 * previously.
 *
 * @note if some other code pushes/pops from the context stack during
 * the lifetime of this class, the pop-on-destruction may fail, or
 * succeed but pop some other context handle than the one originally.
 * pushed.
 *
 */
class scoped_override_t;

/**
 * This macro will set the current device for the remainder of the scope in which it is
 * invoked, and will change it back to the previous value when exiting the scope. Use
 * it as an opaque command, which does not explicitly expose the variable defined under
 * the hood to effect this behavior.
 */
#define CUDA_CONTEXT_FOR_THIS_SCOPE(_cuda_context) \
	::cuda::context::current::scoped_override_t scoped_context_override{ _cuda_context }


inline bool push_if_not_on_top(const context_t& context);
inline void push(const context_t& context);

inline void synchronize()
{
	auto status = cuCtxSynchronize();
	if (not is_success(status)) {
		throw cuda::runtime_error(status, "Failed synchronizing current context");
	}
}

namespace detail_ {

inline void synchronize(context::handle_t handle)
{
	auto status = cuCtxSynchronize();
	if (not is_success(status)) {
		throw cuda::runtime_error(status,"Failed synchronizing "
			+ context::detail_::identify(handle));
	}
}

inline void synchronize(device::id_t device_id, context::handle_t handle)
{
	auto status = cuCtxSynchronize();
	if (not is_success(status)) {
		throw cuda::runtime_error(status, "Failed synchronizing "
			+ context::detail_::identify(handle, device_id));
	}
}

} // namespace detail

} // namespace current

} // namespace context

} // namespace cuda

#endif // CUDA_API_WRAPPERS_CURRENT_CONTEXT_HPP_
