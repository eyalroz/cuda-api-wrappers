/**
 * @file
 */

#ifndef CUDA_API_WRAPPERS_PRIMARY_CONTEXT_HPP_
#define CUDA_API_WRAPPERS_PRIMARY_CONTEXT_HPP_

#include "current_context.hpp"
#include "context.hpp" // A primary context is a context, so can't avoid this

namespace cuda {

namespace device {

///@cond
class primary_context_t;
///@endcond

namespace primary_context {

namespace detail_ {

struct state_t {
	context::flags_t flags;
	int              is_active; // non-zero value means true
};

inline state_t raw_state(device::id_t device_id)
{
	state_t result;
	auto status = cuDevicePrimaryCtxGetState(device_id, &result.flags, &result.is_active);
	throw_if_error(status, "Failed obtaining the state of the primary context for "
		+ device::detail_::identify(device_id));
	// Note: Not sanitizing the flags from having CU_CTX_MAP_HOST set
	return result;
}

inline context::flags_t flags(device::id_t device_id)
{
	return raw_state(device_id).flags & ~CU_CTX_MAP_HOST;
}

inline bool is_active(device::id_t device_id)
{
	return raw_state(device_id).is_active;
}

// We used this wrapper for a one-linear to track PC releases
inline status_t decrease_refcount_nothrow(device::id_t device_id) noexcept
{
	auto result = cuDevicePrimaryCtxRelease(device_id);
	return result;
}

inline void decrease_refcount(device::id_t device_id)
{
	auto status = decrease_refcount_nothrow(device_id);
	throw_if_error_lazy(status, "Failed releasing the reference to the primary context for " + device::detail_::identify(device_id));
}

inline handle_t obtain_and_increase_refcount(device::id_t device_id)
{
	handle_t primary_context_handle;
	auto status = cuDevicePrimaryCtxRetain(&primary_context_handle, device_id);
	throw_if_error_lazy(status,
		"Failed obtaining (and possibly creating, and adding a reference count to) the primary context for "
		+ device::detail_::identify(device_id));
	return primary_context_handle;
}

inline void increase_refcount(device::id_t device_id)
{
	obtain_and_increase_refcount(device_id);
}

} // namespace detail_

/**
 * @returns true if the device's primary context is active (i.e. has resources allocated for it),
 * which implies we are holding a refcount unit for it somewhere.
 *
 * @note recall a primary context being active does not mean that it is the _current_ context
 */
inline bool is_active(const device_t& device);

/**
 * @brief Destroy and clean up all resources associated with the specified device's primary context
 *
 * @param device The device whose primary context is to be destroyed
 */
void destroy(const device_t& device);

namespace detail_ {

inline primary_context_t wrap(
	device::id_t       device_id,
	context::handle_t  handle,
	bool               decrease_refcount_on_destruct) noexcept;

} // namespace detail_

} // namespace primary_context

/**
 * A class for holding the primary context of a CUDA device.
 *
 * @note Since the runtime API tends to make such contexts active and not
 * let them go inactive very easily, this class assumes the primary context
 * is active already on construction. Limiting constructor accessibility
 * will help ensure this invariant is indeed maintained.
 */
class primary_context_t : public context_t {

protected: // data members

	/**
	 * @brief Responsibility for a unit of reference to the primary context.
	 *
	 * When true, the object is "responsible" for decreasing, at some point,
	 * the number of references registered with the CUDA driver to the
	 * device's primary context.
	 *
	 * When false, it is assumed whoever constructed the object has full
	 * responsibility for the CUDA driver reference count, will not let
	 * it drop to 0 while this object is alive, and does not need this
	 * object to reduce the reference count itself.
	 *
	 * These two values correspond to different use-cases of constructing
	 * an object of this type, in particular - construction by lvalue and
	 * rvalue/temporary device_t proxy objects.
	 */
	bool owns_refcount_unit_;

protected: // constructors

	primary_context_t(
			device::id_t       device_id,
			context::handle_t  handle,
			bool               decrease_refcount_on_destruction) noexcept
	: context_t(device_id, handle, false),
	  owns_refcount_unit_(decrease_refcount_on_destruction) { }
		// Note we are _not_ increasing the reference count; we assume
		// the primary context is already active. Also, like any context
		// proxy, we are not making this context current on construction
		// nor expecting it to be current throughout its lifetime.

public:

	/// @return a stream object for the default-ID stream of the device, which
	/// is pre-created and on which actions are scheduled when the runtime API
	/// is used and no stream is specified.
	stream_t default_stream() const noexcept;

public: // friendship

	friend class device_t;
	friend primary_context_t device::primary_context::detail_::wrap(device::id_t, context::handle_t, bool) noexcept;

public: // constructors and destructor

	primary_context_t(const primary_context_t& other)
	: context_t(other), owns_refcount_unit_(other.owns_refcount_unit_)
	{
		if (owns_refcount_unit_) {
			primary_context::detail_::obtain_and_increase_refcount(device_id_);
		}
	}

	primary_context_t(primary_context_t&& other) noexcept = default;

	~primary_context_t() NOEXCEPT_IF_NDEBUG
	{
		if (owns_refcount_unit_) {
#ifndef NDEBUG
			device::primary_context::detail_::decrease_refcount_nothrow(device_id_);
			// Swallow any error to avoid termination on throwing from a dtor
#else
			primary_context::detail_::decrease_refcount(device_id_);
#endif
		}
	}

public: // operators

	primary_context_t& operator=(const primary_context_t& other) = delete;
	primary_context_t& operator=(primary_context_t&& other) = default;
};

namespace primary_context {

namespace detail_ {

// Note the refcount semantics here, they're a bit tricky
inline context::handle_t get_handle(device::id_t device_id, bool with_refcount_increase = false)
{
	auto handle = obtain_and_increase_refcount(device_id);
	if (not with_refcount_increase) {
		decrease_refcount(device_id);
	}
	return handle;
}

} // namespace detail_

/**
 * Obtain a handle to the primary context of a given device - creating
 * it ("activating" it) if it doesn't exist.
 *
 * @note This method and its returned object will "perform their own"
 * reference accounting vis-a-vis the driver, i.e. the caller does not
 * need to worry about increasing or decreasing the CUDA driver reference
 * count for the primary context. Naturally, though, the caller must not
 * interfere with this reference accounting by decreasing the reference
 * count arbitrarily by more than it has increased it, by destroying
 * the primary context etc.
 *
 * @param device The device whose primary context is to be proxied
 * @return A proxy object for the specified device
 */
primary_context_t get(const device_t& device);

namespace detail_ {

/**
 * Like `get()`, but never holds a refcount unit, and if the primary
 * context was inactive - activates it and leaks the refcount unit.
 *
 * @todo DRY with @ref context::current::detail_::get_with_fallback_push()

 */
primary_context_t leaky_get(device::id_t device_id);

// Note that destroying the wrapped instance decreases the refcount,
// meaning that the handle must have been obtained with an "unmatched"
// refcount increase
inline device::primary_context_t wrap(
	id_t      device_id,
	handle_t  handle,
	bool      decrease_refcount_on_destruct) noexcept
{
	return {device_id, handle, decrease_refcount_on_destruct};
}

} // namespace detail_



} // namespace primary_context
} // namespace device

namespace context {

namespace detail_ {

/**
 * Checks if a context is the primary one for a device.
 *
 * @param handle Handle to the potentially-primary context
 * @param device_id Index of the device whose primary context we're interested in
 *
 * @note avoid using this if not really necessary - it may cause
 * the primary context to be created.
 */
bool is_primary_for_device(handle_t handle, device::id_t device_id);

handle_t get_primary_for_same_device(handle_t handle, bool assume_active = false);


inline bool is_primary(handle_t handle)
{
	return is_primary_for_device(handle, get_device_id(handle));
}

} // namespace detail_

} // namespace context

namespace device {

namespace primary_context {

namespace detail_ {

inline bool is_current(device::id_t device_id)
{
	auto current_context = context::current::detail_::get_handle();
	return context::detail_::is_primary_for_device(current_context, device_id);
}

} // namespace detail

/// @return true if the current context is its device's primary context
inline bool is_current()
{
	auto device_id = context::current::detail_::get_device_id();
	return detail_::is_current(device_id);
}

} // namespace primary_context

} // namespace device

} // namespace cuda


// Should we want to let the primary context reset itself? That's something which regular
// contexts can't do. We might preclude this, and allow it through the device_t class,
// instead - it already has device_t::reset(), which should be the exact same thing. We
// don't really care if that destroys contexts we're holding on to, because: 1. It won't
// cause segmentation violations - we're not dereferencing freed pointers and 2. it's the
// user's problem, not ours.


#endif /* CUDA_API_WRAPPERS_PRIMARY_CONTEXT_HPP_ */
