/**
 * @file
 *
 * @brief A CUDA event wrapper class and some associated
 * free-standing functions.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_EVENT_HPP_
#define CUDA_API_WRAPPERS_EVENT_HPP_

#include "types.hpp"

#include <cuda_runtime_api.h>

#include <chrono> // for duration types
#include "constants.hpp"
#include "current_device.hpp"
#include "error.hpp"
#include "ipc.hpp"

namespace cuda {

///@cond
class device_t;
class stream_t;
///@endcond

namespace event {

namespace detail_ {

inline void destroy(
	handle_t           handle,
	device::id_t       device_id,
	context::handle_t  context_handle);

inline void enqueue_in_current_context(stream::handle_t stream_handle, handle_t event_handle)
{
	auto status = cuEventRecord(event_handle, stream_handle);
	throw_if_error_lazy(status,
		"Failed recording " + event::detail_::identify(event_handle)
		+ " on " + stream::detail_::identify(stream_handle));
}

/**
 * Schedule a specified event to occur (= to fire) when all activities
 * already scheduled on the stream have concluded.
 *
 * @param stream_handle handle of the stream (=queue) where to enqueue the event occurrence
 * @param event_handle Event to be made to occur on stream @ref stream_handle
 */
inline void enqueue(context::handle_t context_handle, stream::handle_t stream_handle, handle_t event_handle) {
	context::current::detail_::scoped_ensurer_t { context_handle };
	enqueue_in_current_context(stream_handle, event_handle);
}

using flags_t = unsigned int;

constexpr flags_t inline make_flags(bool uses_blocking_sync, bool records_timing, bool interprocess)
{
	return
		  ( uses_blocking_sync  ? CU_EVENT_BLOCKING_SYNC : 0  )
		| ( records_timing      ? 0 : CU_EVENT_DISABLE_TIMING )
		| ( interprocess        ? CU_EVENT_INTERPROCESS : 0  );
}

} // namespace detail_

} // namespace event

///@cond
class event_t;
///@endcond

namespace event {

/**
 * @brief Wrap an existing CUDA event in a @ref event_t instance.
 *
 * @note This is a named constructor idiom, existing of direct access to the ctor
 * of the same signature, to emphasize that a new event is _not_ created.
 *
 * @param context_handle Handle of the context in which this event was created
 * @param event_handle handle of the pre-existing event
 * @param take_ownership When set to `false`, the CUDA event
 * will not be destroyed along with proxy; use this setting
 * when temporarily working with a stream existing irrespective of
 * the current context and outlasting it. When set to `true`,
 * the proxy class will act as it does usually, destroying the event
 * when being destructed itself.
 * @return an event wrapper associated with the specified event
 */
event_t wrap(
	device::id_t       device_id,
	context::handle_t  context_handle,
	handle_t           event_handle,
	bool               take_ownership = false,
	bool               hold_pc_refcount_unit = false) noexcept;

::std::string identify(const event_t& event);

} // namespace event

/**
 * Have the calling thread wait - either busy-waiting or blocking - and
 * return only after this event has occurred (see @ref event_t::has_occurred()
 *
 * @todo figure out what happens if the event has not been recorded
 * before this call is made.
 *
 * @note the waiting will occur either passively (e.g. like waiting for
 * information on a file descriptor), or actively (by busy-waiting) -
 * depending on the flag with which the event was created.
 *
 * @param event the event for whose occurrence to wait; must be scheduled
 * to occur on some stream (possibly the different stream)
 */
inline void wait(const event_t& event);

/**
 * @brief Wrapper class for a CUDA event
 *
 * Use this class - built around an event handle - to perform almost, if not all,
 * event-related operations the CUDA Runtime API is capable of.
 *
 * @note By default this class has RAII semantics, i.e. it has the runtime create
 * an event on construction and destroy it on destruction, and isn't merely
 * an ephemeral wrapper one could apply and discard; but this second kind of
 * semantics is also (sort of) supported, through the @ref event_t::owning field.
 */
class event_t {

public: // data member non-mutator getters
	/// The raw CUDA ID for the device w.r.t. which the event is defined
	device::id_t      device_id()       const noexcept { return device_id_; };

	/// The raw CUDA handle for the context in which the represented stream is defined.
	context::handle_t context_handle()  const noexcept { return context_handle_; }

	/// The raw CUDA handle for this event
	event::handle_t   handle()          const noexcept { return handle_; }

	/// True if this wrapper is responsible for telling CUDA to destroy the event upon the wrapper's own destruction
	bool              is_owning()       const noexcept { return owning; }

	/// True if this wrapper has been associated with an increase of the device's primary context's reference count
	bool              holds_primary_context_reference()
	                                    const noexcept { return holds_pc_refcount_unit; }

	/// The device w.r.t. which the event is defined
	device_t          device()          const;

	/// The context in which this stream was defined.
	context_t         context()         const;



public: // other non-mutator methods

	/**
	 * Has this event already occurred, or is it still pending on a stream?
	 *
	 * @note an event can occur multiple times, but in the context of this
	 * method, it only has two states: pending (on a stream) and has_occured.
	 *
	 * @return if all work on the stream with which the event was recorded
	 * has been completed, returns true; if there is pending work on that stream
	 * before the point of recording, returns false; if the event has not
	 * been recorded at all, returns true.
	 */
	bool has_occurred() const
	{
		auto status = cuEventQuery(handle_);
		if (status == cuda::status::success) return true;
		if (status == cuda::status::async_operations_not_yet_completed) return false;
		throw cuda::runtime_error(status,
			"Could not determine whether " + event::detail_::identify(handle_)
			+ "has already occurred or not");
	}

	/**
	 * An alias for {@ref event_t::has_occurred()} - to conform to how the CUDA runtime
	 * API names this functionality
	 */
	bool query() const { return has_occurred(); }

public: // other mutator methods

	/**
	 * Schedule a specified event to occur (= to fire) when all activities
	 * already scheduled on the event's device's default stream have concluded.
	 *
	 * @note No protection against repeated calls.
	 */
	void record() const
	{
		event::detail_::enqueue(context_handle_, stream::default_stream_handle, handle_);
	}

	/**
	 * Schedule a specified event to occur (= to fire) when all activities
	 * already scheduled on the stream have concluded.
	 *
	 * @note No protection against repeated calls.
	 */
	void record(const stream_t& stream) const;

	/**
	 * Records the event and ensures it has occurred before returning
	 * (by synchronizing the stream).
	 *
	 * @note No protection against repeated calls.
	 */
	void fire(const stream_t& stream) const;

	/**
	 * See @see cuda::wait() .
	 */
	void synchronize() const
	{
		return cuda::wait(*this);
	}

protected: // constructors

	event_t(
		device::id_t device_id,
		context::handle_t context_handle,
		event::handle_t event_handle,
		bool take_ownership,
		bool hold_pc_refcount_unit) noexcept
	:
		device_id_(device_id),
		context_handle_(context_handle),
		handle_(event_handle),
		owning(take_ownership),
		holds_pc_refcount_unit(hold_pc_refcount_unit) { }

public: // friendship

	friend event_t event::wrap(
		device::id_t       device,
		context::handle_t  context_handle,
		event::handle_t    event_handle,
		bool               take_ownership,
		bool               hold_pc_refcount_unit) noexcept;

public: // constructors and destructor

	// Events cannot be copied, despite our allowing non-owning class instances.
	// The reason is that we might inadvertently copy an owning instance, creating
	// a non-owning instance and letting the original owning instance go out of scope -
	// thus destructing the C++ object, and destroying the underlying CUDA object.
	// Essentially, that is like passing a reference to a local variable - which we
	// may not do.
	event_t(const event_t& other) = delete;

	event_t(event_t&& other) noexcept : event_t(
		other.device_id_, other.context_handle_, other.handle_, other.owning, other.holds_pc_refcount_unit)
	{
		other.owning = false;
		other.holds_pc_refcount_unit = false;
	};

	~event_t() noexcept(false)
	{
		if (owning) {
#ifdef NDEBUG
			cuEventDestroy(handle_);
				// Note: "Swallowing" any potential error to avoid ::std::terminate(); also,
				// because the event cannot possibly exist after this call.
#else
			event::detail_::destroy(handle_, device_id_, context_handle_);
#endif
		}
		// TODO: DRY
		if (holds_pc_refcount_unit) {
#ifdef NDEBUG
			device::primary_context::detail_::decrease_refcount_nothrow(device_id_);
				// Note: "Swallowing" any potential error to avoid ::std::terminate(); also,
				// because a failure probably means the primary context is inactive already
#else
			device::primary_context::detail_::decrease_refcount(device_id_);
#endif
		}
	}

public: // operators

	event_t& operator=(const event_t&) = delete;
	event_t& operator=(event_t&& other) noexcept
	{
		::std::swap(device_id_, other.device_id_);
		::std::swap(context_handle_, other.context_handle_);
		::std::swap(handle_, other.handle_);
		::std::swap(owning, other.owning);
		::std::swap(holds_pc_refcount_unit, holds_pc_refcount_unit);
		return *this;
	}

protected: // data members
	device::id_t       device_id_;
	context::handle_t  context_handle_;
	event::handle_t    handle_;
	bool               owning;
		// this field is mutable only for enabling move construction; other
		// than in that case it must not be altered
	bool               holds_pc_refcount_unit;
		// When context_handle_ is the handle of a primary context, this event may
		// be "keeping that context alive" through the refcount - in which case
		// it must release its refcount unit on destruction
};

namespace event {

/**
 * @brief The type used by the CUDA Runtime API to represent the time difference
 * between pairs of events.
 */
using duration_t = ::std::chrono::duration<float, ::std::milli>;

/**
 * Determine (inaccurately) the elapsed time between two events
 *
 * @note  Q: Why the weird output type?
 *        A: This is what the CUDA Runtime API itself returns
 *
 * @param start first timepoint event
 * @param end second, later, timepoint event
 * @return the difference in the (inaccurately) measured time, in msec
 */
inline duration_t time_elapsed_between(const event_t& start, const event_t& end)
{
	float elapsed_milliseconds;
	auto status = cuEventElapsedTime(&elapsed_milliseconds, start.handle(), end.handle());
	throw_if_error_lazy(status, "determining the time elapsed between events");
	return duration_t { elapsed_milliseconds };
}

inline duration_t time_elapsed_between(const ::std::pair<const event_t&, const event_t&>& event_pair)
{
	return time_elapsed_between(event_pair.first, event_pair.second);
}

inline event_t wrap(
	device::id_t       device_id,
	context::handle_t  context_handle,
	handle_t           event_handle,
	bool               take_ownership,
	bool               hold_pc_refcount_unit) noexcept
{
	return { device_id, context_handle, event_handle, take_ownership, hold_pc_refcount_unit };
}

namespace detail_ {

inline ::std::string identify(const event_t& event)
{
	return identify(event.handle(), event.context_handle(), event.device_id());
}

inline handle_t create_raw_in_current_context(flags_t flags = 0u)
{
	cuda::event::handle_t new_event_handle;
	auto status = cuEventCreate(&new_event_handle, flags);
	throw_if_error_lazy(status, "Failed creating a CUDA event");
	return new_event_handle;
}

// Notes:
// * For now, event_t's need their device's ID - even if it's the current device;
//   that explains the requirement in this function's interface.
// * Similarly, this function does not know whether the context is primary or
//   not, and it is up to the caller to know that and decide whether the event
//   proxy should decrease the primary context refcount on destruction
inline event_t create_in_current_context(
	device::id_t       current_device_id,
	context::handle_t  current_context_handle,
	bool               hold_pc_refcount_unit,
	bool               uses_blocking_sync,
	bool               records_timing,
	bool               interprocess)
{
	auto flags = make_flags(uses_blocking_sync, records_timing, interprocess);
	auto new_event_handle = create_raw_in_current_context(flags);
	return wrap(current_device_id, current_context_handle, new_event_handle, do_take_ownership, hold_pc_refcount_unit);
}

inline void destroy_in_current_context(
	handle_t           handle,
	device::id_t       current_device_id,
	context::handle_t  current_context_handle)
{
	auto status = cuEventDestroy(handle);
	throw_if_error_lazy(status, "Failed destroying " +
		identify(handle, current_context_handle, current_device_id));
}

/**
 * @note see @ref cuda::event::create()
 */

inline event_t create(
	device::id_t       device_id,
	context::handle_t  context_handle,
	bool               hold_pc_refcount_unit,
	bool               uses_blocking_sync,
	bool               records_timing,
	bool               interprocess)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);

	return detail_::create_in_current_context(
		device_id, context_handle,
		hold_pc_refcount_unit,
		uses_blocking_sync, records_timing, interprocess);
}

inline void destroy(
	handle_t           handle,
	device::id_t       device_id,
	context::handle_t  context_handle)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	destroy_in_current_context(handle, device_id, context_handle);
}

} // namespace detail_

/**
 * @brief creates a new event on (the primary execution context of) a device.
 *
 * @param device              The device on which to create the new stream
 * @param uses_blocking_sync  When synchronizing on this new event, shall a thread busy-wait for it, or block?
 * @param records_timing      Can this event be used to record time values (e.g. duration between events)?
 * @param interprocess        Can multiple processes work with the constructed event?
 * @return The constructed event proxy
 *
 * @note The created event will keep the device's primary context active while it exists.
 */
event_t create(
	const device_t&  device,
	bool             uses_blocking_sync = sync_by_busy_waiting, // Yes, that's the runtime default
	bool             records_timing     = do_record_timings,
	bool             interprocess       = not_interprocess);

/**
 * @brief creates a new event.
 *
 * @param context             The CUDA execution context in which to create the event
 * @param uses_blocking_sync  When synchronizing on this new event, shall a thread busy-wait for it, or block?
 * @param records_timing      Can this event be used to record time values (e.g. duration between events)?
 * @param interprocess        Can multiple processes work with the constructed event?
 * @return The constructed event proxy
 *
 * @note Even if the context happens to be primary, the created event will _not_ keep this context alive.
 */
inline event_t create(
	const context_t&  context,
	bool              uses_blocking_sync = sync_by_busy_waiting,
	bool              records_timing     = do_record_timings,
	bool              interprocess       = not_interprocess);

} // namespace event

inline void wait(const event_t& event)
{
	auto context_handle = event.context_handle();
	auto event_handle = event.handle();
	context::current::detail_::scoped_override_t context_for_this_scope(context_handle);
	auto status = cuEventSynchronize(event_handle);
	throw_if_error_lazy(status, "Failed synchronizing " + event::detail_::identify(event));
}

inline void synchronize(const event_t& event)
{
	return wait(event);
}

} // namespace cuda

#endif // CUDA_API_WRAPPERS_EVENT_HPP_
