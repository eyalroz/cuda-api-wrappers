/**
 * @file event.hpp
 *
 * @brief A CUDA event wrapper class and some associated
 * free-standing functions.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_EVENT_HPP_
#define CUDA_API_WRAPPERS_EVENT_HPP_

#include <cuda/api/constants.hpp>
#include <cuda/api/current_device.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/ipc.hpp>
#include <cuda/common/types.hpp>

#include <cuda_runtime_api.h>

#include <chrono> // for duration types

namespace cuda {

///@cond
class device_t;
class stream_t;
///@endcond

namespace event {

namespace detail_ {

/**
 * Schedule a specified event to occur (= to fire) when all activities
 * already scheduled on the stream have concluded.
 *
 * @param stream_id id of the stream (=queue) where to enqueue the event occurrence
 * @param event_id Event to be made to occur on stream @ref stream_id
 */
inline void enqueue(stream::id_t stream_id, id_t event_id) {
	auto status = cudaEventRecord(event_id, stream_id);
	cuda::throw_if_error(status,
		"Failed recording event " + cuda::detail_::ptr_as_hex(event_id)
		+ " on stream " + cuda::detail_::ptr_as_hex(stream_id));
}

constexpr unsigned inline make_flags(bool uses_blocking_sync, bool records_timing, bool interprocess)
{
	return
		  ( uses_blocking_sync  ? cudaEventBlockingSync : 0  )
		| ( records_timing      ? 0 : cudaEventDisableTiming )
		| ( interprocess        ? cudaEventInterprocess : 0  );
}

} // namespace detail_

} // namespace event

///@cond
class event_t;
///@endcond

namespace event {

namespace detail_ {
/**
 * @brief Wrap an existing CUDA event in a @ref event_t instance.
 *
 * @param device_id ID of the device for which the stream is defined
 * @param event_id ID of the pre-existing event
 * @param take_ownership When set to `false`, the CUDA event
 * will not be destroyed along with proxy; use this setting
 * when temporarily working with a stream existing irrespective of
 * the current context and outlasting it. When set to `true`,
 * the proxy class will act as it does usually, destroying the event
 * when being destructed itself.
 * @return The constructed `cuda::event_t`.
 */
event_t wrap(
	device::id_t  device_id,
	id_t          event_id,
	bool          take_ownership = false) noexcept;

} // namespace detail_

} // namespace event

inline void synchronize(const event_t& event);

/**
 * @brief Proxy class for a CUDA event
 *
 * Use this class - built around an event id - to perform almost, if not all,
 * event-related operations the CUDA Runtime API is capable of.
 *
 * @note By default this class has RAII semantics, i.e. it has the runtime create
 * an event on construction and destroy it on destruction, and isn't merely
 * an ephemeral wrapper one could apply and discard; but this second kind of
 * semantics is also (sort of) supported, through the @ref event_t::owning field.
 *
 * @note this is one of the three main classes in the Runtime API wrapper library,
 * together with @ref cuda::device_t and @ref cuda::stream_t
 */
class event_t {
public: // data member non-mutator getters
	/**
	 * The CUDA runtime API ID this object is wrapping
	 */
	event::id_t  id()                 const noexcept{ return id_;                 }
	/**
	 * The device with which this event is associated (i.e. on whose stream
	 * this event can be enqueued)
	 */
	device::id_t device_id()          const noexcept { return device_id_;          }
	device_t     device() const noexcept;
	/**
	 * Is this wrapper responsible for having the CUDA Runtime API destroy
	 * the event when it destructs?
	 */
	bool         is_owning()          const noexcept { return owning;              }

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
		auto status = cudaEventQuery(id_);
		if (status == cuda::status::success) return true;
		if (status == cuda::status::not_ready) return false;
		throw cuda::runtime_error(status,
			"Could not determine whether event " + detail_::ptr_as_hex(id_)
			+ "has already occurred or not.");
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
	void record()
	{
		event::detail_::enqueue(stream::default_stream_id, id_);
	}

	/**
	 * Schedule a specified event to occur (= to fire) when all activities
	 * already scheduled on the stream have concluded.
	 *
	 * @note No protection against repeated calls.
	 */
	void record(const stream_t& stream);

	/**
	 * Records the event and ensures it has occurred before returning
	 * (by synchronizing the stream).
	 *
	 * @note No protection against repeated calls.
	 */
	void fire(const stream_t& stream);

	/**
	 * Have the calling thread wait - either busy-waiting or blocking - and
	 * return only after this event has occurred (see @ref has_occurred() ).
	 */
	void synchronize()
	{
		return cuda::synchronize(*this);
	}

protected: // constructors

	event_t(device::id_t device_id, event::id_t event_id, bool take_ownership) noexcept
	: device_id_(device_id), id_(event_id), owning(take_ownership) { }

public: // friendship

	friend event_t event::detail_::wrap(device::id_t device_id, event::id_t event_id, bool take_ownership) noexcept;

public: // constructors and destructor

	event_t(const event_t& other) noexcept : event_t(other.device_id_, other.id_, false) { }

	event_t(event_t&& other) noexcept :
		event_t(other.device_id_, other.id_, other.owning)
	{
		other.owning = false;
	};

	~event_t()
	{
		if (owning) { cudaEventDestroy(id_); }
	}

public: // operators

	event_t& operator=(const event_t& other) = delete;
	event_t& operator=(event_t&& other) = delete;

protected: // data members
	const device::id_t  device_id_;
	const event::id_t   id_;
	bool                owning;
		// this field is mutable only for enabling move construction; other
		// than in that case it must not be altered
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
	auto status = cudaEventElapsedTime(&elapsed_milliseconds, start.id(), end.id());
	cuda::throw_if_error(status, "determining the time elapsed between events");
	return duration_t { elapsed_milliseconds };
}

namespace detail_ {

/**
 * Obtain a proxy object for an already-existing CUDA event
 *
 * @note This is a named constructor idiom instead of direct access to the ctor of the same
 * signature, to emphase whot this construction means - a new event is not
 * created.
 *
 * @param device_id Device to which the event is related
 * @param event_id of the event for which to obtain a proxy
 * @param take_ownership when true, the wrapper will have the CUDA Runtime API destroy
 * the event when it destructs (making an "owning" event wrapper; otherwise, it is
 * assume that some other code "owns" the event and will destroy it when necessary
 * (and not while the wrapper is being used!)
 * @return an event wrapper associated with the specified event
 */
inline event_t wrap(
	device::id_t  device_id,
	id_t          event_id,
	bool          take_ownership) noexcept
{
	return event_t(device_id, event_id, take_ownership);
}

// Note: For now, event_t's need their device's ID - even if it's the current device;
// that explains the requirement in this function's interface
inline event_t create_on_current_device(
	device::id_t  current_device_id,
	bool          uses_blocking_sync,
	bool          records_timing,
	bool          interprocess)
{
	auto flags = make_flags(uses_blocking_sync, records_timing, interprocess);
	cuda::event::id_t new_event_id;
	auto status = cudaEventCreateWithFlags(&new_event_id, flags);
	cuda::throw_if_error(status, "failed creating a CUDA event associated with the current device");
	// Note: We're trusting CUDA to actually have succeeded if it reports success,
	// so we're not checking the newly-created event id - which is really just
	// a pointer - for nullness
	return wrap(current_device_id, new_event_id, do_take_ownership);
}

/**
 * @note see @ref cuda::event::create()
 */

inline event_t create(
	device::id_t  device_id,
	bool          uses_blocking_sync,
	bool          records_timing,
	bool          interprocess)
{
	device::current::detail_::scoped_override_t
		set_device_for_this_scope(device_id);
	return detail_::create_on_current_device(device_id, uses_blocking_sync, records_timing, interprocess);
}

} // namespace detail_

/**
 * @brief creates a new execution stream on a device.
 *
 * @param device              The device on which to create the new stream
 * @param uses_blocking_sync  When synchronizing on this new event, shall a thread busy-wait for it, or block?
 * @param records_timing      Can this event be used to record time values (e.g. duration between events)?
 * @param interprocess        Can multiple processes work with the constructed event?
 * @return The constructed event proxy
 *
 * @note Creating an event
 */
inline event_t create(
	device_t&  device,
	bool       uses_blocking_sync = sync_by_busy_waiting, // Yes, that's the runtime default
	bool       records_timing     = do_record_timings,
	bool       interprocess       = not_interprocess);

} // namespace event

/**
 * Waits for a specified event to conclude before returning control
 * to the calling code.
 *
 * @todo Determine how this waiting takes place (as opposed to stream
 * synchrnoization).
 *
 * @param event the event for whose occurrence to wait; must be scheduled
 * to occur on some stream (possibly the different stream)
 */
inline void synchronize(const event_t& event)
{
	auto device_id = event.device_id();
	auto event_id = event.id();
	device::current::detail_::scoped_override_t device_for_this_scope(device_id);
	auto status = cudaEventSynchronize(event_id);
	throw_if_error(status, "Failed synchronizing the event with id "
		+ cuda::detail_::ptr_as_hex(event_id) + " on   " + ::std::to_string(device_id));
}

} // namespace cuda

#endif // CUDA_API_WRAPPERS_EVENT_HPP_
