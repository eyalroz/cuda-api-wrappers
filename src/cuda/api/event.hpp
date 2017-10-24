/**
 * @file event.hpp
 *
 * @brief A CUDA event wrapper class, and some free-standing
 * functions for handling events using their IDs.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_EVENT_HPP_
#define CUDA_API_WRAPPERS_EVENT_HPP_

#include <cuda/api/types.h>
#include <cuda/api/constants.h>
#include <cuda/api/error.hpp>
#include <cuda/api/current_device.hpp>

#include <cuda_runtime_api.h>

namespace cuda {

///@cond
template <bool AssumedCurrent> class device_t;
///@endcond

namespace event {

/**
 * Synchronization option for @ref cuda::event_t 's
 */
enum : bool {
	/**
	 * The thread calling event_.synchronize() will enter
	 * a busy-wait loop; this (might) minimize delay between
	 * kernel execution conclusion and control returning to
	 * the thread, but is very wasteful of CPU time.
	 */
	sync_by_busy_waiting = false,
	/**
	 * The thread calling event_.synchronize() will block -
	 * yield control of the CPU and will only become ready
	 * for execution after the kernel has completed its
	 * execution - at which point it would have to wait its
	 * turn among other threads. This does not waste CPU
	 * computing time, but results in a longer delay.
	 */
	sync_by_blocking = true,
};

/**
 * Should the CUDA Runtime API record timing information for
 * events as it schedules them?
 */
enum : bool {
	dont_record_timings = false,
	do_record_timings   = true,
};

/**
 * IPC usability option for {@ref cuda::event_t}'s
 */
enum : bool {
	not_interprocess = false,         //!< Can only be used by the process which created it
	interprocess = true,              //!< Can be shared between processes. Must not be able to record timings.
	single_process = not_interprocess
};

namespace detail {

/**
 * Schedule a specified event to occur (= to fire) when all activities
 * already scheduled on the stream habe concluded.
 *
 * @param stream_id id of the stream (=queue) where to enqueue the event occurrence
 * @param event_id Event to be made to occur on stream @ref stream_id
 */
inline void enqueue(stream::id_t stream_id, id_t event_id) {
	auto status = cudaEventRecord(event_id, stream_id);
	cuda::throw_if_error(status,
		"Failed recording event " + cuda::detail::ptr_as_hex(event_id)
		+ " on stream " + cuda::detail::ptr_as_hex(stream_id));
}

constexpr unsigned inline make_flags(bool uses_blocking_sync, bool records_timing, bool interprocess)
{
	return
		  ( uses_blocking_sync  ? cudaEventBlockingSync : 0  )
		| ( records_timing      ? 0 : cudaEventDisableTiming )
		| ( interprocess        ? cudaEventInterprocess : 0  );
}

} // namespace detail

} // namespace event

///@cond
class event_t;
///@endcond

namespace event {

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
inline event_t wrap(
	device::id_t  device_id,
	id_t          event_id,
	bool          take_ownership = false);

} // namespace event

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
	event::id_t  id()                 const { return id_;                 }
	/**
	 * The device with which this event is associated (i.e. on whose stream
	 * this event can be enqueued)
	 */
	device::id_t device_id()          const { return device_id_;          }
	device_t<detail::do_not_assume_device_is_current> device() const;
	/**
	 * Is this wrapper responsible for having the CUDA Runtime API destroy
	 * the event when it destructs?
	 */
	bool         is_owning()          const { return owning;              }

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
			"Could not determine whether event " + detail::ptr_as_hex(id_)
			+ "has already occurred or not.");
	}

	/**
	 * An alias for {@ref event_t::has_occurred()} - to conform to how the CUDA runtime
	 * API names this functionality
	 */
	bool query() const { return has_occurred(); }


public: // other mutator methods

	/**
	 * @note No protection against repeated calls.
	 */
	void record(stream::id_t stream_id = stream::default_stream_id)
	{
		// TODO: Perhaps check the device ID here, rather than
		// have the Runtime API call fail?
		event::detail::enqueue(stream_id, id_);
	}

	/**
	 * Records the event and ensures it has occurred before returning
	 * (by synchronizing the stream).
	 *
	 * @note with the default argument, and when the default stream
	 * is synchronous, the synchronization will do nothing, and there
	 * will be no difference between @ref record() and this method.
	 *
	 * @note No protection against repeated calls.
	 */
	void fire(stream::id_t stream_id = stream::default_stream_id) {
		record(stream_id);
		stream::wrap(device_id_, stream_id).synchronize();
	}

	/**
	 * Have the calling thread wait - either busy-waiting or blocking - and
	 * return only after this event has occurred (see @ref has_occurred() ).
	 */
	void synchronize()
	{
		auto status = cudaEventSynchronize(id_);
		cuda::throw_if_error(status,
			"Failed synchronizing on event " + detail::ptr_as_hex(id_));
	}

protected: // constructor

	event_t(device::id_t device_id, event::id_t event_id, bool take_ownership)
	: device_id_(device_id), id_(event_id), owning(take_ownership) { }

public: // friendship

	friend event_t event::wrap(device::id_t device_id, event::id_t event_id, bool take_ownership);

public: // constructors and destructor
	event_t(device::id_t device_id, event::id_t event_id) :
		event_t(device_id, event_id, false) { }

	event_t(const event_t& other) :
		device_id_(other.device_id_), id_(other.id_), owning(false){ };

	event_t(event_t&& other) :
		device_id_(other.device_id_), id_(other.id_), owning(other.owning)
	{
		other.owning = false;
	};

	~event_t()
	{
		if (owning) { cudaEventDestroy(id_); }
	}

protected: // data members
	const device::id_t  device_id_;
	const event::id_t   id_;
	bool                owning;
		// this field is mutable only for enabling move construction; other
		// than in that case it must not be altered
};

namespace event {

/**
 * Determine (inaccurately) the elapsed time between two events,
 * by their id's.
 *
 * @note  Q: Why the weird output type?
 *        A: This is what the CUDA Runtime API itself returns
 *
 * @param start first timepoint event id
 * @param end second, later, timepoint event id
 * @return the difference in the (inaccurately) measured time, in msec
 */
inline float milliseconds_elapsed_between(id_t start, id_t end)
{
	float elapsed_milliseconds;
	auto status = cudaEventElapsedTime(&elapsed_milliseconds, start, end);
	cuda::throw_if_error(status, "determining the time elapsed between events");
	return elapsed_milliseconds;
}


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
inline float milliseconds_elapsed_between(const event_t& start, const event_t& end)
{
	return milliseconds_elapsed_between(start.id(), end.id());
}

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
	bool          take_ownership)
{
	return event_t(device_id, event_id, take_ownership);
}

namespace detail {

inline event_t create_on_current_device(
	bool          uses_blocking_sync = sync_by_busy_waiting, // Yes, that's the runtime default
	bool          records_timing     = do_record_timings,
	bool          interprocess       = not_interprocess)
{
	auto flags = make_flags(uses_blocking_sync, records_timing, interprocess);
	id_t new_event_id;
	auto status = cudaEventCreateWithFlags(&new_event_id, flags);
	cuda::throw_if_error(status, "failed creating a CUDA event associated with the current device");
	bool take_ownership = true;
	return wrap(device::current::get_id(), new_event_id, take_ownership);
}

} // namespace detail

/**
 * @brief creates a new execution stream on a device.
 *
 * @param device_id ID of the device on which to create the new stream
 * @param uses_blocking_sync When synchronizing on this new evet,
 * shall a thread busy-wait for it, or
 * @param records_timing Can this event be used to record time
 * values (e.g. duration between events)
 * @param interprocess Can multiple processes work with the constructed
 * event?
 * @return The constructed event proxy class
 */
inline event_t create(
	device::id_t  device_id,
	bool          uses_blocking_sync = sync_by_busy_waiting, // Yes, that's the runtime default
	bool          records_timing     = do_record_timings,
	bool          interprocess       = not_interprocess)
{
	device::current::scoped_override_t<cuda::detail::do_not_assume_device_is_current>
		set_device_for_this_scope(device_id);
	return detail::create_on_current_device(
		uses_blocking_sync, records_timing, interprocess);
}

} // namespace event
} // namespace cuda


#endif /* CUDA_API_WRAPPERS_EVENT_HPP_ */
