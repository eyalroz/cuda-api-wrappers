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

template <bool AssumedCurrent> class device_t;

namespace event {

/**
 * Synchronization option for @ref cuda::event_t 's
 */
enum : bool {
	sync_by_busy_waiting = false,
		/**
		 * The thread calling event_.synchronize() will enter
		 * a busy-wait loop; this (might) minimize delay between
		 * kernel execution conclusion and control returning to
		 * the thread, but is very wasteful of CPU time.
		 */
	sync_by_blocking = true,
		/**
		 * The thread calling event_.synchronize() will block -
		 * yield control of the CPU and will only become ready
		 * for execution after the kernel has completed its
		 * execution - at which point it would have to wait its
		 * turn among other threads. This does not waste CPU
		 * computing time, but results in a longer delay.
		 */
};

enum : bool {
	dont_record_timings = false,
	do_record_timings = true,
};

/**
 * IPC usability option for {@ref cuda::event_t}'s
 */
enum : bool {
	not_interprocess = false,
		//!< Can be shared between processes. Must not be
		//!< able to record timings
	interprocess = true,
		//!< Can only be used by the process which created it
	single_process = not_interprocess
};

namespace detail {

void enqueue(stream::id_t stream_id, id_t event_id) {
	auto status = cudaEventRecord(event_id, stream_id);
	throw_if_error(status,
		"Failed recording event " + cuda::detail::ptr_as_hex(event_id)
		+ " on stream " + cuda::detail::ptr_as_hex(stream_id));
}

constexpr unsigned make_flags(bool uses_blocking_sync, bool records_timing, bool interprocess)
{
	return
		  ( uses_blocking_sync  ? cudaEventBlockingSync : 0  )
		| ( records_timing      ? 0 : cudaEventDisableTiming )
		| ( interprocess        ? cudaEventInterprocess : 0  );
}

id_t create_on_current_device(bool uses_blocking_sync, bool records_timing, bool interprocess)
{
	id_t new_event_id;
	auto flags = make_flags(uses_blocking_sync, records_timing, interprocess);
	auto status = cudaEventCreate(&new_event_id, flags);
	throw_if_error(status, "failed creating a CUDA event associated with the current device");
	return new_event_id;
}

} // namespace detail

} // namespace event

class event_t;

namespace event {

inline event_t wrap(
	device::id_t  device_id,
	id_t          event_id,
	bool          take_ownership = false);

} // namespace event

/**
 * A proxy class for CUDA events. By default, it has
 * RAII semantics, i.e. it has the runtime create an event
 * on construction and destory it on destruction, and isn't
 * merely an ephemeral wrapper one could apply and discard;
 * but this second kind of semantics is also (sort of)
 * supported, throw the owning field.
 */
class event_t {
public: // data member non-mutator getters
	event::id_t  id()                 const { return id_;                 }
	device::id_t device_id()          const { return device_id_;          }
	device_t<detail::do_not_assume_device_is_current> device() const;
	bool         is_owning()          const { return owning;              }

public: // other non-mutator methods

	/**
	 * Checks whether this event has already occurred or whether it's
	 * still pending.
	 *
	 * @return if all work on the stream with which the event was recorded
	 * has been completed, returns true; if there is pending work on that stream
	 * before the point of recording, returns false; if the event has not
	 * been recorded at all, returns true.
	 */
	bool has_occurred() const
	{
		auto status = cudaEventQuery(id_);
		if (status == cuda::error::success) return true;
		if (status == cuda::error::not_ready) return false;
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
	 * Have the calling thread wait - either busy-waiting or blocking - and
	 * return only after this event has occurred (see @ref has_occurred() ).
	 */
	void synchronize()
	{
		auto status = cudaEventSynchronize(id_);
		throw_if_error(status,
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
		owning = false;
	}

protected: // data members
	const device::id_t  device_id_;
	const event::id_t   id_;
	bool                owning;
		// this field is mutable only for enabling move construction; other
		// than in that case it must not be altered
};

namespace event {

float milliseconds_elapsed_between(id_t start, id_t end)
{
	float elapsed_milliseconds;
	auto status = cudaEventElapsedTime(&elapsed_milliseconds, start, end);
	throw_if_error(status, "determining the time elapsed between events");
	return elapsed_milliseconds;
}
float milliseconds_elapsed_between(const event_t& start, const event_t& end)
{
	return milliseconds_elapsed_between(start.id(), end.id());
}

/**
 * @brief Wrap an existing CUDA event in a @ref event_t instance.
 *
 * @param device_id ID of the device for which the stream is defined
 * @param event_id ID of the pre-existing event
 * @param take_ownership When set to {@code false}, the CUDA event
 * will not be destroyed along with proxy; use this setting
 * when temporarily working with a stream existing irrespective of
 * the current context and outlasting it. When set to {@code true},
 * the proxy class will act as it does usually, destroying the event
 * when being destructed itself.
 * @return The constructed {@code cuda::event_t}.
 */
inline event_t wrap(
	device::id_t  device_id,
	id_t          event_id,
	bool          take_ownership /* = false, see declaration */)
{
	return event_t(device_id, event_id, take_ownership);
}

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
inline event_t make(
	device::id_t  device_id,
	bool          uses_blocking_sync = sync_by_busy_waiting, // Yes, that's the runtime default
	bool          records_timing     = do_record_timings,
	bool          interprocess       = not_interprocess)
{
	auto new_event_id = detail::create_on_current_device(
		uses_blocking_sync, records_timing, interprocess);
	bool take_ownership = true;
	return wrap(device_id, new_event_id, take_ownership);
}

/**
 * @note The reason this doesn't take the three boolean parameters
 * is avoiding an implicit cast from {@code bool} to {@code cuda::device::id_t},
 * which would take us to the other variant of @ref cuda::event::make() with a
 * possibly invalid device ID.
 */
inline event_t make()
{
	return make(device::current::get_id());
}

} // namespace event
} // namespace cuda


#endif /* CUDA_API_WRAPPERS_EVENT_HPP_ */
