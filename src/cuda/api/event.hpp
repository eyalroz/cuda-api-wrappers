#pragma once
#ifndef CUDA_API_WRAPPERS_EVENT_HPP_
#define CUDA_API_WRAPPERS_EVENT_HPP_

#include "cuda/api/types.h"
#include "cuda/api/constants.h"
#include "cuda/api/error.hpp"

#include <cuda_runtime_api.h>

namespace cuda {
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
 * IPC usability option for @ref cuda::event_t 's
 */
enum : bool {
	not_interprocess = false,
		//!< Can be shared between processes. Must not be
		//!< able to record timings
	interprocess = true,
		//!< Can only be used by the process which created it
};

namespace detail {

void enqueue(stream::id_t stream_id, id_t event_id) {
	auto status = cudaEventRecord(event_id, stream_id);
	throw_if_error(status,
		"Failed recording event " + cuda::detail::ptr_as_hex(event_id)
		+ " on stream " + cuda::detail::ptr_as_hex(stream_id));
}

void enqueue(device::id_t device_id, stream::id_t stream_id, id_t event_id) {
	device::current::scoped_override_t<> device_setter(device_id);
	enqueue(stream_id, event_id);
}

} // namespace detail
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
public: // constructors and destructor
	event_t() : owning(true)
	{
		auto status = cudaEventCreate(&id_);
		throw_if_error(status, "failed creating a CUDA event");
	}
	event_t(
		bool uses_blocking_sync,
			// I would have liked to default to false, but this would
			// occlude the trivial (argument-less) constructor)
		bool records_timing = event::do_record_timings,
		bool interprocess = event::not_interprocess)
	:
		owning(true), blocking_sync(uses_blocking_sync),
		records_timing_(records_timing), interprocess_(interprocess)
	{
		auto status = cudaEventCreate(&id_, flags());
		throw_if_error(status, "failed creating a CUDA event");
	}

	event_t(const event_t& other) :
		id_(other.id_), owning(false),
		blocking_sync(other.blocking_sync),
		records_timing_(other.records_timing_),
		interprocess_(other.interprocess_) { };

	event_t(event_t&& other) : owning(other.owning), id_(other.id_),
		blocking_sync(other.blocking_sync),
		records_timing_(other.records_timing_),
		interprocess_(other.interprocess_) { other.owning = false; };

	~event_t() { if (owning) cudaEventDestroy(id_); }

protected:
	unsigned flags() const {
		return
			  ( blocking_sync    ? cudaEventBlockingSync : 0  )
			| ( records_timing_  ? 0 : cudaEventDisableTiming )
			| ( interprocess_    ? cudaEventInterprocess : 0  )
		;
	}

public: // data member getter
	event::id_t id()                  const { return id_;             }
	bool        is_owning()           const { return owning;          }
	bool        interprocess()        const { return interprocess_;   }
	bool        records_timing()      const { return records_timing_; }
	bool        uses_blocking_sync()  const { return blocking_sync;   }

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
	 * An alias for @ref has_occurred() - to conform to how the CUDA runtime
	 * API names this functionality
	 */
	bool query() const { return has_occurred(); }


public: // other mutator methods

	/**
	 * @note No protection against repeated calls.
	 */
	void record(stream::id_t stream_id = stream::default_stream_id)
	{
		// TODO: Don't I need a device ID here?
		event::detail::enqueue(stream_id, id_);
	}

	/**
	 * Have the calling thread wait - either busy-waiting or blocking - and
	 * return only after this event has occurred (see @ref has_occrred ).
	 */
	void synchronize()
	{
		auto status = cudaEventSynchronize(id_);
		throw_if_error(status,
			"Failed synchronizing on event " + detail::ptr_as_hex(id_));
	}

protected: // data members
	bool           owning; // not const, to enable move assignment / construction
	event::id_t    id_; // it can't be a const, since we're making
	                    // a CUDA API call during construction
	const bool blocking_sync    { false };
	const bool records_timing_  { false };
	const bool interprocess_    { false };
};

namespace event {

float milliseconds_elapsed_between(id_t start, id_t end)
{
	float elapsed_ms;
	auto result = cudaEventElapsedTime(&elapsed_ms, start, end);
	throw_if_error(result, "determining the time elapsed between events");
	return elapsed_ms;
}
float milliseconds_elapsed_between(const event_t& start, const event_t& end)
{
	return milliseconds_elapsed_between(start.id(), end.id());
}

inline event_t make()
{
	return event_t();
}

inline event_t make(
	bool uses_blocking_sync,
	bool records_timing = do_record_timings,
	bool interprocess = not_interprocess)
{
	return event_t(uses_blocking_sync, records_timing, interprocess);
}

} // namespace event

} // namespace cuda


#endif /* CUDA_API_WRAPPERS_EVENT_HPP_ */
