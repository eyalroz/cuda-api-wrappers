#pragma once
#ifndef CUDA_API_WRAPPERS_EVENT_HPP_
#define CUDA_API_WRAPPERS_EVENT_HPP_

#include "cuda/api/types.h"
#include "cuda/api/error.hpp"

#include <cuda_runtime_api.h>

namespace cuda {

namespace event {
enum : bool {
	dont_use_blocking_sync = false,
	use_blocking_sync = true,
};
enum : bool {
	dont_record_timings = false,
	do_record_timings = true,
};
enum : bool {
	not_interprocess = false,
	interprocess = true,
};
} // namespace event

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
		owning(true), uses_blocking_sync_(uses_blocking_sync),
		records_timing_(records_timing), interprocess_(interprocess)
	{
		auto status = cudaEventCreate(&id_, flags());
		throw_if_error(status, "failed creating a CUDA event");
	}

	event_t(const event_t& other) :
		id_(other.id_), owning(false),
		uses_blocking_sync_(other.uses_blocking_sync_),
		records_timing_(other.records_timing_),
		interprocess_(other.interprocess_) { };

	event_t(event_t&& other) : owning(other.owning), id_(other.id_),
		uses_blocking_sync_(other.uses_blocking_sync_),
		records_timing_(other.records_timing_),
		interprocess_(other.interprocess_) { other.owning = false; };

	~event_t() { if (owning) cudaEventDestroy(id_); }

protected:
	unsigned flags() const {
		return
			  (uses_blocking_sync_  ? cudaEventBlockingSync : 0)
			| (records_timing_      ? 0 : cudaEventDisableTiming)
			| (interprocess_        ? cudaEventInterprocess : 0)
		;
	}

public: // data member getter
	event::id_t id()                  const { return id_;                 }
	bool        is_owning()           const { return owning;              }
	bool        interprocess()        const { return interprocess_;       }
	bool        records_timing()      const { return records_timing_;     }
	bool        uses_blocking_sync()  const { return uses_blocking_sync_; }

public: // other mutator methods
	status_t query()
	{
		// Note we can't throw on failure since this is a destructor... let's
		// home for the best.
		return cudaEventQuery(id_);
	}

	/**
	 * @note No protection against repeated calls.
	 *
	 * @note Not using cuda::stream::Default to avoid the extra include
	 */
	void record(stream::id_t stream = (stream::id_t) nullptr)
	{
		throw_if_error(cudaEventRecord(id_, stream), "failed recording an event");
	}
	void synchronize()
	{
		throw_if_error(cudaEventSynchronize(id_), "failed synchronizing an event");
	}

protected: // data members
	bool           owning; // not const, to enable move assignment / construction
	event::id_t    id_; // it can't be a const, since we're making
	                    // a CUDA API call during construction
	const bool uses_blocking_sync_ { false };
	const bool records_timing_  { false };
	const bool interprocess_ { false };
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

} // namespace event

} // namespace cuda


#endif /* CUDA_API_WRAPPERS_EVENT_HPP_ */
