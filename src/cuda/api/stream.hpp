#pragma once
#ifndef CUDA_STREAM_H_
#define CUDA_STREAM_H_

#include "cuda/api/types.h"
#include "cuda/api/error.hpp"
#include "cuda/api/kernel_launch.cuh"
#include "cuda/api/current_device.hpp"
#include "cuda/api/device_count.hpp"

#include <cuda_runtime_api.h>

#include <string>
#include <functional>

namespace cuda {

namespace stream {

/**
 * Creates a new stream on the current device.
 *
 * @return the created stream's id
 */
inline id_t create(
	priority_t    priority = default_priority,
	bool          synchronizes_with_default_stream = true)
{
	unsigned int flags = synchronizes_with_default_stream ?
		cudaStreamDefault : cudaStreamNonBlocking;
	stream::id_t new_stream_id;
	auto result = cudaStreamCreateWithPriority(&new_stream_id, flags, priority);
	throw_if_error(result,
		std::string("Failed creating a new stream on CUDA device ")
		+ std::to_string(device::current::get_id()));
	return new_stream_id;
}

/**
 * Creates a new stream on an artibrary device
 * (which may or may not be the current one)
 *
 * @return the created stream's id
 */
inline id_t create(
	device::id_t  device_id,
	priority_t    priority = default_priority,
	bool          synchronizes_with_default_stream = true)
{
	device::current::ScopedDeviceOverride<> set_device_for_this_scope(device_id);
	return create(priority, synchronizes_with_default_stream);
}

} // namespace stream

template <bool AssumesDeviceIsCurrent = detail::do_not_assume_device_is_current>
class stream_t {

public: // type definitions
	using callback_t = std::function<void(stream::id_t, status_t)>;

	using priority_t = stream::priority_t;

	enum : bool {
		doesnt_synchronizes_with_default_stream  = false,
		does_synchronize_with_default_stream     = true,
	};

protected: // type definitions
	using DeviceSetter = ::cuda::device::current::ScopedDeviceOverride<AssumesDeviceIsCurrent>;

public: // statics

	/**
	 * Check whethers a certain stream is associated with a specific device.
	 *
	 * @note the stream_t class includes information regarding a stream's
	 * device association, so this function only makes sense for CUDA stream
	 * identifiers
	 *
	 * @param stream_id the CUDA runtime API identifier for the stream whose
	 * association is to be checked
	 * @param device_id a CUDA device identifier
	 * @return true if the specified stream is associated with the specified
	 * device, false if they are unassociated
	 * @throws if the association check returns anything weird
	 */
	static bool associated_with(stream::id_t stream_id, device::id_t device_id)
	{
		DeviceSetter set_device_for_this_scope(device_id);
		auto result = cudaStreamQuery(stream_id);
		switch(result) {
		case cudaSuccess:
		case cudaErrorNotReady:
			return true;
		case cudaErrorInvalidResourceHandle:
			return false;
		default:
			throw(std::logic_error("unexpected status returned from cudaStreamQuery()"));
		}
	}

public: // const getters
	stream::id_t id() const { return id_; }
	device::id_t device_id() const { return device_id_; }

public: // other non-mutators

	/**
	 * Strangely enough, CUDA won't tell you which device a stream is associated with,
	 * while it can - supposedly - tell this itself when querying stream status. So,
	 * let's use that. This is ugly and possibly buggy, but it _might_ just work.
	 *
	 * @param stream_id a stream identifier
	 * @return the identifier of the device for which the stream was created.
	 */
	device::id_t  get_associated_device(stream::id_t stream_id) const
	{
		if (stream_id == cuda::stream::default_stream_id) {
			throw std::invalid_argument("Cannot determine device association The default/null stream");
		}
		for(device::id_t  d = 0; d < device::count(); d++) {
			if (associated_with(stream_id, d)) { return d; }
		}
		throw std::runtime_error("Could not find any device associated with a specified stream");
	}

	/**
	 * When true, work running in the created stream may run concurrently with
	 * work in stream 0 (the NULL stream), and there is no implicit
	 * synchronization performed between it and stream 0.
	 */
	bool synchronizes_with_default_stream() const
	{
		unsigned int flags;
		auto result = cudaStreamGetFlags(id_, &flags);
		throw_if_error(result,
			std::string("Failed obtaining flags for a stream")
			+ " on CUDA device " + std::to_string(device_id_));
		return flags & cudaStreamNonBlocking;
	}

	priority_t priority() const
	{
		int the_priority;
		auto result = cudaStreamGetPriority(id_, &the_priority);
		throw_if_error(result,
			std::string("Failure obtaining priority for a stream")
			+ " on CUDA device " + std::to_string(device_id_));
		return the_priority;
	}

	/**
	 * Check whether the queue has any incomplete operations
	 *
	 * @todo What if there are incomplete operations, but they're all waiting on
	 * something on another queue? Should the queue count as "busy" then?
	 *
	 * @return true if all enqueued operations have been completed; false if there
	 * are any incomplete enqueued operations
	 */
	bool busy() const
	{
		DeviceSetter set_device_for_this_scope(device_id_);
		auto result = cudaStreamQuery(id_);
		switch(result) {
		case cudaSuccess:
			return true;
		case cudaErrorNotReady:
			return false;
		default:
			throw(std::logic_error("unexpected status returned from cudaStreamQuery()"));
		}
	}

protected: // static methods

	/**
	 * Untested!
	 */
	static void callback_adapter(stream::id_t stream_id, status_t status, void *type_erased_callback)
	{
		auto retyped_callback = reinterpret_cast<callback_t*>(type_erased_callback);
		(*retyped_callback)(stream_id, status);
	}


public: // mutators
	template<typename KernelFunction, typename... KernelParameters>
	void launch(
		const KernelFunction&       kernel_function,
		launch_configuration_t      launch_configuration,
		KernelParameters...         parameters)
	{
		return ::cuda::launch(kernel_function, launch_configuration, id_, parameters...);
	}

	// TODO: Duplicate the above better?
	template<typename KernelFunction, typename... KernelParameters>
	void enqueue(
		const KernelFunction&       kernel_function,
		launch_configuration_t      launch_configuration,
		KernelParameters...         parameters)
	{
		return launch(kernel_function, launch_configuration, id_, parameters...);
	}

	__host__ void synchronize()
	{
		DeviceSetter set_device_for_this_scope(id_);
		// TODO: some kind of string representation for the stream
		auto result = cudaStreamSynchronize(id_);
		throw_if_error(result,
			std::string("Failed synchronizing a stream")
			+ " on CUDA device " + std::to_string(device_id_));
	}

	__host__ void add_callback (callback_t& callback)
	{
		DeviceSetter set_device_for_this_scope(id_);

		// The nVIDIA runtime API (upto v.8) requires flags to be 0, see
		// http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM
		enum : unsigned int { fixed_flags = 0 };

		// This always registers the static function callback_adapter as the callback -
		// but what that one will do is call the actual callback we were passed; note
		// that since you can one can have a lambda capture data and wrap that in an
		// std::function, there's not much need (it would seem) for an extra inner
		// user_data parameter to callback_t.
		auto result = cudaStreamAddCallback(id_, &callback_adapter, &callback, fixed_flags);
		throw_if_error(result,
			std::string("Failed adding a callback to stream ")
			+ " on CUDA device " + std::to_string(device_id_));
	}

	// TODO: wrap this with a nicer interface taking a lambda...
	/**
	 * Attaches a region of managed memory (i.e. in an address space visible
	 * on all CUDA devices and the host) to this specific stream on its specific device.
	 * This is actually a commitment vis-a-vis the CUDA driver and the GPU itself that
	 * it doesn't need to worry about accesses to this memory from elsewhere, and can
	 * optimize accordingly. Also, the host will be allowed to read from this memory
	 * region whenever no kernels are pending on this stream
	 *
	 * @note This happens asynchronously, as an operation on this stream, i.e.
	 * the attachment goes into effect (some time after) after previous stream
	 * operations have concluded.
	 *
	 * @param managed_region_start a pointer to the beginning of the managed memory region.
	 * This cannot be a pointer to anywhere in the middle of an allocated region - you must
	 * pass whatever cudaMallocManaged (@ref memory::managed:allocate
	 */
	void enqueue_memory_attachment(const void* managed_region_start)
	{
		DeviceSetter set_device_for_this_scope(id_);
		// This fixed value is required by the CUDA Runtime API,
		// to indicate that the entire memory region, rather than a part of it, will be
		// attached to this stream
		constexpr const size_t length = 0;
		auto result =  cudaStreamAttachMemAsync(id_, managed_region_start, length, cudaMemAttachSingle);
		throw_if_error(result,
			std::string("Failed attaching a managed memory region to a stream")
			+ " on CUDA device " + std::to_string(device_id_));
	}


	void wait(event::id_t event_id)
	{
		// Required by the CUDA runtime API
		constexpr const unsigned int  flags = 0;
		auto result = cudaStreamWaitEvent(id_, event_id, flags);
		throw_if_error(result,
			std::string("Failed waiting for an event")
			+ " on CUDA device " + std::to_string(device_id_));
	}

	// TODO: wait() with an event_t - possible?

public: // constructors and destructor

	stream_t(cuda::device::id_t device_id, stream::id_t stream_id) : device_id_(device_id), id_(stream_id)
	{
		// TODO: Should we check that the stream is actually associated with the device?
	}
	~stream_t() { if (is_owning) cudaStreamDestroy(id_); }

public: // named constructor idioms

	stream_t<detail::do_not_assume_device_is_current> create(
		device::id_t  device_id,
		priority_t    priority = stream::default_priority,
		bool          synchronizes_with_default_stream = true)
	{
		return stream_t<detail::do_not_assume_device_is_current>(
			AssumesDeviceIsCurrent ?
				stream::create(priority, synchronizes_with_default_stream) :
				stream::create(device_id, priority, synchronizes_with_default_stream)
		);
	}

protected: // constructors

	// The ctors here are called by named constructor idioms, see below; we don't
	// want to "just" create streams willy-nilly just by writing "stream_t my_stream;"...

	/**
	 *
	 *
	 * @param device_id
	 * @param priority a value in the device's allowed priority range
	 * (@ref cudaDeviceGetStreamPriorityRange
	 * @param synchronizes_with_default_stream
	 */
	stream_t(
		cuda::device::id_t  device_id,
		priority_t priority = stream::default_priority,
		bool synchronizes_with_default_stream = true) : device_id_(device_id), is_owning(true)
	{
		stream::create(device_id, priority, synchronizes_with_default_stream);
	}
	stream_t() : stream_t(cuda::device::current::get_id()) { };

protected: // data members
	cuda::device::id_t  device_id_;
	stream::id_t        id_;
	bool               is_owning { false };

};

template <bool AssumesDeviceIsCurrent = false>
using queue_t = stream_t<AssumesDeviceIsCurrent>;

using queue_id_t = stream::id_t;

} // namespace cuda


#endif /* CUDA_STREAM_H_ */
