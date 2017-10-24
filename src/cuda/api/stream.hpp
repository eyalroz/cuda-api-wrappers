/**
 * @file stream.hpp
 *
 * @brief A proxy class for CUDA streams, providing access to
 * all Runtime API calls involving their use and management.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_STREAM_HPP_
#define CUDA_API_WRAPPERS_STREAM_HPP_

#include <cuda/api/types.h>
#include <cuda/api/error.hpp>
#include <cuda/api/memory.hpp>
#include <cuda/api/kernel_launch.cuh>
#include <cuda/api/current_device.hpp>
#include <cuda/api/device_count.hpp>

#include <cuda_runtime_api.h>

#include <string>
#include <functional>

namespace cuda {

template <bool AssumedCurrent> class device_t;

template <bool AssumesDeviceIsCurrent = false> class stream_t;

namespace stream {

// Use this for the second argument to create_on_current_device()
enum : bool {
	implicitly_synchronizes_with_default_stream = true,
	no_implicit_synchronization_with_default_stream = false,
	sync = implicitly_synchronizes_with_default_stream,
	async = no_implicit_synchronization_with_default_stream,
};

namespace detail {

inline id_t create_on_current_device(
	bool          synchronizes_with_default_stream,
	priority_t    priority = stream::default_priority
)
{
	unsigned int flags = (synchronizes_with_default_stream == sync) ?
		cudaStreamDefault : cudaStreamNonBlocking;
	id_t new_stream_id;
	auto status = cudaStreamCreateWithPriority(&new_stream_id, flags, priority);
	cuda::throw_if_error(status,
		std::string("Failed creating a new stream on CUDA device ")
		+ std::to_string(device::current::get_id()));
	return new_stream_id;
}

} // namespace detail

/**
 * Check whether a certain stream is associated with a specific device.
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
inline bool is_associated_with(stream::id_t stream_id, device::id_t device_id)
{
	device::current::scoped_override_t<cuda::detail::do_not_assume_device_is_current>
		set_device_for_this_scope(device_id);
	auto status = cudaStreamQuery(stream_id);
	switch(status) {
	case cudaSuccess:
	case cudaErrorNotReady:
		return true;
	case cudaErrorInvalidResourceHandle:
		return false;
	default:
		throw(std::logic_error("unexpected status returned from cudaStreamQuery()"));
	}
}

/**
 * @brief Obtains the device ID with which a stream with a given ID is associated
 *
 * Strangely enough, CUDA won't tell you which device a stream is associated with,
 * while it can - supposedly - tell this itself when querying stream status. So,
 * let's use that. This is ugly and possibly buggy, but it _might_ just work.
 *
 * @param stream_id a stream identifier
 * @return the identifier of the device for which the stream was created.
 */
inline device::id_t associated_device(stream::id_t stream_id)
{
	if (stream_id == cuda::stream::default_stream_id) {
		throw std::invalid_argument("Cannot determine device association for the default/null stream");
	}
	for(device::id_t device_index = 0; device_index < device::count(); device_index++) {
		if (is_associated_with(stream_id, device_index)) { return device_index; }
	}
	throw std::runtime_error(
		"Could not find any device associated with stream " + cuda::detail::ptr_as_hex(stream_id));
}

/**
 * Wraps a CUDA stream ID in a stream_t proxy instance,
 * possibly also taking on the responsibility of eventually
 * destroying the stream
 *
 * @return a stream_t proxy for the CUDA stream
 */
inline stream_t<> wrap(
	device::id_t  device_id,
	id_t          stream_id,
	bool          take_ownership = false);

} // namespace stream

/**
 * @brief Proxy class for a CUDA stream
 *
 * Use this class - built around an event ID - to perform almost, if not all
 * operations related to CUDA events,
 *
 * @tparam AssumesDeviceCurrent - when true, the code performs no setting of the
 * device ID before acting
 *
 * @note this is one of the three main classes in the Runtime API wrapper library,
 * together with @ref cuda::device_t and @ref cuda::event_t
 */
template <bool AssumesDeviceIsCurrent /* = false , see template declaration */>
class stream_t {

public: // type definitions
	using callback_t = std::function<void(stream::id_t, status_t)>;

	using priority_t = stream::priority_t;

	enum : bool {
		doesnt_synchronizes_with_default_stream  = false,
		does_synchronize_with_default_stream     = true,
	};

protected: // type definitions
	using DeviceSetter = device::current::scoped_override_t<AssumesDeviceIsCurrent>;


public: // const getters
	stream::id_t id() const { return id_; }
	device::id_t device_id() const { return device_id_; }
	device_t<detail::do_not_assume_device_is_current> device() const;
	bool is_owning() const { return owning; }

public: // other non-mutators

	/**
	 * When true, work running in the created stream may run concurrently with
	 * work in stream 0 (the NULL stream), and there is no implicit
	 * synchronization performed between it and stream 0.
	 */
	bool synchronizes_with_default_stream() const
	{
		unsigned int flags;
		auto status = cudaStreamGetFlags(id_, &flags);
		throw_if_error(status,
			std::string("Failed obtaining flags for a stream")
			+ " on CUDA device " + std::to_string(device_id_));
		return flags & cudaStreamNonBlocking;
	}

	priority_t priority() const
	{
		int the_priority;
		auto status = cudaStreamGetPriority(id_, &the_priority);
		throw_if_error(status,
			std::string("Failure obtaining priority for a stream")
			+ " on CUDA device " + std::to_string(device_id_));
		return the_priority;
	}

	/**
	 * Determines whether all work on this stream has been completed
	 *
	 * @note having work is _not_ the same as being busy executing that work!
	 *
	 * @todo What if there are incomplete operations, but they're all waiting on
	 * something on another queue? Should the queue count as "busy" then?
	 *
	 * @return true if there is still work pending, false otherwise
	 */
	bool has_work_remaining() const
	{
		DeviceSetter set_device_for_this_scope(device_id_);
		auto status = cudaStreamQuery(id_);
		switch(status) {
		case cudaSuccess:
			return false;
		case cudaErrorNotReady:
			return true;
		default:
			throw(cuda::runtime_error(status,
				"unexpected status returned from cudaStreamQuery() for stream "
				+ detail::ptr_as_hex(id_)));
		}
	}

	/**
	 * The opposite of @ref has_work()
	 *
	 * @return true if there is no work pending, false if all
	 * previously-scheduled work has been completed
	 */
	bool is_clear() const { return !has_work_remaining(); }

	/**
	 * An alias for @ref is_clear() - to conform to how the CUDA runtime
	 * API names this functionality
	 */
	bool query() const { return is_clear(); }


protected: // static methods

	/**
	 * A function used internally by this class as the immediate CUDA callback; see
	 * @ref enqueue_t::callback
	 *
	 * @param stream_id the ID of the stream for which a callback was triggered - this
	 * will be passed by the CUDA runtime
	 * @param status the CUDA status when the callback is triggered - this
	 * will be passed by the CUDA runtime
	 * @param type_erased_callback the callback which was passed to @ref enqueue_t::callback,
	 * and which the programmer actually wants to be called
	 */
	static void callback_adapter(stream::id_t stream_id, status_t status, void *type_erased_callback)
	{
		auto retyped_callback = reinterpret_cast<callback_t*>(type_erased_callback);
		(*retyped_callback)(stream_id, status);
	}

public: // mutators

	class enqueue_t {
	protected:
		const device::id_t& device_id_;
		const stream::id_t& stream_id_;

	public:
		enqueue_t(const device::id_t& device_id, const stream::id_t& stream_id)
		: device_id_(device_id), stream_id_(stream_id) {}

		template<typename KernelFunction, typename... KernelParameters>
		void kernel_launch(
			const KernelFunction&       kernel_function,
			launch_configuration_t      launch_configuration,
			KernelParameters...         parameters)
		{
			// Kernel executions cannot be enqueued in streams associated
			// with devices other than the current one, see:
			// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
			DeviceSetter set_device_for_this_scope(device_id_);
			return cuda::enqueue_launch(kernel_function, stream_id_, launch_configuration, parameters...);
		}

		/**
		 * Have the CUDA device perform an I/O operation between two specified
		 * memory regions (on or off the actual device)
		 *
		 * @param destination destination region into which to copy. May be
		 * anywhere in which memory can be mapped to the device's memory space (e.g.
		 * the device's global memory, host memory or the global memory of another device)
		 * @param source destination region from which to copy. May be
		 * anywhere in which memory can be mapped to the device's memory space (e.g.
		 * the device's global memory, host memory or the global memory of another device)
		 * @param num_bytes size of the region to copy
		 **/
		void copy(void *destination, const void *source, size_t num_bytes)
		{
			// It is not necessary to make the device current, according to:
			// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
			memory::async::copy(destination, source, num_bytes, stream_id_);
		}

		/**
		 * Set all bytes of a certain region in device memory (or unified memory,
	 	 * but using the CUDA device to do it) to a single fixed value.
		 *
		 * @param destination Beginning of the region to fill
		 * @param byte_value the value with which to fill the memory region bytes
		 * @param num_bytes size of the region to fill
		 */
		void memset(void *destination, int byte_value, size_t num_bytes)
		{
			// Is it necessary to set the device? I wonder.
			DeviceSetter set_device_for_this_scope(device_id_);
			memory::device::async::set(destination, byte_value, num_bytes, stream_id_);
		}

		/**
		 * Have an event 'fire', i.e. marked as having occurred,
		 * after all hereto-scheduled work on this stream has been completed.
		 * Threads which are @ref stream_t::wait_on() 'ing the event will become available
		 * for continued execution.
		 *
		 * @param event_id CUDA runtime API ID of the event to have occuring on
		 * completion of the hereto-scheduled work on this stream
		 **/
		void event(cuda::event::id_t event_id) {
			// TODO: ensure the stream and the event are associated with the same device

			// Not calling event::detail::enqueue to avoid dependency on event.hpp
			auto status = cudaEventRecord(event_id, stream_id_);
			throw_if_error(status,
				"Failed scheduling event " + cuda::detail::ptr_as_hex(event_id) + " to occur"
				+ " on stream " + cuda::detail::ptr_as_hex(stream_id_)
				+ " on CUDA device " + std::to_string(device_id_));
		}

		/**
		 * Execute the specified function on the calling host thread once all
		 * hereto-scheduled work on this stream has been completed.
		 *
		 * @param callback a function to execute on the host. Its signature
		 * must being with `(cuda::stream::id_t stream_id, cuda::event::id_t event_id`
		 */
		void callback(callback_t callback)
		{
			DeviceSetter set_device_for_this_scope(device_id_);

			// The nVIDIA runtime API (upto v8.0) requires flags to be 0, see
			// http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
			//
			enum : unsigned int { fixed_flags = 0 };

			// This always registers the static function callback_adapter as the callback -
			// but what that one will do is call the actual callback we were passed; note
			// that since you can can have a lambda capture data and wrap that in the
			// std::function, there's not much need (it would seem) for an extra inner
			// user_data parameter to callback_t
			auto status = cudaStreamAddCallback(stream_id_, &callback_adapter, &callback, fixed_flags);
			throw_if_error(status,
				std::string("Failed scheduling a callback function to be launched")
				+ " on stream " + cuda::detail::ptr_as_hex(stream_id_)
				+ " on CUDA device " + std::to_string(device_id_));
		}

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
		 * pass whatever @ref cuda::memory::managed::allocate() (or `cudaMallocManaged()`)
		 * returned.
		 */
		void memory_attachment(const void* managed_region_start)
		{
			DeviceSetter set_device_for_this_scope(device_id_);
			// This fixed value is required by the CUDA Runtime API,
			// to indicate that the entire memory region, rather than a part of it, will be
			// attached to this stream
			constexpr const size_t length = 0;
			auto status =  cudaStreamAttachMemAsync(
				stream_id_, managed_region_start, length, cudaMemAttachSingle);
			throw_if_error(status,
				std::string("Failed scheduling an attachment of a managed memory region")
				+ " on stream " + cuda::detail::ptr_as_hex(stream_id_)
				+ " on CUDA device " + std::to_string(device_id_));
		}

		/**
		 * Will pause all further activity on the stream until the specified event has
		 * occurred  (i.e. has fired, i.e. has had all preceding scheduled work
		 * on the stream on which it was recorded completed).
		 *
		 * @note this call will not delay any already-enqueued work on the stream,
		 * only work enqueued _after_ the call.
		 *
		 * @param event_id ID of the event for whose occurrence to wait; the event
		 * would typically be recorded on another stream.
		 *
		 */
		void wait(event::id_t event_id)
		{
			// Required by the CUDA runtime API; the flags value is
			// currently unused
			constexpr const unsigned int  flags = 0;
			auto status = cudaStreamWaitEvent(stream_id_, event_id, flags);
			throw_if_error(status,
				std::string("Failed scheduling a wait for event ") + cuda::detail::ptr_as_hex(event_id)
				+ " on stream " + cuda::detail::ptr_as_hex(stream_id_)
				+ " on CUDA device " + std::to_string(device_id_));
		}

	}; // class enqueue_t

	/**
	 * Block or busy-wait until all previously-scheduled work
	 * on this stream has been completed
	 */
	void synchronize()
	{
		DeviceSetter set_device_for_this_scope(device_id_);
		// TODO: some kind of string representation for the stream
		auto status = cudaStreamSynchronize(id_);
		throw_if_error(status,
			std::string("Failed synchronizing a stream")
			+ " on CUDA device " + std::to_string(device_id_));
	}

public: // constructors and destructor

	stream_t(const stream_t& other) :
		device_id_(other.device_id_), id_(other.id_), owning(false) { };

	stream_t(stream_t&& other) :
		device_id_(other.device_id_), id_(other.id_), owning(other.owning)
	{
		other.owning = false;
	};

	// TODO: Perhaps drop this in favor of just the protected constructor,
	// and let all wrapping construction be done by the stream::wrap() function?
	stream_t(device::id_t device_id, stream::id_t stream_id)
	: stream_t(device_id, stream_id, false) { }

	~stream_t()
	{
		if (owning) {
			device::current::scoped_override_t<> set_device_for_this_scope(device_id_);
			cudaStreamDestroy(id_);
		}
		owning = false;
	}

public: // operators

	// TODO: Do we really want to allow assignments? Hmm... probably not, it's
	// too risky - someone might destroy one of the streams and use the others
	stream_t& operator=(const stream_t<AssumesDeviceIsCurrent>& other) = delete;
	stream_t& operator=(const stream_t<not AssumesDeviceIsCurrent>& other) = delete;
	stream_t& operator=(stream_t<AssumesDeviceIsCurrent>& other) = delete;
	stream_t& operator=(stream_t<not AssumesDeviceIsCurrent>& other) = delete;

protected: // constructor

	stream_t(device::id_t device_id, stream::id_t stream_id, bool take_ownership)
	: device_id_(device_id), id_(stream_id), owning(take_ownership) { }

public: // friendship

	friend stream_t<> stream::wrap(device::id_t device_id, id_t stream_id, bool take_ownership);

protected: // data members
	const device::id_t  device_id_;
	const stream::id_t  id_;
	bool                owning;

public: // data members - which only exist in lieu of namespaces
	enqueue_t     enqueue { device_id_, id_ };

};

inline bool operator==(const stream_t<>& lhs, const stream_t<>& rhs)
{
	return lhs.device_id() == rhs.device_id() and lhs.id() == rhs.id();
}

inline bool operator!=(const stream_t<>& lhs, const stream_t<>& rhs)
{
	return not (lhs == rhs);
}

namespace stream {

/**
 * Use these for the third argument of @ref cuda::stream::wrap()
 */
enum : bool {
	dont_take_ownership = false,
	take_ownership      = true,
};

/**
 * @brief Wrap an existing stream in a @ref stream_t instance.
 *
 * @param device_id ID of the device for which the stream is defined
 * @param stream_id ID of the pre-existing stream
 * @param take_ownership When set to `false`, the stream
 * will not be destroyed along with the wrapper; use this setting
 * when temporarily working with a stream existing irrespective of
 * the current context and outlasting it. When set to `true`,
 * the proxy class will act as it does usually, destroying the stream
 * when being destructed itself.
 * @return an instance of the stream proxy class, with the specified
 * device-stream combination.
 */
inline stream_t<> wrap(
	device::id_t  device_id,
	id_t          stream_id,
	bool          take_ownership /* = false, see declaration */)
{
	return stream_t<>(device_id, stream_id, take_ownership);
}

inline stream_t<> create(
	device::id_t  device_id,
	bool          synchronizes_with_default_stream,
	priority_t    priority = stream::default_priority)
{
	device::current::scoped_override_t<> set_device_for_this_scope(device_id);
	auto new_stream_id = cuda::stream::detail::create_on_current_device(
		priority, synchronizes_with_default_stream);
	return wrap(device_id, new_stream_id, take_ownership);
}

} // namespace stream


template <bool AssumesDeviceIsCurrent = false>
using queue_t = stream_t<AssumesDeviceIsCurrent>;

using queue_id_t = stream::id_t;

} // namespace cuda


#endif /* CUDA_API_WRAPPERS_STREAM_HPP_ */
