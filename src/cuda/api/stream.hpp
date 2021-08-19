/**
 * @file stream.hpp
 *
 * @brief A proxy class for CUDA streams, providing access to
 * all Runtime API calls involving their use and management.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_STREAM_HPP_
#define CUDA_API_WRAPPERS_STREAM_HPP_

#include <cuda/api/current_device.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/kernel_launch.hpp>
#include <cuda/api/memory.hpp>
#include <cuda/api/miscellany.hpp>
#include <cuda/common/types.hpp>

#include <cuda_runtime_api.h>

#include <string>
#include <memory>
#include <utility>
#include <tuple>

namespace cuda {

class device_t;
class event_t;
class stream_t;

namespace stream {

// Use this for the second argument to create_on_current_device()
enum : bool {
	implicitly_synchronizes_with_default_stream = true,
	no_implicit_synchronization_with_default_stream = false,
	sync = implicitly_synchronizes_with_default_stream,
	async = no_implicit_synchronization_with_default_stream,
};

namespace detail_ {

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
		::std::string("Failed creating a new stream on CUDA device ")
		+ ::std::to_string(device::current::detail_::get_id()));
	return new_stream_id;
}

/**
 * Check whether a certain stream is associated with a specific device.
 *
 * @note the stream_t class includes information regarding a stream's
 * device association, so this function only makes sense for CUDA stream
 * identifiers.
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
	device::current::detail_::scoped_override_t set_device_for_this_scope(device_id);
	auto status = cudaStreamQuery(stream_id);
	switch(status) {
	case cudaSuccess:
	case cudaErrorNotReady:
		return true;
	case cudaErrorInvalidResourceHandle:
		return false;
	default:
		throw(::std::logic_error("unexpected status returned from cudaStreamQuery()"));
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
		throw ::std::invalid_argument("Cannot determine device association for the default/null stream");
	}
	for(device::id_t device_index = 0; device_index < device::count(); device_index++) {
		if (is_associated_with(stream_id, device_index)) { return device_index; }
	}
	throw ::std::runtime_error(
		"Could not find any device associated with stream " + cuda::detail_::ptr_as_hex(stream_id));
}

inline void record_event_on_current_device(device::id_t current_device_id, stream::id_t stream_id, event::id_t event_id);

/**
 * Wraps a CUDA stream ID in a stream_t proxy instance,
 * possibly also taking on the responsibility of eventually
 * destroying the stream
 *
 * @return a stream_t proxy for the CUDA stream
 */
stream_t wrap(
	device::id_t  device_id,
	id_t          stream_id,
	bool          take_ownership = false) noexcept;

} // namespace detail_

} // namespace stream

inline void synchronize(const stream_t& stream);

/**
 * @brief Proxy class for a CUDA stream
 *
 * Use this class - built around an event ID - to perform almost, if not all
 * operations related to CUDA events,
 *
 * @note this is one of the three main classes in the Runtime API wrapper library,
 * together with @ref cuda::device_t and @ref cuda::event_t
 */
class stream_t {

public: // type definitions
	using priority_t = stream::priority_t;

	enum : bool {
		doesnt_synchronizes_with_default_stream  = false,
		does_synchronize_with_default_stream     = true,
	};

protected: // type definitions
	using device_setter_type = device::current::detail_::scoped_override_t;


public: // const getters
	stream::id_t id() const noexcept { return id_; }
	device_t device() const noexcept;
	bool is_owning() const noexcept { return owning; }

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
			::std::string("Failed obtaining flags for a stream")
			+ " on CUDA device " + ::std::to_string(device_id_));
		return flags & cudaStreamNonBlocking;
	}

	priority_t priority() const
	{
		int the_priority;
		auto status = cudaStreamGetPriority(id_, &the_priority);
		throw_if_error(status,
			::std::string("Failure obtaining priority for a stream")
			+ " on CUDA device " + ::std::to_string(device_id_));
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
		device_setter_type set_device_for_this_scope(device_id_);
		auto status = cudaStreamQuery(id_);
		switch(status) {
		case cudaSuccess:
			return false;
		case cudaErrorNotReady:
			return true;
		default:
			throw(cuda::runtime_error(status,
				"unexpected status returned from cudaStreamQuery() for stream "
				+ detail_::ptr_as_hex(id_)));
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
	 * A function used internally by this class as the host function to call directly; see
	 * @ref enqueue_t::host_function_call - but only with CUDA version 10.0 and later.
	 *
	 * @param stream_id the ID of the stream for which a host function call was triggered - this
	 * will be passed by the CUDA runtime
	 * @param device_id_stream_id_and_callable a 3-tuple, containing the ID of the device to which the stream launching
	 * the callable is associated, the ID of that launching stream, and the callable callback which was passed to
	 * @ref enqueue_t::host_function_call, and which the programmer actually wants to be called.

	 */
	template <typename Callable>
	static void stream_launched_host_function_adapter(void * device_id_stream_id_and_callable)
	{
		using triplet_type = ::std::tuple<device::id_t, stream::id_t, Callable>;
		auto* triplet_ptr = reinterpret_cast<triplet_type*>(device_id_stream_id_and_callable);
		auto unique_ptr = ::std::unique_ptr<triplet_type>{triplet_ptr}; // Ensures deletion when we leave this function.
		auto device_id = ::std::get<0>(*triplet_ptr);
		auto stream_id = ::std::get<1>(*triplet_ptr);
		auto& callable = ::std::get<2>(*triplet_ptr);
		callable( stream_t{device_id, stream_id, do_not_take_ownership} );
	}

	/**
	 * @brief A function to @ref `host_function_launch_adapter`, for use with the old-style CUDA Runtime API call,
	 * which passes more arguments to the callable - and calls the host function even on device failures.
	 *
	 * @param stream_id the ID of the stream for which a host function call was triggered - this
	 * will be passed by the CUDA runtime
	 * @note status indicates the status the CUDA status when the host function call is triggered; anything
	 * other than @ref `cuda::status::success` means there's been a device error previously - but
	 * in that case, we won't invoke the callable, as such execution is deprecated; see:
	 * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM
	 * @param device_id_and_callable a pair-value, containing the ID of the device to which the stream launching
	 * the host function call is associated, as well as the callable callback which was passed to
	 * @ref enqueue_t::host_function_call, and which the programmer actually wants to be called.
	 */
	template <typename Callable>
	static void callback_launch_adapter(
		stream::id_t  stream_id,
		status_t      status,
		void *        device_id_stream_id_and_callable)
	{
		(void) stream_id; // it's redundant
		if (status != cuda::status::success) {
			using triplet_type = ::std::tuple<device::id_t, stream::id_t, Callable>;
			delete reinterpret_cast<triplet_type*>(device_id_stream_id_and_callable);
			return;
		}
		stream_launched_host_function_adapter<Callable>(device_id_stream_id_and_callable);
	}

public: // mutators

	/**
	 * @brief A gadget through which commands are enqueued on the stream.
	 *
	 * @note this class exists solely as a form of "syntactic sugar", allowing for code such as
	 *
	 *   my_stream.enqueue.copy(foo, bar, my_size)
	 */
	class enqueue_t {
	protected:
		const stream_t& associated_stream;

	public:
		enqueue_t(const stream_t& stream) : associated_stream(stream) {}

		template<typename KernelFunction, typename... KernelParameters>
		void kernel_launch(
			bool                        thread_block_cooperativity,
			const KernelFunction&       kernel_function,
			launch_configuration_t      launch_configuration,
			KernelParameters...         parameters)
		{
			// Kernel executions cannot be enqueued in streams associated
			// with devices other than the current one, see:
			// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#stream-and-event-behavior
			device_setter_type set_device_for_this_scope(associated_stream.device_id_);
			return cuda::enqueue_launch(
				thread_block_cooperativity,
				kernel_function,
				associated_stream,
				launch_configuration,
				parameters...);
		}

		template<typename KernelFunction, typename... KernelParameters>
		void kernel_launch(
			const KernelFunction&       kernel_function,
			launch_configuration_t      launch_configuration,
			KernelParameters...         parameters)
		{
			// TODO: Somehow I can't avoid code duplication with the previous variant of kernel_launch;
			// why is that?
			//
			// return kernel_launch(cuda::thread_blocks_cant_cooperate,
			// 	kernel_function, stream_id_, launch_configuration, parameters...);
			//

			device_setter_type set_device_for_this_scope(associated_stream.device_id_);
			return cuda::enqueue_launch(
				cuda::thread_blocks_may_not_cooperate,
				kernel_function, associated_stream, launch_configuration, parameters...);
		}

		/**
		 * Have the CUDA device perform an I/O operation between two specified
		 * memory regions (on or off the actual device)
		 *
		 */

		///@{
		/**
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
			memory::async::detail_::copy(destination, source, num_bytes, associated_stream.id_);
		}

		void copy(void* destination, memory::const_region_t source, size_t num_bytes)
		{
#ifndef NDEBUG
			if (source.size() < num_bytes) {
				throw ::std::logic_error("Attempt to copy more than the source region's size");
			}
#endif
			copy(destination, source.start(), num_bytes);
		}

		void copy(memory::region_t destination, memory::const_region_t source, size_t num_bytes)
		{
			copy(destination.start(), source, num_bytes);
		}

		void copy(memory::region_t destination, memory::const_region_t source)
		{
			copy(destination, source, source.size());
		}
		///@}

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
			device_setter_type set_device_for_this_scope(associated_stream.device_id_);
			memory::device::async::detail_::set(destination, byte_value, num_bytes, associated_stream.id_);
		}

		/**
		 * Set all bytes of a certain region in device memory (or unified memory,
		 * but using the CUDA device to do it) to zero.
		 *
		 * @note this is a separate method, since the CUDA runtime has a separate
		 * API call for setting to zero; does that mean there are special facilities
		 * for zero'ing memory faster? Who knows.
		 *
		 * @param destination Beginning of the region to fill
		 * @param num_bytes size of the region to fill
		 */
		void memzero(void *destination, size_t num_bytes)
		{
			// Is it necessary to set the device? I wonder.
			device_setter_type set_device_for_this_scope(associated_stream.device_id_);
			memory::device::async::detail_::zero(destination, num_bytes, associated_stream.id_);
		}

		/**
		 * Have an event 'fire', i.e. marked as having occurred,
		 * after all hereto-scheduled work on this stream has been completed.
		 * Threads which are @ref stream_t::wait_on() 'ing the event will become available
		 * for continued execution.
		 *
		 * @param existing_event A pre-created CUDA event (for the stream's device); any existing
		 * "registration" of the event to occur elsewhere is overwritten.
		 *
		 * @note It is possible to wait for events across devices, but it is _not_ possible to
		 * trigger events across devices.
		 **/
		event_t& event(event_t& existing_event);

		/**
		 * Have an event 'fire', i.e. marked as having occurred,
		 * after all hereto-scheduled work on this stream has been completed.
		 * Threads which are @ref stream_t::wait_on() 'ing the event will become available
		 * for continued execution.
		 *
		 * @note the parameters are the same as for @ref event::create()
		 *
		 * @note It is possible to wait for events across devices, but it is _not_ possible to
		 * trigger events across devices.
		 *
		 **/
		event_t event(
			bool          uses_blocking_sync = event::sync_by_busy_waiting,
			bool          records_timing     = event::do_record_timings,
			bool          interprocess       = event::not_interprocess);

		/**
		 * Execute the specified function on the calling host thread once all
		 * hereto-scheduled work on this stream has been completed.
		 *
		 * @param callable_ a function to execute on the host. It must be callable
		 * with two parameters: `cuda::stream::id_t stream_id, cuda::event::id_t event_id`
		 */
		template <typename Callable>
		void host_function_call(Callable callable_)
		{
			device_setter_type set_device_for_this_scope(associated_stream.device_id_);


			// Since callable_ will be going out of scope after the enqueueing,
			// and we don't know anything about the scope of the original argument with
			// which we were called, we must make a copy of `callable_` on the heap
			// and pass that as the user-defined data. We also add information about
			// the enqueueing stream.
			auto raw_callable_extra_argument = new
				::std::tuple<device::id_t, stream::id_t, Callable>(
					associated_stream.device_id_,
					associated_stream.id(),
					Callable(::std::move(callable_))
				);

			// While we always register the same static function, `callback_adapter` as the
			// callback - what it will actually _do_ is invoke the callback we were passed.

#if CUDART_VERSION >= 10000
			auto status = cudaLaunchHostFunc(
				associated_stream.id_, &stream_launched_host_function_adapter<Callable>, raw_callable_extra_argument);
#else
			// The nVIDIA runtime API (at least up to v10.2) requires passing 0 as the flags
			// variable, see:
			// http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
			constexpr const unsigned fixed_flags { 0u };
			auto status = cudaStreamAddCallback(
				associated_stream.id_, &callback_launch_adapter<Callable>, raw_callable_extra_argument, fixed_flags);
#endif

			throw_if_error(status,
				::std::string("Failed scheduling a callback to be launched")
				+ " on stream " + cuda::detail_::ptr_as_hex(associated_stream.id_)
				+ " on CUDA device " + ::std::to_string(associated_stream.device_id_));
		}


		/**
		 * Allocate a specified amount of memory.
		 *
		 * @param num_bytes amount of memory to allocate
		 * @return a region whose location is set when scheduling, but the memory of which
		 * only become allocated for use once the allocation task is actually reached by
		 * the stream and completed.
		 */
		memory::region_t memory_allocation(size_t num_bytes) {
			return memory::device::async::allocate(associated_stream, num_bytes);
		}

		/**
		 * Sets the attachment of a region of managed memory (i.e. in the address space visible
		 * on all CUDA devices and the host) in one of several supported attachment modes.
		 *
		 * The attachmentis actually a commitment vis-a-vis the CUDA driver and the GPU itself
		 * that it doesn't need to worry about accesses to this memory from devices other than
		 * its object of attachment, so that the driver can optimize scheduling accordingly.
		 *
		 * @note by default, the memory region is attached to this specific stream on its
		 * specific device. In this case, the host will be allowed to read from this memory
		 * region whenever no kernels are pending on this stream.
		 *
		 * @note Attachment happens asynchronously, as an operation on this stream, i.e.
		 * the attachment goes into effect (some time after) previous scheduled actions have
		 * concluded.
		 */
		///@{
		/**
		 * @param managed_region_start a pointer to the beginning of the managed memory region.
		 * This cannot be a pointer to anywhere in the middle of an allocated region - you must
		 * pass whatever @ref cuda::memory::managed::allocate() (or `cudaMallocManaged()`)
		 * returned.
		 */
		void memory_attachment(
			const void* managed_region_start,
			memory::managed::attachment_t attachment = memory::managed::attachment_t::single_stream)
		{
			device_setter_type set_device_for_this_scope(associated_stream.device_id_);
			// This fixed value is required by the CUDA Runtime API,
			// to indicate that the entire memory region, rather than a part of it, will be
			// attached to this stream
			constexpr const size_t length = 0;
			auto flags = static_cast<unsigned>(attachment);
			auto status =  cudaStreamAttachMemAsync(
				associated_stream.id_, managed_region_start, length, flags);
			throw_if_error(status,
				"Failed scheduling an attachment of a managed memory region"
				" on stream " + cuda::detail_::ptr_as_hex(associated_stream.id_)
				+ " on CUDA device " + ::std::to_string(associated_stream.device_id_));
		}

		/**
		 * @param region the managed memory region to attach; it cannot be a sub-region -
		 * you must pass whatever @ref cuda::memory::managed::allocate() returned.
		 */
		void memory_attachment(
			memory::managed::region_t region,
			memory::managed::attachment_t attachment = memory::managed::attachment_t::single_stream)
		{
			memory_attachment(region.start(), attachment);
		}
		///@}


		/**
		 * Will pause all further activity on the stream until the specified event has
		 * occurred  (i.e. has fired, i.e. has had all preceding scheduled work
		 * on the stream on which it was recorded completed).
		 *
		 * @note this call will not delay any already-enqueued work on the stream,
		 * only work enqueued _after_ the call.
		 *
		 * @param event_ the event for whose occurrence to wait; the event
		 * would typically be recorded on another stream.
		 *
		 */
		void wait(const event_t& event_);

	}; // class enqueue_t

	friend class enqueue_t;

	/**
	 * Block or busy-wait until all previously-scheduled work
	 * on this stream has been completed
	 */
	void synchronize() const
	{
		cuda::synchronize(*this);
	}

protected: // constructor

	stream_t(device::id_t device_id, stream::id_t stream_id, bool take_ownership = false) noexcept
	: device_id_(device_id), id_(stream_id), owning(take_ownership) { }

public: // constructors and destructor

	stream_t(const stream_t& other) noexcept :
	stream_t(other.device_id_, other.id_, false) { }

	stream_t(stream_t&& other) noexcept : 
		stream_t(other.device_id_, other.id_, other.owning)
	{
		other.owning = false;
	}

	~stream_t()
	{
		if (owning) {
			device_setter_type set_device_for_this_scope(device_id_);
			cudaStreamDestroy(id_);
		}
	}

public: // operators

	stream_t& operator=(const stream_t& other) = delete;
	stream_t& operator=(stream_t& other) = delete;

public: // friendship

	friend stream_t stream::detail_::wrap(device::id_t device_id, stream::id_t stream_id, bool take_ownership) noexcept;

	friend inline bool operator==(const stream_t& lhs, const stream_t& rhs) noexcept
	{
		return lhs.device_id_ == rhs.device_id_ and lhs.id() == rhs.id();
	}

protected: // data members
	const device::id_t  device_id_;
	const stream::id_t  id_;
	bool                owning;

public: // data members - which only exist in lieu of namespaces
	enqueue_t     enqueue { *this };
		// The use of *this here is safe, since enqueue_t doesn't do anything with it
		// on its own. Any use of enqueue only happens through, well, *this - and
		// after construction.
};

inline bool operator!=(const stream_t& lhs, const stream_t& rhs) noexcept
{
	return not (lhs == rhs);
}

namespace stream {

namespace detail_ {
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
inline stream_t wrap(
	device::id_t  device_id,
	id_t          stream_id,
	bool          take_ownership /* = false, see declaration */) noexcept
{
	return stream_t(device_id, stream_id, take_ownership);
}

inline stream_t create(
	device::id_t  device_id,
	bool          synchronizes_with_default_stream,
	priority_t    priority = stream::default_priority)
{
	device::current::detail_::scoped_override_t set_device_for_this_scope(device_id);
	auto new_stream_id = cuda::stream::detail_::create_on_current_device(
		synchronizes_with_default_stream, priority);
	return wrap(device_id, new_stream_id, do_take_ownership);
}

} // namespace detail_

/**
 * @brief Create a new stream (= queue) on a CUDA device.
 *
 * @param device the device on which a stream is to be created
 * @param synchronizes_with_default_stream if true, no work on this stream
 * will execute concurrently with work from the default stream (stream 0)
 * @param priority priority of tasks on the stream, relative to other streams,
 * for execution scheduling; lower numbers represent higher properties. Each
 * device has a range of priorities, which can be obtained using
 * @ref device_t::stream_priority_range() .
 * @return The newly-created stream
 */
stream_t create(
	device_t     device,
	bool         synchronizes_with_default_stream,
	priority_t   priority = stream::default_priority);

} // namespace stream

using queue_t = stream_t;
using queue_id_t = stream::id_t;

inline void synchronize(const stream_t& stream)
{
	auto status = cudaStreamSynchronize(stream.id());
	throw_if_error(status,
		::std::string("Failed synchronizing a stream")
		+ " on CUDA device " + ::std::to_string(stream.device().id()));
}


} // namespace cuda

#endif // CUDA_API_WRAPPERS_STREAM_HPP_
