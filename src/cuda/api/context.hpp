/**
 * @file
 *
 * @brief Contains a proxy class for CUDA execution contexts.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_CONTEXT_HPP_
#define CUDA_API_WRAPPERS_CONTEXT_HPP_

#include <cuda/api/current_context.hpp>
#include <cuda/api/versions.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/constants.hpp>
#include <cuda/api/types.hpp>

#include <cuda.h>
#include <string>
#include <utility>

namespace cuda {

///@cond
class device_t;
class event_t;
class context_t;
class stream_t;
class module_t;
///@endcond

namespace link {
struct options_t;
} // namespace link

namespace context {

using limit_t = CUlimit;
using limit_value_t = size_t;
using shared_memory_bank_size_t = CUsharedconfig;

/**
 * A range of priorities supported by a CUDA context; ranges from the
 * higher numeric value to the lower.
 */
struct stream_priority_range_t {
	stream::priority_t least; /// Higher numeric value, lower priority
	stream::priority_t greatest; /// Lower numeric value, higher priority

	/**
	 * When true, stream prioritization is not supported, i.e. all streams have
	 * "the same" priority - the default one.
	 */
	constexpr bool is_trivial() const
	{
		return least == stream::default_priority and greatest == stream::default_priority;
	}
};

/**
 * Obtain a wrapper for an already-existing CUDA context
 *
 * @note This is a named constructor idiom instead of direct access to the ctor of the same
 * signature, to emphase what this construction means - a new context is _not_
 * created.
 *
 * @param device_id Device with which the context is associated
 * @param context_id id of the context to wrap with a proxy
 * @param take_ownership when true, the wrapper will have the CUDA driver destroy
 * the context when the wrapper itself destruct; otherwise, it is assumed
 * that the context is "owned" elsewhere in the code, and that location or entity
 * is responsible for destroying it when relevant (possibly after this wrapper
 * ceases to exist)
 * @return a context wrapper associated with the specified context
 */
context_t wrap(
	device::id_t       device_id,
	context::handle_t  context_id,
	bool               take_ownership = false) noexcept;

namespace detail_ {

::std::string identify(const context_t& context);

inline limit_value_t get_limit(limit_t limit_id)
{
	limit_value_t limit_value;
	auto status = cuCtxGetLimit(&limit_value, limit_id);
	throw_if_error_lazy(status, "Failed obtaining CUDA context limit value");
	return limit_value;
}

inline void set_limit(limit_t limit_id, limit_value_t new_value)
{
	auto status = cuCtxSetLimit(limit_id, new_value);
	throw_if_error_lazy(status, "Failed obtaining CUDA context limit value");
}

constexpr flags_t inline make_flags(
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
	bool                                   keep_larger_local_mem_after_resize)
{
	return
		  synch_scheduling_policy // this enum value is also a valid bitmask
		| (keep_larger_local_mem_after_resize    ? CU_CTX_LMEM_RESIZE_TO_MAX : 0);
}

// consider renaming this: device_id_of
inline device::id_t get_device_id(handle_t context_handle)
{
	auto needed_push = current::detail_::push_if_not_on_top(context_handle);
	auto device_id = current::detail_::get_device_id();
	if (needed_push) {
		current::detail_::pop();
	}
	return device_id;
}


context_t from_handle(
	context::handle_t  context_handle,
	bool               take_ownership = false);

inline size_t total_memory(handle_t handle)
{
	size_t total_mem_in_bytes;
	auto status = cuMemGetInfo(nullptr, &total_mem_in_bytes);
	throw_if_error_lazy(status, "Failed determining amount of total memory for " + identify(handle));
	return total_mem_in_bytes;

}

inline size_t free_memory(handle_t handle)
{
	size_t free_mem_in_bytes;
	auto status = cuMemGetInfo(&free_mem_in_bytes, nullptr);
	throw_if_error_lazy(status, "Failed determining amount of free memory for " + identify(handle));
	return free_mem_in_bytes;
}

inline void set_cache_preference(handle_t handle, multiprocessor_cache_preference_t preference)
{
	auto status = cuCtxSetCacheConfig(static_cast<CUfunc_cache>(preference));
	throw_if_error_lazy(status,
		"Setting the multiprocessor L1/Shared Memory cache distribution preference to " +
		::std::to_string((unsigned) preference) + " for " + identify(handle));
}

inline multiprocessor_cache_preference_t cache_preference(handle_t handle)
{
	CUfunc_cache preference;
	auto status = cuCtxGetCacheConfig(&preference);
	throw_if_error_lazy(status,
		"Obtaining the multiprocessor L1/Shared Memory cache distribution preference for " + identify(handle));
	return (multiprocessor_cache_preference_t) preference;
}

inline shared_memory_bank_size_t shared_memory_bank_size(handle_t handle)
{
	CUsharedconfig bank_size;
	auto status = cuCtxGetSharedMemConfig(&bank_size);
	throw_if_error_lazy(status, "Obtaining the multiprocessor shared memory bank size for " + identify(handle));
	return static_cast<shared_memory_bank_size_t>(bank_size);
}

inline void set_shared_memory_bank_size(handle_t handle, shared_memory_bank_size_t bank_size)
{
	auto status = cuCtxSetSharedMemConfig(static_cast<CUsharedconfig>(bank_size));
	throw_if_error_lazy(status, "Setting the multiprocessor shared memory bank size for " + identify(handle));
}

inline void synchronize(context::handle_t handle)
{
	context::current::detail_::scoped_override_t set_context_for_this_scope(handle);
	context::current::detail_::synchronize(handle);
}

inline void synchronize(device::id_t device_id, context::handle_t handle)
{
	context::current::detail_::scoped_override_t set_context_for_this_scope(handle);
	context::current::detail_::synchronize(device_id, handle);
}

inline void destroy(handle_t handle)
{
	auto status = cuCtxDestroy(handle);
	throw_if_error_lazy(status, "Failed destroying " + identify(handle));
}

inline void destroy(handle_t handle, device::id_t device_index)
{
	auto status = cuCtxDestroy(handle);
	throw_if_error_lazy(status, "Failed destroying " + identify(handle, device_index));
}

inline context::flags_t get_flags(handle_t handle)
{
	current::detail_::scoped_override_t set_context_for_this_scope{handle};
	return context::current::detail_::get_flags();
}

} // namespace detail_

} // namespace context

inline void synchronize(const context_t& context);

/**
 * @brief Wrapper class for a CUDA context
 *
 * Use this class - built around a context id - to perform all
 * context-related operations the CUDA Driver (or, in fact, Runtime) API is capable of.
 *
 * @note By default this class has RAII semantics, i.e. it creates a
 * context on construction and destroys it on destruction, and isn't merely
 * an ephemeral wrapper one could apply and discard; but this second kind of
 * semantics is also supported, through the @ref context_t::holds_refcount_unit_ field.
 *
 * @note A context is a specific to a device; see, therefore, also @ref device_t .
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to properties of the context is a const-respecting operation on this class.
 */
class context_t {
public: // types
	using scoped_setter_type = context::current::detail_::scoped_override_t;
	using flags_type = context::flags_t;

	static_assert(
		::std::is_same<::std::underlying_type<CUsharedconfig>::type, ::std::underlying_type<cudaSharedMemConfig>::type>::value,
		"Unexpected difference between enumerators used for the same purpose by the CUDA runtime and the CUDA driver");

public: // inner classes

	/**
	 * @brief A class to create a faux member in a @ref device_t, in lieu of an in-class
	 * namespace (which C++ does not support); whenever you see a function
	 * `my_dev.memory::foo()`, think of it as a `my_dev::memory::foo()`.
	 */
	class global_memory_type {
	protected: // data members
		const device::id_t device_id_;
		const context::handle_t context_handle_;

	public:
		global_memory_type(device::id_t device_id, context::handle_t context_handle)
			: device_id_(device_id), context_handle_(context_handle)
		{}
		///@endcond

		device_t associated_device() const;

		context_t associated_context() const;

		/**
		 * Allocate a region of memory on the device
		 *
		 * @param size_in_bytes size in bytes of the region of memory to allocate
		 * @return a non-null (device-side) pointer to the allocated memory
		 */
		memory::region_t allocate(size_t size_in_bytes);

		/**
		 * Allocates memory on the device whose pointer is also visible on the host,
		 * and possibly on other devices as well - with the same address. This is
		 * nVIDIA's "managed memory" mechanism.
		 *
		 * @note Managed memory isn't as "strongly associated" with a single device
		 * as the result of allocate(), since it can be read or written from any
		 * device or from the host. However, the actual space is allocated on
		 * some device, so its creation is a device (device_t) object method.
		 *
		 * @note for a more complete description see the
		 * <a href="http://docs.nvidia.com/cuda/cuda-runtime-api/">CUDA Runtime API
		 * reference</a>)
		 *
		 * @param size_in_bytes Size of memory region to allocate
		 * @param initial_visibility if this equals ,to_supporters_of_concurrent_managed_access\ only the host (and the
		 * allocating device) will be able to utilize the pointer returned; if false,
		 * it will be made usable on all CUDA devices on the systems.
		 * @return the allocated pointer; never returns null (throws on failure)
		 */
		memory::region_t allocate_managed(
			size_t size_in_bytes,
			cuda::memory::managed::initial_visibility_t initial_visibility =
			cuda::memory::managed::initial_visibility_t::to_supporters_of_concurrent_managed_access);

		/**
		 * Amount of total global memory on the CUDA device's primary context.
		 */
		size_t amount_total() const
		{
			scoped_setter_type set_context_for_this_scope(context_handle_);
			return context::detail_::total_memory(context_handle_);
		}

		/**
		 * Amount of free global memory on the CUDA device's primary context.
		 */
		size_t amount_free() const
		{
			scoped_setter_type set_context_for_this_scope(context_handle_);
			return context::detail_::free_memory(context_handle_);
		}
	}; // class global_memory_type


public: // data member non-mutator getters

	/**
	 * The CUDA context ID this object is wrapping
	 */
	context::handle_t handle() const noexcept { return handle_; }

	/**
	 * The device with which this context is associated
	 */
	device::id_t device_id() const noexcept { return device_id_; }
	device_t device() const;

	/**
	 * Is this wrapper responsible for having the wrapped CUDA context destroyed on destruction?
	 */
	bool is_owning() const noexcept { return owning_;  }

	/**
	 * The amount of total global device memory available to this context, including
	 * memory already allocated.
	 */
	size_t total_memory() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::total_memory(handle_);
	}

	/**
	 *  The amount of unallocated global device memory available to this context
	 *  and not yet allocated.
	 *
	 *  @note It is not guaranteed that this entire amount can actually be succefully allocated.
	 */
	size_t free_memory() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::free_memory(handle_);
	}

public: // other non-mutator methods

	/**
	 * Determines the balance between L1 space and shared memory space set
	 * for kernels executing within this context.
	 */
	multiprocessor_cache_preference_t cache_preference() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::cache_preference(handle_);
	}

	/**
	 * @return the stack size in bytes of each GPU thread
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	size_t stack_size() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::get_limit(CU_LIMIT_STACK_SIZE);
	}

	/**
	 * @return the size of the FIFO (first-in, first-out) buffer used by the printf() function available to device kernels
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	context::limit_value_t printf_buffer_size() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::get_limit(CU_LIMIT_PRINTF_FIFO_SIZE);
	}

	/**
	 * @return the size in bytes of the heap used by the malloc() and free() device system calls.
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	context::limit_value_t memory_allocation_heap_size() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::get_limit(CU_LIMIT_MALLOC_HEAP_SIZE);
	}

	/**
	 * @return the maximum grid depth at which a thread can issue the device
	 * runtime call `cudaDeviceSynchronize()` / `cuda::device::synchronize()`
     * to wait on child grid launches to complete.
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	context::limit_value_t maximum_depth_of_child_grid_synch_calls() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::get_limit(CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH);
	}

	global_memory_type memory() const
	{
		return { device_id_, handle_ };
	}

	/**
	 * @return maximum number of outstanding device runtime launches that can be made from this context.
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	context::limit_value_t maximum_outstanding_kernel_launches() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::get_limit(CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT);
	}

#if CUDA_VERSION >= 10000
	/**
	 * @return maximum granularity of fetching from the L2 cache
	 *
	 * @note A value between 0 and 128; it is apparently a "hint" somehow.
	 *
	 * @todo Is this really a feature of the context? Not of the device?
	 */
	context::limit_value_t l2_fetch_granularity() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::get_limit(CU_LIMIT_MAX_L2_FETCH_GRANULARITY);
	}
#endif

	/**
	 * @brief Returns the shared memory bank size, as described in
	 * <a href="https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/">this Parallel-for-all blog entry</a>
	 *
	 * @return the shared memory bank size in bytes
	 */
	context::shared_memory_bank_size_t shared_memory_bank_size() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::shared_memory_bank_size(handle_);
	}

	/**
	 * Determine if this context is the system's current CUDA context.
	 */
	bool is_current() const
	{
		return context::current::detail_::is_(handle_);
	}

	/**
	 * Determine if this context is the primary context for its associated device.
	 */
	bool is_primary() const;

	/**
	 *
	 * @todo isn't this a feature of devices?
	 */
	context::stream_priority_range_t stream_priority_range() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		context::stream_priority_range_t result;
		auto status = cuCtxGetStreamPriorityRange(&result.least, &result.greatest);
		throw_if_error_lazy(status, "Obtaining the priority range for streams within " +
			context::detail_::identify(*this));
		return result;
	}

	context::limit_value_t get_limit(context::limit_t limit_id) const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::get_limit(limit_id);
	}

	version_t api_version() const
	{
		unsigned int raw_version;
		auto status = cuCtxGetApiVersion(handle_, &raw_version);
		throw_if_error_lazy(status, "Failed obtaining the API version for " + context::detail_::identify(*this));
		return version_t::from_single_number((int) raw_version);
	}

protected:
	context::flags_t flags() const
	{
		return context::detail_::get_flags(handle_);
	}


public: // methods which mutate the context, but not its wrapper
	/**
	 * Gets the synchronization policy to be used for threads synchronizing
	 * with this CUDA context.
	 *
	 * @note see @ref host_thread_synch_scheduling_policy_t
	 * for a description of the various policies.
	 */
	context::host_thread_synch_scheduling_policy_t synch_scheduling_policy() const
	{
		return context::host_thread_synch_scheduling_policy_t(flags() & CU_CTX_SCHED_MASK);
	}

	bool keeping_larger_local_mem_after_resize() const
	{
		return flags() & CU_CTX_LMEM_RESIZE_TO_MAX;
	}

	/**
	 * See @ref cuda::stream::create()
	 */
	stream_t create_stream(
		bool                will_synchronize_with_default_stream,
		stream::priority_t  priority = cuda::stream::default_priority);

	/**
	 * See @ref cuda::event::create()
	 */
	event_t create_event(
		bool uses_blocking_sync = event::sync_by_busy_waiting, // Yes, that's the runtime default
		bool records_timing     = event::do_record_timings,
		bool interprocess       = event::not_interprocess);

	template <typename ContiguousContainer,
		cuda::detail_::enable_if_t<detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool> = true>
	module_t create_module(ContiguousContainer module_data, const link::options_t& link_options) const;

	template <typename ContiguousContainer,
		cuda::detail_::enable_if_t<detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool> = true>
	module_t create_module(ContiguousContainer module_data) const;

public: // Methods which don't mutate the context, but affect the device itself


	void enable_access_to(const context_t& peer) const;

	void disable_access_to(const context_t& peer) const;

	void reset_persisting_l2_cache() const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
#if (CUDA_VERSION >= 11000)
		auto status = cuCtxResetPersistingL2Cache();
		throw_if_error_lazy(status, "Failed resetting/clearing the persisting L2 cache memory");
#endif
		throw cuda::runtime_error(
			cuda::status::insufficient_driver,
			"Resetting/clearing the persisting L2 cache memory is not supported when compiling CUDA versions lower than 11.0");
	}

public: // other methods which don't mutate this class as a reference, but do mutate the context

	/**
	 * @brief Sets the shared memory bank size, described in
 	 * <a href="https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/">this Parallel-for-all blog entry</a>
 	 *
 	 * @param bank_size the shared memory bank size to set
 	 */
	void set_shared_memory_bank_size(context::shared_memory_bank_size_t bank_size) const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		context::detail_::set_shared_memory_bank_size(handle_, bank_size);
	}

	/**
	 * Controls the balance between L1 space and shared memory space for
	 * kernels executing within this context.
	 *
	 * @param preference the preferred balance between L1 and shared memory
	 */
	void set_cache_preference(multiprocessor_cache_preference_t preference) const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		context::detail_::set_cache_preference(handle_, preference);
	}

	void set_limit(context::limit_t limit_id, context::limit_value_t new_value) const
	{
		scoped_setter_type set_context_for_this_scope(handle_);
		return context::detail_::set_limit(limit_id, new_value);
	}

	void stack_size(context::limit_value_t new_value) const
	{
		return set_limit(CU_LIMIT_STACK_SIZE, new_value);
	}

	void printf_buffer_size(context::limit_value_t new_value) const
	{
		return set_limit(CU_LIMIT_PRINTF_FIFO_SIZE, new_value);
	}

	void memory_allocation_heap_size(context::limit_value_t new_value) const
	{
		return set_limit(CU_LIMIT_MALLOC_HEAP_SIZE, new_value);
	}

	void set_maximum_depth_of_child_grid_synch_calls(context::limit_value_t new_value) const
	{
		return set_limit(CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH, new_value);
	}

	void set_maximum_outstanding_kernel_launches(context::limit_value_t new_value) const
	{
		return set_limit(CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT, new_value);
	}

	/**
	 * Have the calling thread wait - either busy-waiting or blocking - and
	 * return only after all pending actions within this context have concluded.
	 */
	void synchronize() const
	{
		cuda::synchronize(*this);
	}

protected: // constructors

	context_t(
		device::id_t       device_id,
		context::handle_t  context_id,
		bool               take_ownership) noexcept
		: device_id_(device_id), handle_(context_id), owning_(take_ownership)
	{ }

public: // friendship

	friend context_t context::wrap(
		device::id_t       device_id,
		context::handle_t  context_id,
		bool               take_ownership) noexcept;

public: // constructors and destructor

	context_t(const context_t& other) :
		context_t(other.device_id_, other.handle_, false)
	{ };

	context_t(context_t&& other) noexcept:
		context_t(other.device_id_, other.handle_, other.owning_)
	{
		other.owning_ = false;
	};

	~context_t()
	{
		if (owning_) {
			cuCtxDestroy(handle_);
			// Note: "Swallowing" any potential error to avoid ::std::terminate(); also,
			// because the context cannot possibly exist after this call.
		}
	}

public: // operators

	context_t& operator=(const context_t&) = delete;
	context_t& operator=(context_t&& other) noexcept
	{
		::std::swap(device_id_, other.device_id_);
		::std::swap(handle_, other.handle_);
		::std::swap(owning_, other.owning_);
		return *this;
	}

protected: // data members
	device::id_t       device_id_;
	context::handle_t  handle_;
	bool               owning_;
		// this field is mutable only for enabling move construction; other
		// than in that case it must not be altered

	// TODO: Should we hold a field indicating whether this context is
	// primary or not?
};

inline bool operator==(const context_t& lhs, const context_t& rhs)
{
	return lhs.handle() == rhs.handle();
}

inline bool operator!=(const context_t& lhs, const context_t& rhs)
{
	return lhs.handle() != rhs.handle();
}

namespace context {

inline context_t wrap(
	device::id_t       device_id,
	handle_t           context_id,
	bool               take_ownership) noexcept
{
	return { device_id, context_id, take_ownership };
}

namespace detail_ {

inline context_t from_handle(
	context::handle_t  context_handle,
	bool               take_ownership)
{
	device::id_t device_id = get_device_id(context_handle);
	return wrap(device_id, context_handle, take_ownership);
}

inline handle_t create_and_push(
	device::id_t                           device_id,
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy = automatic,
	bool                                   keep_larger_local_mem_after_resize = false)
{
	auto flags = context::detail_::make_flags(
		synch_scheduling_policy,
		keep_larger_local_mem_after_resize);
	handle_t handle;
	auto status = cuCtxCreate(&handle, flags, device_id);
	throw_if_error_lazy(status, "failed creating a CUDA context associated with "
		+ device::detail_::identify(device_id));
	return handle;
}

} // namespace detail_

/**
 * @brief creates a new context on a given device
 *
 * @param device              The device on which to create the new stream
 * @param synch_scheduling_policy
 * @param keep_larger_local_mem_after_resize
 * @return
 * @note Until CUDA 11, there used to also be a flag for enabling/disabling
 * the ability of mapping pinned host memory to device addresses. However, it was
 * being ignored since CUDA 3.2 already, with the minimum CUDA version supported
 * by these wrappers being later than that, so - no sense in keeping it.
 */
context_t create(
	device_t                               device,
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy = heuristic,
	bool                                   keep_larger_local_mem_after_resize = false);

context_t create_and_push(
	device_t                               device,
	host_thread_synch_scheduling_policy_t  synch_scheduling_policy = heuristic,
	bool                                   keep_larger_local_mem_after_resize = false);

namespace current {

/**
 * Obtain the current CUDA context, if one exists.
 *
 * @throws ::std::runtime_error in case there is no current context
 */
inline context_t get()
{
	auto handle = detail_::get_handle();
	if (handle == context::detail_::none) {
		throw ::std::runtime_error("Attempt to obtain the current CUDA context when no context is current.");
	}
	return context::detail_::from_handle(handle);
}

inline void set(const context_t& context)
{
	return detail_::set(context.handle());
}

inline bool push_if_not_on_top(const context_t& context)
{
	return context::current::detail_::push_if_not_on_top(context.handle());
}

inline void push(const context_t& context)
{
	return context::current::detail_::push(context.handle());
}

inline context_t pop()
{
	static constexpr const bool do_not_take_ownership { false };
	// Unfortunately, since we don't store the device IDs of contexts
	// on the stack, this incurs an extra API call beyond just the popping...
	auto handle = context::current::detail_::pop();
	auto device_id = context::detail_::get_device_id(handle);
	return context::wrap(device_id, handle, do_not_take_ownership);
}

namespace detail_ {

/**
 * If now current context exists, push the current device's primary context onto the stack
 */
handle_t push_default_if_missing();

/**
 * Ensures that a current context exists by pushing the current device's primary context
 * if necessary, and returns the current context
 *
 * @throws ::std::runtime_error in case there is no current context
 */
inline context_t get_with_fallback_push()
{
	auto handle = push_default_if_missing();
	return context::detail_::from_handle(handle);
}


} // namespace detail_

} // namespace current

bool is_primary(const context_t& context);

namespace detail_ {

inline ::std::string identify(const context_t& context)
{
	return identify(context.handle(), context.device_id());
}

} // namespace detail_

} // namespace context

inline void synchronize(const context_t& context)
{
	context::detail_::synchronize(context.device_id(), context.handle());
}

} // namespace cuda

#endif // CUDA_API_WRAPPERS_CONTEXT_HPP_
