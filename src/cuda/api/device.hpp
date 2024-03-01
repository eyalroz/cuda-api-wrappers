/**
 * @file
 *
 * @brief A proxy class for CUDA devices, providing access to
 * all Runtime API calls involving their use and management; and
 * some device-related standalone functions.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DEVICE_HPP_
#define CUDA_API_WRAPPERS_DEVICE_HPP_

#include "types.hpp"
#include "current_device.hpp"
#include "device_properties.hpp"
#include "memory.hpp"
#include "pci_id.hpp"
#include "primary_context.hpp"
#include "error.hpp"

#include <cuda_runtime_api.h>

#include <string>
#include <cstring>
#include <type_traits>

namespace cuda {

///@cond
class event_t;
class stream_t;
class device_t;
namespace memory {
class pool_t;
} // namespace memory
///@endcond

/**
 * @brief Waits for all previously-scheduled tasks on all streams (= queues)
 * on a specified device to conclude.
 *
 * Depending on the host_thread_sync_scheduling_policy_t set for this
 * device, the thread calling this method will either yield, spin or block
 * until all tasks scheduled previously scheduled on this device have been
 * concluded.
 */
void synchronize(const device_t& device);

namespace device {

///@cond
class primary_context_t;
///@cendond

using limit_t = context::limit_t;
using limit_value_t = context::limit_value_t;
using shared_memory_bank_size_t = context::shared_memory_bank_size_t;

namespace detail_ {

/**
 * Construct a @ref device_t wrapper class instance for a given device ID
 *
 * @param id Numeric id (mostly an ordinal) of the device to wrap
 * @param primary_context_handle if this is not the "none" value, the wrapper object
 * will be owning a reference unit to the device's primary context, which it
 * will release on destruction. Use this to allow runtime-API-style code, which does
 * not explicitly construct contexts, to be able to function with a primary context
 * being made and kept active.
 */
device_t wrap(
	id_t id,
	primary_context::handle_t primary_context_handle = context::detail_::none,
	bool holds_primary_context_refcount_unit = false) NOEXCEPT_IF_NDEBUG;

} // namespace detail

/**
 * Returns a wrapper for the CUDA device with a given id
 *
 * @param id the ID of the device for which to obtain the wrapper
 *
 * @note Effectively, this is an alias of the @ref get function.
 */
device_t wrap(id_t id) NOEXCEPT_IF_NDEBUG;

using stream_priority_range_t = context::stream_priority_range_t;

namespace detail_ {

inline ::std::string get_name(id_t id)
{
	using size_type = int; // Yes, an int, that's what cuDeviceName takes
	static constexpr const size_type initial_size_reservation { 100 };
	static constexpr const size_type larger_size { 1000 }; // Just in case
	char stack_buffer[initial_size_reservation];
	auto buffer_size = static_cast<size_type>((sizeof(stack_buffer) / sizeof(char)));
	auto try_getting_name = [&](char* buffer, size_type buffer_size_) -> size_type {
		auto status = cuDeviceGetName(buffer, buffer_size-1, id);
		throw_if_error_lazy(status, "Failed obtaining the CUDA device name of device " + ::std::to_string(id));
		buffer[buffer_size_-1] = '\0';
		return static_cast<size_type>(::std::strlen(buffer));
	};
	auto prospective_name_length = try_getting_name(stack_buffer, initial_size_reservation);
	if (prospective_name_length < buffer_size - 1) {
		return { stack_buffer, static_cast<::std::string::size_type>(prospective_name_length) };
	}
	::std::string result;
	result.reserve(prospective_name_length);
	prospective_name_length = try_getting_name(&result[0], buffer_size);
		// We can't use result.data() since it's const until C++20ץץץ
	if (prospective_name_length >= buffer_size - 1) {
		throw ::std::runtime_error("CUDA device name longer than expected maximum size " + ::std::to_string(larger_size));
	}
	return result;
}

} // namespace detail

} // namespace device

/**
 * @brief Wrapper class for a CUDA device
 *
 * Use this class - built around a device ID, or for the current device - to
 * perform almost, if not all, device-related operations, as opposed to passing
 * the device ID around all that time.
 *
 * @note this is one of the three main classes in the Runtime API wrapper library,
 * together with @ref cuda::stream_t and @ref cuda::event_t
 *
 * @note obtaining device LUID's is not supported (those are too graphics-specific)
 * @note This class is a "reference type", not a "value type". Therefore, making changes
 * to properties of the device is a const-respecting operation on this class.
 */
class device_t {
public: // types
	using properties_t = device::properties_t;
	using attribute_value_t = device::attribute_value_t;
	using flags_type = device::flags_t;

	/**
	 *
	 * @note The memory proxy regards the device's primary context.
	 *
	 * @todo Consider a scoped/unscoped dichotomy.
	 */
	context_t::global_memory_type memory() const {
		return primary_context().memory();
	}

protected: // types

public:
	/**
	 * @brief Determine whether this device can access the global memory
	 * of another CUDA device.
	 *
	 * @param peer the device which is to be accessed
	 * @return true iff acesss is possible
	 */
	bool can_access(const device_t& peer) const
	{
		CAW_SET_SCOPE_CONTEXT(primary_context_handle());
		int result;
		auto status = cuDeviceCanAccessPeer(&result, id(), peer.id());
		throw_if_error_lazy(status, "Failed determining whether "
			+ device::detail_::identify(id_) + " can access "
			+ device::detail_::identify(peer.id_));
		return (result == 1);
	}

	/**
	 * @brief Enable access by this device to the global memory of another device
	 *
	 * @param peer the device to which to enable access
	 */
	void enable_access_to(const device_t& peer) const
	{
		primary_context().enable_access_to(peer.primary_context());
	}

	/**
	 * @brief Disable access by this device to the global memory of another device
	 *
	 * @param peer the device to which to disable access
	 */
	void disable_access_to(const device_t& peer) const
	{
		primary_context().disable_access_to(peer.primary_context());
	}


#if CUDA_VERSION >= 9020
	uuid_t uuid () const {
		uuid_t result;
		auto status = cuDeviceGetUuid(&result, id_);
		throw_if_error_lazy(status, "Failed obtaining UUID for " + device::detail_::identify(id_));
		return result;
	}
#endif // CUDA_VERSION >= 9020

protected:
	void cache_and_ensure_primary_context_activation() const {
		if (primary_context_handle_ == context::detail_::none) {
			primary_context_handle_ = device::primary_context::detail_::obtain_and_increase_refcount(id_);
			holds_pc_refcount_unit_ = true;
		}
	}

	context::handle_t primary_context_handle() const
	{
		cache_and_ensure_primary_context_activation();
		return primary_context_handle_;
	}


public:
	/**
	 * Produce a proxy for the device's primary context - the one used by runtime API calls.
	 *
	 * @param scoped When true, the primary proxy object returned will not perform its
	 * own reference accounting, and will assume the primary context is active while
	 * this device object exists. When false, the returned primary context proxy object
	 * _will_ take care of its own reference count unit, and can outlive this object.
	 */
	device::primary_context_t primary_context(bool hold_pc_refcount_unit = false) const;

	void set_flags(flags_type new_flags) const
	{
		auto status = cuDevicePrimaryCtxSetFlags(id(), new_flags);
		throw_if_error_lazy(status, "Failed setting (primary context) flags for device " + device::detail_::identify(id_));
	}

#if CUDA_VERSION >= 11020
	memory::pool_t default_memory_pool() const;
#endif
public:

	/**
	 * Obtains the (mostly) non-numeric properties for this device.
	 *
	 * @todo get rid of this in favor of individual properties only.
	 */
	properties_t properties() const
	{
		properties_t properties;
		auto status = cudaGetDeviceProperties(&properties, id());
		throw_if_error_lazy(status, "Failed obtaining device properties for " + device::detail_::identify(id_));
		return properties;
	}

	static device_t choose_best_match(const properties_t& properties) {
		device::id_t id;
		auto status = cudaChooseDevice(&id, &properties);
		throw_if_error_lazy(status, "Failed choosing a best matching device by a a property set.");
		return device::wrap(id);
	}

	/**
	 * Obtains this device's human-readable name, e.g. "GeForce GTX 650 Ti BOOST".
	 */
	::std::string name() const
	{
		// If I were lazy, I would just write:
		// return properties().name;
		// and let you wait for all of that to get populated. But not me!
		return cuda::device::detail_::get_name(id_);
	}

	/**
	 * Obtain a numeric-value attribute of the device
	 *
	 * @note See @ref device::attribute_t for explanation about attributes,
	 * properties and flags.
	 */
	attribute_value_t get_attribute(device::attribute_t attribute) const
	{
		attribute_value_t attribute_value;
		auto status = cuDeviceGetAttribute(&attribute_value, attribute, id_);
		throw_if_error_lazy(status, "Failed obtaining device properties for " + device::detail_::identify(id_));
		return attribute_value;
	}

	grid::block_dimension_t maximum_threads_per_block() const
	{
		return get_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
	}

	/**
	 * Obtains this device's location on the PCI express bus in terms of
	 * domain, bus and device id, e.g. (0, 1, 0)
	 */
	device::pci_location_t pci_id() const
	{
		auto pci_domain_id = get_attribute(CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID);
		auto pci_bus_id    = get_attribute(CU_DEVICE_ATTRIBUTE_PCI_BUS_ID);
		auto pci_device_id = get_attribute(CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID);
		return {pci_domain_id, pci_bus_id, pci_device_id};
	}

	device::multiprocessor_count_t multiprocessor_count() const
	{
		return get_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
	}

#if CUDA_VERSION >= 10020
	/**
	 * True if the device supports the facilities under namespace @ref memory::virtual_
	 * including the separation of memory allocation from address range mapping, and
	 * the possibility of changing mapping after allocation.
	 */
	bool supports_virtual_memory_management() const
	{
#if CUDA_VERSION >= 11030
		return get_attribute(CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED);
#else
		return get_attribute(CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED);
#endif // CUDA_VERSION >= 11030
	}
#endif // CUDA_VERSION >= 10020

	/**
	 * Obtains the device's hardware architecture generation numeric
	 * designator see @ref cuda::device::compute_architecture_t
	 */
	device::compute_architecture_t architecture() const
	{
		unsigned major = get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
		return { major };
	}

	/**
	 * Obtains the device's compute capability; see @ref cuda::device::compute_capability_t
	 */
	device::compute_capability_t compute_capability() const
	{
		auto major = architecture();
		unsigned minor = get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
		return {major, minor};
	}

	/**
	 * Determine whether this device can coherently access managed memory
	 * concurrently with the CPU
	 */
	bool supports_concurrent_managed_access() const
	{
		return (get_attribute(CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS) != 0);
	}

	/**
	 * True if this device supports executing kernels in which blocks can
	 * directly cooperate beyond the use of global-memory atomics.
	 */
	bool supports_block_cooperation() const
	{
		return get_attribute(CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH);
	}

#if CUDA_VERSION >= 12000
	/**
	 * True if this device supports "clusters" of grid blocks,
	 * which can pool their shared memory together
	 */
	bool supports_block_clustering() const
	{
		return get_attribute(CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH);
	}
#endif

#if CUDA_VERSION >= 11020
	/**
	 * True if this device supports integrated memory pool and
	 * stream ordered memory allocator.
	 */
	bool supports_memory_pools() const
	{
		return get_attribute(CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED);
	}

#endif // CUDA_VERSION >= 11020

	/**
	 * Obtains the upper limit on the amount of a certain kind of
	 * resource this device offers.
	 *
	 * @param resource which resource's limit to obtain
	 */
	device::limit_value_t get_limit(device::limit_t limit) const
	{
		return primary_context().get_limit(limit);
	}

	/**
	 * Set the upper limit of one of the named numeric resources
	 * on this device
	 */
	void set_limit(device::limit_t limit, device::limit_value_t new_value) const
	{
		primary_context().set_limit(limit, new_value);
	}

	/**
	 * @brief Waits for all previously-scheduled tasks on all streams (= queues)
	 * on this device to conclude
	 *
	 * Depending on the host_thread_sync_scheduling_policy_t set for this
	 * device, the thread calling this method will either yield, spin or block
	 * until all tasks scheduled previously scheduled on this device have been
	 * concluded.
	 */
	const device_t& synchronize() const
	{
		cuda::synchronize(*this);
		return *this;
	}

	device_t& synchronize()
	{
		cuda::synchronize(*this);
		return *this;
	}

	const device_t& make_current() const
	{
		device::current::set(*this);
		return *this;
	}

	device_t& make_current()
	{
		device::current::set(*this);
		return *this;
	}

	/**
	 * Invalidates all memory allocations and resets all state regarding this
	 * CUDA device on the current operating system process.
	 *
	 * @todo Determine whether this actually performs a hardware reset or not
	 */
	void reset() const
	{
		// Notes:
		//
		// 1. We _cannot_ use cuDevicePrimaryCtxReset() - because that one only affects
		// the device's primary context, while cudaDeviceReset() destroys _all_ contexts for
		// the device.
		// 2. We don't need the primary context to be active here, so not using the usual
		//    primary_context_handle() getter mechanism.

		auto pc_handle = (primary_context_handle_ == context::detail_::none) ?
			device::primary_context::detail_::obtain_and_increase_refcount(id_) :
			primary_context_handle_;
		CAW_SET_SCOPE_CONTEXT(pc_handle);
		auto status = cudaDeviceReset();
		throw_if_error_lazy(status, "Resetting " + device::detail_::identify(id_));
	}

	/**
	 * Controls the balance between L1 space and shared memory space for
	 * kernels executing on this device.
	 *
	 * @param preference the preferred balance between L1 and shared memory
	 */
	void set_cache_preference(multiprocessor_cache_preference_t preference) const
	{
		primary_context().set_cache_preference(preference);
	}

	/**
	 * Determines the balance between L1 space and shared memory space set
	 * for kernels executing on this device.
	 */
	multiprocessor_cache_preference_t cache_preference() const
	{
		return primary_context().cache_preference();
	}

	/**
	 * @brief Sets the shared memory bank size, described in
	 * <a href="https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/">this Parallel-for-all blog entry</a>
	 *
	 * @param new_bank_size the shared memory bank size to set, in bytes
	 */
	void set_shared_memory_bank_size(device::shared_memory_bank_size_t new_bank_size) const
	{
		primary_context().set_shared_memory_bank_size(new_bank_size);
	}

	/**
	 * @brief Returns the shared memory bank size, as described in
	 * <a href="https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/">this Parallel-for-all blog entry</a>
	 *
	 * @return the shared memory bank size in bytes
	 */
	device::shared_memory_bank_size_t shared_memory_bank_size() const
	{
		return primary_context().shared_memory_bank_size();
	}

	// For some reason, there is no cudaFuncGetCacheConfig. Weird.
	//
	// template <typename KernelFunction>
	// inline multiprocessor_cache_preference_t kernel_cache_preference(
	// 	const KernelFunction* kernel, multiprocessor_cache_preference_t preference);

	/**
	 * Return the proxied device's ID
	 *
	 */
	device::id_t id() const noexcept
	{
		return id_;
	}

	/**
	 * Obtain a wrapper for the (always-existing) default stream within
	 * the device' primary context.
	 *
	 * @param hold_primary_context_refcount_unit when true, the returned stream
	 *     wrapper will keep the device' primary context in existence during
	 *     its lifetime.
	 */
	stream_t default_stream(bool hold_primary_context_refcount_unit = false) const;

	/// See @ref cuda::stream::create()
	stream_t create_stream(
		bool                will_synchronize_with_default_stream,
		stream::priority_t  priority = cuda::stream::default_priority) const;

	/// See @ref cuda::event::create()
	event_t create_event(
		bool uses_blocking_sync = event::sync_by_busy_waiting, // Yes, that's the runtime default
		bool records_timing     = event::do_record_timings,
		bool interprocess       = event::not_interprocess);

	/// See @ref cuda::context::create()
	context_t create_context(
		context::host_thread_sync_scheduling_policy_t   sync_scheduling_policy = context::heuristic,
		bool                                            keep_larger_local_mem_after_resize = false) const;

#if CUDA_VERSION >= 11020

	/// See @ref cuda::memory::pool::create()
	template <memory::pool::shared_handle_kind_t Kind = memory::pool::shared_handle_kind_t::no_export>
	memory::pool_t create_memory_pool() const;

#endif

	/**
	 * Launch a kernel on the default stream of the device' primary context
	 *
	 * @tparam Kernel May be either a plain function type (for a `__global__` function
	 *     accessible to the translation unit, or (a reference to) any subclass of
	 * `   `cuda::kernel_t`.
	 * @param kernel_function
	 *     the kernel to launch; may be either a (`__global__`) function pointer,
	 *     or a kernel proxy class.
	 * @param launch_configuration
	 *     the configuration with which to launch the kernel;
	 * @param arguments
	 *     the arguments with which to launch @p kernel (but note that references
	 *     are not maintained).
	 */
	template<typename Kernel, typename ... KernelParameters>
	void launch(
		Kernel                  kernel,
		launch_configuration_t  launch_configuration,
		KernelParameters...     arguments) const;

	/**
	 * Determines the range of possible priorities for streams on this device.
	 *
	 * @return a priority range, whose semantics are a bit confusing; see @ref priority_range_t . If
	 * the device does not support stream priorities, a 'trivial' range  of priority values will be
	 * returned.
	 */
	device::stream_priority_range_t stream_priority_range() const
	{
		return primary_context().stream_priority_range();
	}

public:
	context::flags_t flags() const
	{
		return device::primary_context::detail_::flags(id_);
	}

	context::host_thread_sync_scheduling_policy_t sync_scheduling_policy() const
	{
		return context::host_thread_sync_scheduling_policy_t(flags() & CU_CTX_SCHED_MASK);
	}

	void set_sync_scheduling_policy(context::host_thread_sync_scheduling_policy_t new_policy)
	{
		auto other_flags = flags() & ~CU_CTX_SCHED_MASK;
		set_flags(other_flags | static_cast<flags_type>(new_policy));
	}

	bool keeping_larger_local_mem_after_resize() const
	{
		return flags() & CU_CTX_LMEM_RESIZE_TO_MAX;
	}

	void keep_larger_local_mem_after_resize(bool keep = true)
	{
		auto other_flags = flags() & ~CU_CTX_LMEM_RESIZE_TO_MAX;
		flags_type new_flags = other_flags | (keep ? CU_CTX_LMEM_RESIZE_TO_MAX : 0);
		set_flags(new_flags);
	}

	void dont_keep_larger_local_mem_after_resize()
	{
		keep_larger_local_mem_after_resize(false);
	}

protected:
	void maybe_decrease_primary_context_refcount() const
	{
		if (holds_pc_refcount_unit_) {
			device::primary_context::detail_::decrease_refcount(id_);
		}
	}

public: 	// constructors and destructor

	friend void swap(device_t& lhs, device_t& rhs) noexcept
	{
		::std::swap(lhs.id_, rhs.id_);
		::std::swap(lhs.primary_context_handle_, rhs.primary_context_handle_);
		::std::swap(lhs.holds_pc_refcount_unit_, rhs.holds_pc_refcount_unit_);
	}

	~device_t() NOEXCEPT_IF_NDEBUG
	{
#ifndef NDEBUG
		maybe_decrease_primary_context_refcount();
#else
		if (holds_pc_refcount_unit_)  {
			device::primary_context::detail_::decrease_refcount_nothrow(id_);
				// Swallow any error to avoid termination on throwing from a dtor
		}
#endif
	}

	device_t(device_t&& other) noexcept : id_(other.id_)
	{
		swap(*this, other);
	}

	device_t(const device_t& other) noexcept : id_(other.id_) { }
		// Device proxies are not owning - as devices aren't allocated nor de-allocated.
		// Also, the proxies don't hold any state (except for one bit regarding whether
		// or not the device proxy has increased the primary context refcount); it's
		// the devices _themselves_ which have state; so there's no problem copying
		// the proxies around. This is unlike events and streams, which get created
		// and destroyed.

	device_t& operator=(const device_t& other) noexcept
	{
		maybe_decrease_primary_context_refcount();
		id_ = other.id_;
		primary_context_handle_ = other.primary_context_handle_;
		holds_pc_refcount_unit_ = false;
		return *this;
	}

	device_t& operator=(device_t&& other) noexcept
	{
		swap(*this, other);
		return *this;
	}

protected: // constructors

	/**
	 * @note Only @ref device::wrap() and @ref device::get() should be
	 * calling this one.
	 */
	explicit device_t(
		device::id_t device_id,
		device::primary_context::handle_t primary_context_handle = context::detail_::none,
		bool hold_primary_context_refcount_unit = false) NOEXCEPT_IF_NDEBUG
	:
		id_(device_id),
		primary_context_handle_(primary_context_handle),
		holds_pc_refcount_unit_(hold_primary_context_refcount_unit)
	{
#ifndef NDEBUG
		if (id_ < 0) {
			throw ::std::invalid_argument("Attempt to construct a CUDA device object for a negative device ID of " + ::std::to_string(id_));
		}
#endif
	}

public: // friends
	friend device_t device::detail_::wrap(
		device::id_t,
		device::primary_context::handle_t handle,
		bool hold_primary_context_refcount_unit) NOEXCEPT_IF_NDEBUG;

protected: // data members
	device::id_t id_; /// Numeric ID of the proxied device.
	mutable device::primary_context::handle_t primary_context_handle_ { context::detail_::none };
		/// Most work involving a device actually occurs using its primary context; we cache the handle
		/// to this context here - albeit not necessary on construction
	mutable bool holds_pc_refcount_unit_ {false };
		/// Since we're allowed to cache the primary context handle on constant device_t's, we
		/// also need to keep track of whether this object "owns" this reference.
};

///@cond
inline bool operator==(const device_t& lhs, const device_t& rhs)
{
	return lhs.id() == rhs.id();
}

inline bool operator!=(const device_t& lhs, const device_t& rhs)
{
	return lhs.id() != rhs.id();
}
///@endcond

namespace device {

namespace detail_ {

inline device_t wrap(
	id_t id,
	primary_context::handle_t primary_context_handle,
	bool hold_primary_context_refcount_unit) NOEXCEPT_IF_NDEBUG
{
	return device_t{ id, primary_context_handle, hold_primary_context_refcount_unit };
}

} // namespace detail_

inline device_t wrap(id_t id) NOEXCEPT_IF_NDEBUG
{
	return detail_::wrap(id);
}

/**
 * Returns a proxy for the CUDA device with a given id
 *
 * @param device_id the ID for which to obtain the device proxy
 * @note direct constructor access is blocked so that you don't get the
 * idea you're actually creating devices
 */
inline device_t get(id_t id)
{
#ifndef NDEBUG
	if (id < 0) {
		throw ::std::invalid_argument("Attempt to obtain a CUDA device with a negative device ID " + ::std::to_string(id));
	}
#endif
	ensure_driver_is_initialized(); // The device_t class mostly assumes the driver has been initialized
	return wrap(id);
}

/**
 * A named constructor idiom for a "dummy" CUDA device representing the CPU.
 *
 * @note Only use this idiom when comparing the results of functions returning
 * locations, which can be either a GPU device or the CPU; any other use will likely
 * result in a runtime error being thrown.
 */
inline device_t cpu() { return get(CU_DEVICE_CPU); }

namespace current {

/**
 * Obtains (a proxy for) the device which the CUDA runtime API considers to be current.
 */
inline device_t get()
{
	ensure_driver_is_initialized();
	auto id = detail_::get_id();
	auto pc_handle = primary_context::detail_::obtain_and_increase_refcount(id);
	return device::detail_::wrap(id, pc_handle);
}

inline void set(const device_t& device)
{
	auto pc = device.primary_context();
	context::current::detail_::set(pc.handle());
}

} // namespace current

/**
 * @brief Obtain a proxy to a device using its PCI bus location
 *
 * @param pci_id The domain-bus-device triplet locating the GPU on the PCI bus
 * @return a device_t proxy object for the device at the specified location
 */
inline device_t get(pci_location_t pci_id)
{
	auto resolved_id = device::detail_::resolve_id(pci_id);
	return get(resolved_id);
}

/**
 * @brief Obtain a proxy to a device using a string with its PCI bus location
 *
 * @param pci_id_str A string listing of the GPU's location on the PCI bus
 * @return a device_t proxy object for the device at the specified location
 *
 * @note I'm not very happy about the assumption that get-device-by-string
 * means get-device-by-pci-location; that's not such a great assumption to
 * make IMHO. But - it's convenient for now and there's no immediate risk
 * from some other obvious source of CUDA-device-identifying strings.
 */
inline device_t get(const ::std::string& pci_id_str)
{
	auto parsed_pci_id = pci_location_t::parse(pci_id_str);
	return get(parsed_pci_id);
}

} // namespace device

} // namespace cuda

#endif // CUDA_API_WRAPPERS_DEVICE_HPP_
