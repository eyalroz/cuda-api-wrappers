/**
 * @file device.hpp
 *
 * @brief A proxy class for CUDA devices, providing access to
 * all Runtime API calls involving their use and management; and
 * some device-related standalone functions.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DEVICE_HPP_
#define CUDA_API_WRAPPERS_DEVICE_HPP_

#include <cuda/api/current_device.hpp>
#include <cuda/api/device_properties.hpp>
#include <cuda/api/memory.hpp>
#include <cuda/api/pci_id.hpp>
#include <cuda/common/types.hpp>

#include <cuda_runtime_api.h>

#include <string>
#include <type_traits>

namespace cuda {

class event_t;
class stream_t;
class device_t;

namespace device {

/**
 * Returns a proxy for the CUDA device with a given id
 *
 * @param device_id the ID for which to obtain the device proxy
 * @note direct constructor access is blocked so that you don't get the
 * idea you're actually creating devices
 */
device_t get(id_t device_id) noexcept;

namespace current {

/**
 * Returns the current device in a wrapper which assumes it is indeed
 * current, i.e. which will not set the current device before performing any
 * other actions.
 */
device_t get();

} // namespace current

namespace peer_to_peer {

/**
 * @brief The value of type for all CUDA device "attributes"; see also @ref cuda::device::attribute_t.
 */
using attribute_value_t = int;

/**
 * @brief An identifier of a integral-numeric-value attribute of a CUDA device.
 *
 * @note Somewhat annoyingly, CUDA devices have attributes, properties and flags.
 * Attributes have integral number values; properties have all sorts of values,
 * including arrays and limited-length strings (see
 * @ref cuda::device::properties_t), and flags are either binary or
 * small-finite-domain type fitting into an overall flags value (see
 * @ref cuda::device_t::flags_t). Flags and properties are obtained all at once,
 * attributes are more one-at-a-time.
 */
using attribute_t = cudaDeviceP2PAttr;

/**
 * Aliases for all CUDA device attributes
 */
enum : ::std::underlying_type<attribute_t>::type {
		link_performance_rank = cudaDevP2PAttrPerformanceRank, /**< A relative value indicating the performance of the link between two devices */                       //!< link_performance_rank
		access_support = cudaDevP2PAttrAccessSupported, /**< 1 if access is supported, 0 otherwise */                                                                    //!< access_support
		native_atomics_support = cudaDevP2PAttrNativeAtomicSupported /**< 1 if the first device can perform native atomic operations on the second device, 0 otherwise *///!< native_atomics_support
};

/**
 * @brief Get one of the numeric attributes for a(n ordered) pair of devices,
 * relating to their interaction
 *
 * @note This is the device-pair equivalent of @ref device_t::get_attribute()
 *
 * @param attribute identifier of the attribute of interest
 * @param source source device
 * @param destination destination device
 * @return the numeric attribute value
 */
inline attribute_value_t get_attribute(attribute_t attribute, id_t source, id_t destination)
{
	attribute_value_t value;
	auto status = cudaDeviceGetP2PAttribute(&value, attribute, source, destination);
	throw_if_error(status,
		"Failed obtaining peer-to-peer device attribute for device pair (" + ::std::to_string(source) + ", "
			+ ::std::to_string(destination) + ')');
	return value;
}

attribute_value_t get_attribute(
	attribute_t  attribute,
	device_t     source,
	device_t     destination);

} // namespace peer_to_peer

/**
 * A range of priorities supported by a CUDA device; ranges from the
 * higher numeric value to the lower.
 */
struct stream_priority_range_t {
	stream::priority_t least; /// Higher numeric value, lower priority
	stream::priority_t greatest; /// Lower numeric value, higher priority

	/**
	 * When true, stream prioritization is not supported, i.e. all streams have
	 * "the same" priority - the default one.
	 */
	constexpr bool is_trivial() const {
		return least == stream::default_priority and greatest == stream::default_priority;
	}
};

} // namespace device


/**
 * @brief Suspends execution until all previously-scheduled tasks on
 * the specified device (all contexts, all streams) have concluded.
 *
 * Depending on the host_thread_synch_scheduling_policy_t set for this
 * device, the thread calling this method will either yield, spin or block
 * until this completion.
 */
inline void synchronize(device_t& device);

/**
 * @brief Proxy class for a CUDA device
 *
 * Use this class - built around a device ID, or for the current device - to
 * perform almost, if not all, device-related operations, as opposed to passing
 * the device ID around all that time.
 *
 * @note this is one of the three main classes in the Runtime API wrapper library,
 * together with @ref cuda::stream_t and @ref cuda::event_t
 */
class device_t {
public: // types
	using properties_t = device::properties_t;
	using attribute_value_t = device::attribute_value_t;
	using resource_limit_t = size_t;
	using shared_memory_bank_size_t = cudaSharedMemConfig;

	using resource_id_t = cudaLimit;

protected: // types
	using scoped_setter_t = device::current::detail_::scoped_override_t;
	using flags_t = unsigned;

	///@cond

	/**
	 * Used by wrapper classes to better "hide" their protected constructors, which take plain numbers,
	 * from user code - so that users don't get "constructor is protected" error if they mistake a number
	 * for a wrapper object when passing arguments.
	 */
	struct wrapping_construction {};

	///@endcond

public:	// types

	/**
	 * @brief A class to create a faux member in a @ref device_t, in lieu of an in-class
	 * namespace (which C++ does not support); whenever you see a function
	 * `my_dev.memory::foo()`, think of it as a `my_dev::memory::foo()`.
	 */
	class global_memory_t {
	protected:
		const device::id_t device_id_;

		using deleter = memory::device::detail_::deleter;
		using allocator = memory::device::detail_::allocator;

	public:
		///@cond
		global_memory_t(device::id_t id) : device_id_(id) { }
		///@endcond

		cuda::device::id_t device_id() const { return device_id_; }

		/**
		 * Allocate a region of memory on the device
		 *
		 * @param size_in_bytes size in bytes of the region of memory to allocate
		 * @return a non-null (device-side) pointer to the allocated memory
		 */
		memory::region_t allocate(size_t size_in_bytes)
		{
			scoped_setter_t set_device_for_this_scope(device_id_);
			return memory::device::detail_::allocate(size_in_bytes);
		}

		// Perhaps drop this? it should really go into a managed namespace
		using initial_visibility_t  = cuda::memory::managed::initial_visibility_t;

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
			initial_visibility_t initial_visibility =
				initial_visibility_t::to_supporters_of_concurrent_managed_access)
		{
			scoped_setter_t set_device_for_this_scope(device_id_);
			return cuda::memory::managed::detail_::allocate(size_in_bytes, initial_visibility);
		}

		/**
		 * Amount of total global memory on the CUDA device.
		 */
		size_t amount_total() const
		{
			scoped_setter_t set_device_for_this_scope(device_id_);
			size_t total_mem_in_bytes;
			auto status = cudaMemGetInfo(nullptr, &total_mem_in_bytes);
			throw_if_error(status,
				::std::string("Failed determining amount of total memory "
					"for CUDA device ") + device_id_as_str(device_id_));
			return total_mem_in_bytes;
		}

		/**
		 * Amount of memory on the CUDA device which is free and may be
		 * allocated for other uses.
		 *
		 * @note No guarantee of this free memory being contigous.
		 */
		size_t amount_free() const
		{
			scoped_setter_t set_device_for_this_scope(device_id_);
			size_t free_mem_in_bytes;
			auto status = cudaMemGetInfo(&free_mem_in_bytes, nullptr);
			throw_if_error(status, "Failed determining amount of "
				"free memory for CUDA device " + device_id_as_str(device_id_));
			return free_mem_in_bytes;
		}
	}; // class global_memory_t

	/**
	 * @brief Determine whether this device can access the global memory
	 * of another CUDA device.
	 *
	 * @param peer the device which is to be accessed
	 * @return true iff acesss is possible
	 */
	bool can_access(device_t peer) const
	{
		int result;
		auto status = cudaDeviceCanAccessPeer(&result, id(), peer.id());
		throw_if_error(status,
			"Failed determining whether CUDA device " + ::std::to_string(id()) + " can access CUDA device "
				+ ::std::to_string(peer.id()));
		return (result == 1);
	}

	/**
	 * @brief Enable access by this device to the global memory of another device
	 *
	 * @param peer the device to which to enable access
	 */
	void enable_access_to(device_t peer)
	{
		enum : unsigned {fixed_flags = 0 };
		// No flags are supported as of CUDA 8.0
		scoped_setter_t set_device_for_this_scope(id());
		auto status = cudaDeviceEnablePeerAccess(peer.id(), fixed_flags);
		throw_if_error(status,
			"Failed enabling access of device " + ::std::to_string(id()) + " to device " + ::std::to_string(peer.id()));
	}

	/**
	 * @brief Disable access by this device to the global memory of another device
	 *
	 * @param peer the device to which to disable access
	 */
	void disable_access_to(device_t peer)
	{
		scoped_setter_t set_device_for_this_scope(id());
		auto status = cudaDeviceDisablePeerAccess(peer.id());
		throw_if_error(status,
			"Failed disabling access of device " + ::std::to_string(id()) + " to device " + ::std::to_string(peer.id()));
	}

protected:
	void set_flags(flags_t new_flags)
	{
		scoped_setter_t set_device_for_this_scope(id_);
		auto status = cudaSetDeviceFlags(new_flags);
		throw_if_error(status, "Failed setting the flags for " + device_id_as_str());
	}

	void set_flags(
		host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
		bool                                   keep_larger_local_mem_after_resize,
		bool                                   allow_pinned_mapped_memory_allocation)
	{
		set_flags( (flags_t)
			  synch_scheduling_policy // this enum value is also a valid bitmask
			| (keep_larger_local_mem_after_resize    ? cudaDeviceLmemResizeToMax : 0)
			| (allow_pinned_mapped_memory_allocation ? cudaDeviceMapHost         : 0));
	}

	static ::std::string device_id_as_str(device::id_t id)
	{
		return "device " + ::std::to_string(id);
	}

	::std::string device_id_as_str() const
	{
		return device_id_as_str(id_);
	}

	flags_t flags() const
	{
		scoped_setter_t set_device_for_this_scope(id_);
		flags_t flags;
		auto status = cudaGetDeviceFlags(&flags);
		throw_if_error(status, "Failed obtaining the flags for  " + device_id_as_str());
		return flags;
	}

public:
	/**
	 * @brief Obtains a proxy for the device's global memory
	 */
	global_memory_t memory() { return global_memory_t(id_); };

	/**
	 * Obtains the (mostly) non-numeric properties for this device.
	 */
	properties_t properties() const
	{
		properties_t properties;
		auto status = cudaGetDeviceProperties(&properties, id());
		throw_if_error(status, "Failed obtaining device properties for " + device_id_as_str());
		return properties;
	}

	static device_t choose_best_match(const properties_t& properties) {
		device::id_t id;
		auto status = cudaChooseDevice(&id, &properties);
		throw_if_error(status, "Failed choosing a best matching device by a a property set.");
		return device::get(id);
	}

	/**
	 * Obtains this device's human-readable name, e.g. "GeForce GTX 650 Ti BOOST".
	 */
	::std::string name() const
	{
		// I could get the name directly, but that would require
		// direct use of the driver, and I'm not ready for that
		// just yet
		return properties().name;
	}

	/**
	 * Obtains this device's location on the PCI express bus in terms of
	 * domain, bus and device id, e.g. (0, 1, 0)
	 */
	device::pci_location_t pci_id() const
	{
		auto pci_domain_id = get_attribute(cudaDevAttrPciDomainId);
		auto pci_bus_id = get_attribute(cudaDevAttrPciBusId);
		auto pci_device_id = get_attribute(cudaDevAttrPciDeviceId);
		return {pci_domain_id, pci_bus_id, pci_device_id};
	}

	/**
	 * Obtains the device's hardware architecture generation numeric
	 * designator see @ref cuda::device::compute_architecture_t
	 */
	device::compute_architecture_t architecture() const
	{
		unsigned major = get_attribute(cudaDevAttrComputeCapabilityMajor);
		return {major};
	}

	/**
	 * Obtains the device's compute capability; see @ref cuda::device::compute_capability_t
	 */
	device::compute_capability_t compute_capability() const
	{
		auto arch = architecture();
		unsigned minor = get_attribute(cudaDevAttrComputeCapabilityMinor);
		return {arch, minor};
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
		auto ret = cudaDeviceGetAttribute(&attribute_value, attribute, id());
		throw_if_error(ret, "Failed obtaining device properties for " + device_id_as_str());
		return attribute_value;
	}

	/**
	 * Determine whether this device can coherently access managed memory
	 * concurrently with the CPU
	 */
	bool supports_concurrent_managed_access() const
	{
		return (get_attribute(cudaDevAttrConcurrentManagedAccess) != 0);
	}

	/**
	 * Obtains the upper limit on the amount of a certain kind of
	 * resource this device offers.
	 *
	 * @param resource which resource's limit to obtain
	 */
	resource_limit_t get_resource_limit(resource_id_t resource) const
	{
		resource_limit_t limit;
		auto status = cudaDeviceGetLimit(&limit, resource);
		throw_if_error(status, "Failed obtaining a resource limit for " + device_id_as_str());
		return limit;
	}

	/**
	 * Set the upper limit of one of the named numeric resources
	 * on this device
	 */
	void set_resource_limit(resource_id_t resource, resource_limit_t new_limit)
	{
		auto status = cudaDeviceSetLimit(resource, new_limit);
		throw_if_error(status, "Failed setting a resource limit for  " + device_id_as_str());
	}

	/**
	 * @brief Waits for all previously-scheduled tasks on all streams (= queues)
	 * on this device to conclude
	 *
	 * Depending on the host_thread_synch_scheduling_policy_t set for this
	 * device, the thread calling this method will either yield, spin or block
	 * until all tasks scheduled previously scheduled on this device have been
	 * concluded.
	 */
	void synchronize()
	{
		cuda::synchronize(*this);
	}

	/**
	 * Invalidates all memory allocations and resets all state regarding this
	 * CUDA device on the current operating system process.
     *
     * @todo Determine whether this actually performs a hardware reset or not
	 */
	void reset()
	{
		scoped_setter_t set_device_for_this_scope(id_);
		status_t status = cudaDeviceReset();
		throw_if_error(status, "Resetting  " + device_id_as_str());
	}

	/**
	 * Controls the balance between L1 space and shared memory space for
	 * kernels executing on this device.
	 *
	 * @param preference the preferred balance between L1 and shared memory
	 */
	void set_cache_preference(multiprocessor_cache_preference_t preference)
	{
		scoped_setter_t set_device_for_this_scope(id_);
		auto status = cudaDeviceSetCacheConfig((cudaFuncCache) preference);
		throw_if_error(status,
			"Setting the multiprocessor L1/Shared Memory cache distribution preference for " + device_id_as_str());
	}

	/**
	 * Determines the balance between L1 space and shared memory space set
	 * for kernels executing on this device.
	 */
	multiprocessor_cache_preference_t cache_preference() const
	{
		scoped_setter_t set_device_for_this_scope(id_);
		cudaFuncCache raw_preference;
		auto status = cudaDeviceGetCacheConfig(&raw_preference);
		throw_if_error(status,
			"Obtaining the multiprocessor L1/Shared Memory cache distribution preference for " + device_id_as_str());
		return (multiprocessor_cache_preference_t) raw_preference;
	}

	/**
	 * @brief Sets the shared memory bank size, described in
	 * <a href="https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/">this Parallel-for-all blog entry</a>
	 *
	 * @param new_bank_size the shared memory bank size to set, in bytes
	 */
	void set_shared_memory_bank_size(shared_memory_bank_size_t new_bank_size)
	{
		scoped_setter_t set_device_for_this_scope(id_);
		auto status = cudaDeviceSetSharedMemConfig(new_bank_size);
		throw_if_error(status, "Setting the multiprocessor shared memory bank size for " + device_id_as_str());
	}

	/**
	 * @brief Returns the shared memory bank size, as described in
	 * <a href="https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/">this Parallel-for-all blog entry</a>
	 *
	 * @return the shared memory bank size in bytes
	 */
	shared_memory_bank_size_t shared_memory_bank_size() const
	{
		scoped_setter_t set_device_for_this_scope(id_);
		shared_memory_bank_size_t bank_size;
		auto status = cudaDeviceGetSharedMemConfig(&bank_size);
		throw_if_error(status, "Obtaining the multiprocessor shared memory bank size for  " + device_id_as_str());
		return bank_size;
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
	device::id_t id() const
	{
		return id_;
	}

	stream_t default_stream() const noexcept;

	// I'm a worried about the creation of streams with the assumption
	// that theirs is the current device, so I'm just forbidding it
	// outright here - even though it's very natural to want to write
	//
	//   cuda::device::curent::get().create_stream()
	//
	// (sigh)... safety over convenience I guess
	//
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

	template<typename KernelFunction, typename ... KernelParameters>
	void launch(
		bool thread_block_cooperativity,
		KernelFunction kernel_function,
		launch_configuration_t launch_configuration,
		KernelParameters ... parameters);

	template<typename KernelFunction, typename ... KernelParameters>
	void launch(
		const KernelFunction& kernel_function, launch_configuration_t launch_configuration,
		KernelParameters ... parameters)
	{
		return launch(
			cuda::thread_blocks_may_not_cooperate, kernel_function, launch_configuration, parameters...);
	}

	/**
	 * Determines the range of possible priorities for streams on this device.
	 *
	 * @return a priority range, whose semantics are a bit confusing; see @ref priority_range_t . If
	 * the device does not support stream priorities, a 'trivial' range  of priority values will be
	 * returned.
	 */
	device::stream_priority_range_t stream_priority_range() const
	{
		scoped_setter_t set_device_for_this_scope(id_);
		stream::priority_t least, greatest;
		auto status = cudaDeviceGetStreamPriorityRange(&least, &greatest);
		throw_if_error(status, "Failed obtaining stream priority range for " + device_id_as_str());
		return {least, greatest};
	}

public:

	host_thread_synch_scheduling_policy_t synch_scheduling_policy() const
	{
		return (host_thread_synch_scheduling_policy_t) (flags() & cudaDeviceScheduleMask);
	}

	void set_synch_scheduling_policy(host_thread_synch_scheduling_policy_t new_policy)
	{
		auto other_flags = flags() & ~cudaDeviceScheduleMask;
		set_flags(other_flags | (flags_t) new_policy);
	}

	bool keeping_larger_local_mem_after_resize() const
	{
		return flags() & cudaDeviceLmemResizeToMax;
	}

	void keep_larger_local_mem_after_resize(bool keep = true)
	{
		auto flags_ = flags();
		if (keep) {
			flags_ |= cudaDeviceLmemResizeToMax;
		} else {
			flags_ &= ~cudaDeviceLmemResizeToMax;
		}
		set_flags(flags_);
	}

	void dont_keep_larger_local_mem_after_resize()
	{
		keep_larger_local_mem_after_resize(false);
	}

	/**
	 * Can we allocated mapped pinned memory on this device?
	 */
	bool can_map_host_memory() const
	{
		return flags() & cudaDeviceMapHost;
	}

	/**
	 * Control whether this device will support allocation of mapped pinned memory
	 */
	void enable_mapping_host_memory(bool allow = true)
	{
		auto flags_ = flags();
		if (allow) {
			flags_ |= cudaDeviceMapHost;
		} else {
			flags_ &= ~cudaDeviceMapHost;
		}
		set_flags(flags_);
	}

	/**
	 * See @ref enable_mapping_host_memory
	 */
	void disable_mapping_host_memory()
	{
		enable_mapping_host_memory(false);
	}

public:

	/**
	 * @brief Makes this device the CUDA Runtime API's current device
	 *
	 * @note a non-current device becoming current will not stop its methods from
	 * always expressly setting the current device before doing anything(!)
	 */
	device_t& make_current()
	{
		device::current::detail_::set(id());
		return *this;
	}

public: 	// constructors and destructor

	~device_t() noexcept = default;
	device_t(device_t&& other) noexcept = default;
	device_t(const device_t& other) noexcept = default;
		// Device proxies are not owning - as devices aren't allocated nor de-allocated.
		// Also, the proxies don't hold any state - it's the devices _themselves_ which
		// have state. ; so there's no problem copying the proxies around. This is
		// unlike events and streams, which get created and destroyed.
	device_t& operator=(const device_t& other) noexcept = default;
	device_t& operator=(device_t&& other) noexcept = default;

protected: // constructors

	/**
	 * @note Only @ref device::current::get() and @ref device::get() should be
	 * calling this one.
	 */
	device_t(wrapping_construction, device::id_t device_id) noexcept : id_( device_id ) { }

public: // friends
	friend device_t device::get(device::id_t) noexcept;

protected:
	// data members
	/**
	 * The numeric ID of the proxied device.
	 */
	device::id_t id_;
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

/**
 * Returns a proxy for the CUDA device with a given id
 *
 * @param device_id the ID for which to obtain the device proxy
 * @note direct constructor access is blocked so that you don't get the
 * idea you're actually creating devices
 */
inline device_t get(id_t device_id) noexcept
{
	return device_t(device_t::wrapping_construction{}, device_id);
}

namespace current {

/**
 * Obtains (a proxy for) the device which the CUDA runtime API considers to be current.
 */
inline device_t get() { return device::get(detail_::get_id()); }

/**
 * Tells the CUDA runtime API to consider the specified device as the current one.
 */
inline void set(device_t device) { detail_::set(device.id()); }

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

inline void synchronize(device_t& device)
{
	auto device_id = device.id();
	device::current::detail_::scoped_override_t set_device_for_this_scope(device_id);
	auto status = cudaDeviceSynchronize();
	throw_if_error(status, "Failed synchronizing " + ::std::to_string(device_id));
}


} // namespace cuda

#endif // CUDA_API_WRAPPERS_DEVICE_HPP_
