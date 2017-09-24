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

#include <cuda/api/types.h>
#include <cuda/api/device_properties.hpp>
#include <cuda/api/memory.hpp>
#include <cuda/api/current_device.hpp>
#include <cuda/api/pci_id.h>
#include <cuda/api/unique_ptr.hpp>

#include <cuda_runtime_api.h>
#include <string>
#include <type_traits>

namespace cuda {

class event_t;
template<bool DeviceAssumedCurrent> class stream_t;
template<bool AssumedCurrent> class device_t;
template<bool AssumedCurrent> class device_t;

namespace device {

/**
 * Returns a proxy for the CUDA device with a given id
 *
 * @param id the ID for which to obtain the device proxy
 * @note direct constructor access is blocked so that you don't get the
 * idea you're actually creating devices
 */
device_t<detail::do_not_assume_device_is_current> get(id_t id);

namespace current {

/**
 * Returns the current device in a wrapper which assumes it is indeed
 * current, i.e. which will not set the current device before performing any
 * other actions.
 */
device_t<detail::assume_device_is_current> get();

} // namespace current

namespace peer_to_peer {

/**
 * @brief The value of type for all CUDA device "attributes"; see also @ref attribute_t.
 */
using attribute_value_t = int;

/**
 * @brief An identifier of a integral-numeric-value attribute of a CUDA device.
 *
 * @note Somewhat annoyingly, CUDA devices have attributes, properties and flags.
 * Attributes have integral number values; properties have all sorts of values,
 * including arrays and limited-length strings (see
 * @ref cuda::device::properties_t), and flags are either binary or
 * small-finite-domain type fitting into an overall flagss value (see
 * @ref cuda::device_t::flags_t).
 */
using attribute_t = cudaDeviceP2PAttr;

/**
 * Aliases for all CUDA device attributes
 */
enum : std::underlying_type<attribute_t>::type {
		link_performance_rank = cudaDevP2PAttrPerformanceRank, /**< A relative value indicating the performance of the link between two devices */                       //!< link_performance_rank
		access_support = cudaDevP2PAttrAccessSupported, /**< 1 if access is supported, 0 otherwise */                                                                    //!< access_support
		native_atomics_support = cudaDevP2PAttrNativeAtomicSupported /**< 1 if the first device can perform native atomic operations on the second device, 0 otherwise *///!< native_atomics_support
};

/**
 * @brief Determine whether one CUDA device can access the global memory
 * of another CUDA device.
 *
 * @param accessor id of the device interested in making a remote access
 * @param peer id of the device which is to be accessed
 * @return true iff acesss is possible
 */
inline bool can_access(id_t accessor, id_t peer)
{
	int result;
	auto status = cudaDeviceCanAccessPeer(&result, accessor, peer);
	throw_if_error(status,
		"Failed determining whether CUDA device " + std::to_string(accessor) + " can access CUDA device "
			+ std::to_string(peer));
	return (result == 1);
}

/**
 * @brief Determine whether one CUDA device can access the global memory
 * of another CUDA device.
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 * @return true iff acess is possible
 */
template<bool FirstIsAssumedCurrent, bool SecondIsAssumedCurrent>
inline bool can_access(const device_t<FirstIsAssumedCurrent>& accessor, const device_t<SecondIsAssumedCurrent>& peer);

/**
 * @brief Enable access by one CUDA device to the global memory of another
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 */
void enable_access(id_t accessor_id, id_t peer_id)
{
	enum
		: unsigned {fixed_flags = 0
	};
	// No flags are supported as of CUDA 8.0
	device::current::scoped_override_t<> set_device_for_this_scope(accessor_id);
	auto status = cudaDeviceEnablePeerAccess(peer_id, fixed_flags);
	throw_if_error(status,
		"Failed enabling access of device " + std::to_string(accessor_id) + " to device " + std::to_string(peer_id));
}

/**
 * @brief Disable access by one CUDA device to the global memory of another
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 */
void disable_access(id_t accessor_id, id_t peer_id)
{
	device::current::scoped_override_t<> set_device_for_this_scope(accessor_id);
	auto status = cudaDeviceDisablePeerAccess(peer_id);
	throw_if_error(status,
		"Failed disabling access of device " + std::to_string(accessor_id) + " to device " + std::to_string(peer_id));
}

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
		"Failed obtaining peer-to-peer device attribute for device pair (" + std::to_string(source) + ", "
			+ std::to_string(destination) + ')');
	return value;
}

template<bool FirstIsAssumedCurrent, bool SecondIsAssumedCurrent>
inline attribute_value_t get_attribute(attribute_t attribute, const device_t<FirstIsAssumedCurrent>& source,
	const device_t<SecondIsAssumedCurrent>& destination);

} // namespace peer_to_peer

} // namespace device

/**
 * @brief Proxy class for a CUDA device
 *
 * Use this class - built around a device ID, or for the current device - to
 * perform almost, if not all, device-related operations, as opposed to passing
 * the device ID around all that time.
 *
 * @tparam AssumedCurrent - when true, the code performs no setting of the
 *
 * @note this is one of the three main classes in the Runtime API wrapper library,
 * together with @ref cuda::stream_t and @ref cuda::event_t
 */
template<bool AssumedCurrent = detail::do_not_assume_device_is_current>
class device_t {
public:
	// types
	using scoped_setter_t = device::current::scoped_override_t<AssumedCurrent>;

protected:
	// types
	using properties_t = device::properties_t;
	using attribute_t = device::attribute_t;
	using attribute_value_t = device::attribute_value_t;
	using flags_t = unsigned;
	using resource_id_t = cudaLimit;
	using resource_limit_t = size_t;
	using shared_memory_bank_size_t = cudaSharedMemConfig;
	using priority_range_t = std::pair<stream::priority_t, stream::priority_t>;

	// This relies on valid CUDA device IDs being non-negative, which is indeed
	// always the case for CUDA <= 8.0 and unlikely to change; however, it's
	// a bit underhanded to make that assumption just to twist this class to do
	// our bidding (see below)
	enum : device::id_t { invalid_id = -1 };

	struct immutable_id_holder_t {
		operator const device::id_t&() const { return id; }
		void set(device::id_t) const { }
		;
		device::id_t id;
	};
	struct mutable_id_holder_t {
		operator const device::id_t&() const { return id; }
		void set(device::id_t new_id) const { id = new_id; }
		mutable device::id_t id { invalid_id };
	};

	using id_holder_type = typename std::conditional<
		AssumedCurrent,
		mutable_id_holder_t,
		immutable_id_holder_t
	>::type;
		// class device_t has a field with the device's id; when assuming the device is the
		// current one, we want to be able to set the device_id after construction; but
		// when not making that assumption, we need a const field, initialized on construction;
		// this hack allows both the options to coexist without the compiler complaining
		// about assigning to a const lvalue (we use it since std::conditional can't
		// be used to select between making a field mutable or not).

public:	// types

	/**
	 * @brief A class to create a faux member in a @ref device_t, in lieu of an in-class
	 * namespace (which C++ does not support); whenever you see a function
	 * `my_dev.memory::foo()`, think of it as a `my_dev::memory::foo()`.
	 */
	class memory_t {
	protected:
		const id_holder_type& device_id;

		using deleter = memory::device::detail::deleter;
		using allocator = memory::device::detail::allocator;

		std::string device_id_as_str() const
		{
			return device_t::device_id_as_str(device_id);
		}

	public:
		///@cond
		memory_t(const device_t::id_holder_type& id) : device_id(id) { }
		///@endcond

		/**
		 * Allocate a region of memory on the device
		 *
		 * @param size_in_bytes size in bytes of the region of memory to allocate
		 * @return a non-null (device-side) pointer to the allocated memory
		 */
		template<typename T = void>
		__host__ T* allocate(size_t size_in_bytes)
		{
			scoped_setter_t set_device_for_this_scope(device_id);
			return memory::device::detail::malloc<T>(size_in_bytes);
		}

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
		 * @param initially_visible_to_host_only if true, only the host (and the
		 * allocating device) will be able to utilize the pointer returned; if false,
		 * it will be made usable on all CUDA devices on the system
		 * @return the allocated pointer; never returns null (throws on failure)
		 */
		template<typename T = void>
		__host__ T* allocate_managed(size_t size_in_bytes, bool initially_visible_to_host_only = false)
		{
			scoped_setter_t set_device_for_this_scope(device_id);
			T* allocated = nullptr;
			auto flags = initially_visible_to_host_only ?
				cudaMemAttachHost : cudaMemAttachGlobal;
			// Note: the typed version also takes its size in bytes, apparently,
			// not in number of elements
			auto status = cudaMallocManaged<T>(&allocated, size_in_bytes, flags);
			if (is_success(status) && allocated == nullptr) {
				// Can this even happen? hopefully not
				status = cudaErrorUnknown;
			}
			throw_if_error(status,
				"Failed allocating " + std::to_string(size_in_bytes) + " bytes of global memory on "
					+ device_id_as_str());
			return allocated;
		}

		using region_pair = ::cuda::memory::mapped::region_pair;

		/**
		 * @brief Allocate a pair of mapped regions, in device-global memory and
		 * in host memory.
		 *
		 * @note see @ref memory::mapped::region_pair for an explanation of how
		 * region pairs work.
		 *
		 * @param size_in_bytes of the region to allocate
		 * @param options to be passed to the CUDA memory allocation API
		 * @return a non-null pair of allocated regions
		 */
		template<typename T = void>
		__host__ region_pair allocate_region_pair(size_t size_in_bytes, region_pair::allocation_options options = {
			region_pair::isnt_portable_across_cuda_contexts, region_pair::without_cpu_write_combining })
		{
			scoped_setter_t set_device_for_this_scope(device_id);
			region_pair allocated;
			auto flags = cuda::memory::mapped::detail::make_cuda_host_alloc_flags(options);
			// Note: the typed cudaHostAlloc also takes its size in bytes, apparently,
			// not in number of elements
			auto status = cudaHostAlloc<T>(static_cast<T**>(&allocated.host_side), size_in_bytes, flags);
			if (is_success(status) && (allocated.host_side == nullptr)) {
				// Can this even happen? hopefully not
				status = cudaErrorUnknown;
			}
			if (is_success(status)) {
				auto get_device_pointer_flags = 0u; // see CUDA runtime API 7.5 documentation
				status = cudaHostGetDevicePointer(static_cast<T**>(&allocated.device_side), allocated.host_side,
					get_device_pointer_flags);
			}
			throw_if_error(status,
				"Failed allocating a mapped pair of memory regions of size " + std::to_string(size_in_bytes)
					+ " bytes of global memory on " + device_id_as_str());
			return allocated;
		}

		/**
		 * Amount of total global memory on the CUDA device.
		 */
		size_t amount_total() const
		{
			scoped_setter_t set_device_for_this_scope(device_id);
			size_t total_mem_in_bytes;
			auto status = cudaMemGetInfo(nullptr, &total_mem_in_bytes);
			throw_if_error(status,
				std::string("Failed determining amount of total memory for CUDA device ") + std::to_string(device_id));
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
			scoped_setter_t set_device_for_this_scope(device_id);
			size_t free_mem_in_bytes;
			auto status = cudaMemGetInfo(&free_mem_in_bytes, nullptr);
			throw_if_error(status, "Failed determining amount of free memory for " + device_id_as_str());
			return free_mem_in_bytes;
		}
	}; // class memory_t

	/**
	 * @brief A class to create a faux member in a @ref device_t, in lieu of an in-class
	 * namespace (which C++ does not support); whenever you see a function
	 * `my_dev.peer_access::foo()`, think of it as a `my_dev::peer_access::foo()`.
	 */
	class peer_access_t {
	protected:
		const device_t::id_holder_type& device_id;

		std::string device_id_as_str() const
		{
			return device_t::device_id_as_str(device_id);
		}

	public:
		///@cond
		peer_access_t(const device_t::id_holder_type& holder) : device_id(holder) { }
		///@endcond

		/**
		 * @brief Determine whether this device can access the global memory
		 * of another CUDA device.
		 *
		 * @param peer id of the device which is to be accessed
		 * @return true iff acesss is possible
		 */
		bool can_access(id_t peer_id) const
		{
			scoped_setter_t set_device_for_this_scope(device_id);
			return device::peer_to_peer::can_access(device_id, peer_id);
		}

		// This won't compile, for some reason:
		//
		// inline bool can_access(const device_t<detail::do_not_assume_device_is_current>& peer) const
		// {
		// 	return this->can_access(peer.id());
		// }

		/**
		 * @brief Enable access by this device to the global memory of another device
		 * with a specified id
		 *
		 * @param peer_id id of the device to which to enable access
		 */
		void enable_to(id_t peer_id)
		{
			if (AssumedCurrent) {
				enum : unsigned {fixed_flags = 0 };
				// No flags are supported as of CUDA 8.0
				auto status = cudaDeviceEnablePeerAccess(peer_id, fixed_flags);
				throw_if_error(status, "Failed enabling access of current device to device " + std::to_string(peer_id));
			} else
				device::peer_to_peer::enable_access(device_id, peer_id);
		}

		/**
		 * @brief Enable access by this device to the global memory of another device
		 *
		 * @param peer device to have access access to
		 */
		void enable_to(device_t<> peer) { enable_to(peer.id()); }

		/**
		 * @brief Disable access by this device to the global memory of another device
		 * with a specified id
		 *
		 * @param peer_id id of the device to which to disable access
		 */
		void disable_to(id_t peer_id)
		{
			if (AssumedCurrent) {
				auto status = cudaDeviceDisablePeerAccess(peer_id);
				throw_if_error(status,
					"Failed disabling access of current device to device " + std::to_string(peer_id));
			} else
			device::peer_to_peer::disable_access(device_id, peer_id);
		}

		/**
		 * @brief Disable access by this device to the global memory of another device
		 *
		 * @param peer device to have access disabled to
		 */
		void disable_to(device_t<> peer) { disable_to(peer.id()); }

	}; // class peer_access_t

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

	static std::string device_id_as_str(device::id_t id)
	{
		return AssumedCurrent ? "current device" : "device " + std::to_string(id);
	}

	std::string device_id_as_str() const
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
	 * Obtains the (mostly) non-numeric properties for this device.
	 */
	properties_t properties() const
	{
		properties_t properties;
		auto status = cudaGetDeviceProperties(&properties, id());
		throw_if_error(status, "Failed obtaining device properties for " + device_id_as_str());
		return properties;
	}

	/**
	 * Obtains this device's human-readable name, e.g. "GeForce GTX 650 Ti BOOST".
	 */
	std::string name() const
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
	device::pci_id_t pci_id() const
	{
		auto pci_domain_id = get_attribute(cudaDevAttrPciDomainId);
		auto pci_bus_id = get_attribute(cudaDevAttrPciBusId);
		auto pci_device_id = get_attribute(cudaDevAttrPciDeviceId);
		return {pci_domain_id, pci_bus_id, pci_device_id};
	}

	/**
	 * Obtains the device's compute capability; see @ref cuda::device::compute_capability_t
	 */
	device::compute_capability_t compute_capability() const
	{
		auto major = get_attribute(cudaDevAttrComputeCapabilityMajor);
		auto minor = get_attribute(cudaDevAttrComputeCapabilityMinor);
		return {major, minor};
	}

	/**
	 * Obtains the device's hardware architecture generation numeric
	 * designator see @ref cuda::device::compute_architecture_t
	 */
	device::compute_architecture_t architecture() const
	{
		auto major = get_attribute(cudaDevAttrComputeCapabilityMajor);
		return {major};
	}

	/**
	 * Obtain a numeric-value attribute of the device
	 *
	 * @note See @ref device::attribute_t for explanation about attributes,
	 * properties and flags.
	 */
	attribute_value_t get_attribute(attribute_t attribute) const
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
		scoped_setter_t set_device_for_this_scope(id_);
		auto status = cudaDeviceSynchronize();
		throw_if_error(status, "Failed synchronizing " + device_id_as_str());
	}

	/**
	 * @brief Waits for all previously-scheduled tasks on a certain stream
	 * (queue) of this device to conclude.
	 *
	 * Depending on the host_thread_synch_scheduling_policy_t set for this
	 * device, the thread calling this method will either yield, spin or block
	 * until all tasks scheduled on the stream with @p stream_id have
	 * been completed.
	 *
	 * @param stream_id The stream for whose currently-scheduled tasks'
	 * conclusion to wait for
	 */
	void synchronize_stream(stream::id_t stream_id)
	{
		scoped_setter_t set_device_for_this_scope(id_);
		auto status = cudaStreamSynchronize(stream_id);
		throw_if_error(status, "Failed synchronizing a stream on " + device_id_as_str());
	}

	/**
	 *
	 * @param stream
	 */
	void synchronize(stream_t<detail::do_not_assume_device_is_current>& stream);

	/**
	 * Waits for a specified event to conclude before returning control
	 * to the calling code.
	 *
	 * @todo Determine how this waiting takes place (as opposed to stream
	 * synchrnoization).
	 *
	 * @param event_id of the event to wait for
	 */
	void synchronize_event(event::id_t event_id)
	{
		scoped_setter_t set_device_for_this_scope(id_);
		auto status = cudaEventSynchronize(event_id);
		throw_if_error(status, "Failed synchronizing an event on   " + device_id_as_str());
	}

	/**
	 *
	 * @param event
	 */
	void synchronize(event_t& event);

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
	 * @brief Sets the shared memory bank size, described here:
	 * @url https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
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
	 * @brief Returns the shared memory bank size, described here:
	 * @url https://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
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
	 */
	device::id_t id() const
	{
		if (AssumedCurrent) {
			// This is the first time in which we need to get the device ID
			// for the current device - as we've constructed this instance of
			// device_t<assume_device_is_current> without the ID
			if (id_.id == invalid_id) {
				id_.set(device::current::get_id());
			}
		}
		return id_.id;
	}

	stream_t<AssumedCurrent> default_stream() const;

	// I'm a worried about the creation of streams with the assumption
	// that theirs is the current device, so I'm just forbidding it
	// outright here - even though it's very natural to want to write
	//
	//   cuda::device::curent::get().create_stream()
	//
	// (sigh)... safety over convenience I guess
	//
	stream_t<detail::do_not_assume_device_is_current> create_stream(bool will_synchronize_with_default_stream,
		stream::priority_t priority = cuda::stream::default_priority);

	template<typename KernelFunction, typename ... KernelParameters>
	void launch(const KernelFunction& kernel_function, launch_configuration_t launch_configuration,
		KernelParameters ... parameters)
	{
		return default_stream().enqueue.kernel_launch(kernel_function, launch_configuration, parameters...);
	}

	priority_range_t stream_priority_range() const
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

	 // I'm of two minds regarding whether or not we should have this method at all;
	 // I'm worried people might assume the proxy object will start behaving like
	 // what they get with cuda::device::current::get(), i.e. a device_t<AssumedCurrent>.
	/**
	 * @brief Makes this device the CUDA Runtime API's current device
	 *
	 * @note a non-current device becoming current will not stop its methods from
	 * always expressly setting the current device before doing anything(!)
	 */
	 void make_current() {
		 static_assert(AssumedCurrent == false,
			 "Attempt to set a device, assumed to be current, to be current");
		 device::current::set(id());
	 }
	device_t& operator=(const device_t& other) = delete;

public: 	// destructor

	~device_t() = default;

protected: // constructors

	/**
	 * @note Only @ref device::current::get() and @ref device::get() should be
	 * calling this one.
	 */
	device_t(device::id_t device_id) : id_( { device_id })
	{
	}

	/**
	 * @note Have a look at how @ref mutable_id_holder is default-constructed.
	 */
	device_t() :
		id_()
	{
		static_assert(AssumedCurrent,
			"Attempt to instantiate a device proxy for a device not known to be "
			"current, without a specific device id");
	}

public: // friends
	friend device_t<detail::assume_device_is_current> device::current::get();
	friend device_t<detail::do_not_assume_device_is_current> device::get(device::id_t id);

protected:
	// data members
	/**
	 * The numeric ID of the proxied device.
	 *
	 * @note  The cannot simply have type {@code const device::id_t}, as
	 * when this class is used for the current device (AssumedCurrent is
	 * true), we don't actually know the device ID on construction - only
	 * later.
	 */
	const id_holder_type id_;

public:
	// faux data members (used as surrogates for internal namespaces)
	///@cond
	memory_t memory { id_ };
	peer_access_t peer_access { id_ };
	// don't worry, these two will not actually use the id_ value
	// without making sure it's been correctly determined
	///@endcond
};

template<bool LHSAssumedCurrent, bool RHSAssumedCurrent>
bool operator==(const device_t<LHSAssumedCurrent>& lhs, const device_t<RHSAssumedCurrent>& rhs)
{
	return lhs.id() == rhs.id();
}

template<bool LHSAssumedCurrent, bool RHSAssumedCurrent>
bool operator!=(const device_t<LHSAssumedCurrent>& lhs, const device_t<RHSAssumedCurrent>& rhs)
{
	return lhs.id() != rhs.id();
}

namespace device {
namespace current {

/**
 * Returns a proxy for the CUDA device the runtime API has set as the current
 *
 * @note direct constructor access is blocked so that you don't get the
 * idea you're actually creating a device
 */
inline cuda::device_t<detail::assume_device_is_current> get()
{
	return {};
}

} // namespace current

/**
 * Returns a proxy for the CUDA device with a given id
 *
 * @param id the ID for which to obtain the device proxy
 * @note direct constructor access is blocked so that you don't get the
 * idea you're actually creating devices
 */
inline device_t<detail::do_not_assume_device_is_current> get(id_t device_id)
{
	return device_t<detail::do_not_assume_device_is_current>{device_id};
}

/**
 * @brief Obtain a proxy to a device using its PCI bus location
 *
 * @param pci_id The domain-bus-device triplet locating the GPU on the PCI bus
 * @return a device_t proxy object for the device at the specified location
 */
inline device_t<detail::do_not_assume_device_is_current> get(pci_id_t pci_id)
{
	auto resolved_id = pci_id.resolve_device_id();
	return get(resolved_id);
}

/**
 * @brief Obtain a proxy to a device using a string with its PCI bus location
 *
 * @param pci_id_str A string listing of the GPU's location on the PCI bus
 * @return a device_t proxy object for the device at the specified location
 */
inline cuda::device_t<detail::do_not_assume_device_is_current> get(const std::string& pci_id_str)
{
	auto parsed_pci_id = pci_id_t::parse(pci_id_str);
	return get(parsed_pci_id);
}

namespace peer_to_peer {

/**
 * @brief Determine whether one CUDA device can access the global memory
 * of another CUDA device.
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 * @return true iff acess is possible
 */
template<bool FirstIsAssumedCurrent, bool SecondIsAssumedCurrent>
inline bool can_access(const device_t<FirstIsAssumedCurrent>& accessor, const device_t<SecondIsAssumedCurrent>& peer)
{
	return can_access(accessor.id(), peer.id());
}

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
template<bool FirstIsAssumedCurrent, bool SecondIsAssumedCurrent>
inline attribute_value_t get_attribute(attribute_t attribute, const device_t<FirstIsAssumedCurrent>& source,
	const device_t<SecondIsAssumedCurrent>& destination)
{
	return get_attribute(attribute, source.id(), destination.id());
}

} // namespace peer_to_peer

} // namespace device

namespace memory {
namespace device {

/**
 * @brief Create a variant of std::unique_pointer for an array in
 * device-global memory
 *
 * @tparam T  the type of individual array elements
 * @tparam AssumedCurrent when false, the current device needs to be
 * set before the allocation is made
 *
 * @param device  on which to construct the array of T elements
 * @param n       the number of elements of type T
 * @return an std::unique_ptr pointing to the allocated memory
 */
template<typename T, bool AssumedCurrent = cuda::detail::do_not_assume_device_is_current>
inline unique_ptr<T> make_unique(device_t<AssumedCurrent>& device, size_t n)
{
	typename device_t<AssumedCurrent>::scoped_setter_t set_device_for_this_scope(device.id());
	return cuda::memory::detail::make_unique<T, detail::allocator, detail::deleter>(n);
}

/**
 * @brief Create a variant of std::unique_pointer for a single value
 * in device-global memory
 *
 * @tparam T  the type of value to construct in device memory
 * @tparam AssumedCurrent when false, the current device needs to be
 * set before the allocation is made
 *
 * @param device  on which to construct the T element
 * @return an std::unique_ptr pointing to the allocated memory
 */
template<typename T, bool AssumedCurrent = cuda::detail::do_not_assume_device_is_current>
inline unique_ptr<T> make_unique(device_t<AssumedCurrent>& device)
{
	typename device_t<AssumedCurrent>::scoped_setter_t set_device_for_this_scope(device.id());
	return cuda::memory::detail::make_unique<T, detail::allocator, detail::deleter>();
}

} // namespace device
} // namespace memory

} // namespace cuda

#endif /* CUDA_API_WRAPPERS_DEVICE_HPP_ */
