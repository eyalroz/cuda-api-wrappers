#pragma once
#ifndef CUDA_DEVICE_H_
#define CUDA_DEVICE_H_

#include "cuda/api/types.h"
#include "cuda/api/device_properties.hpp"
#include "cuda/api/memory.hpp"
#include "cuda/api/current_device.hpp"
#include "cuda/api/stream.hpp" // For default_stream() and launch() only

#include <cuda_runtime_api.h>
#include <string>

namespace cuda {

namespace device {

struct pci_id_t {
	// This is simply what we get in CUDA's cudaDeviceProp structure
	int domain;
	int bus;
	int device;

	operator std::string() const;
	// This is not a ctor so as to maintain the plain-old-structness
	static pci_id_t parse(const std::string& id_str);
	device::id_t get_cuda_device_id() const;
};

inline bool can_access_peer(id_t accessor_id, id_t peer_id)
{
	int result;
	auto status = cudaDeviceCanAccessPeer(&result, accessor_id, peer_id);
	throw_if_error(status,
		"Failed determining whether CUDA device " + std::to_string(accessor_id) +
		" can access CUDA device " + std::to_string(peer_id));
	return (result == 1);
}


} // namespace device

// TODO: Consider making this a multiton, with one possible instance per device_id
template <bool AssumedCurrent = detail::do_not_assume_device_is_current>
class device_t {

protected: // types
	using device_setter              = device::current::ScopedDeviceOverride<AssumedCurrent>;
	using properties_t               = device::properties_t;
	using attribute_t                = device::attribute_t;
	using attribute_value_t          = device::attribute_value_t;
	using flags_t                    = device::flags_t;
	using pci_id_t                   = device::pci_id_t;
	using resource_id_t              = cudaLimit;
	using resource_limit_t           = size_t;
	using shared_memory_bank_size_t  = cudaSharedMemConfig;
	using priority_range_t           = std::pair<stream::priority_t, stream::priority_t>;

public: // types

	class memory_t {
	protected:
		const device::id_t& device_id;

	public:
		memory_t(device::id_t  device_id) : device_id(device_id) {}

		template <typename T = void>
		__host__ T* allocate(size_t size_in_bytes)
		{
			device_setter set_device_for_this_scope(device_id);
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
		 * @note for a more complete description see
		 * {@link http://docs.nvidia.com/cuda/cuda-runtime-api/}
		 *
		 *
		 * @param size_in_bytes Size of memory region to allocate
		 * @param initially_visible_to_host_only if true, only the host (and the
		 * allocating device) will be able to utilize the pointer returned; if false,
		 * it will be made usable on all CUDA devices on the system
		 * @return the allocated pointer; never returns null (throws on failure)
		 */
		// TODO: managed - similar to owner
		template <typename T = void>
		__host__ T* allocate_managed(
			size_t size_in_bytes, bool initially_visible_to_host_only = false)
		{
			device_setter set_device_for_this_scope(device_id);
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
				"Failed allocating " + std::to_string(size_in_bytes) +
				" bytes of global memory on CUDA device " + std::to_string(device_id));
			return allocated;
		}

		using region_pair = ::cuda::memory::mapped::region_pair;
		template <typename T = void>
		__host__ region_pair allocate_region_pair(
			size_t size_in_bytes, region_pair::allocation_options options = {false, false})
		{
			device_setter set_device_for_this_scope(device_id);
			region_pair allocated;
			auto flags = cuda::memory::mapped::detail::make_cuda_host_alloc_flags(options);
			// Note: the typed cudaHostAlloc also takes its size in bytes, apparently,
			// not in number of elements
			auto status = cudaHostAlloc<T>(
					static_cast<T**>(&allocated.host_side), size_in_bytes, flags);
			if (is_success(status) && (allocated.host_side == nullptr)) {
				// Can this even happen? hopefully not
				status = cudaErrorUnknown;
			}
			if (is_success(status)) {
				auto get_device_pointer_flags = 0u; // see CUDA runtime API 7.5 documentation
				status = cudaHostGetDevicePointer(
					static_cast<T**>(&allocated.device_side),
					allocated.host_side, get_device_pointer_flags);
			}
			throw_if_error(status,
				"Failed allocating a mapped pair of memory regions of size " +
				std::to_string(size_in_bytes) +
				" bytes of global memory on CUDA device " + std::to_string(device_id));
			return allocated;
		}

		size_t amount_total() const
		{
			device_setter set_device_for_this_scope(device_id);
			size_t total_mem_in_bytes;
			auto status = cudaMemGetInfo(&total_mem_in_bytes, nullptr);
			throw_if_error(status,
				std::string("Failed determining amount of free memory for CUDA device ") +
				std::to_string(device_id));
			return total_mem_in_bytes;
		}

		size_t amount_free() const
		{
			device_setter set_device_for_this_scope(device_id);
			size_t free_mem_in_bytes;
			auto status = cudaMemGetInfo(&free_mem_in_bytes, nullptr);
			throw_if_error(status,
				std::string("Failed determining amount of free memory for CUDA device ") +
				std::to_string(device_id));
			return free_mem_in_bytes;
		}
	}; // class memory_t

	class peer_access_t {
	protected:
		const device::id_t& device_id;

	public:
		peer_access_t(device::id_t  device_id) : device_id(device_id) {}

		bool can_access(id_t peer_id) const
		{
			device_setter set_device_for_this_scope(device_id);
			return device::can_access_peer(device_id, peer_id);
		}

		void enable_to(id_t peer_id) {
			device_setter set_device_for_this_scope(device_id);
			enum : unsigned { fixed_flags = 0}; // No flags are supported as of CUDA 8.0
			auto status = cudaDeviceEnablePeerAccess(peer_id, fixed_flags);
			throw_if_error(status,
				"Failed enabling access of device " + std::to_string(device_id)
				+ " to device " + std::to_string(peer_id));
		}

		void disable_to(id_t peer_id) {
			device_setter set_device_for_this_scope(device_id);
			auto status = cudaDeviceDisablePeerAccess(peer_id);
			throw_if_error(status,
				"Failed disabling access of device " + std::to_string(id_)
				+ " to device " + std::to_string(peer_id));

		}
	}; // class peer_access_t


protected:
	/**
	 * Code using the device_t class should use
	 */
	void set_flags(flags_t new_flags)
	{
		device_setter set_device_for_this_scope(id_);
		auto status = cudaSetDeviceFlags(new_flags);
		throw_if_error(status,
			"Failed setting the flags for CUDA device " + std::to_string(id_));
	}

public: // methods
	// TODO: Add free() and free_region_pair() as methods?

	properties_t properties() const
	{
		properties_t properties;
		auto status = cudaGetDeviceProperties(&properties, id_);
		throw_if_error(status,
			std::string("Failed obtaining device properties for CUDA device ") + std::to_string(id_));
		return properties;
	}

	std::string name() const { return properties().name; }

	pci_id_t pci_id() const
	{
		auto p = properties();
		return { p.pciDomainID, p.pciBusID, p.pciDeviceID };
	}

	/**
	 * Yes, properties vs attributes is a bit confusing, but that's what CUDA has.
	 */
	attribute_value_t get_attribute(attribute_t attribute) const
	{
		attribute_value_t attribute_value;
		auto ret = cudaDeviceGetAttribute(&attribute_value, attribute, id_);
		throw_if_error(ret,
			std::string("Failed obtaining device properties for CUDA device ") + std::to_string(id_));
		return attribute_value;
	}

	resource_limit_t resource_limit(resource_id_t resource) const
	{
		resource_limit_t limit;
		auto status = cudaDeviceGetLimit(&limit, resource);
		throw_if_error(status, std::string("Failed obtaining a resource limit "
			"for CUDA device ") + std::to_string(id_));
		return limit;
	}

	void set_resource_limit(resource_id_t resource, resource_limit_t new_limit)
	{
		auto status = cudaDeviceSetLimit(resource, new_limit);
		throw_if_error(status, std::string("Failed setting a resource limit "
			"for CUDA device ") + std::to_string(id_));
	}

	inline void synchronize()
	{
		device_setter set_device_for_this_scope(id_);
		auto status = cudaDeviceSynchronize();
		throw_if_error(status,
			std::string("Failed synchronizing CUDA device ") + std::to_string(id_));
	}
	inline void synchronize_stream(stream::id_t stream_id)
	{
		device_setter set_device_for_this_scope(id_);
		auto status = cudaStreamSynchronize(stream_id);
		throw_if_error(status, "Failed synchronizing a stream  on CUDA device " + std::to_string(id_));
	}

	inline void synchronize_event(event::id_t event_id)
	{
		device_setter set_device_for_this_scope(id_);
		auto status = cudaEventSynchronize(event_id);
		throw_if_error(status, "Failed synchronizing an event on CUDA device " + std::to_string(id_));
	}

	inline void reset()
	{
		device_setter set_device_for_this_scope(id_);
		status_t status = cudaDeviceReset();
		throw_if_error(status, "Resetting CUDA device " + std::to_string(id_));
	}

	inline void set_cache_preference(multiprocessor_cache_preference_t preference) {
		device_setter set_device_for_this_scope(id_);
		auto status = cudaDeviceSetCacheConfig((cudaFuncCache) preference);
		throw_if_error(status,
			"Setting the multiprocessor L1/Shared Memory cache distribution preference for device " +
			std::to_string(id_));
	}

	inline multiprocessor_cache_preference_t cache_preference() const {
		device_setter set_device_for_this_scope(id_);
		cudaFuncCache raw_preference;
		auto status = cudaDeviceGetCacheConfig(&raw_preference);
		throw_if_error(status,
			"Obtaining the multiprocessor L1/Shared Memory cache distribution preference for device " +
			std::to_string(id_));
		return (multiprocessor_cache_preference_t) raw_preference;
	}

	inline void set_shared_memory_bank_size(shared_memory_bank_size_t new_bank_size) {
		device_setter set_device_for_this_scope(id_);
		auto status = cudaDeviceSetSharedMemConfig(new_bank_size);
		throw_if_error(status,
			"Setting the multiprocessor shared memory bank size for device " +
			std::to_string(id_));
	}

	inline shared_memory_bank_size_t shared_memory_bank_size() const {
		device_setter set_device_for_this_scope(id_);
		shared_memory_bank_size_t bank_size;
		auto status = cudaDeviceGetSharedMemConfig(&bank_size);
		throw_if_error(status,
			"Obtaining the multiprocessor shared memory bank size for device " +
			std::to_string(id_));
		return bank_size;
	}

	// For some reason, there is no cudaFuncGetCacheConfig. Weird.
	//
	// template <typename KernelFunction>
	// inline multiprocessor_cache_preference_t kernel_cache_preference(
	// 	const KernelFunction* kernel, multiprocessor_cache_preference_t preference);

	device::id_t  id() const { return id_; }
	// Note: There is _no_ method to make this device the current one,
	// since I do not want users to hang on to these objects and
	// manipulate them this way or that

	stream_t<AssumedCurrent> default_stream()
	{
		return stream_t<AssumedCurrent>(id_, stream::default_stream_id);
	}

	stream::id_t create_stream(
		stream::priority_t  priority = stream::default_priority,
		bool                synchronizes_with_default_stream = true)
	{
		return AssumedCurrent ?
			stream::create(priority, synchronizes_with_default_stream) :
			stream::create(id_, priority, synchronizes_with_default_stream);
	}

	stream_t<detail::do_not_assume_device_is_current> create_stream_proxy(
		stream::priority_t  priority = stream::default_priority,
		bool                synchronizes_with_default_stream = true)
	{
		return stream_t<AssumedCurrent>::create(id_, priority, synchronizes_with_default_stream);
	}

	template<typename KernelFunction, typename... KernelParameters>
	void launch(
		const KernelFunction&       kernel_function,
		launch_configuration_t      launch_configuration,
		KernelParameters...         parameters)
	{
		return default_stream().launch(kernel_function, launch_configuration, parameters...);
	}

	priority_range_t stream_priority_range() const
	{
		device_setter set_device_for_this_scope(id_);
		stream::priority_t least, greatest;
		auto status = cudaDeviceGetStreamPriorityRange(&least, &greatest);
		throw_if_error(status,
			"Failed obtaining stream priority range for CUDA device " + std::to_string(id_));
		return { least, greatest };
	}

	flags_t flags() const
	{
		device_setter set_device_for_this_scope(id_);
		flags_t flags;
		auto status = cudaGetDeviceFlags(&flags);
		throw_if_error(status,
			"Failed obtaining the flags for CUDA device " + std::to_string(id_));
		return flags;
	}

	host_thread_synch_scheduling_policy_t synch_scheduling_policy() const
	{
		return (host_thread_synch_scheduling_policy_t) (flags() & cudaDeviceScheduleMask);
	}

	void set_synch_scheduling_policy(host_thread_synch_scheduling_policy_t new_policy)
	{
		set_flags(flags() & (flags_t) new_policy);
	}

	bool keeping_local_mem_allocation_after_launch() const
	{
		return flags() & cudaDeviceLmemResizeToMax;
	}

	void keep_local_mem_allocation_after_launch(bool keep = true)
	{
		if (keep) { set_flags(flags() & cudaDeviceLmemResizeToMax); }
	}

	void discard_local_mem_allocation_after_launch()
	{
		set_flags(flags() & ~cudaDeviceLmemResizeToMax);
	}

	bool pinned_mapped_memory_allocation_is_allowed() const
	{
		return flags() & cudaDeviceMapHost;
	}

	void allow_pinned_mapped_memory_allocation(bool allow = true)
	{
		if (allow) { set_flags(flags() & cudaDeviceMapHost); }
	}

	void prevent_pinned_mapped_memory_allocation()
	{
		set_flags(flags() & ~cudaDeviceMapHost);
	}

	void set_flags(
		host_thread_synch_scheduling_policy_t synch_scheduling_policy,
		bool keep_local_mem_allocation_after_launch,
		bool allow_pinned_mapped_memory_allocation)
	{
		return set_flags((flags_t)
			  synch_scheduling_policy // this enum value is also a valid flags value
			| (keep_local_mem_allocation_after_launch ? cudaDeviceLmemResizeToMax : 0)
			| (allow_pinned_mapped_memory_allocation  ? cudaDeviceMapHost         : 0)
		);
	}

	void make_current() {
		static_assert(AssumedCurrent == false,
			"Attempt to manipulate current device from a device proxy assumed to be current");
		device::current::set(id_);
	}

	device_t& operator=(const device_t& other) = delete;
	// TODO: Consider limiting the copy ctor acess to prevent users from
	// messing up stuff; problem is, removing it prevents the device::get()
	// function from working and friend'ing it doesn't seem to work.
	// device_t(const device_t& other) = default;

public: // constructors and destructor

	// TODO: Consider making the ctor accessible only by the device::get() function,
	device_t(device::id_t  device_id) : id_(device_id), memory(device_id), peer_access(device_id) { };
	// TODO: How safe is it that we allow copy construction?
	~device_t() { };

protected: // data members
	const device::id_t   id_;

public: // faux data members (used as surrogates for internal namespaces)
	memory_t             memory;
	peer_access_t        peer_access;
};

namespace device {
namespace current {

/**
 * Returns the current device in a wrapper which assumes it is indeed
 * current, i.e. which will not set the device before performing any
 * other actions.
 */
inline device_t<detail::assume_device_is_current> get()
{
	return device_t<detail::assume_device_is_current>(current::get_id());
}

} // namespace current

/**
 * A convenience method to avoid explicitly constructing a device_t object
 * just for using it once. Thus, you could write:
 *
 *   cuda::device::get(my_favorite_device_id).synchronize();
 *
 * instead of
 *
 *   cuda::device_t d(my_favorite_device_id);
 *   d.synchronize();
 *
 * and instead of the non-wrapped-API equivalent:
 *
 *   cudaError_t status;
 *   int prev_device;
 *   status = cudaGetDevice();
 *   if (status != cudaSuccess) // error handling
 *   status cudaSetDevice(my_favorite_device_id);
 *   if (status != cudaSuccess) // error handling
 *   status = cudaDeviceSynchronize();
 *   if (status != cudaSuccess) // error handling
 *   status cudaSetDevice(prev_device);
 *   if (status != cudaSuccess) // error handling
 *
 * with the API directly.
 *
 * @param device_id Which device to "get"
 * @return a devie device_t object, which is only intended to call methods for,
 * not for manipulation/passing arround etc.
 */
inline device_t<detail::do_not_assume_device_is_current> get(device::id_t device_id)
{
	return device_t<detail::do_not_assume_device_is_current>(device_id);
}

} // namespace device

} // namespace cuda

std::istream& operator>>(std::istream& is, cuda::device::pci_id_t& pci_id);


#endif /* CUDA_DEVICE_H_ */
