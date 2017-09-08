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

namespace cuda {

class event_t;
template <bool DeviceAssumedCurrent> class stream_t;

namespace device {

template <bool AssumedCurrent = detail::do_not_assume_device_is_current> class device_t;

namespace peer_to_peer {

using attribute_value_t = int;
using attribute_t       = cudaDeviceP2PAttr;

inline bool can_access(id_t accessor, id_t peer)
{
	int result;
	auto status = cudaDeviceCanAccessPeer(&result, accessor, peer);
	throw_if_error(status,
		"Failed determining whether CUDA device " + std::to_string(accessor) +
		" can access CUDA device " + std::to_string(peer));
	return (result == 1);
}


template <bool FirstIsAssumedCurrent, bool SecondIsAssumedCurrent>
inline bool can_access(
	const device_t<FirstIsAssumedCurrent>& accessor,
	const device_t<SecondIsAssumedCurrent>& peer);



inline attribute_value_t get_attribute(
	attribute_t   attribute,
	id_t          source,
	id_t          destination)
{
	attribute_value_t value;
	auto status = cudaDeviceGetP2PAttribute(&value, attribute, source, destination);
	throw_if_error(status,
		"Failed obtaining peer-to-peer device attribute for device pair ("
		+ std::to_string(source) + ", " + std::to_string(destination) + ')');
	return value;
}

template <bool FirstIsAssumedCurrent, bool SecondIsAssumedCurrent>
inline attribute_value_t get_attribute(
	attribute_t                              attribute,
	const device_t<FirstIsAssumedCurrent >&  source,
	const device_t<SecondIsAssumedCurrent>&  destination);

} // namespace peer_to_peer

namespace current {

/**
 * Returns the current device in a wrapper which assumes it is indeed
 * current, i.e. which will not set the current device before performing any
 * other actions.
 */
inline device_t<detail::do_not_assume_device_is_current> get(device::id_t device_id);

} // namespace current

} // namespace device

// TODO: Consider making this a multiton, with one possible instance per device_id
template <bool AssumedCurrent = detail::do_not_assume_device_is_current>
class device_t {
public: // types
	using scoped_setter              = device::current::scoped_override_t<AssumedCurrent>;

protected: // types
	using properties_t               = device::properties_t;
	using attribute_t                = device::attribute_t;
	using attribute_value_t          = device::attribute_value_t;
	using flags_t                    = unsigned;
	using resource_id_t              = cudaLimit;
	using resource_limit_t           = size_t;
	using shared_memory_bank_size_t  = cudaSharedMemConfig;
	using priority_range_t           = std::pair<stream::priority_t, stream::priority_t>;

	// This relies on valid CUDA device IDs being non-negative, which is indeed
	// always the case for CUDA <= 8.0 and unlikely to change; however, it's
	// a bit underhanded to make that assumption just to twist this class to do
	// our bidding (see below)
	enum : device::id_t { invalid_id = -1 };

	struct immutable_id_holder {
		operator const device::id_t&() const { return id; }
		void set(device::id_t) const { };
		device::id_t id;
	};
	struct mutable_id_holder {
		operator const device::id_t&() const { return id; }
		void set(device::id_t new_id) const { id = new_id; };
		mutable device::id_t id { invalid_id };
	};

	using id_holder_type              = typename std::conditional<
		AssumedCurrent, mutable_id_holder, immutable_id_holder>::type;
		// class device_t has a field with the device's id; when assuming the device is the
		// current one, we want to be able to set the device_id after construction; but
		// when not making that assumption, we need a const field, initialized on construction;
		// this hack allows both the options to coexist without the compiler complaining
		// about assigning to a const lvalue (we use it since std::conditional can't
		// be used to select between making a field mutable or not).

public: // types

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

		memory_t(const device_t::id_holder_type& id) : device_id(id) {}

		template <typename T = void>
		__host__ T* allocate(size_t size_in_bytes)
		{
			scoped_setter set_device_for_this_scope(device_id);
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
		// TODO: managed - similar to owner
		template <typename T = void>
		__host__ T* allocate_managed(
			size_t size_in_bytes, bool initially_visible_to_host_only = false)
		{
			scoped_setter set_device_for_this_scope(device_id);
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
				" bytes of global memory on " + device_id_as_str());
			return allocated;
		}

		using region_pair = ::cuda::memory::mapped::region_pair;
		template <typename T = void>
		__host__ region_pair allocate_region_pair(
			size_t size_in_bytes, region_pair::allocation_options options = {false, false})
		{
			scoped_setter set_device_for_this_scope(device_id);
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
				std::to_string(size_in_bytes) + " bytes of global memory on " +
				device_id_as_str());
			return allocated;
		}

		size_t amount_total() const
		{
			scoped_setter set_device_for_this_scope(device_id);
			size_t total_mem_in_bytes;
			auto status = cudaMemGetInfo(nullptr, &total_mem_in_bytes);
			throw_if_error(status,
				std::string("Failed determining amount of total memory for CUDA device ") +
				std::to_string(device_id));
			return total_mem_in_bytes;
		}

		size_t amount_free() const
		{
			scoped_setter set_device_for_this_scope(device_id);
			size_t free_mem_in_bytes;
			auto status = cudaMemGetInfo(&free_mem_in_bytes, nullptr);
			throw_if_error(status,
				"Failed determining amount of free memory for " +
				device_id_as_str());
			return free_mem_in_bytes;
		}
	}; // class memory_t

	class peer_access_t {
	protected:
		const device_t::id_holder_type& device_id;

		std::string device_id_as_str() const
		{
			return device_t::device_id_as_str(device_id);
		}

	public:
		peer_access_t(const device_t::id_holder_type& holder) : device_id(holder) {}

		bool can_access(id_t peer_id) const
		{
			scoped_setter set_device_for_this_scope(device_id);
			return device::peer_to_peer::can_access(device_id, peer_id);
		}

		// This won't compile, for some reason:
		//
		// inline bool can_access(const device_t<detail::do_not_assume_device_is_current>& peer) const
		// {
		// 	return this->can_access(peer.id());
		// }


		void enable_to(id_t peer_id) {
			scoped_setter set_device_for_this_scope(device_id);
			enum : unsigned { fixed_flags = 0}; // No flags are supported as of CUDA 8.0
			auto status = cudaDeviceEnablePeerAccess(peer_id, fixed_flags);
			throw_if_error(status,
				"Failed enabling access of " + device_id_as_str() +
				" to device " + std::to_string(peer_id));
		}

		void disable_to(id_t peer_id) {
			scoped_setter set_device_for_this_scope(device_id);
			auto status = cudaDeviceDisablePeerAccess(peer_id);
			throw_if_error(status,
				"Failed disabling access of " + device_id_as_str() +
				" to device " + std::to_string(peer_id));

		}
	}; // class peer_access_t


protected:
	void set_flags(flags_t new_flags)
	{
		scoped_setter set_device_for_this_scope(id_);
		auto status = cudaSetDeviceFlags(new_flags);
		throw_if_error(status,
			"Failed setting the flags for " + device_id_as_str());
	}

	void set_flags(
		host_thread_synch_scheduling_policy_t synch_scheduling_policy,
		bool keep_local_mem_allocation_after_launch,
		bool allow_pinned_mapped_memory_allocation)
	{
		set_flags((flags_t)
			  synch_scheduling_policy // this enum value is also a valid flags value
			| (keep_local_mem_allocation_after_launch ? cudaDeviceLmemResizeToMax : 0)
			| (allow_pinned_mapped_memory_allocation  ? cudaDeviceMapHost         : 0)
		);
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
		scoped_setter set_device_for_this_scope(id_);
		flags_t flags;
		auto status = cudaGetDeviceFlags(&flags);
		throw_if_error(status,
			"Failed obtaining the flags for  " + device_id_as_str());
		return flags;
	}

public:
	// TODO: Add free() and free_region_pair() as methods?

	properties_t properties() const
	{
		properties_t properties;
		auto status = cudaGetDeviceProperties(&properties, id());
		throw_if_error(status,
			"Failed obtaining device properties for " + device_id_as_str());
		return properties;
	}

	std::string name() const {
		// I could get the name directly, but that would require
		// direct use of the driver, and I'm not ready for that
		// just yet
		return properties().name;
	}
	device::pci_id_t pci_id() const
	{
		auto pci_domain_id = get_attribute(cudaDevAttrPciDomainId);
		auto pci_bus_id    = get_attribute(cudaDevAttrPciBusId);
		auto pci_device_id = get_attribute(cudaDevAttrPciDeviceId);
		return { pci_domain_id, pci_bus_id, pci_device_id };
	}

	device::compute_capability_t compute_capability() const
	{
		auto major = get_attribute(cudaDevAttrComputeCapabilityMajor);
		auto minor = get_attribute(cudaDevAttrComputeCapabilityMinor);
		return { major, minor };
	}

	/**
	 * Yes, properties vs attributes is a bit confusing, but that's what CUDA has.
	 */
	attribute_value_t get_attribute(attribute_t attribute) const
	{
		attribute_value_t attribute_value;
		auto ret = cudaDeviceGetAttribute(&attribute_value, attribute, id());
		throw_if_error(ret,
			"Failed obtaining device properties for " + device_id_as_str());
		return attribute_value;
	}

	bool supports_concurrent_managed_access() const
	{
		return (get_attribute(cudaDevAttrConcurrentManagedAccess) != 0);
	}

	resource_limit_t get_resource_limit(resource_id_t resource) const
	{
		resource_limit_t limit;
		auto status = cudaDeviceGetLimit(&limit, resource);
		throw_if_error(status,
			"Failed obtaining a resource limit for " + device_id_as_str());
		return limit;
	}

	void set_resource_limit(resource_id_t resource, resource_limit_t new_limit)
	{
		auto status = cudaDeviceSetLimit(resource, new_limit);
		throw_if_error(status,
			"Failed setting a resource limit for  " + device_id_as_str());
	}

	inline void synchronize()
	{
		scoped_setter set_device_for_this_scope(id_);
		auto status = cudaDeviceSynchronize();
		throw_if_error(status, "Failed synchronizing " + device_id_as_str());
	}
	inline void synchronize_stream(stream::id_t stream_id)
	{
		scoped_setter set_device_for_this_scope(id_);
		auto status = cudaStreamSynchronize(stream_id);
		throw_if_error(status, "Failed synchronizing a stream on " + device_id_as_str());
	}

	inline void synchronize(stream_t<detail::do_not_assume_device_is_current>& stream);

	inline void synchronize_event(event::id_t event_id)
	{
		scoped_setter set_device_for_this_scope(id_);
		auto status = cudaEventSynchronize(event_id);
		throw_if_error(status, "Failed synchronizing an event on   " + device_id_as_str());
	}

	inline void synchronize(event_t& event);

	inline void reset()
	{
		scoped_setter set_device_for_this_scope(id_);
		status_t status = cudaDeviceReset();
		throw_if_error(status, "Resetting  " + device_id_as_str());
	}

	inline void set_cache_preference(multiprocessor_cache_preference_t preference) {
		scoped_setter set_device_for_this_scope(id_);
		auto status = cudaDeviceSetCacheConfig((cudaFuncCache) preference);
		throw_if_error(status,
			"Setting the multiprocessor L1/Shared Memory cache distribution preference for " +
			 device_id_as_str());
	}

	inline multiprocessor_cache_preference_t cache_preference() const {
		scoped_setter set_device_for_this_scope(id_);
		cudaFuncCache raw_preference;
		auto status = cudaDeviceGetCacheConfig(&raw_preference);
		throw_if_error(status,
			"Obtaining the multiprocessor L1/Shared Memory cache distribution preference for " +
			 device_id_as_str());
		return (multiprocessor_cache_preference_t) raw_preference;
	}

	inline void set_shared_memory_bank_size(shared_memory_bank_size_t new_bank_size) {
		scoped_setter set_device_for_this_scope(id_);
		auto status = cudaDeviceSetSharedMemConfig(new_bank_size);
		throw_if_error(status,
			"Setting the multiprocessor shared memory bank size for " + device_id_as_str());
	}

	inline shared_memory_bank_size_t shared_memory_bank_size() const {
		scoped_setter set_device_for_this_scope(id_);
		shared_memory_bank_size_t bank_size;
		auto status = cudaDeviceGetSharedMemConfig(&bank_size);
		throw_if_error(status,
			"Obtaining the multiprocessor shared memory bank size for  " + device_id_as_str());
		return bank_size;
	}

	// For some reason, there is no cudaFuncGetCacheConfig. Weird.
	//
	// template <typename KernelFunction>
	// inline multiprocessor_cache_preference_t kernel_cache_preference(
	// 	const KernelFunction* kernel, multiprocessor_cache_preference_t preference);

	device::id_t id() const
	{
		if (AssumedCurrent) {
			// This is the first time in which we need to get the device ID
			// for the current device - as we've constructed this instance of
			// device_t<assume_device_is_current> without the ID
			if (id_.id == invalid_id) { id_.set(device::current::get_id()); }
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
	stream_t<detail::do_not_assume_device_is_current> create_stream(
		bool                will_synchronize_with_default_stream,
		stream::priority_t  priority = cuda::stream::default_priority);

	template<typename KernelFunction, typename... KernelParameters>
	void launch(
		const KernelFunction&       kernel_function,
		launch_configuration_t      launch_configuration,
		KernelParameters...         parameters)
	{
		return default_stream().enqueue.kernel_launch(kernel_function, launch_configuration, parameters...);
	}

	priority_range_t stream_priority_range() const
	{
		scoped_setter set_device_for_this_scope(id_);
		stream::priority_t least, greatest;
		auto status = cudaDeviceGetStreamPriorityRange(&least, &greatest);
		throw_if_error(status,
			"Failed obtaining stream priority range for " + device_id_as_str());
		return { least, greatest };
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

	bool keeping_local_mem_allocation_after_launch() const
	{
		return flags() & cudaDeviceLmemResizeToMax;
	}

	void keep_local_mem_allocation_after_launch(bool keep = true)
	{
		auto flags_ =  flags();
		if (keep) { flags_ |= cudaDeviceLmemResizeToMax; }
		else      { flags_ &= ~cudaDeviceLmemResizeToMax; }
		set_flags(flags_);
	}

	void discard_local_mem_allocation_after_launch()
	{
		keep_local_mem_allocation_after_launch(false);
	}

	bool can_map_host_memory() const
	{
		return flags() & cudaDeviceMapHost;
	}

	void enable_mapping_host_memory(bool allow = true)
	{
		auto flags_ =  flags();
		if (allow) { flags_ |= cudaDeviceMapHost; }
		else       { flags_ &= ~cudaDeviceMapHost; }
		set_flags(flags_);
	}

	void disable_mapping_host_memory()
	{
		enable_mapping_host_memory(false);
	}

public:
	// I'm of two minds regarding whether or not we should have this method at all;
	// I'm worried people might assume the proxy object will start behaving like
	// what they get with cuda::device::current::get(), i.e. a device_t<AssumedCurrent>.
	void make_current() {
		static_assert(AssumedCurrent == false,
			"Attempt to manipulate current device from a device proxy assumed to be current");
		device::current::set(id());
		// But note this change will not make methods of this device_t forego their
		// cudaSetDevice()'ing !
	}

	device_t& operator=(const device_t& other) = delete;

public: // constructors and destructor

	// TODO: Consider making the ctor accessible only by the device::get()
	// and device::current::get() functions
	device_t(device::id_t  device_id) : id_({device_id}) { }
	~device_t() { };

protected: // constructors
	/**
	 * @note Have a look at how mutable_id_holder is default-constructed.
	 */
	device_t() : id_()
	{
		static_assert(AssumedCurrent,
			"Attempt to instantiate a device proxy for a device not known to be "
			"current, without a specific device id");
	}

	friend device_t<detail::assume_device_is_current> device::current::get();

protected: // data members
	const id_holder_type id_;

public: // faux data members (used as surrogates for internal namespaces)
	memory_t             memory { id_ };
	peer_access_t        peer_access { id_ };
		// don't worry, these two will not actually use the id_ value
		// without making sure it's been correctly determined
};

template<bool LHSAssumedCurrent, bool RHSAssumedCurrent>
bool operator==(
	const device_t<LHSAssumedCurrent>& lhs,
	const device_t<RHSAssumedCurrent>& rhs)
{
	return lhs.id() == rhs.id();
}
template<bool LHSAssumedCurrent, bool RHSAssumedCurrent>
bool operator!=(
	const device_t<LHSAssumedCurrent>& lhs,
	const device_t<RHSAssumedCurrent>& rhs)
{
	return lhs.id() != rhs.id();
}


namespace device {
namespace current {

inline cuda::device_t<detail::assume_device_is_current> get()
{
	return cuda::device_t<detail::assume_device_is_current>();
}

} // namespace current

/**
 * A convenience method to avoid "knowing" about device_t's, so you can write:
 *
 *   cuda::device::get(my_favorite_device_id).synchronize();
 *
 * instead of
 *
 *   cuda::device_t(my_favorite_device_id).synchronize();
 *
 */
inline cuda::device_t<detail::do_not_assume_device_is_current> get(device::id_t device_id)
{
	return cuda::device_t<detail::do_not_assume_device_is_current>(device_id);
}

inline cuda::device_t<detail::do_not_assume_device_is_current> get(pci_id_t pci_id)
{
	auto resolved_id = pci_id.resolve_device_id();
	return get(resolved_id);
}

inline cuda::device_t<detail::do_not_assume_device_is_current> get(const std::string& pci_id_str)
{
	return get(pci_id_t::parse(pci_id_str));
}



namespace peer_to_peer {

template <bool FirstIsAssumedCurrent, bool SecondIsAssumedCurrent>
inline bool can_access(
	const device_t<FirstIsAssumedCurrent>& accessor,
	const device_t<SecondIsAssumedCurrent>& peer)
{
	return can_access(accessor.id(), peer.id());
}

template <bool FirstIsAssumedCurrent, bool SecondIsAssumedCurrent>
inline attribute_value_t get_attribute(
	attribute_t                              attribute,
	const device_t<FirstIsAssumedCurrent >&  source,
	const device_t<SecondIsAssumedCurrent>&  destination)
{
	return get_attribute(attribute, source.id(), destination.id());
}

} // namespace peer_to_peer




} // namespace device


namespace memory {
namespace device {

template<typename T, bool AssumedCurrent = cuda::detail::do_not_assume_device_is_current>
inline unique_ptr<T> make_unique(device_t<AssumedCurrent>& device, size_t n)
{
	typename device_t<AssumedCurrent>::scoped_setter set_device_for_this_scope(device.id());
	return cuda::memory::detail::make_unique<T, detail::allocator, detail::deleter>(n);
}

template<typename T, bool AssumedCurrent = cuda::detail::do_not_assume_device_is_current>
inline unique_ptr<T> make_unique(device_t<AssumedCurrent>& device)
{
	typename device_t<AssumedCurrent>::scoped_setter set_device_for_this_scope(device.id());
	return cuda::memory::detail::make_unique<T, detail::allocator, detail::deleter>();
}

} // namespace device
} // namespace memory

} // namespace cuda

#endif /* CUDA_API_WRAPPERS_DEVICE_HPP_ */
