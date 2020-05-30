/**
 * @file device_function.hpp
 *
 * @brief Functions for querying information and making settings
 * regarding device-side functions - kernels or otherwise.
 *
 * @note This file does _not_ have device-side functions itself,
 * nor is it about the device-side part of the runtime API (i.e.
 * API functions which may be called from the device).
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_DEVICE_FUNCTION_HPP_
#define CUDA_API_WRAPPERS_DEVICE_FUNCTION_HPP_

#include <cuda/api/types.hpp>
#include <cuda/api/device_properties.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/current_device.hpp>

#include <cuda_runtime_api.h>

namespace cuda {

///@cond
class device_t;
///@endcond

namespace device_function {

/**
 * @brief a wrapper around @ref cudaFuncAttributes, offering
 * a few convenience member functions.
 */
struct attributes_t : cudaFuncAttributes {

	cuda::device::compute_capability_t ptx_version() const noexcept {
		return device::compute_capability_t::from_combined_number(ptxVersion);
	}

	cuda::device::compute_capability_t binary_compilation_target_architecture() const noexcept {
		return device::compute_capability_t::from_combined_number(binaryVersion);
	}
};

/**
 * @brief Calculate the effective maximum size of allocatable (dynamic)
 * shared memory in a grid block
 *
 * @param attributes Attributes of the `__global__` kernel function
 * for which we wish to determine the allocation limit
 * @param compute_capability the GPU device's compute capability figure (e.g. 3.5
 * or 5.0), which fully determines the maximum allocation size
 */
inline memory::shared::size_t maximum_dynamic_shared_memory_per_block(
	attributes_t attributes, device::compute_capability_t compute_capability)
{
	auto available_without_static_allocation = compute_capability.max_shared_memory_per_block();
	auto statically_allocated_shared_mem = attributes.sharedSizeBytes;
	if (statically_allocated_shared_mem > available_without_static_allocation) {
		throw std::logic_error("More static shared memory has been allocated for a device function"
		" than seems to be available on devices with the specified compute capability.");
	}
	return available_without_static_allocation - statically_allocated_shared_mem;
}

} // namespace device_function

/**
 * A non-owning wrapper class for CUDA `__global__` functions
 *
 * @note The association of a `device_function_t` with an individual device is somewhat tenuous.
 * That is, the same function pointer could be used with any other device (provided the kernel
 * was compiled appropriately). However, many/most of the features, attributes and settings
 * are device-specific.
 */
class device_function_t {
public: // getters
	const void* ptr() const noexcept { return ptr_; }
	const device_t device() const noexcept;

protected:
	device::id_t device_id() const noexcept { return device_id_; }

public: // type_conversions
	operator const void*() noexcept { return ptr_; }

public: // non-mutators

	inline device_function::attributes_t attributes() const;

/*
	// The following are commented out because there are no CUDA API calls for them!
	// You may uncomment them if you'd rather get an exception...

	multiprocessor_cache_preference_t                cache_preference() const;
	multiprocessor_shared_memory_bank_size_option_t  shared_memory_bank_size() const;
*/

	/**
	 * @brief Calculates the number of grid blocks which may be "active" on a given GPU
	 * multiprocessor simultaneously (i.e. with warps from any of these block
	 * being schedulable concurrently)
	 *
	 * @param device
	 * @param device_function
	 * @param num_threads_per_block
	 * @param dynamic_shared_memory_per_block
	 * @param disable_caching_override On some GPUs, the choice of whether to
	 * cache memory reads affects occupancy. But what if this caching results in 0
	 * potential occupancy for a kernel? There are two options, controlled by this flag.
	 * When it is set to false - the calculator will assume caching is off for the
	 * purposes of its work; when set to true, it will return 0 for such device functions.
	 * See also the "Unified L1/Texture Cache" section of the
	 * <a href="http://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html">Maxwell
	 * tuning guide</a>.
	 */
	grid::dimension_t maximum_active_blocks_per_multiprocessor(
		grid::block_dimension_t   num_threads_per_block,
		memory::shared::size_t    dynamic_shared_memory_per_block,
		bool                      disable_caching_override = false);

public: // mutators

	void set_attribute(cudaFuncAttribute attribute, int value);

	/**
	 * @brief Change the hardware resource carve-out between L1 cache and shared memory
	 * for launches of the device_function to allow for at least the specified amount of
	 * shared memory.
	 *
	 * On several nVIDIA GPU micro-architectures, the L1 cache and the shared memory in each
	 * symmetric multiprocessor (=physical core) use the same hardware resources. The
	 * carve-out between the two uses has a device-wide value (which can be changed), but can
	 * also be set on the individual device-function level, by specifying the amount of shared
	 * memory the kernel may require.
	 */
	void opt_in_to_extra_dynamic_memory(cuda::memory::shared::size_t amount_required_by_kernel);

	/**
	 * @brief Indicate the desired carve-out between shared memory and L1 cache when launching
	 * this kernel - with fine granularity.
	 *
	 * On several nVIDIA GPU micro-architectures, the L1 cache and the shared memory in each
	 * symmetric multiprocessor (=physical core) use the same hardware resources. The
	 * carve-out between the two uses has a device-wide value (which can be changed), but the
	 * driver can set another value for a specific function. This function doesn't make a demand
	 * from the CUDA runtime (as in @p opt_in_to_extra_dynamic_memory), but rather indicates
	 * what is the fraction of L1 to shared memory it would like the kernel scheduler to carve
	 * out.
	 *
	 * @param shared_mem_percentage The percentage - from 0 to 100 - of the combined L1/shared
	 * memory space the user wishes to assign to shared memory.
	 *
	 * @note similar to @ref set_cache_preference() - but with finer granularity.
	 */
	void set_preferred_shared_mem_fraction(unsigned shared_mem_percentage);

	/**
	 * @brief Indicate the desired carve-out between shared memory and L1 cache when launching
	 * this kernel - with coarse granularity.
	 *
	 * On several nVIDIA GPU micro-architectures, the L1 cache and the shared memory in each
	 * symmetric multiprocessor (=physical core) use the same hardware resources. The
	 * carve-out between the two uses has a device-wide value (which can be changed), but the
	 * driver can set another value for a specific function. This function doesn't make a demand
	 * from the CUDA runtime (as in @p opt_in_to_extra_dynamic_memory), but rather indicates
	 * what is the fraction of L1 to shared memory it would like the kernel scheduler to carve
	 * out.
	 *
	 * @param preference one of: as much shared memory as possible, as much
	 * L1 as possible, or no preference (i.e. using the device default).
	 *
	 * @note similar to @ref set_preferred_shared_mem_fraction() - but with coarser granularity.
	 */
	void set_cache_preference(multiprocessor_cache_preference_t preference);

	/**
	 * @brief Sets a device function's preference of shared memory bank size preference
	 * (for the current device probably)
	 *
	 * @param device the CUDA device for execution on which the preference is set
	 * @param config bank size setting to make
	 */
	void set_shared_memory_bank_size(multiprocessor_shared_memory_bank_size_option_t config);


protected: // ctors & dtor
	device_function_t(device::id_t device_id, const void* f) : device_id_(device_id), ptr_(f)
	{
		// TODO: Consider checking whether this actually is a device function
		// TODO: Consider performing a check for nullptr
	}

public: // ctors & dtor
	template <typename DeviceFunction>
	device_function_t(const device_t& device, DeviceFunction f);
	~device_function_t() { };

protected: // data members
	const device::id_t device_id_;
	const void* const ptr_;
};

namespace device_function {




/**
 * @brief A 'version' of
 * @ref cuda::compute_capability_t::maximum_dynamic_shared_memory_per_block()
 * for use with a specific device function - which will take its use of
 * static shared memory into account.
 *
 * @param device_function The (`__global__` or `__device__`)
 * function for which to calculate
 * the effective available shared memory per block
 * @param compute_capability on which kind of device the kernel function is to
 * be launched;
 * @return the maximum amount of shared memory per block which a launch of the
 * specified function can require

 * @todo It's not clear whether this is actually necessary given the {@ref device_function_t}
 * pointer.
 *
 */
inline memory::shared::size_t maximum_dynamic_shared_memory_per_block(
	const device_function_t& device_function, device::compute_capability_t compute_capability)
{
	return device_function::maximum_dynamic_shared_memory_per_block(
		device_function.attributes(), compute_capability);
}


namespace detail {

template<bool...> struct bool_pack;
template<bool... bs>
using all_true = std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

template<typename... KernelParameters>
struct raw_device_function_typegen {
	static_assert(all_true<std::is_same<KernelParameters, ::cuda::detail::kernel_parameter_decay_t<KernelParameters>>::value...>::value,
		"Invalid kernel parameter types" );
	using type = void(*)(KernelParameters...);
		// Why no decay? After all, CUDA kernels only takes parameters by value, right?
		// Well, we're inside `detail::`. You shouldn't call this
		// nice simple types we can use with CUDA kernels; you should not pass such
		// parameter types here in the first place.
};

template<typename KernelFunction, typename... KernelParameters>
typename raw_device_function_typegen<KernelParameters...>::type unwrap_inner(std::true_type, device_function_t wrapped)
{
	using raw_device_function_t = typename raw_device_function_typegen<KernelParameters ...>::type;
	return reinterpret_cast<raw_device_function_t>(wrapped.ptr());
}

template<typename KernelFunction, typename... KernelParameters>
KernelFunction unwrap_inner(std::false_type, KernelFunction raw_function)
{
	static_assert(
		std::is_function<typename std::decay<KernelFunction>::type>::value or
		(std::is_pointer<KernelFunction>::value and  std::is_function<typename std::remove_pointer<KernelFunction>::type>::value)
		, "Invalid KernelFunction type - it must be either a function or a pointer-to-a-function");
	return raw_function;
}

} // namespace detail

template<typename KernelFunction, typename... KernelParameters>
auto unwrap(KernelFunction f) -> typename std::conditional<
	std::is_same<typename std::decay<KernelFunction>::type, device_function_t>::value,
	typename detail::raw_device_function_typegen<KernelParameters...>::type,
	KernelFunction>::type
{
	using got_a_device_function_t =
		std::integral_constant<bool, std::is_same<typename std::decay<KernelFunction>::type, device_function_t>::value>;
	return detail::unwrap_inner<KernelFunction, KernelParameters...>(got_a_device_function_t{}, f);
}

} // namespace device_function
} // namespace cuda

#endif // CUDA_API_WRAPPERS_DEVICE_FUNCTION_HPP_
