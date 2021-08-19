/**
 * @file kernel.hpp
 *
 * @brief Functions for querying information and making settings
 * regarding CUDA kernels (`__global__` functions).
 *
 * @note This file does _not_ define any kernels  itself.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_KERNEL_HPP_
#define CUDA_API_WRAPPERS_KERNEL_HPP_

#include <cuda/api/current_device.hpp>
#include <cuda/api/device_properties.hpp>
#include <cuda/api/error.hpp>
#include <cuda/common/types.hpp>

#include <cuda_runtime_api.h>

namespace cuda {

///@cond
class device_t;
class stream_t;
///@endcond

namespace kernel {

/**
 * @brief a wrapper around `cudaFuncAttributes`, offering
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

} // namespace kernel

/**
 * A non-owning wrapper class for CUDA `__global__` functions
 *
 * @note The association of a `kernel_t` with an individual device is somewhat tenuous.
 * That is, the same function pointer could be used with any other device (provided the kernel
 * was compiled appropriately). However, many/most of the features, attributes and settings
 * are device-specific.
 */
class kernel_t {
public: // getters
	const void* ptr() const noexcept { return ptr_; }
	device_t device() const noexcept;
	bool thread_block_cooperation() const noexcept { return thread_block_cooperation_; }

protected:
	device::id_t device_id() const noexcept { return device_id_; }

public: // type_conversions
	operator const void*() noexcept { return ptr_; }

public: // non-mutators

	inline kernel::attributes_t attributes() const;

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
	 * for launches of the kernel to allow for at least the specified amount of
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
	 *
	 * @param dynamic_shared_memory_size The amount of dynamic shared memory each grid block will
	 * need.
	 * @param block_size_limit do not return a block size above this value; the default, 0,
	 * means no limit on the returned block size.
	 * @param disable_caching_override On platforms where global caching affects occupancy,
	 * and when enabling caching would result in zero occupancy, the occupancy calculator will
	 * calculate the occupancy as if caching is disabled. Setting this to true makes the
	 * occupancy calculator return 0 in such cases. More information can be found about this
	 * feature in the "Unified L1/Texture Cache" section of the
	 * <a href="https://docs.nvidia.com/cuda/maxwell-tuning-guide/index.html">Maxwell tuning guide</a>.
	 *
	 * @return A pair, with the second element being the maximum achievable block size
	 * (1-dimensional), and the first element being the minimum number of such blocks necessary
	 * for keeping the GPU "busy" (again, in a 1-dimensional grid).
	 */
	::std::pair<grid::dimension_t, grid::block_dimension_t>
	min_grid_params_for_max_occupancy(
		memory::shared::size_t   dynamic_shared_memory_size = no_dynamic_shared_memory,
		grid::block_dimension_t  block_size_limit = 0,
		bool                     disable_caching_override = false);

	template <typename UnaryFunction>
	::std::pair<grid::dimension_t, grid::block_dimension_t>
	min_grid_params_for_max_occupancy(
		UnaryFunction            block_size_to_dynamic_shared_mem_size,
		grid::block_dimension_t  block_size_limit = 0,
		bool                     disable_caching_override = false);

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
	 * @param config bank size setting to make
	 */
	void set_shared_memory_bank_size(multiprocessor_shared_memory_bank_size_option_t config);


protected: // ctors & dtor
	kernel_t(device::id_t device_id, const void* f, bool thread_block_cooperation = false)
	: device_id_(device_id), ptr_(f), thread_block_cooperation_(thread_block_cooperation)
	{
		// TODO: Consider checking whether this actually is a device function
		// TODO: Consider performing a check for nullptr
	}

public: // ctors & dtor
	template <typename DeviceFunction>
	kernel_t(const device_t& device, DeviceFunction f, bool thread_block_cooperation = false);
	~kernel_t() { };

protected: // data members
	const device::id_t device_id_;
	const void* const ptr_;
	const bool thread_block_cooperation_;
};

namespace kernel {

namespace detail_ {

template<bool...> struct bool_pack;
template<bool... bs>
using all_true = ::std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

template<typename... KernelParameters>
struct raw_kernel_typegen {
	static_assert(all_true<::std::is_same<KernelParameters, ::cuda::detail_::kernel_parameter_decay_t<KernelParameters>>::value...>::value,
		"Invalid kernel parameter types" );
	using type = void(*)(KernelParameters...);
		// Why no decay? After all, CUDA kernels only takes parameters by value, right?
		// Well, we're inside `detail_::`. You should be careful to only instantiate this class with
		// nice simple types we can pass to CUDA kernels.
};

template<typename Kernel, typename... KernelParameters>
typename raw_kernel_typegen<KernelParameters...>::type unwrap_inner(::std::true_type, kernel_t wrapped)
{
	using raw_kernel_t = typename raw_kernel_typegen<KernelParameters ...>::type;
	return reinterpret_cast<raw_kernel_t>(const_cast<void*>(wrapped.ptr()));
		// The inner cast here is because we store the pointer as const void* - as an extra precaution
		// against anybody trying to write through it. Now, function pointers can't get written through,
		// but are still for some reason not considered const.
}

template<typename Kernel, typename... KernelParameters>
Kernel unwrap_inner(::std::false_type, Kernel raw_function)
{
	static_assert(
		::std::is_function<typename ::std::decay<Kernel>::type>::value or
		(::std::is_pointer<Kernel>::value and ::std::is_function<typename ::std::remove_pointer<Kernel>::type>::value)
		, "Invalid Kernel type - it must be either a function or a pointer-to-a-function");
	return raw_function;
}

} // namespace detail_

/**
 * Obtain the raw function pointer of any type acceptable as a launchable kernel
 * by {@ref enqueue_launch}.
 */
template<typename Kernel, typename... KernelParameters>
auto unwrap(Kernel f) -> typename ::std::conditional<
	::std::is_same<typename ::std::decay<Kernel>::type, kernel_t>::value,
	typename detail_::raw_kernel_typegen<KernelParameters...>::type,
	Kernel>::type
{
	using got_a_kernel_t =
		::std::integral_constant<bool, ::std::is_same<typename ::std::decay<Kernel>::type, kernel_t>::value>;
	return detail_::unwrap_inner<Kernel, KernelParameters...>(got_a_kernel_t{}, f);
}

} // namespace kernel

} // namespace cuda

#endif // CUDA_API_WRAPPERS_KERNEL_HPP_
