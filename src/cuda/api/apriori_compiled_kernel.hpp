/**
 * @file apriori_compiled_kernel.hpp
 *
 * @brief An implementation of a subclass of @ref `kernel_t` for kernels
 * compiled together with the host-side program.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_APRIORI_COMPILED_KERNEL_HPP_
#define CUDA_API_WRAPPERS_APRIORI_COMPILED_KERNEL_HPP_

#include <cuda/api/kernel.hpp>
#include <cuda/api/current_context.hpp>
#include <type_traits>

namespace cuda {

///@cond
class device_t;
class apriori_compiled_kernel_t;
///@nocond

namespace kernel {

namespace detail_ {

inline handle_t get_handle(const void *kernel_function_ptr, const char* name = nullptr)
{
	handle_t handle;
	auto status = cudaGetFuncBySymbol(&handle, kernel_function_ptr);
	throw_if_error(status, "Failed obtaining a CUDA function handle for "
		+ ((name == nullptr) ? ::std::string("a kernel function") : ::std::string("kernel function ") + name)
		+ " at " + cuda::detail_::ptr_as_hex(kernel_function_ptr));
	return handle;
}

apriori_compiled_kernel_t wrap(
	device::id_t device_id,
	context::handle_t context_id,
	kernel::handle_t f,
	const void* ptr);


} // namespace detail_
} // namespace kernel

/**
 * @brief A subclass of the @ref `kernel_t` interface for kernels being
 * functions marked as __global__ in source files and compiled apriori.
 */
class apriori_compiled_kernel_t final : public kernel_t {
public: // getters
	const void *ptr() const noexcept { return ptr_; }
	const void *get() const noexcept { return ptr_; }

public: // type_conversions
	explicit operator const void *() noexcept { return ptr_; }

public: // non-mutators

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
		grid::block_dimension_t num_threads_per_block,
		memory::shared::size_t dynamic_shared_memory_per_block,
		bool disable_caching_override = false);

protected: // ctors & dtor
	apriori_compiled_kernel_t(device::id_t device_id, context::handle_t context_handle,
		kernel::handle_t handle, const void *f)
		: kernel_t(device_id, context_handle, handle), ptr_(f) {
		// TODO: Consider checking whether this actually is a device function, at all and in this context
#ifndef NDEBUG
		assert(f != nullptr && "Attempt to construct a kernel object for a nullptr kernel function pointer");
#endif
	}
	apriori_compiled_kernel_t(device::id_t device_id, context::handle_t context_handle, const void *f)
		: apriori_compiled_kernel_t(device_id, context_handle, kernel::detail_::get_handle(f), f) { }

public: // ctors & dtor
	apriori_compiled_kernel_t(const apriori_compiled_kernel_t&) = default;
	apriori_compiled_kernel_t(apriori_compiled_kernel_t&&) = default;

public: // friends
	friend apriori_compiled_kernel_t kernel::detail_::wrap(device::id_t, context::handle_t, kernel::handle_t, const void*);

protected: // data members
	const void *const ptr_;
};

inline grid::dimension_t apriori_compiled_kernel_t::maximum_active_blocks_per_multiprocessor(
	grid::block_dimension_t num_threads_per_block,
	memory::shared::size_t dynamic_shared_memory_per_block,
	bool disable_caching_override)
{
	context::current::detail_::scoped_override_t set_context_for_this_context(context_handle_);
	int result;
	unsigned int flags = disable_caching_override ?
						 cudaOccupancyDisableCachingOverride : cudaOccupancyDefault;
	auto status = cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
		&result, ptr_, (int) num_threads_per_block,
		dynamic_shared_memory_per_block, flags);
	throw_if_error(status, "Failed calculating the maximum occupancy "
						   "of device function blocks per multiprocessor");
	return result;
}

namespace kernel {
namespace detail_ {

inline apriori_compiled_kernel_t wrap(
	device::id_t device_id,
	context::handle_t context_id,
	kernel::handle_t f,
	const void *ptr)
{
	return {device_id, context_id, f, ptr};
}

} // namespace detail

template<typename KernelFunctionPtr>
apriori_compiled_kernel_t get(device_t device, KernelFunctionPtr function_ptr);

template<typename KernelFunctionPtr>
apriori_compiled_kernel_t get(context_t context, KernelFunctionPtr function_ptr);

} // namespace kernel

} // namespace cuda

#endif // CUDA_API_WRAPPERS_APRIORI_COMPILED_KERNEL_HPP_
