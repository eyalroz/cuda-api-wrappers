/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * of kernel-launch-related functions.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_LAUNCH_HPP_
#define MULTI_WRAPPER_IMPLS_LAUNCH_HPP_

#include "kernel.hpp"
#include "../types.hpp"
#include "../memory.hpp"
#include "../stream.hpp"
#include "../kernel_launch.hpp"
#include "../pointer.hpp"
#include "../device.hpp"

namespace cuda {

namespace detail_ {

template<typename... KernelParameters>
void enqueue_launch_helper<apriori_compiled_kernel_t, KernelParameters...>::operator()(
	const apriori_compiled_kernel_t&  wrapped_kernel,
	const stream_t &                  stream,
	launch_configuration_t            launch_configuration,
	KernelParameters &&...            parameters) const
{
	using raw_kernel_t = typename kernel::detail_::raw_kernel_typegen<KernelParameters ...>::type;
	auto unwrapped_kernel_function = reinterpret_cast<raw_kernel_t>(const_cast<void *>(wrapped_kernel.ptr()));
	// Notes:
	// 1. The inner cast here is because we store the pointer as const void* - as an extra
	//    precaution against anybody trying to write through it. Now, function pointers
	//    can't get written through, but are still for some reason not considered const.
	// 2. We rely on the caller providing us with more-or-less the correct parameters -
	//    corresponding to the compiled kernel function's. I say "more or less" because the
	//    `KernelParameter` pack may contain some references, arrays and so on - which CUDA
	//    kernels cannot accept; so we massage those a bit.

	detail_::enqueue_raw_kernel_launch_in_current_context(
		unwrapped_kernel_function,
		stream.handle(),
		launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

template<typename... KernelParameters>
::std::array<const void*, sizeof...(KernelParameters)>
marshal_dynamic_kernel_arguments(KernelParameters&&... parameters)
{
	return ::std::array<const void*, sizeof...(KernelParameters)> { &parameters... };
}


// Note: The last (valid) element of marshalled_arguments must be null
inline void launch_type_erased_in_current_context(
	kernel::handle_t        kernel_function_handle,
	device::id_t            device_id,
	context::handle_t       context_handle,
	stream::handle_t        stream_handle,
	launch_configuration_t  launch_config,
	const void**            marshalled_arguments)
{
	status_t status;
	const auto&lc = launch_config; // alias for brevity
	if (launch_config.block_cooperation)
		status = cuLaunchCooperativeKernel(
			kernel_function_handle,
			lc.dimensions.grid.x,  lc.dimensions.grid.y,  lc.dimensions.grid.z,
			lc.dimensions.block.x, lc.dimensions.block.y, lc.dimensions.block.z,
			lc.dynamic_shared_memory_size,
			stream_handle,
			const_cast<void**>(marshalled_arguments)
		);
	else {
		static constexpr const auto no_arguments_in_alternative_format = nullptr;
		// TODO: Consider passing marshalled_arguments in the alternative format
		status = cuLaunchKernel(
			kernel_function_handle,
			lc.dimensions.grid.x,  lc.dimensions.grid.y,  lc.dimensions.grid.z,
			lc.dimensions.block.x, lc.dimensions.block.y, lc.dimensions.block.z,
			lc.dynamic_shared_memory_size,
			stream_handle,
			const_cast<void**>(marshalled_arguments),
			no_arguments_in_alternative_format
		);
	}
	throw_if_error_lazy(status,
		(lc.block_cooperation ? "Cooperative " : "") +
		::std::string(" kernel launch failed for ") + kernel::detail_::identify(kernel_function_handle)
		+ " on " + stream::detail_::identify(stream_handle, context_handle, device_id));
}


template<typename... KernelParameters>
struct enqueue_launch_helper<kernel_t, KernelParameters...> {
	void operator()(
	const kernel_t&                       wrapped_kernel,
	const stream_t &                      stream,
	launch_configuration_t                launch_config,
	KernelParameters &&...                arguments) const
	{
		auto marshalled_arguments { marshal_dynamic_kernel_arguments(::std::forward<KernelParameters>(arguments)...) };
		auto function_handle = wrapped_kernel.handle();
		context::current::detail_::scoped_override_t set_context_for_this_scope{stream.context_handle()};
		launch_type_erased_in_current_context(
			function_handle, stream.device_id(), stream.context_handle(),
			stream.handle(), launch_config, marshalled_arguments.data());
	}

};

template<typename RawKernelFunction, typename... KernelParameters>
void enqueue_launch(
	::std::integral_constant<bool, false>, // Got a raw kernel function
	RawKernelFunction&&       kernel_function,
	const stream_t&           stream,
	launch_configuration_t    launch_configuration,
	KernelParameters&&...     parameters)
{
	// Note: Unfortunately, even though CUDA should be aware of which context a stream belongs to,
	// and not have trouble enqueueing into a stream in another context - it balks at doing so under
	// certain conditions, so we must place ourselves in the stream's context.
	context::current::detail_::scoped_override_t set_context_for_this_scope{stream.context_handle()};
	detail_::enqueue_raw_kernel_launch_in_current_context<RawKernelFunction, KernelParameters...>(
		kernel_function, stream.handle(), launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

template<typename Kernel, typename... KernelParameters>
void enqueue_launch(
::std::integral_constant<bool, true>, // a kernel wrapped in a kernel_t (sub)class
	Kernel&&                kernel,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	enqueue_launch_helper<typename ::std::decay<Kernel>::type, KernelParameters...>{}(
		::std::forward<Kernel>(kernel), stream, launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

} // namespace detail_

template<typename Kernel, typename... KernelParameters>
inline void launch(
	Kernel&&                kernel,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	auto primary_context = detail_::get_implicit_primary_context(std::forward<Kernel>(kernel));
	auto stream = primary_context.default_stream();

	// Note: If Kernel is a kernel_t, and its associated device is different
	// than the current device, the next call will fail:

	enqueue_launch(kernel, stream, launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

template <typename SpanOfConstVoidPtrLike>
inline void launch_type_erased(
	const kernel_t&         kernel,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	SpanOfConstVoidPtrLike  marshalled_arguments)
{
	static_assert(
		::std::is_same<typename SpanOfConstVoidPtrLike::value_type, void*>::value or
		::std::is_same<typename SpanOfConstVoidPtrLike::value_type, const void*>::value,
		"The element type of the marshalled arguments container type must be either void* or const void*");
#ifndef NDEBUG
	if (*(marshalled_arguments.end() - 1) != nullptr) {
		throw ::std::invalid_argument("marshalled arguments for a kernel launch must end with a nullptr element");
	}
#endif
	context::current::detail_::scoped_override_t set_context_for_this_scope{stream.context_handle()};
	return detail_::launch_type_erased_in_current_context(
		kernel.handle(),
		stream.device_id(),
		stream.context_handle(),
		stream.handle(),
		launch_configuration,
		static_cast<const void**>(marshalled_arguments.data()));
}

#if ! CAN_GET_APRIORI_KERNEL_HANDLE

#if defined(__CUDACC__)

// Unfortunately, the CUDA runtime API does not allow for computation of the grid parameters for maximum occupancy
// from code compiled with a host-side-only compiler! See cuda_runtime.h for details

#if CUDA_VERSION >= 10000
namespace detail_ {

template <typename UnaryFunction>
inline grid::composite_dimensions_t min_grid_params_for_max_occupancy(
	const void *             ptr,
	device::id_t             device_id,
	UnaryFunction            block_size_to_dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
	int min_grid_size_in_blocks { 0 };
	int block_size { 0 };
		// Note: only initializing the values her because of a
		// spurious (?) compiler warning about potential uninitialized use.
	auto result = cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
		&min_grid_size_in_blocks, &block_size,
		ptr,
		block_size_to_dynamic_shared_mem_size,
		static_cast<int>(block_size_limit),
		disable_caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault
	);
	throw_if_error_lazy(result,
		"Failed obtaining parameters for a minimum-size grid for kernel " + detail_::ptr_as_hex(ptr) +
			" on device " + ::std::to_string(device_id) + ".");
	return { (grid::dimension_t) min_grid_size_in_blocks, (grid::block_dimension_t) block_size };
}

inline grid::composite_dimensions_t min_grid_params_for_max_occupancy(
	const void *             ptr,
	device::id_t             device_id,
	memory::shared::size_t   dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
	auto always_need_same_shared_mem_size =
		[dynamic_shared_mem_size](::size_t) { return dynamic_shared_mem_size; };
	return min_grid_params_for_max_occupancy(
		ptr, device_id, always_need_same_shared_mem_size, block_size_limit, disable_caching_override);
}

} // namespace detail_

inline grid::composite_dimensions_t min_grid_params_for_max_occupancy(
	const apriori_compiled_kernel_t&  kernel,
	memory::shared::size_t            dynamic_shared_memory_size,
	grid::block_dimension_t           block_size_limit,
	bool                              disable_caching_override)
{
	return detail_::min_grid_params_for_max_occupancy(
		kernel.ptr(), kernel.device().id(), dynamic_shared_memory_size, block_size_limit, disable_caching_override);
}

template <typename UnaryFunction>
grid::composite_dimensions_t min_grid_params_for_max_occupancy(
	const apriori_compiled_kernel_t&  kernel,
	UnaryFunction                     block_size_to_dynamic_shared_mem_size,
	grid::block_dimension_t           block_size_limit,
	bool                              disable_caching_override)
{
	return detail_::min_grid_params_for_max_occupancy(
		kernel.ptr(), kernel.device_id(), block_size_to_dynamic_shared_mem_size, block_size_limit, disable_caching_override);
}
#endif // CUDA_VERSION >= 10000

#endif // defined(__CUDACC__)
#endif // ! CAN_GET_APRIORI_KERNEL_HANDLE

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_LAUNCH_HPP_

