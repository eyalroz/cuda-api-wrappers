/**
 * @file kernel_launch.cuh
 *
 * @brief Variadic, chevron-less wrappers for the CUDA kernel launch mechanism.
 *
 * This file has two stand-alone functions used for launching kernels - by
 * application code directly and by other API wrappers (e.g. @ref cuda::device_t
 * and @ref cuda::stream_t ).
 *
 * <p>The wrapper functions have two goals:
 *
 * <ul>
 * <li>Avoiding the annoying triple-chevron syntax, e.g.
 *
 *   my_kernel<<<launch, config, stuff>>>(real, args)
 *
 * and sticking to proper C++; in other words, the wrappers are "ugly"
 * instead of client code having to be.
 * <li>Avoiding some of the "parameter soup" of launching a kernel: It's
 * rather easy to mix up shared memory sizes with stream IDs; grid and
 * block dimensions with each other; and even grid/block dimensions with
 * the scalar parameters - since a `dim3` is constructible from
 * integral values. Instead, we enforce a launch configuration structure:
 * {@ref cuda::launch_configuration_t}.
 * </ul>
 *
 * @note You'd probably better avoid launching kernels using these
 * function directly, and go through the @ref cuda::stream_t or @ref cuda::device_t
 * proxy classes' launch mechanism (e.g.
 * `my_stream.enqueue.kernel_launch(...)`).
 *
 * @note Even though when you use this wrapper, your code will not have the silly
 * chevron, you can't use it from regular `.cpp` files compiled with your host
 * compiler. Hence the `.cuh` extension. You _can_, however, safely include this
 * file from your `.cpp` for other definitions. Theoretically, we could have
 * used the `cudaLaunchKernel` API function, by creating an array on the stack
 * which points to all of the other arguments, but that's kind of redundant.
 *
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_KERNEL_LAUNCH_CUH_
#define CUDA_API_WRAPPERS_KERNEL_LAUNCH_CUH_

#include <cuda/api/types.h>
#include <cuda/api/constants.h>
#include <cuda/api/device_function.hpp>

#if (__CUDACC_VER_MAJOR__ >= 9)
#include <cooperative_groups.h>
#endif

#include <type_traits>
#include <utility>

namespace cuda {

enum : shared_memory_size_t { no_shared_memory = 0 };
constexpr grid_dimensions_t single_block() { return 1; }
constexpr grid_block_dimensions_t single_thread_per_block() { return 1; };

namespace detail {

template<typename Fun>
struct is_function_ptr: std::integral_constant<bool,
    std::is_pointer<Fun>::value and std::is_function<typename std::remove_pointer<Fun>::type>::value> { };

template <typename Arg>
void collect_argument_addresses(void** collected_addresses, Arg&& arg)
{
	collected_addresses[0] = static_cast<void*>(&arg);
}

template <typename Arg1, typename Arg2>
void collect_argument_addresses(void** collected_addresses, Arg1&& arg1, Arg2&& arg2)
{
	collect_argument_addresses(collected_addresses,     std::forward<Arg1>(arg1));
	collect_argument_addresses(collected_addresses + 1, std::forward<Arg2>(arg2));
}

template <typename Arg1, typename Arg2, typename Arg3>
void collect_argument_addresses(
	void** collected_addresses, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3)
{
	collect_argument_addresses(collected_addresses,     std::forward<Arg1>(arg1));
	collect_argument_addresses(collected_addresses + 1, std::forward<Arg2>(arg2));
	collect_argument_addresses(collected_addresses + 2, std::forward<Arg3>(arg3));
}

template <typename Arg1, typename Arg2, typename Arg3, typename Arg4>
void collect_argument_addresses(
	void** collected_addresses, Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4)
{
	collect_argument_addresses(
		collected_addresses,std::forward<Arg1>(arg1), std::forward<Arg2>(arg2));
	collect_argument_addresses(
		collected_addresses + 2, std::forward<Arg3>(arg3), std::forward<Arg4>(arg4));
}

template <typename Arg1, typename Arg2, typename Arg3, typename Arg4, typename... Args>
void collect_argument_addresses(
	void** collected_addresses, 
	Arg1&& arg1, Arg2&& arg2, Arg3&& arg3, Arg4&& arg4, Args&&... args)
{
	collect_argument_addresses(
		collected_addresses, 
		std::forward<Arg1, Arg2, Arg3, Arg4>(arg1, arg2, arg3, arg4));
	collect_argument_addresses(collected_addresses + 4, std::forward<Args>(args)...);
}

} // namespace detail

/**
 * @brief Enqueues a kernel on a stream (=queue) on the current CUDA device.
 *
 * CUDA's 'chevron' kernel launch syntax cannot be compiled in proper C++. Thus, every kernel launch must
 * at some point reach code compiled with CUDA's nvcc. Naively, every single different kernel (perhaps up
 * to template specialization) would require writing its own wrapper C++ function, launching it. This
 * function, however, constitutes a single minimal wrapper around the CUDA kernel launch, which may be
 * called from proper C++ code (across translation unit boundaries - the caller is compiled with a C++
 * compiler, the callee compiled by nvcc).
 *
 * <p>This function is similar to C++17's `std::apply`, or to a a beta-reduction in Lambda calculus:
 * It applies a function to its arguments; the difference is in the nature of the function (a CUDA kernel)
 * and in that the function application requires setting additional CUDA-related launch parameters,
 * additional to the function's own.
 *
 * <p>As kernels do not return values, neither does this function. It also contains no hooks, logging
 * commands etc. - if you want those, write an additional wrapper (perhaps calling this one in turn).
 *
 * @param cooperative if true, use CUDA's "cooperative launch" mechanism which enables more flexible
 * synchronization capabilities (see CUDA C Programming Guide C.3. Grid Synchronization)
 * @param kernel_function the kernel to apply. Pass it just as-it-is, as though it were any other function. Note:
 * If the kernel is templated, you must pass it fully-instantiated.
 * @param stream_id the CUDA hardware command queue on which to place the command to launch the kernel (affects
 * the scheduling of the launch and the execution)
 * @param launch_configuration a kernel is launched on a grid of blocks of thread, and with an allowance of
 * shared memory per block in the grid; this defines how the grid will look and what the shared memory
 * allowance will be (see {@ref cuda::launch_configuration_t})
 * @param parameters whatever parameters @p kernel_function takes
 */
template<typename KernelFunction, typename... KernelParameters>
inline void enqueue_launch(
	bool                        thread_block_cooperation,
	const KernelFunction&       kernel_function,
	stream::id_t                stream_id,
	launch_configuration_t      launch_configuration,
	KernelParameters...         parameters)
#ifndef __CUDACC__
// If we're not in CUDA's NVCC, this can't run properly anyway, so either we throw some
// compilation error, or we just do nothing. For now it's option 2.
	;
#else
{
	static_assert(std::is_function<KernelFunction>::value or
	    (detail::is_function_ptr<KernelFunction>::value),
	    "Only a bona fide function can be a CUDA kernel and be launched; "
	    "you were attempting to enqueue a launch of something other than a function");

	if (thread_block_cooperation == thread_blocks_may_not_cooperate) {
		// regular plain vanilla launch
		kernel_function <<<
			launch_configuration.grid_dimensions,
			launch_configuration.block_dimensions,
			launch_configuration.dynamic_shared_memory_size,
			stream_id
			>>>(parameters...);
	}
	else {
#if __CUDACC_VER_MAJOR__ >= 9
		// Cooperative launches cannot be made using the triple-chevron syntax,
		// nor is there a variadic-template of the launch API call, so we need to
		// a bit of useless work here. We could have done exactly the same thing
		// for the non-cooperative case, mind you.

		void* argument_ptrs[sizeof...(KernelParameters)];
		// fill the argument array with our parameters. Yes, the use
		// of the two terms is confusing here and depends on how you
		// look at things.
		detail::collect_argument_addresses(argument_ptrs, parameters...);
		auto status = cudaLaunchCooperativeKernel(
			(const void*) kernel_function,
			launch_configuration.grid_dimensions,
			launch_configuration.block_dimensions,
			argument_ptrs,
			launch_configuration.dynamic_shared_memory_size,
			stream_id);
		throw_if_error(status, "Cooperative launch failed");

#else
		throw cuda::runtime_error(status::not_supported,
			"Only CUDA versions 9.0 and later support launching kernels \"cooperatively\"");
#endif
	}
}
#endif

template<typename KernelFunction, typename... KernelParameters>
inline void enqueue_launch(
	const KernelFunction&       kernel_function,
	stream::id_t                stream_id,
	launch_configuration_t      launch_configuration,
	KernelParameters...         parameters)
{
	enqueue_launch(thread_blocks_may_not_cooperate, kernel_function, stream_id, launch_configuration, parameters...);
}

/**
 * @brief Enqueues a kernel on a stream (=queue) on the current CUDA device.
 *
 * @note It is up to the caller to ensure the device function's signature
 * matches the argument types.
 *
 * @param wrapped_device_function the kernel to apply, as a @ref device_function_t.
 * @param stream_id the CUDA hardware command queue on which to place the command to launch the kernel (affects
 * the scheduling of the launch and the execution)
 * @param launch_configuration a kernel is launched on a grid of blocks of thread, and with an allowance of
 * shared memory per block in the grid; this defines how the grid will look and what the shared memory
 * allowance will be (see {@ref cuda::launch_configuration_t})
 * @param parameters whatever parameters @p wrapped_device_function takes
 */
template<typename... KernelParameters>
inline void enqueue_launch(
	device_function_t           wrapped_kernel_function,
	stream::id_t                stream_id,
	launch_configuration_t      launch_configuration,
	KernelParameters...         parameters)
{
	using kernel_function_type = void (*)(KernelParameters...);
	auto unwrapped_kernel_function = reinterpret_cast<kernel_function_type>(wrapped_kernel_function.ptr());
	return enqueue_launch(unwrapped_kernel_function, stream_id, launch_configuration, parameters...);
}

/**
 * Variant of @ref enqueue_launch for use with the default stream on the current device.
 *
 * @note This isn't called `enqueue` since the default stream is synchronous.
 */
template<typename KernelFunction, typename... KernelParameters>
inline void launch(
	const KernelFunction&       kernel_function,
	launch_configuration_t      launch_configuration,
	KernelParameters...         parameters)
{
	enqueue_launch(kernel_function, stream::default_stream_id, launch_configuration, parameters...);
}

/**
 * Variant of @ref enqueue_launch for use with the default stream on the current device.
 *
 * @note This isn't called `enqueue` since the default stream is synchronous.
 *
 */
template<typename... KernelParameters>
inline void launch(
	device_function_t           device_function,
	launch_configuration_t      launch_configuration,
	KernelParameters...         parameters)
{
	enqueue_launch(device_function, stream::default_stream_id, launch_configuration, parameters...);
}

} // namespace cuda

#endif /* CUDA_KERNEL_LAUNCH_H_ */
