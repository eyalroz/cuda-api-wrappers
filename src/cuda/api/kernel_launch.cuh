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
 * and stick to proper C++; in other words, the wrappers are "ugly" instead
 * of your code having to be.
 * <li>Avoiding some of the "parameter soup" of launching a kernel: It's
 * rather easy to mix up shared memory sizes with stream IDs; grid and
 * block dimensions with each other; and even grid/block dimensions with
 * the scalar parameters - since a {@code dim3} is constructible from
 * integral values. Instead, we enforce a launch configuration structure:
 * {@ref cuda::launch_configuration_t}.
 * </ul>
 *
 * @note You'd probably better avoid launching kernels using these
 * function directly, and go through the @ref cuda::stream_t or @ref cuda::device_t
 * proxy classes' launch mechanism (e.g.
 * {@code my_stream.enqueue.kernel_launch(...)}).
 *
 * @note Even though when you use this wrapper, your code will not have the silly
 * chevron, you can't use it from regular {@code .cpp} files compiled with your host
 * compiler. Hence the {@code .cuh} extension. You _can_, however, safely include this
 * file from your {@code .cpp} for other definitions. Theoretically, we could have
 * used the {@code cudaLaunchKernel} API function, by creating an array on the stack
 * which points to all of the other arguments, but that's kind of redundant.
 *
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_KERNEL_LAUNCH_CUH_
#define CUDA_API_WRAPPERS_KERNEL_LAUNCH_CUH_

#include "cuda/api/types.h"
#include "cuda/api/constants.h"

namespace cuda {

enum : shared_memory_size_t { no_shared_memory = 0 };
constexpr grid_dimensions_t single_block() { return 1; }
constexpr grid_block_dimensions_t single_thread_per_block() { return 1; };

/**
* CUDA's 'chevron' kernel launch syntax cannot be compiled in proper C++. Thus, every kernel launch must
* at some point reach code compiled with CUDA's nvcc. Naively, every single different kernel (perhaps up
* to template specialization) would require writing its own wrapper C++ function, launching it. This
* function, however, constitutes a single minimal wrapper around the CUDA kernel launch, which may be
* called from proper C++ code (across translation unit boundaries - the caller is compiled with a C++
* compiler, the callee compiled by nvcc).
*
* <p>This function is similar to C++17's {@code std::apply}, or to a a beta-reduction in Lambda calculus:
* It applies a function to its arguments; the difference is in the nature of the function (a CUDA kernel)
* and in that the function application requires setting additional CUDA-related launch parameters,
* additional to the function's own.
*
* <p>As kernels do not return values, neither does this function. It also contains no hooks, logging
* commands etc. - if you want those, write an additional wrapper (perhaps calling this one in turn).
*
* @param[in] kernel_function the kernel to apply. Pass it just as-it-is, as though it were any other function. Note:
* If the kernel is templated, you must pass it fully-instantiated.
* @param[in] launch_configuration a kernel is launched on a grid of blocks of thread, and with an allowance of
* shared memory per block in the grid; this defines how the grid will look and what the shared memory
* allowance will be (see {@ref cuda::launch_configuration_t})
* @param[in] launch_configuration The grid and shared memory configuration parameters for this launch
* in the apprioriate structure
* @param[in] stream_id the CUDA hardware command queue on which to place the command to launch the kernel (affects
* the scheduling of the launch and the execution)
* @param[in] parameters whatever parameters @p kernel_function takes
*/
template<typename KernelFunction, typename... KernelParameters>
inline void enqueue_launch(
	const KernelFunction&       kernel_function,
	launch_configuration_t      launch_configuration,
	stream::id_t                stream_id,
	KernelParameters...         parameters)
#ifndef __CUDACC__
// If we're not in CUDA's NVCC, this can't run properly anyway, so either we throw some
// compilation error, or we just do nothing. For now it's option 2.
	;
#else
{
	kernel_function <<<
		launch_configuration.grid_dimensions,
		launch_configuration.block_dimensions,
		launch_configuration.dynamic_shared_memory_size,
		stream_id
		>>>(parameters...);
}
#endif

/**
 * Variant of enqueue_launch, above, for use with the default stream on the current device;
 * it's not also called 'enqueue' since the default stream is synchronous.
 */
template<typename KernelFunction, typename... KernelParameters>
inline void launch(
	const KernelFunction&       kernel_function,
	launch_configuration_t      launch_configuration,
	KernelParameters...         parameters)
{
	enqueue_launch(kernel_function, launch_configuration, stream::default_stream_id, parameters...);
}

} // namespace cuda

#endif /* CUDA_KERNEL_LAUNCH_H_ */
