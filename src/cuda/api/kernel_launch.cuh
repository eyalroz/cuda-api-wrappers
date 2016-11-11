/**
 * Variadic wrapper function for the CUDA kernel launch mechanism.
 * The basic intention is to avoid the annoying triple-chevron syntax, e.g.
 *
 *   my_kernel<<<launch, config, stuff>>>(real, args)
 *
 * and stick to proper C++. So essentially these wrapper will be ugly instead
 * of your code being ugly. Additionally, we're avoiding some of the parameter
 * soup using the launch_configuration_t structure. Still, you should probably
 * call the launch() method of {@ref stream_t}'s.
 *
 * @note Even though when you use this wrapper, your code will not have the silly chevron,
 * you can't use it from regular .cpp files compiled with your host compiler.
 * Hence the .cuh extension.
 *
 */

#pragma once
#ifndef CUDA_KERNEL_LAUNCH_CUH_
#define CUDA_KERNEL_LAUNCH_CUH_

#include "cuda/api/types.h"
#include "cuda/api/constants.h"

namespace cuda {

enum : shared_memory_size_t { NoSharedKernelMemory = 0 };
const grid_dimensions_t       SingleBlock          { 1 };
const grid_block_dimensions_t SingleThreadPerBlock { 1 };


/**
* CUDA's kernel launching mechanism cannot be compiled in C++ - both for syntactic and semantic reasons.
* Thus, every kernel launch must at some point reach code compiled with CUDA's nvcc. Naively, every single
* different kernel (perhaps up to template specialization) would require writing its own wrapper C++ function,
* launching it. This function, however, constitutes a single minimal wrapper around the CUDA kernel launch,
* which may be called from proper C++ code.
*
* <p>This function is similar to C++17's std::apply, or to a a beta-reduction in Lambda calculus: It applies
* a function to its arguments; the difference is in the nature of the function (a CUDA kernel) and in that
* the function application requires setting additional CUDA-related launch parameters, other than the
* function's own.
*
* <p>As kernels do not return values, neither does this function. It also contains
* no hooks, logging commands etc. - if you want those, write your own wrapper (perhaps calling this one in
* turn).
*
* @param[in] kernel_function the kernel to apply. Pass it just as-it-is, as though it were any other function. Note:
* If the kernel is templated, you must pass it fully-instantiated.
* @param[in] launch_configuration a kernel is launched on a grid of blocks of thread, and with an allowance of
* shared memory per block in the grid; this defines how the grid will look and what the shared memory
* allowance will be; {@see launch_configuration_t}
* @param[in] block_dimensions the number of CUDA threads (a.k.a. hardware threads, or 'CUDA cores') in every
* execution grid block, in each of upto 3 dimensions.
* @param[in] shared_memory_size the amount, in bytes, of shared memory to allocate for common use by each execution
* block in the grid; limited by your specific GPU's capabilities and typically <= 48 Ki.
* @param[in] stream the CUDA hardware command queue on which to place the command to launch the kernel (affects
* the scheduling of the launch and the execution)
* @param[in] parameters whatever parameters {@kernel_function} takes
*/
template<typename KernelFunction, typename... KernelParameters>
inline void launch(
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
 * Variant of launch, above, for use with the default stream on the current device.
 */
template<typename KernelFunction, typename... KernelParameters>
inline void launch(
	const KernelFunction&       kernel_function,
	launch_configuration_t      launch_configuration,
	KernelParameters...         parameters)
{
	launch(kernel_function, launch_configuration, stream::default_stream_id, parameters...);
}

namespace linear_grid {

enum : grid_dimension_t       { SingleBlock = 1 };
enum : grid_block_dimension_t { SingleThreadPerBlock = 1 };

class launch_configuration_t {
public:
	grid_dimension_t       grid_length; // in threads
	grid_block_dimension_t block_length; // in threads
	shared_memory_size_t   dynamic_shared_memory_size; // in bytes

	launch_configuration_t()
		: grid_length(SingleBlock), block_length(SingleThreadPerBlock),
		  dynamic_shared_memory_size(NoSharedKernelMemory) { }
	launch_configuration_t(
			grid_dimension_t grid_length_,
			grid_block_dimension_t block_length_,
			unsigned short dynamic_shared_memory_size = NoSharedKernelMemory)
		: grid_length(grid_length_), block_length(block_length_),
		  dynamic_shared_memory_size(dynamic_shared_memory_size) { }

	// Allows linear launch configs to be passed
	// to the launcher function
	operator cuda::launch_configuration_t()
	{
		return {grid_length, block_length, dynamic_shared_memory_size};
	}
};

} // linear_grid

} // namespace cuda

#endif /* CUDA_KERNEL_LAUNCH_H_ */
