/**
 * @file constants.h
 *
 * @brief Fundamental CUDA-related constants and enumerations,
 * not dependent on any more complex abstractions, placed
 * in relevant namespaces.
 *
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_CONSTANTS_H_
#define CUDA_API_WRAPPERS_CONSTANTS_H_

#include <cuda/api/types.h>

namespace cuda {

/**
* CUDA's NVCC allows use the use of the warpSize identifier, without having
* to define it. Un(?)fortunately, warpSize is not a compile-time constant; it
* is replaced at some point with the appropriate immediate value which goes into,
* the SASS instruction as a literal. This is apparently due to the theoretical
* possibility of different warp sizes in the future. However, it is useful -
* both for host-side and more importantly for device-side code - to have the
* warp size available at compile time. This allows all sorts of useful
* optimizations, as well as its use in constexpr code.
*
* If nVIDIA comes out with 64-lanes-per-warp GPUs - we'll refactor this.
*/
enum : native_word_t { warp_size          = 32 };

namespace stream {

// Would have called it "default" but that's a reserved word;
// Would have liked to make this an enum, but pointers are
// not appropriate for that
/**
 * The CUDA runtime provides a default stream on which work
 * is scheduled when no stream is specified; for those API calls
 * where you need to specify the relevant stream's ID, and want to
 * specify the default, this is what you use.
 */
const stream::id_t default_stream_id = nullptr;

} // namespace stream

namespace device {

enum : device::id_t {
	/**
	 * If the CUDA runtime has not been set to a specific device, this
	 * is the ID of the device it defaults to.
	 */
	default_device_id = 0
};

} // namespace device

} // namespace cuda

#endif /* CUDA_API_WRAPPERS_CONSTANTS_H_ */
