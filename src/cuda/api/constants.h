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

// warpSize is not a compile-time constant, because theoretically different
// devices could have different constants; but for a specific target architecture,
// it is compiled into an immediate constant in the machine instruction. Anyway,
// we're using a constant of our own here, for ease of use. If nVIDIA comes
// out with 64-lanes-per-warp GPUs, we'll need to refactor this

enum : native_word_t { warp_size          = 32 };
enum : native_word_t { half_warp_size     = warp_size / 2 };
enum : native_word_t { log_warp_size      = 5 };

// For the time being, all CUDA-enabled GPUs are little-endian, and this constant
// reflects that fact
static const endianness_t compilation_target_endianness = endianness_t::little;

namespace stream {

// Would have called it "default" but that's a reserved word;
// Would have liked to make this an enum, but pointers are
// not appropriate for that
const stream::id_t default_stream_id = nullptr;

} // namespace stream

namespace device {

enum : device::id_t {  default_device_id = 0 };

} // namespace device

} // namespace cuda

#endif /* CUDA_API_WRAPPERS_CONSTANTS_H_ */
