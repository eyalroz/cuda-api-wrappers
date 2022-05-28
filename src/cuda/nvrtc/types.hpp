/**
 * @file
 *
 * @brief Type definitions used in CUDA real-time compilation work wrappers.
 */
#pragma once
#ifndef SRC_CUDA_NVRTC_TYPES_HPP_
#define SRC_CUDA_NVRTC_TYPES_HPP_

#include <cuda/api/types.hpp>

#include <vector>

#if __cplusplus >= 201703L
// #include <filesystem>
#include <string_view>
namespace cuda {
using string_view = ::std::string_view;
// namespace filesystem = ::std::filesystem;
}
#else
#include <cuda/nvrtc/detail/string_view.hpp>
namespace cuda {
using string_view = bpstd::string_view;
}
#endif

namespace cuda {

// The C++ standard library doesn't offer ::std::dynarray (although it almost did),
// and we won't introduce our own here. So...
template <typename T>
using dynarray = ::std::vector<T>;
//
// An easy alternative might be using a non-initializing allocator;
// see: https://stackoverflow.com/a/15966795/1593077
// this is not entirely sufficient, as we should probably not
// provide a container which may then be resized.

} // namespace cuda

#endif /* SRC_CUDA_NVRTC_TYPES_HPP_ */
