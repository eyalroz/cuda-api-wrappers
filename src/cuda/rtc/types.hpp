/**
 * @file
 *
 * @brief Type definitions used in CUDA real-time compilation work wrappers.
 */
#pragma once
#ifndef SRC_CUDA_NVRTC_TYPES_HPP_
#define SRC_CUDA_NVRTC_TYPES_HPP_

#include <cuda/api/types.hpp>

#include <nvrtc.h>
#if CUDA_VERSION >= 11010
#include <nvPTXCompiler.h>
#endif // CUDA_VERSION >= 11010

#include <vector>

#if __cplusplus >= 201703L
#include <string_view>
namespace cuda {
using string_view = ::std::string_view;
}
#else
#include <cuda/rtc/detail/string_view.hpp>
namespace cuda {
using string_view = bpstd::string_view;
}
#endif

namespace cuda {

enum source_kind_t {
	cuda_cpp = 0,
	ptx = 1
};

// The C++ standard library doesn't offer ::std::dynarray (although it almost did),
// and we won't introduce our own here. So...
template <typename T>
using dynarray = ::std::vector<T>;
//
// An easy alternative might be using a non-initializing allocator;
// see: https://stackoverflow.com/a/15966795/1593077
// this is not entirely sufficient, as we should probably not
// provide a container which may then be resized.

/**
 * @brief Real-time compilation of CUDA programs using the NVIDIA NVRTC library.
 */
namespace rtc {

using const_cstrings_span = span<const char* const>;
using const_cstring_pairs_span = span<::std::pair<const char* const, const char* const>>;

namespace program {

namespace detail_ {

constexpr char const *kind_name(source_kind_t kind)
{
	return (kind == cuda_cpp) ? "CUDA C++" : "PTX";
}

} // namespace detail_

} // namespace program

namespace detail_ {

template <source_kind_t Kind> struct types {};

template <> struct types<cuda_cpp> {
	using handle_type = nvrtcProgram;
	using status_type = nvrtcResult;
	enum named_status : ::std::underlying_type<status_type>::type {
		success = NVRTC_SUCCESS,
		out_of_memory = NVRTC_ERROR_OUT_OF_MEMORY,
		program_creation_failure = NVRTC_ERROR_OUT_OF_MEMORY,
		invalid_input = NVRTC_ERROR_PROGRAM_CREATION_FAILURE,
		invalid_program = NVRTC_ERROR_INVALID_PROGRAM,
		invalid_option = NVRTC_ERROR_INVALID_OPTION,
		compilation_failure = NVRTC_ERROR_COMPILATION,
		builtin_operation_failure = NVRTC_ERROR_BUILTIN_OPERATION_FAILURE,
		no_registered_globals_after_compilation = NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION,
		no_lowered_names_before_compilation = NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION,
		invalid_expression_to_register_as_global = NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID,
		internal_error = NVRTC_ERROR_INTERNAL_ERROR,
	};
};

#if CUDA_VERSION >= 11010
template <> struct types<ptx> {
	using handle_type = nvPTXCompilerHandle;
	using status_type = nvPTXCompileResult;
	enum named_status : ::std::underlying_type<status_type>::type {
		success = NVPTXCOMPILE_SUCCESS,
		invalid_handle = NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE,
		invalid_program_handle = NVPTXCOMPILE_ERROR_INVALID_COMPILER_HANDLE,
		out_of_memory = NVPTXCOMPILE_ERROR_OUT_OF_MEMORY,
		invalid_input = NVPTXCOMPILE_ERROR_INVALID_INPUT,
		compilation_invocation_incomplete = NVPTXCOMPILE_ERROR_COMPILER_INVOCATION_INCOMPLETE,
		compilation_failure = NVPTXCOMPILE_ERROR_COMPILATION_FAILURE,
		unsupported_ptx_version = NVPTXCOMPILE_ERROR_UNSUPPORTED_PTX_VERSION,
		internal_error = NVPTXCOMPILE_ERROR_INTERNAL,
	};
};
#endif // CUDA_VERSION >= 11010

} // namespace detail_

namespace program {

template <source_kind_t Kind>
using handle_t = typename cuda::rtc::detail_::types<Kind>::handle_type;

} // namespace program

template <source_kind_t Kind>
using status_t = typename detail_::types<Kind>::status_type;

} // namespace rtc

} // namespace cuda

#endif /* SRC_CUDA_NVRTC_TYPES_HPP_ */
