/**
 * @file
 *
 * @brief Definitions and utility functions relating to run-time compilation (RTC)
 * of CUDA code using the NVRTC library
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NVRTC_COMPILATION_OPTIONS_HPP_
#define CUDA_API_WRAPPERS_NVRTC_COMPILATION_OPTIONS_HPP_

#include <cuda/api/device_properties.hpp>
#include <cuda/api/device.hpp> // for set_target taking a device
#include "detail/marshalled_options.hpp"

#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <limits>
#include <iterator>

namespace cuda {

namespace detail_ {

template <class F>
void for_each_argument(F) { }

template <class F, class... Args>
void for_each_argument(F f, Args&&... args) {
    using arrT = int[];
    static_cast<void>(arrT{(f(::std::forward<Args>(args)), 0)...});
// This:
//	[](...){}((f(::std::forward<Args>(args)), 0)...);
// doesn't guarantee execution order
}

/*
// Allocator adapter which should hopefully let
// vectors skip setting character values to 0 when
// resizing.
template <typename T, typename Allocator=::std::allocator<T>>
class default_init_allocator : public Allocator {
  typedef ::std::allocator_traits<Allocator> a_t;
public:
  template <typename U> struct rebind {
    using other =
      default_init_allocator<
        U, typename a_t::template rebind_alloc<U>
      >;
  };

  using Allocator::A;

  template <typename U>
  void construct(U* ptr)
    noexcept(::std::is_nothrow_default_constructible<U>::value) {
    ::new(static_cast<void*>(ptr)) U;
  }
  template <typename U, typename...Args>
  void construct(U* ptr, Args&&... args) {
    a_t::construct(static_cast<Allocator&>(*this),
                   ptr, ::std::forward<Args>(args)...);
  }
};
*/

} // namespaced detail_

namespace rtc {

enum class cpp_dialect_t {
	cpp03 = 0,
	cpp11 = 1,
	cpp14 = 2,
	cpp17 = 3,
	last = cpp17
};

namespace detail_ {

static constexpr const size_t language_dialect_name_length { 5 };
constexpr const char* cpp_dialect_names[] =  {
	"c++03",
	"c++11",
	"c++14",
	"c++17",
};

cpp_dialect_t cpp_dialect_from_name(const char* dialect_name) noexcept(false)
{
	for(auto known_dialect = (int) cpp_dialect_t::cpp03;
		known_dialect <= (int) cpp_dialect_t::last;
		known_dialect++)
	{
		if (strcmp(detail_::cpp_dialect_names[known_dialect], dialect_name) == 0) {
			return static_cast<cpp_dialect_t>(known_dialect);
		}
	}
	throw ::std::invalid_argument(::std::string("No C++ dialect named \"") + dialect_name + '"');
}

} // namespace detail_

struct compilation_options_t {

	static constexpr const size_t do_not_set_register_count { 0 };

	/**
	 * Target devices in terms of CUDA compute capability.
	 *
	 * @note Given a compute capability X.Y, the compilation API call
	 * will be passed "sm_XY", _not_ "compute_XY". The distinction between
	 * the two is not currently supported.
	 *
	 * @note Not all compute capabilities are supported! As of CUDA 11.0,
	 * the minimum supported compute capability is 3.5 .
	 *
	 * @note As of CUDA 11.0, the default is compute_52.
	 *
	 * @todo Use something less fancy than ::std::unordered_set, e.g.
	 * a vector-backed ordered set or a dynamic bit-vector for membership.
	 */
    ::std::unordered_set<cuda::device::compute_capability_t> targets_;

    /**
     * Generate relocatable code that can be linked with other relocatable device code. It is equivalent to
     *
     * @note equivalent to "--relocatable-device-code" or "-rdc" for NVCC.
     */
    bool generate_relocatable_code { false };

    /**
     * Do extensible whole program compilation of device code.
     *
     * @todo explain what that is.
     */
    bool compile_extensible_whole_program { false };

    /**
     *  Generate debugging information (and perhaps limit optimizations?); see also @ref generate_line_info
     */
    bool debug { false };

    /**
     *  Generate information for translating compiled code line numbers to source code line numbers.
     */
    bool generate_line_info { false };

    /**
     * Allow the use of the 128-bit `__int128` type in the code.
     */
    bool support_128bit_integers { false };

    /**
     *  emit a remark when a function is inlined
     */
    bool indicate_function_inlining { false };

    /**
     *  Print a self-identification string indicating which
     *  compiler produced the code, in the compilation result
     */
    bool compiler_self_identification { false };

    /**
     * Specify the maximum amount of registers that GPU functions can use. Until a function-specific limit, a
     * higher value will generally increase the performance of individual GPU threads that execute this
     * function. However, because thread registers are allocated from a global register pool on each GPU,
     * a higher value of this option will also reduce the maximum thread block size, thereby reducing the
     * amount of thread parallelism. Hence, a good maxrregcount value is the result of a trade-off.
     * If this option is not specified, then no maximum is assumed. Value less than the minimum registers
     * required by ABI will be bumped up by the compiler to ABI minimum limit.
     *
     * @note Set this to @ref do_not_set_register_count to not pass this as a compilation option.
     *
     * @todo Use ::std::optional
     */
    size_t maximum_register_count { do_not_set_register_count };

    /**
     * When performing single-precision floating-point operations, flush denormal values to zero.
     *
     * @Setting @ref use_fast_math implies setting this to true.
     */
    bool flush_denormal_floats_to_zero { false };

    /**
     * For single-precision floating-point square root, use IEEE round-to-nearest mode or use a faster approximation.
     *
     * @Setting @ref use_fast_math implies setting this to false.
     */
    bool use_precise_square_root { true };

    /**
     * For single-precision floating-point division and reciprocals, use IEEE round-to-nearest mode or use a faster approximation.
     *
     * @Setting @ref use_fast_math implies setting this to false.
     */
    bool use_precise_division { true };

    /**
     * Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA).
     *
     * @Setting @ref use_fast_math implies setting this to false.
     */
    bool use_fused_multiply_add { true };

    /**
     * Make use of fast math operations. Implies use_fused_multiply_add,
     * not use_precise_division and not use_precise_square_root.
     */
    bool use_fast_math { false };

    /**
     * Do not compile fully into PTX/Cubin. Instead, only generate NVVM (the LLVM IR variant), which is
     * combined with other NVVM pieces from LTO-compiled "objects", at device link time.
     */
    bool link_time_optimization { false };

    /**
     * Enables more aggressive device code vectorization in the NVVM optimizer.
     */
    bool extra_device_vectorization { false };

	// TODO: switch to optional<cpp_dialect_t> when the library starts depending on C++14
    bool specify_language_dialect { false };
    /**
     * Set language dialect to C++03, C++11, C++14 or C++17.
     *
     */
    cpp_dialect_t language_dialect { cpp_dialect_t::cpp03 };

    ::std::unordered_set<::std::string> no_value_defines;

    ::std::unordered_map<::std::string,::std::string> valued_defines;

    bool disable_warnings { false };

    /**
     * Treat all kernel pointer parameters as if they had the `restrict` (or `__restrict`) qualifier.
     */
    bool assume_restrict { false };

    /**
     * Assume functions without an explicit specification of their execution space are `__device__`
     * rather than `__host__` functions.
     */
    bool default_execution_space_is_device { false };

    /**
     * Display (error) numbers for warning (and error?) messages, in addition to the message itself.
     */
    bool display_error_numbers { true };

    /**
     * A sequence of directories to be searched for headers. These paths are searched _after_ the
     * list of headers given to nvrtcCreateProgram.
     *
     * @note The members here are `std::string`'s rather than `const char*` or `std::string_view`'s,
     * since this class is a value-type, and cannot rely someone else keeping these strings alive.
     *
     * @todo In C++17, consider making the elements `std::filesystem::path`'s.
     */
    ::std::vector<::std::string> additional_include_paths;

    /**
     * Header files to preinclude during preprocessing of the source.
     *
     * @note The members here are `std::string`'s rather than `const char*` or `std::string_view`'s,
     * since this class is a value-type, and cannot rely someone else keeping these strings alive.
     *
     * @todo In C++17, consider making the elements `std::filesystem::path`'s.
     *
     * @todo Check how these strings are interpreted. Do they need quotation marks? brackets? full paths?
     */
    ::std::vector<::std::string> preinclude_files;

    /**
     * Provide builtin definitions of @ref ::std::move and @ref ::std::forward.
     *
     * @note Only relevant when the dialect is C++11 or later.
     */
    bool builtin_move_and_forward { true };

    /**
	 * Use @ref setrlimit() to increase the stack size to the maximum the OS allows.
     * The limit is reverted to its previous value after compilation.
	 *
	 * @note :
     *  1. Only works on Linux
     *  2. Affects the entire process, not just the thread invoking the compilation
     *     command.
	 */
    bool increase_stack_limit_to_max { true };

    /**
     * Provide builtin definitions of ::std::initializer_list class and member functions.
     *
     * @note Only relevant when the dialect is C++11 or later.
     */
    bool builtin_initializer_list { true };

	/**
	 * Support for additional, arbitrary options which may not be covered by other fields
	 * in this class (e.g. due to newer CUDA versions providing them)
	 *
	 * @note These are appended to the command-line verbatim (so, no prefixing with `-`
	 * signs, no combining pairs of consecutive elements as opt=value etc.)
	 */
	::std::vector<::std::string> extra_options;

protected:
    template <typename T>
    void process(T& opts) const;

public: // "shorthands" for more complex option setting
	// TODO: Drop the following methods and make targets a custom
	// inner class which can assigned, added to or subtracted from

	/**
	 * Have the compilation also target a specific compute capability.
	 *
	 * @note previously-specified compute capabilities will be targeted in
	 * addition to the one specified.
	 */
	compilation_options_t& add_target(device::compute_capability_t compute_capability)
	{
		targets_.clear();
		targets_.insert(compute_capability);
		return *this;
	}

	/**
	 * Have the compilation target one one specific compute capability.
	 *
	 * @note any previous target settings are dropped, i.e. no other compute
	 * capability will be targeted.
	 */
	compilation_options_t& set_target(device::compute_capability_t compute_capability)
	{
		targets_.clear();
		add_target(compute_capability);
		return *this;
	}

	compilation_options_t& set_target(device_t device)
	{
		return set_target(device.compute_capability());
	}

	compilation_options_t& set_language_dialect(cpp_dialect_t dialect)
	{
		specify_language_dialect = true;
		language_dialect = dialect;
		return *this;
	}

	compilation_options_t& clear_language_dialect()
	{
		specify_language_dialect = false;
		return *this;
	}

	compilation_options_t& set_language_dialect(const char* dialect_name)
	{
		return (dialect_name == nullptr or *dialect_name == '\0') ?
			clear_language_dialect() :
			set_language_dialect(detail_::cpp_dialect_from_name(dialect_name));
	}

	compilation_options_t& set_language_dialect(const ::std::string& dialect_name)
	{
		return dialect_name.empty() ?
			clear_language_dialect() :
			set_language_dialect(dialect_name.c_str());
	}

};


namespace detail_ {

const char* true_or_false(bool b) { return b ? "true" : "false"; }

}

/**
 * Use the left-shift operator (<<) to render a delimited sequence of
 * command-line-argument-like options (with or without a value as relevant)
 * into some target entity - which could be a buffer or a more complex
 * structure.
 *
 */
template <typename MarshalTarget, typename Delimiter>
void process(const compilation_options_t& opts, MarshalTarget& marshalled, Delimiter optend)
{
	// TODO: Consider taking an option to be verbose, and push_back option values which are compiler
	// defaults.
	if (opts.generate_relocatable_code)         { marshalled << "--relocatable-device-code=true" << optend;      }
	if (opts.compile_extensible_whole_program)  { marshalled << "--extensible-whole-program=true" << optend;     }
	if (opts.debug)                             { marshalled << "--device-debug" << optend;                      }
	if (opts.generate_line_info)                { marshalled << "--generate-line-info" << optend;                }
	if (opts.support_128bit_integers)           { marshalled << "--device-int128" << optend;                     }
	if (opts.indicate_function_inlining)        { marshalled << "--optimization-info=inline" << optend;          }
	if (opts.compiler_self_identification)      { marshalled << "--version-ident=true" << optend;                }
	if (not opts.builtin_initializer_list)      { marshalled << "--builtin-initializer-list=false" << optend;    }
	if (opts.extra_device_vectorization)        { marshalled << "--extra-device-vectorization" << optend;        }
	if (opts.disable_warnings)                  { marshalled << "--disable-warnings" << optend;                  }
	if (opts.assume_restrict)                   { marshalled << "--restrict" << optend;                          }
	if (opts.default_execution_space_is_device) { marshalled << "--device-as-default-execution-space" << optend; }
	if (not opts.display_error_numbers)         { marshalled << "--no-display-error-number" << optend;           }
	if (not opts.builtin_move_and_forward)      { marshalled << "--builtin-move-forward=false" << optend;        }
	if (not opts.increase_stack_limit_to_max)   { marshalled << "--modify-stack-limit=false" << optend;          }
	if (opts.link_time_optimization)            { marshalled << "--dlink-time-opt" << optend;                    }
	if (opts.use_fast_math)                     { marshalled << "--use_fast_math" << optend;                     }
	else {
		if (opts.flush_denormal_floats_to_zero) { marshalled << "--ftz" << optend;                               }
		if (not opts.use_precise_square_root)   { marshalled << "--prec-sqrt=false" << optend;                   }
		if (not opts.use_precise_division)      { marshalled << "--prec-div=false" << optend;                    }
		if (not opts.use_fused_multiply_add)    { marshalled << "--fmad=false" << optend;                        }
	}

	if (opts.specify_language_dialect) {
		marshalled << "--std=" << detail_::cpp_dialect_names[(unsigned) opts.language_dialect] << optend;
	}

	if (opts.maximum_register_count != compilation_options_t::do_not_set_register_count) {
		marshalled << "--maxrregcount" << opts.maximum_register_count << optend;
	}

	// Multi-value options

	for(const auto& target : opts.targets_) {
#if CUDA_VERSION < 11000
		marshalled << "--gpu-architecture=compute_" << target.as_combined_number() << optend;
#else
		marshalled << "--gpu-architecture=sm_" << target.as_combined_number() << optend;
#endif
	}

	for(const auto& def : opts.no_value_defines) {
		marshalled << "-D" << def << optend;
		// Note: Could alternatively use "--define-macro" instead of "-D"
	}

	for(const auto& def : opts.valued_defines) {
		marshalled << "-D" << def.first << '=' << def.second << optend;
	}

	for(const auto& path : opts.additional_include_paths) {
		marshalled << "--include-path=" << path << optend;
	}

	for(const auto& preinclude_file : opts.preinclude_files) {
		marshalled << "--pre-include=" << preinclude_file << optend;
	}

	for(const auto& extra_opt : opts.extra_options) {
		marshalled << extra_opt << optend;
	}
}

marshalled_options_t marshal(const compilation_options_t& opts)
{
	marshalled_options_t mo;
	// TODO: Can we easily determine the max number of options here?
	process(opts, mo, detail_::optend);
	return mo;
}

::std::string render(const compilation_options_t& opts)
{
	::std::ostringstream oss;
	process(opts, oss, ' ');
	if (oss.tellp() > 0) {
		// Remove the last, excessive, delimiter
		oss.seekp(-1,oss.cur);
	}
	return oss.str();
}

} // namespace rtc

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_COMPILATION_OPTIONS_HPP_
