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
	throw std::invalid_argument(std::string("No C++ dialect named \"") + dialect_name + '"');
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
	 * @todo Use something less fancy than std::unordered_set, e.g.
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

    ::std::unordered_map<::std::string,std::string> valued_defines;

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
     */
    ::std::vector<::std::string> additional_include_paths;

    /**
     * Header files to preinclude during preprocessing of the source.
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

	compilation_options_t& set_language_dialect(const char* dialect_name)
	{
		return set_language_dialect(detail_::cpp_dialect_from_name(dialect_name));
	}

public:
	marshalled_options_t marshal() const;
};

namespace detail_ {

const char* true_or_false(bool b) { return b ? "true" : "false"; }

}

template <typename T>
void compilation_options_t::process(T& opt_struct) const
{
	// TODO: Consider taking an option to be verbose, and push_back option values which are compiler
	// defaults.
	if (generate_relocatable_code)         { opt_struct.push_back("--relocatable-device-code=true");      }
	if (compile_extensible_whole_program)  { opt_struct.push_back("--extensible-whole-program=true");     }
	if (debug)                             { opt_struct.push_back("--device-debug");                      }
	if (generate_line_info)                { opt_struct.push_back("--generate-line-info");                }
	if (support_128bit_integers)           { opt_struct.push_back("--device-int128");                     }
	if (indicate_function_inlining)        { opt_struct.push_back("--optimization-info=inline");          }
	if (compiler_self_identification)      { opt_struct.push_back("--version-ident=true");                }
	if (not builtin_initializer_list)      { opt_struct.push_back("--builtin-initializer-list=false");    }
	if (extra_device_vectorization)        { opt_struct.push_back("--extra-device-vectorization");        }
	if (disable_warnings)                  { opt_struct.push_back("--disable-warnings");                  }
	if (assume_restrict)                   { opt_struct.push_back("--restrict");                          }
	if (default_execution_space_is_device) { opt_struct.push_back("--device-as-default-execution-space"); }
	if (not display_error_numbers)         { opt_struct.push_back("--no-display-error-number");           }
	if (not builtin_move_and_forward)      { opt_struct.push_back("--builtin-move-forward=false");        }
	if (not increase_stack_limit_to_max)   { opt_struct.push_back("--modify-stack-limit=false");          }
	if (link_time_optimization)            { opt_struct.push_back("--dlink-time-opt");                    }
	if (use_fast_math)                     { opt_struct.push_back("--use_fast_math");                     }
	else {
		if (flush_denormal_floats_to_zero) { opt_struct.push_back("--ftz");                               }
		if (not use_precise_square_root)   { opt_struct.push_back("--prec-sqrt=false");                   }
		if (not use_precise_division)      { opt_struct.push_back("--prec-div=false");                    }
		if (not use_fused_multiply_add)    { opt_struct.push_back("--fmad=false");                        }
	}

	if (specify_language_dialect) {
		opt_struct.push_back("--std=", detail_::cpp_dialect_names[(unsigned) language_dialect]);
	}

	if (maximum_register_count != do_not_set_register_count) {
		opt_struct.push_back("--maxrregcount", maximum_register_count);
	}

	// Multi-value options

	for(const auto& target : targets_) {
#if CUDA_VERSION < 11000
		opt_struct.push_back("--gpu-architecture=compute_", target.as_combined_number());
#else
		opt_struct.push_back("--gpu-architecture=sm_", target.as_combined_number());
#endif
	}

	for(const auto& def : no_value_defines) {
		opt_struct.push_back("-D", def);
			// Note: Could alternatively use "--define-macro" instead of "-D"
	}

	for(const auto& def : valued_defines) {
		opt_struct.push_back("-D",def.first, '=', def.second);
	}

	for(const auto& path : additional_include_paths) {
		opt_struct.push_back("--include-path=", path);
	}

	for(const auto& preinclude_file : preinclude_files) {
		opt_struct.push_back("--pre-include=", preinclude_file);
	}
}

marshalled_options_t compilation_options_t::marshal() const
{
	detail_::marshalled_options_size_computer_t size_computer;
	process(size_computer);
	marshalled_options_t marshalled(size_computer.num_options(), size_computer.buffer_size());
	process(marshalled);
	return marshalled;
}


} // namespace rtc

} // namespace cuda

#endif // CUDA_API_WRAPPERS_NVRTC_COMPILATION_OPTIONS_HPP_
