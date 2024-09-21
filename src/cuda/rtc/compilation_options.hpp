/**
 * @file
 *
 * @brief Definitions and utility functions relating to run-time compilation (RTC)
 * of CUDA code using the NVRTC library
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_RTC_COMPILATION_OPTIONS_HPP_
#define CUDA_API_WRAPPERS_RTC_COMPILATION_OPTIONS_HPP_

#include "cuda/api/detail/option_marshalling.hpp"

#include "../api/device_properties.hpp"
#include "../api/device.hpp"
#include "../api/common_ptx_compilation_options.hpp"

#include <unordered_map>
#include <unordered_set>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>
#include <limits>
#include <iterator>

namespace cuda {

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

inline cpp_dialect_t cpp_dialect_from_name(const char* dialect_name) noexcept(false)
{
	for(auto known_dialect = static_cast<int>(cpp_dialect_t::cpp03);
		known_dialect <= static_cast<int>(cpp_dialect_t::last);
		known_dialect++)
	{
		if (strcmp(detail_::cpp_dialect_names[known_dialect], dialect_name) == 0) {
			return static_cast<cpp_dialect_t>(known_dialect);
		}
	}
	throw ::std::invalid_argument(::std::string("No C++ dialect named \"") + dialect_name + '"');
}

} // namespace detail_

namespace error {

/// Possible ways of handling a potentially problematic finding by the compiler 
/// in the program source code
enum handling_method_t { raise_error = 0, suppress = 1, warn = 2 };

/// Errors, or problematic findings, by the compiler are identified by a number of this type
using number_t = unsigned;

namespace detail_ {

inline const char* option_name_part(handling_method_t method)
{
	static constexpr const char* parts[] = { "error", "suppress", "warn" };
	return parts[method];
}

} // namespace detail_

} // namespace error

/// Compilation options common to all kinds of JIT-compilable programs
template <source_kind_t Kind>
struct compilation_options_base_t {
	template <typename T>
	using optional = cuda::optional<T>;

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

public:
	// TODO: Drop the following methods and make targets a custom
	// inner class which can assigned, added to or subtracted from

	/**
	 * Have the compilation also target a specific compute capability.
	 *
	 * @note previously-specified compute capabilities will be targeted in
	 * addition to the one specified.
	 */
	compilation_options_base_t& add_target(device::compute_capability_t compute_capability)
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
	///@{
	compilation_options_base_t& set_target(device::compute_capability_t compute_capability)
	{
		targets_.clear();
		add_target(compute_capability);
		return *this;
	}

	compilation_options_base_t& set_target(device_t device)
	{
		return set_target(device.compute_capability());
	}
	///@}
}; // compilation_options_base_t

/// Commonly-used phrases regarding the optimization level (e.g. from GCC's
/// command-line arguments), translated into the numeric levels the RTC
/// compilation actually accepts
enum : rtc::optimization_level_t {
	O0 = 0,
	no_optimization = O0,
	O1 = 1,
	O2 = 2,
	O3 = 3,
	maximum_optimization = O3
};

/**
 * Options to be passed to one of the NVIDIA JIT compilers along with a program's source code
 *
 * @note at the raw API level, the options are passed in a simpler form, less convenient for
 * modification. This is handled by the @ref program_t class.
 */
template <source_kind_t Kind>
class compilation_options_t;

/// Options for JIT-compilation of CUDA PTX code
template <>
class compilation_options_t<ptx> final :
	public compilation_options_base_t<ptx>,
	public common_ptx_compilation_options_t {
public:
	///@cond
	using parent = compilation_options_base_t<ptx>;
	using parent::parent;
	///@endcond

	/// Makes the PTX compiler run without producing any CUBIN output (for PTX verification only)
	bool parse_without_code_generation { false };

	/// Allow the JIT compiler to perform expensive optimizations using maximum available resources
	/// (memory and compile-time).
	bool allow_expensive_optimizations_below_O2 { false };

	/**
	 * Compile as patch code for CUDA tools.
     *
	 * @note :
	 *
	 * 1. Cannot Shall not be used in conjunction with @ref parse_without_code_generation
	 *    or {@ref compile_extensible_whole_program}.
	 * 2. Some PTX ISA features may not be usable in this compilation mode.
	 */
	bool compile_as_tools_patch { false };

	/**
	 * Expecting only whole-programs to be directly usable, allow some calls to not be resolved
	 * until device-side linking is performed (see @ref link_t).
	 */
	bool compile_extensible_whole_program { false };

	/// Enable the contraction of multiplcations-followed-by-additions (or subtractions) into single
	/// fused instructions (FMAD, FFMA, DFMA)
	bool use_fused_multiply_add { true };

	/// Print code generation statistics along with the compilation log
	bool verbose { false };

	/**
	 * Prevent the compiler from merging consecutive basic blocks
	 * (@ref https://en.wikipedia.org/wiki/Basic_block) into a single block.
	 *
	 * Normally, the compiler attempts to merge consecutive "basic blocks" as part of its optimization
	 * process. However, for debuggable code this is very confusing.
	 */
	bool dont_merge_basicblocks { false };

	/// The equivalent of suppressing all findings which currently trigger a warning
	bool disable_warnings { false };

	/// Disable use of the "optimizer constant bank" feature
	bool disable_optimizer_constants { false };

	/// Prevents the optimizing away of the return instruction at the end of a program (a kernel?),
	/// making it possible to set a breakpoint just at that point
	bool return_at_end_of_kernel { false };

	/// Generate relocatable references for variables and preserve relocations generated for them in
	/// the linked executable.
	bool preserve_variable_relocations { false };

	/// Warnings about situations likely to result in poor performance
	/// or other problems.
	struct {
		bool double_precision_ops { false };
		bool local_memory_use { false };
		bool registers_spill_to_local_memory { false };
		bool indeterminable_stack_size { true };
		// Does the PTX compiler library actually support this? ptxas does, but the PTX compilation API
		// doesn't mention it
		bool double_demotion { false };
	} situation_warnings;

	/// Limits on the number of registers which generated object code (of different kinds) is allowed
	/// to use
	struct {
		optional<rtc::ptx_register_count_t> kernel {};
		optional<rtc::ptx_register_count_t> device_function {};
	} maximum_register_counts;

	/// Options for fully-specifying a caching mode
	struct caching_mode_spec_t {
		optional<caching_mode_t<memory_operation_t::load>> load {};
		optional<caching_mode_t<memory_operation_t::store>> store {};
	};
	struct {
		/// The caching mode to be used for instructions which don't specify a caching mode
		caching_mode_spec_t default_ {};
		/// A potential forcing of the caching mode, overriding even what instructions themselves
		/// specify
		caching_mode_spec_t forced {};
	} caching_modes;

	/// Get a reference to the caching mode the compiler will be told to use as the default, for load
	/// instructions which don't explicitly specify a particular caching mode.
	optional<caching_mode_t<memory_operation_t::load>>& default_load_caching_mode() override
	{
		return caching_modes.default_.load;
	}

	/// Get the caching mode the compiler will be told to use as the default, for load instructions
	/// which don't explicitly specify a particular caching mode.
	optional<caching_mode_t<memory_operation_t::load>> default_load_caching_mode() const override
	{
		return caching_modes.default_.load;
	}

	/**
	 * Specifies the GPU kernels, or `__global__` functions in CUDA-C++ terms, or `.entry`
	 * functions in PTX terms, for which code must be generated.
	 *
	 * @note The PTX source may contain code for additional `.entry` functions.
	 */
	::std::vector<::std::string> mangled_entry_function_names;

	::std::vector<::std::string>& entries();
	::std::vector<::std::string>& kernels();
	::std::vector<::std::string>& kernel_names();
}; // compilation_options_t<ptx>

/// Options for JIT-compilation of CUDA C++ code
template <>
class compilation_options_t<cuda_cpp> final :
	public compilation_options_base_t<cuda_cpp>,
	public common_ptx_compilation_options_t
{
public:
	using parent = compilation_options_base_t<cuda_cpp>;
	using parent::parent;

	/**
	 * Do extensible whole program compilation of device code.
	 *
	 * @todo explain what that is.
	 */
	bool compile_extensible_whole_program { false };

	/**
	 *  If debug mode is enabled, perform limited optimizations of device code rather than none at all
	 *
	 *  @note It is not possible to force device code optimizations off in NVRTC in non-debug mode with
	 *  '--dopt=off' - that's rejected by NVRTC as an invalid option.
	 */
	bool optimize_device_code_in_debug_mode { false };

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
	 */
	optional<size_t> maximum_register_count { };

	/**
	 * When performing single-precision floating-point operations, flush denormal values to zero.
	 *
	 * @note Setting @ref use_fast_math implies setting this to true.
	 */
	bool flush_denormal_floats_to_zero { false };

	/**
	 * For single-precision floating-point square root, use IEEE round-to-nearest mode or use a faster approximation.
	 *
	 * Setting @ref use_fast_math implies setting this to false.
	 */
	bool use_precise_square_root { true };

	/**
	 * For single-precision floating-point division and reciprocals, use IEEE round-to-nearest mode or use a faster approximation.
	 *
	 * Setting @ref use_fast_math implies setting this to false.
	 */
	bool use_precise_division { true };

	/**
	 * Enables (disables) the contraction of floating-point multiplies and adds/subtracts into floating-point multiply-add operations (FMAD, FFMA, or DFMA).
	 *
	 * Setting @ref use_fast_math implies setting this to false.
	 */
	bool use_fused_multiply_add { true };

	/// Make use of fast math operations. Implies use_fused_multiply_add,
	/// not use_precise_division and not use_precise_square_root.
	bool use_fast_math { false };

	/**
	 * Do not compile fully into PTX/Cubin. Instead, only generate NVIDIA's "LTO IR", which is
	 * combined with other LTO IR pieces from object files compiled with LTO support, at
	 * device link time.
	 */
	bool link_time_optimization { false };

	/// Implicitly add the directories of source files (TODO: Which source files?) as include
	/// file search paths.
	bool source_dirs_in_include_path { true };

	///Enables more aggressive device code vectorization in the LTO IR optimizer.
	bool extra_device_vectorization { false };

	/// The dialect of C++ as which the compiler will be forced to interpret the program source code
	optional<cpp_dialect_t> language_dialect { };

	/// Preprocessor macros to have the compiler define, without specifying a particular value
	::std::unordered_set<::std::string> no_value_defines;

	/// Preprocessor macros to tell the compiler to specifically _un_define.
	::std::unordered_set<::std::string> undefines;

	/// Preprocessor macros to have the compiler define to specific values
	::std::unordered_map<::std::string,::std::string> valued_defines;

	/// Have the compiler treat all warnings as though they were suppressed, and print nothing
	bool disable_warnings { false };

	/// Treat all kernel pointer parameters as if they had the `restrict` (or `__restrict`) qualifier.
	bool assume_restrict { false };

	/// Assume functions without an explicit specification of their execution space are `__device__`
	/// rather than `__host__` functions.
	bool default_execution_space_is_device { false };

	/// Display (error) numbers for warning (and error?) messages, in addition to the message itself.
	bool display_error_numbers { true };

	/// Extra options for the PTX compiler (a.k.a. "PTX optimizing assembler").
	::std::string ptxas;

	/**
	 * A sequence of directories to be searched for headers. These paths are searched _after_ the
	 * list of headers given to nvrtcCreateProgram.
	 *
	 * @note The members here are `::std::string`'s rather than `const char*` or `::std::string_view`'s,
	 * since this class is a value-type, and cannot rely someone else keeping these strings alive.
	 *
	 * @todo In C++17, consider making the elements `::std::filesystem::path`'s.
	 */
	::std::vector<::std::string> additional_include_paths;

	/**
	 * Header files to preinclude during preprocessing of the source.
	 *
	 * @note The members here are `::std::string`'s rather than `const char*` or `::std::string_view`'s,
	 * since this class is a value-type, and cannot rely someone else keeping these strings alive.
	 *
	 * @todo In C++17, consider making the elements `::std::filesystem::path`'s.
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
	 * Use `setrlimit()` to increase the stack size to the maximum the OS allows.
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

	::std::unordered_map<error::number_t, error::handling_method_t> error_handling_overrides;

public: // "shorthands" for more complex option setting

	/// Let the compiler interpret the program source code using its default-assumption for the
	/// C++ language dialect
	compilation_options_t& clear_language_dialect()
	{
		language_dialect = {};
		return *this;
	}

	/// Set which dialect of the C++ language the compiler will try to interpret
	/// the program source code as.
	compilation_options_t& set_language_dialect(cpp_dialect_t dialect)
	{
		language_dialect = dialect;
		return *this;
	}

	/// @copydoc set_language_dialect(cpp_dialect_t)
	compilation_options_t& set_language_dialect(const char* dialect_name)
	{
		return (dialect_name == nullptr or *dialect_name == '\0') ?
			clear_language_dialect() :
			set_language_dialect(detail_::cpp_dialect_from_name(dialect_name));
	}

	/// @copydoc set_language_dialect(cpp_dialect_t)
	compilation_options_t& set_language_dialect(const ::std::string& dialect_name)
	{
		return dialect_name.empty() ?
			clear_language_dialect() :
			set_language_dialect(dialect_name.c_str());
	}

	/// Ignore compiler findings of the specified number (rather than warnings about
	/// them or raising an error)
	compilation_options_t& suppress_error(error::number_t error_number)
	{
		error_handling_overrides[error_number] = error::suppress;
		return *this;
	}

	/// Treat compiler findings of the specified number as an error (rather than
	/// suppressing them or just warning about them)
	compilation_options_t& treat_as_error(error::number_t error_number)
	{
		error_handling_overrides[error_number] = error::raise_error;
		return *this;
	}

	/// Treat compiler findings of the specified number as warnings (rather than
	/// raising an error or ignoring them)
	compilation_options_t& warn_about(error::number_t error_number)
	{
		error_handling_overrides[error_number] = error::warn;
		return *this;
	}
}; // compilation_options_t<cuda_cpp>

template <typename CompilationOptions>
inline ::std::string render(const CompilationOptions& opts)
{
	return marshalling::render(opts);
}

} // namespace rtc

namespace marshalling {

namespace detail_ {

template <typename MarshalTarget, typename Delimiter>
struct gadget<rtc::compilation_options_t<ptx>, MarshalTarget, Delimiter> {
	static void process(
		const rtc::compilation_options_t<ptx> &opts,
		MarshalTarget &marshalled, Delimiter delimiter,
		bool need_delimiter_after_last_option)
	{
		opt_start_t<Delimiter> opt_start { delimiter };
		// TODO: Consider taking an option to be verbose in specifying compilation flags, and setting option values
		//  even when they are the compiler defaults.

		// flags
		if (opts.generate_relocatable_device_code)  { marshalled << opt_start << "--compile-only";                  }
		if (opts.compile_as_tools_patch)            { marshalled << opt_start << "--compile-as-tools-patch";        }
		if (opts.generate_debug_info)               { marshalled << opt_start << "--device-debug";                  }
		if (opts.generate_source_line_info)         { marshalled << opt_start << "--generate-line-info";            }
		if (opts.compile_extensible_whole_program)  { marshalled << opt_start << "--extensible-whole-program";      }
		if (not opts.use_fused_multiply_add)        { marshalled << opt_start << "--fmad false";                    }
		if (opts.verbose)                           { marshalled << opt_start << "--verbose";                       }
		if (opts.dont_merge_basicblocks)            { marshalled << opt_start << "--dont-merge-basicblocks";        }
		{
			const auto& osw = opts.situation_warnings;
			if (osw.double_precision_ops)            { marshalled << opt_start << "--warn-on-double-precision-use";   }
			if (osw.local_memory_use)                { marshalled << opt_start << "--warn-on-local-memory-usage";     }
			if (osw.registers_spill_to_local_memory) { marshalled << opt_start << "--warn-on-spills";                 }
			if (not osw.indeterminable_stack_size)   { marshalled << opt_start << "--suppress-stack-size-warning";    }
			if (osw.double_demotion)                 { marshalled << opt_start << "--suppress-double-demote-warning"; }
		}
		if (opts.disable_warnings)                  { marshalled << opt_start << "--disable-warnings";              }
		if (opts.disable_optimizer_constants)       { marshalled << opt_start << "--disable-optimizer-constants";   }


		if (opts.return_at_end_of_kernel)           { marshalled << opt_start << "--return-at-end";                 }
		if (opts.preserve_variable_relocations)     { marshalled << opt_start << "--preserve-relocs";               }

		// Non-flag single-value options

		if (opts.optimization_level) {
			marshalled << opt_start << "--opt-level" << opts.optimization_level.value();
			if (opts.optimization_level.value() < rtc::O2
				and opts.allow_expensive_optimizations_below_O2)
			{
				marshalled << opt_start << "--allow-expensive-optimizations";
			}
		}

		if (opts.maximum_register_counts.kernel) {
			marshalled << opt_start << "--maxrregcount " << opts.maximum_register_counts.kernel.value();
		}
		if (opts.maximum_register_counts.device_function) {
			marshalled << opt_start << "--device-function-maxrregcount " << opts.maximum_register_counts.device_function.value();
		}

		{
			const auto& ocm = opts.caching_modes;
			if (ocm.default_.load)  { marshalled << opt_start << "--def-load-cache "    << ocm.default_.load.value();  }
			if (ocm.default_.store) { marshalled << opt_start << "--def-store-cache "   << ocm.default_.store.value(); }
			if (ocm.forced.load)    { marshalled << opt_start << "--force-load-cache "  << ocm.forced.load.value();    }
			if (ocm.forced.store)   { marshalled << opt_start << "--force-store-cache " << ocm.forced.store.value();   }
		}

		// Multi-value options

		for(const auto& target : opts.targets_) {
			auto prefix = opts.parse_without_code_generation ? "compute" : "sm";
			marshalled << opt_start << "--gpu-name=" << prefix << '_'  << target.as_combined_number();
		}

		if (not opts.mangled_entry_function_names.empty()) {
			marshalled << opt_start << "--entry";
			bool first = true;
			for (const auto &entry: opts.mangled_entry_function_names) {
				if (first) { first = false; }
				else { marshalled << ','; }
				marshalled << entry;
			}
		}

		if (need_delimiter_after_last_option) {
			marshalled << opt_start; // If no options were marshalled, this does nothing
		}
	}
};

template <typename MarshalTarget, typename Delimiter>
struct gadget<rtc::compilation_options_t<cuda_cpp>, MarshalTarget, Delimiter> {
	static void process(
		const rtc::compilation_options_t<cuda_cpp>& opts, MarshalTarget& marshalled, Delimiter delimiter,
		bool need_delimiter_after_last_option)
	{
		opt_start_t<Delimiter> opt_start { delimiter };
		if (opts.generate_relocatable_device_code)  { marshalled << opt_start << "--relocatable-device-code=true";      }
		if (opts.compile_extensible_whole_program)  { marshalled << opt_start << "--extensible-whole-program=true";     }
		if (opts.generate_debug_info)               { marshalled << opt_start << "--device-debug";                      }
		if (opts.generate_source_line_info)         { marshalled << opt_start << "--generate-line-info";                }
		if (opts.support_128bit_integers)           { marshalled << opt_start << "--device-int128";                     }
		if (opts.indicate_function_inlining)        { marshalled << opt_start << "--optimization-info=inline";          }
		if (opts.compiler_self_identification)      { marshalled << opt_start << "--version-ident=true";                }
		if (not opts.builtin_initializer_list)      { marshalled << opt_start << "--builtin-initializer-list=false";    }
		if (not opts.source_dirs_in_include_path)   { marshalled << opt_start << "--no-source-include ";                }
		if (opts.extra_device_vectorization)        { marshalled << opt_start << "--extra-device-vectorization";        }
		if (opts.disable_warnings)                  { marshalled << opt_start << "--disable-warnings";                  }
		if (opts.assume_restrict)                   { marshalled << opt_start << "--restrict";                          }
		if (opts.default_execution_space_is_device) { marshalled << opt_start << "--device-as-default-execution-space"; }
		if (not opts.display_error_numbers)         { marshalled << opt_start << "--no-display-error-number";           }
		if (not opts.builtin_move_and_forward)      { marshalled << opt_start << "--builtin-move-forward=false";        }
		if (not opts.increase_stack_limit_to_max)   { marshalled << opt_start << "--modify-stack-limit=false";          }
		if (opts.link_time_optimization)            { marshalled << opt_start << "--dlink-time-opt";                    }
		if (opts.use_fast_math)                     { marshalled << opt_start << "--use_fast_math";                     }
		else {
			if (opts.flush_denormal_floats_to_zero) { marshalled << opt_start << "--ftz";                               }
			if (not opts.use_precise_square_root)   { marshalled << opt_start << "--prec-sqrt=false";                   }
			if (not opts.use_precise_division)      { marshalled << opt_start << "--prec-div=false";                    }
			if (not opts.use_fused_multiply_add)    { marshalled << opt_start << "--fmad=false";                        }
		}
		if (opts.optimize_device_code_in_debug_mode) {
			marshalled << opt_start << "--dopt=on";
		}
		if (not opts.ptxas.empty()) {
			marshalled << opt_start << "--ptxas-options=" << opts.ptxas;

		}

		if (opts.language_dialect) {
			marshalled << opt_start << "--std=" << rtc::detail_::cpp_dialect_names[static_cast<unsigned>(opts.language_dialect.value())];
		}

		if (opts.maximum_register_count) {
			marshalled << opt_start << "--maxrregcount=" << opts.maximum_register_count.value();
		}

		// Multi-value options

		for(const auto& target : opts.targets_) {
	#if CUDA_VERSION < 11000
			marshalled << opt_start << "--gpu-architecture=compute_" << target.as_combined_number();
	#else
			marshalled << opt_start << "--gpu-architecture=sm_" << target.as_combined_number();
	#endif
		}

		for(const auto& def : opts.undefines) {
			marshalled << opt_start << "-U" << def;
			// Note: Could alternatively use "--undefine-macro=" instead of "-D"
		}


		for(const auto& def : opts.no_value_defines) {
			marshalled << opt_start << "-D" << def;
			// Note: Could alternatively use "--define-macro=" instead of "-D"
		}

		for(const auto& def : opts.valued_defines) {
			marshalled << opt_start << "-D" << def.first << '=' << def.second;
		}

		for(const auto& path : opts.additional_include_paths) {
			marshalled << opt_start << "--include-path=" << path;
		}

		for(const auto& preinclude_file : opts.preinclude_files) {
			marshalled << opt_start << "--pre-include=" << preinclude_file;
		}

		for(const auto& override : opts.error_handling_overrides) {
			marshalled
				<< opt_start << "--diag-" << rtc::error::detail_::option_name_part(override.second)
				<< '=' << override.first ;
		}

		for(const auto& extra_opt : opts.extra_options) {
			marshalled << opt_start << extra_opt;
		}

		if (need_delimiter_after_last_option) {
			marshalled << opt_start; // If no options were marshalled, this does nothing
		}
	}
};

} // namespace detail_

} // namespace marshalling

} // namespace cuda

#endif // CUDA_API_WRAPPERS_RTC_COMPILATION_OPTIONS_HPP_
