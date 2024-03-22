/**
 * @file
 *
 * @brief Definitions and utility functions relating to just-in-time compilation, assembly
 * and linking of CUDA code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_ASSEMBLY_AND_LINK_OPTIONS_HPP_
#define CUDA_API_WRAPPERS_ASSEMBLY_AND_LINK_OPTIONS_HPP_

#include "common_ptx_compilation_options.hpp"
#include "device_properties.hpp"
#include "types.hpp"

#include <array>

namespace cuda {

///@cond
class module_t;
///@endcond

namespace link {

/// Possible strategies for obtaining fully-compiled binary code for a target device
/// when it is not immediately available.
enum fallback_strategy_for_binary_code_t {
	/// Prefer compiling available PTX code to produce fully-compiled binary code
	prefer_compiling_ptx            = 0,
	/// Prefer using existing fully-compiled (binary) code, for a compatible but
	/// not identical target device
	prefer_using_compatible_binary  = 1,
};

namespace detail_ {

/// The CUDA driver's raw generic JIT-related option type
using option_t = CUjit_option;

/**
 * Mechanism for finalizing options into a format readily usable by the
 * link_t wrapper (and by the `cuLink`- functions - but highly inconvenient
 * for inspection and modification.
 *
 * @note Don't create these yourself unless you have to; use @ref options_t
 * instead, and @ref options_t::marshal() when done, for completing the
 * linking-process. If you must create them - use `push_back()` method
 * repeatedly until done with all options.
 */
struct marshalled_options_t {
	/// The CUDA driver's expected type for number of link-related options
	using size_type = unsigned;

	/// The CUDA driver's enum for option identification has this many values -
	/// and thus, there is need for no more than this many marshalled options
	constexpr static const size_type max_num_options { CU_JIT_NUM_OPTIONS };

protected:
	::std::array<option_t, max_num_options> option_buffer;
	::std::array<void*, max_num_options> value_buffer;
	size_type count_ { 0 };
public:
	void push_back(option_t option)
	{
		if (count_ >= max_num_options) {
			throw ::std::invalid_argument("Attempt to push back the same option a second time");
			// If each option is pushed back at most once, the count cannot exist the number
			// of possible options. In fact, it can't even reach it because some options contradict.
			//
			// Note: This check will not catch all repeat push-backs, nor the case of conflicting
			// options - the cuLink methods will catch those. We just want to avoid overflow.
		}
		option_buffer[count_] = option;
		count_++;
	}
protected:
	template <typename I>
	void* process_value(typename ::std::enable_if<::std::is_integral<I>::value, I>::type value)
	{
		return reinterpret_cast<void*>(static_cast<uintptr_t>(value));
	}

	template <typename T>
	void* process_value(T* value)
	{
		return static_cast<void*>(value);
	}

	void* process_value(bool value) { return process_value<int>(value ? 1 : 0); }

	void* process_value(caching_mode_t<memory_operation_t::load> value)
	{
		using ut = typename ::std::underlying_type<caching_mode_t<memory_operation_t::load>>::type;
		return process_value(static_cast<ut>(value));
	}

public:
	/**
	 * This method (alone) is used to populate this structure.
	 *
	 * @note The class is not a standard container, and this method cannot be
	 * reversed or undone, i.e. there is no `pop_back()` or `pop()`.
	 */
	template <typename T>
	void push_back(option_t option, T value)
	{
		push_back(option);
		process_value(value);
		// Now set value_buffer[count-1]...
		value_buffer[count_-1] = process_value(value);
	}

	/// These three methods yield what the CUDA driver actually expects:
	/// Two matching raw buffers and their count of elements
	///@{
	const option_t* options() const { return option_buffer.data(); }
	const void * const * values() const { return value_buffer.data(); }
	size_type count() const { return count_; }
	///@}
};

} // namespace detail_

/**
 * A convenience class for holding, setting and inspecting options for a CUDA binary code
 * linking process - which may also involve PTX compilation.
 *
 * @note This structure does not let you set those options which the CUDA driver documentation
 * describes as having internal purposes only.
 */
struct options_t final : public rtc::common_ptx_compilation_options_t {

	/// options related to logging the link-process
	struct {
		/// Non-error information regarding the logging process (i.e. its "standard output" stream)
		optional<span<char>> info;

		/// Information regarding errors in the logging process (i.e. its "standard error" stream)
		optional<span<char>> error;

		/// Control whether the info and error logging will be verbose
		bool verbose;
	} logs;

	/// Instead of using explicitly-specified binary target, from
	/// @ref common_ptx_compilation_options_t::specific_target - use the device of the current CUDA
	/// context as the target for binary generation
	bool obtain_target_from_cuda_context { true };

	/// Possible strategy for obtaining fully-compiled binary code when it is not
	/// simply available in the input to the link-process
	optional<fallback_strategy_for_binary_code_t> fallback_strategy_for_binary_code;

	// Ignoring the "internal purposes only" options;
	//
	//   CU_JIT_NEW_SM3X_OPT
	//   CU_JIT_FAST_COMPILE
	//   CU_JIT_GLOBAL_SYMBOL_NAMES
	//   CU_JIT_GLOBAL_SYMBOL_ADDRESSES
	//   CU_JIT_GLOBAL_SYMBOL_COUNT
	//
};

namespace detail_ {

/// Construct a easily-driver-usable link-process options structure from
/// a more user-friendly `options_t` structure.
inline marshalled_options_t marshal(const options_t& link_options)
{
	marshalled_options_t marshalled{};
	const auto& lo = link_options;

	if (lo.max_num_registers_per_thread) {
		marshalled.push_back(CU_JIT_MAX_REGISTERS, lo.max_num_registers_per_thread.value());
	}

	if (lo.min_num_threads_per_block) {
		marshalled.push_back(CU_JIT_THREADS_PER_BLOCK, lo.min_num_threads_per_block.value());
	}

	if (lo.logs.info) {
		marshalled.push_back(CU_JIT_INFO_LOG_BUFFER, lo.logs.info.value().data());
		marshalled.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, lo.logs.info.value().size());
	}

	if (lo.logs.error) {
		marshalled.push_back(CU_JIT_ERROR_LOG_BUFFER, lo.logs.error.value().data());
		marshalled.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, lo.logs.error.value().size());
	}

	if (lo.optimization_level) {
		marshalled.push_back(CU_JIT_OPTIMIZATION_LEVEL, lo.optimization_level.value());
	}

	if (lo.obtain_target_from_cuda_context) {
		marshalled.push_back(CU_JIT_TARGET_FROM_CUCONTEXT);
	}
	else if (lo.specific_target) {
		marshalled.push_back(CU_JIT_TARGET, lo.specific_target.value().as_combined_number());
	}

	if (lo.fallback_strategy_for_binary_code) {
		marshalled.push_back(CU_JIT_FALLBACK_STRATEGY, lo.fallback_strategy_for_binary_code.value());
	}

	if (lo.generate_debug_info) {
		marshalled.push_back(CU_JIT_GENERATE_DEBUG_INFO);
	}

	if (lo.generate_source_line_info) {
		marshalled.push_back(CU_JIT_GENERATE_LINE_INFO);
	}

	if (lo.generate_source_line_info) {
		marshalled.push_back(CU_JIT_GENERATE_LINE_INFO);
	}

	if (lo.logs.verbose) {
		marshalled.push_back(CU_JIT_LOG_VERBOSE);
	}

	if (lo.default_load_caching_mode()) {
		marshalled.push_back(CU_JIT_CACHE_MODE, lo.default_load_caching_mode().value());
	}

	return marshalled;
}

} // namespace detail_

// TODO: Compiler "output options":
//
// threads per block targeted
// compilation wall time
// amount written to info log

} // namespace link

} // namespace cuda

#endif // CUDA_API_WRAPPERS_ASSEMBLY_AND_LINK_OPTIONS_HPP_
