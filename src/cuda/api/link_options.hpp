/**
 * @file
 *
 * @brief Definitions and utility functions relating to just-in-time compilation and linking of CUDA code.
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

enum fallback_strategy_t {
	prefer_ptx     = 0,
	prefer_binary  = 1,
};


using register_index_t = unsigned;
using option_t = CUjit_option;

struct marshalled_options_t {
	using size_type = unsigned;
	constexpr static const size_type max_num_options { CU_JIT_NUM_OPTIONS };

protected:
	::std::array<option_t, max_num_options> option_buffer;
	::std::array<void*, max_num_options> value_buffer;
	size_type count_ { 0 };
public:
	size_type count() const { return count_; }

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

	template <typename T>
	void push_back(option_t option, T value)
	{
		push_back(option);
		process_value(value);
		// Now set value_buffer[count-1]...
		value_buffer[count_-1] = process_value(value);
	}
	const option_t* options() const { return option_buffer.data(); }
	const void * const * values() const { return value_buffer.data(); }
};

struct options_t final : public rtc::common_ptx_compilation_options_t {

	// Note: The sizes are used as parameters too.
	optional<span<char>> info_log;
	optional<span<char>> error_log;

	// Note: When this is true, the specific_target of the base class
	// is overridden
	bool obtain_target_from_cuda_context { true };

	/// fallback behavior if a (matching cubin???) is not found
	optional<fallback_strategy_t> fallback_strategy;

	// It _seems_ that the verbosity is a boolean setting - but this is not clear
	bool verbose_log;

	bool specify_default_load_caching_mode { false };

	// Ignoring the "internal purposes only" options;
	//
	//   CU_JIT_NEW_SM3X_OPT
	//   CU_JIT_FAST_COMPILE
	//   CU_JIT_GLOBAL_SYMBOL_NAMES
	//   CU_JIT_GLOBAL_SYMBOL_ADDRESSES
	//   CU_JIT_GLOBAL_SYMBOL_COUNT
	//

};

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

	if (lo.info_log) {
		auto data = lo.info_log.value().data();
		marshalled.push_back(CU_JIT_INFO_LOG_BUFFER, data);
		marshalled.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, lo.info_log.value().size());
	}

	if (lo.error_log) {
		marshalled.push_back(CU_JIT_ERROR_LOG_BUFFER, lo.error_log.value().data());
		marshalled.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, lo.error_log.value().size());
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

	if (lo.fallback_strategy) {
		marshalled.push_back(CU_JIT_FALLBACK_STRATEGY, lo.fallback_strategy.value());
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

	if (lo.verbose_log) {
		marshalled.push_back(CU_JIT_LOG_VERBOSE);
	}

	if (lo.default_load_caching_mode()) {
		marshalled.push_back(CU_JIT_CACHE_MODE, lo.default_load_caching_mode().value());
	}

	return marshalled;
}

// TODO: Compiler "output options":
//
// threads per block targeted
// compilation wall time
// amount written to info log

} // namespace link

} // namespace cuda

#endif // CUDA_API_WRAPPERS_ASSEMBLY_AND_LINK_OPTIONS_HPP_
