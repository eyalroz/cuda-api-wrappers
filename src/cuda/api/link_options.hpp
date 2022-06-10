/**
 * @file
 *
 * @brief Definitions and utility functions relating to just-in-time compilation and linking of CUDA code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_ASSEMBLY_AND_LINK_OPTIONS_HPP_
#define CUDA_API_WRAPPERS_ASSEMBLY_AND_LINK_OPTIONS_HPP_


#include <cuda/api/types.hpp>

#if __cplusplus >= 202002L
#include <span>
#endif
#include <array>

namespace cuda {

///@cond
class module_t;
///@endcond

namespace link {

enum input_type_t {
	cubin,    /// Compiled device-class-specific device code
	ptx,      /// PTX (microarchitecture-inspecific intermediate representation)
	fatbin,   /// A bundle of multiple cubin and/or PTX inputs; typically
	object,   /// A host-side binary object with embedded device code; a `.o` file
	library,  /// An archive of objects files with embedded device code; a `.a` file
} ;

enum fallback_strategy_t {
	prefer_ptx     = 0,
	prefer_binary  = 1,
};

enum class caching_mode_t  {

	/**
	 * ca - Cache at all levels, likely to be accessed again.
	 *
	 * The default load instruction cache operation is ld.ca,
	 * which allocates cache lines in all levels (L1 and L2) with
	 * normal eviction policy. Global data is coherent at the L2
	 *  level, but multiple L1 caches are not coherent for global
	 *  data.
	 */
	cash_at_all_levels,
	cash_in_l1_and_l2 = cash_at_all_levels,
	ca = cash_at_all_levels,

	/**
	 * Cache at global level (cache in L2 and below, not L1).
	 *
	 * Use ld.cg to cache loads only globally, bypassing the L1
	 * cache, and cache only in the L2 cache.
	 */
	cache_at_global_level,
	cache_in_l2_only = cache_at_global_level,
	cg = cache_at_global_level,

	/**
	 * Cache streaming, likely to be accessed once.
	 *
	 * The ld.cs load cached streaming operation allocates global
	 * lines with evict-first policy in L1 and L2 to limit cache
	 * pollution by temporary streaming data that may be accessed
	 * once or twice. When ld.cs is applied to a Local window
	 * address, it performs the ld.lu operation.
	 */
	cache_as_evict_first,
	cache_streaming = cache_as_evict_first,
	cs = cache_streaming,

	/**
	 * Last use.
	 *
	 * The compiler/programmer may use ld.lu when restoring spilled
	 * registers and popping function stack frames to avoid needless
	 * write-backs of lines that will not be used again. The ld.lu
	 * instruction performs a load cached streaming operation
	 * (ld.cs) on global addresses.
	 */
	last_use,
	lu = last_use,

	/**
	 * Don't cache and fetch again (consider cached system memory
	 * lines stale, fetch again).
	 *
	 * The ld.cv load operation applied to a global System Memory
	 * address invalidates (discards) a matching L2 line and
	 * re-fetches the line on each new load.
	 */
	 fetch_again_and_dont_cache,
	 cv = fetch_again_and_dont_cache,
};

using register_index_t = unsigned;
using optimization_level_t = unsigned;
using option_t = CUjit_option;
constexpr const optimization_level_t maximum_optimization_level { 4 };

struct marshalled_options_t {
	using size_type = unsigned;
	constexpr static const size_type max_num_options { CU_JIT_NUM_OPTIONS };

protected:
	::std::array<option_t, max_num_options> option_buffer;
	::std::array<void*, max_num_options> value_buffer;
	size_type count_ { 0 };
public:
	size_type count() { return count_; }

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

	void* process_value(caching_mode_t value)
	{
		return process_value(static_cast<typename ::std::underlying_type<caching_mode_t>::type>(value));
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

struct options_t {

	static constexpr const register_index_t no_max_registers_limit { 0 };

	/**
	 * Limit the number of registers which a kernel thread may use.
	 *
	 * @todo Use an optional.
	 */
	register_index_t max_num_registers_per_thread { no_max_registers_limit };

	static constexpr const register_index_t no_min_num_threads_per_block { 0 };

	/**
	 * The minimum number of threads per block which the compiler should target
	 * @note can't be combined with a value for the @ref target property.
	 *
	 * @todo Use an optional.
	 */
	grid::block_dimension_t min_num_threads_per_block { no_min_num_threads_per_block };

	// Note: The sizes are used as parameters too.
	span<char> info_log, error_log;

	static constexpr const optimization_level_t dont_set_optimization_level { maximum_optimization_level + 1 };
	/**
	 * Compilation optimization level (as in -O1, -O2 etc.)
	 *
	 * @todo Use an optional.
	 */
	optimization_level_t optimization_level { dont_set_optimization_level };

	/**
	 *
	 * @todo Use a variant or optional+variant.
	 */
	struct {
		bool obtain_from_cuda_context { true };
		bool use_specific { true };
		device::compute_capability_t specific;
	} target; // Can't be combined with CU_JIT_THREADS_PER_BLOCK

	bool specify_fallback_strategy { false };
	/**
	 * @todo Use an optional.
	 */
	fallback_strategy_t fallback_strategy { prefer_ptx }; // fallback behavior if a cubin matching (WHAT?) is not found

	/**
	 *  Whether or not to generate indications of which PTX/SASS instructions correspond to which
	 *  lines of the source code, within the compiled output (-lineinfo)
	 */
	bool generate_debug_information { false }; /// Whether or not to generate debug information within the compiled output (-g)
	bool generate_source_line_number_information { false };

	// It _seems_ that the verbosity is a boolean setting - but this is not clear
	bool verbose_log;

	bool specify_default_load_caching_mode { false };
	/**
	 *  Specifies which of the PTX load caching modes use by default,
	 *  when no caching mode is specified in a PTX instruction  (-dlcm)
	 */
	caching_mode_t default_load_caching_mode;

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
	marshalled_options_t marshalled;
	const auto& lo = link_options;

	if (lo.max_num_registers_per_thread != lo.no_max_registers_limit) {
		marshalled.push_back(CU_JIT_MAX_REGISTERS, lo.max_num_registers_per_thread);
	}

	if (lo.min_num_threads_per_block != lo.no_min_num_threads_per_block) {
		marshalled.push_back(CU_JIT_THREADS_PER_BLOCK, lo.min_num_threads_per_block);
	}

	auto cil = const_cast<span<char>*>(&lo.info_log);
	if (cil->data() != nullptr and cil->size() != 0) {
		marshalled.push_back(CU_JIT_INFO_LOG_BUFFER, cil->data());
		marshalled.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES, cil->size());
	}

	auto cel = const_cast<span<char>*>(&lo.error_log);
	if (cel->data() != nullptr and cel->size() != 0) {
		marshalled.push_back(CU_JIT_ERROR_LOG_BUFFER, cel->data());
		marshalled.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, cel->size());
	}

	if (lo.optimization_level != lo.dont_set_optimization_level) {
		marshalled.push_back(CU_JIT_OPTIMIZATION_LEVEL, lo.optimization_level);
	}

	if (lo.target.obtain_from_cuda_context) {
		marshalled.push_back(CU_JIT_TARGET_FROM_CUCONTEXT);
	}
	else if (lo.target.use_specific) {
		marshalled.push_back(CU_JIT_TARGET, lo.target.specific.as_combined_number());
	}

	if (lo.specify_fallback_strategy) {
		marshalled.push_back(CU_JIT_FALLBACK_STRATEGY, lo.fallback_strategy);
	}

	if (lo.generate_debug_information) {
		marshalled.push_back(CU_JIT_GENERATE_DEBUG_INFO);
	}

	if (lo.generate_source_line_number_information) {
		marshalled.push_back(CU_JIT_GENERATE_LINE_INFO);
	}

	if (lo.generate_source_line_number_information) {
		marshalled.push_back(CU_JIT_GENERATE_LINE_INFO);
	}

	if (lo.verbose_log) {
		marshalled.push_back(CU_JIT_LOG_VERBOSE);
	}

	if (lo.specify_default_load_caching_mode) {
		marshalled.push_back(CU_JIT_CACHE_MODE, lo.default_load_caching_mode);
	}

	return marshalled;
}

// TODO: Compiler "output options":
//
// threads per block targeted
// compilation wall time
// amount written to info log

} // namespace assembly_and_link

} // namespace cuda

#endif // CUDA_API_WRAPPERS_ASSEMBLY_AND_LINK_OPTIONS_HPP_
