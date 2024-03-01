/**
 * @file
 *
 * @brief Definitions and utility functions relating to just-in-time compilation and linking of CUDA code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_COMMON_PTX_COMPILATION_OPTIONS_HPP_
#define CUDA_API_WRAPPERS_COMMON_PTX_COMPILATION_OPTIONS_HPP_


#include "types.hpp"
#include "device.hpp"

#include <array>

namespace cuda {

namespace rtc {

using ptx_register_count_t = int16_t;
using optimization_level_t = int;

} // namespace rtc


enum class memory_operation_t { load, store };

template <memory_operation_t Op> struct caching;

template <> struct caching<memory_operation_t::load> {
	enum mode {
		/**
		 * ca - Cache at all levels, likely to be accessed again.
		 *
		 * The default load instruction cache operation is ld.ca,
		 * which allocates cache lines in all levels (L1 and L2) with
		 * normal eviction policy. Global data is coherent at the L2
		 *  level, but multiple L1 caches are not coherent for global
		 *  data.
		 */
		ca = 0, all = ca, cache_all = ca, cache_at_all_levels = ca, cash_in_l1_and_l2 = ca,

		/**
		 * Cache at global level (cache in L2 and below, not L1).
		 *
		 * Use ld.cg to cache loads only globally, bypassing the L1
		 * cache, and cache only in the L2 cache.
		 */
		cg = 1, global = cg, cache_global = cg, cache_at_global_level = cg, cache_in_l2_only = cache_at_global_level,

		/**
		 * Cache streaming, likely to be accessed once.
		 *
		 * The ld.cs load cached streaming operation allocates global
		 * lines with evict-first policy in L1 and L2 to limit cache
		 * pollution by temporary streaming data that may be accessed
		 * once or twice. When ld.cs is applied to a Local window
		 * address, it performs the ld.lu operation.
		 */
		cs = 2, evict_first = cs, cache_as_evict_first = cs, cache_streaming = cs,

		/**
		 * Last use.
		 *
		 * The compiler/programmer may use ld.lu when restoring spilled
		 * registers and popping function stack frames to avoid needless
		 * write-backs of lines that will not be used again. The ld.lu
		 * instruction performs a load cached streaming operation
		 * (ld.cs) on global addresses.
		 */
		lu = 3, last_use = lu,

		/**
		 * Don't cache and fetch again (consider cached system memory
		 * lines stale, fetch again).
		 *
		 * The ld.cv load operation applied to a global System Memory
		 * address invalidates (discards) a matching L2 line and
		 * re-fetches the line on each new load.
		 */
		cv = 4, dont_cache = cv, fetch_again_and_dont_cache = cv,
	};
	static constexpr const char* mode_names[] = { "ca", "cg", "cs", "lu", "cv" };
};


template <> struct caching<memory_operation_t::store> {
	enum mode {
		wb = 0, write_back = wb, write_back_coherent_levels = wb,
		cg = 1, global = cg, cache_global = cg, cache_at_global_level = cg,
		cs = 2, evict_first = cs, cache_as_evict_first = cs, cache_streaming = cs,
		wt = 3, write_through = wt, write_through_to_system_memory = wt
	};
	static constexpr const char* mode_names[] = { "wb", "cg", "cs", "wt" };
};

template <memory_operation_t Op>
using caching_mode_t = typename caching<Op>::mode;

template <memory_operation_t Op>
const char* name(caching_mode_t<Op> mode)
{
	return caching<Op>::mode_names[static_cast<int>(mode)];
}

template <memory_operation_t Op>
inline ::std::ostream& operator<< (::std::ostream& os, caching_mode_t<Op> lcm)
{
	return os << name(lcm);
}

namespace rtc {

/**
 * The range of optimization level values outside of which the
 * compiler is certain not to support.
 */
constexpr const struct {
	optimization_level_t minimum;
	optimization_level_t maximum;
} valid_optimization_level_range {0, 4};

/**
 * A subset of the options for compiling PTX code into SASS,
 * usable both with the CUDA driver and with NVIDIA's PTX compilation
 * library.
 */
struct common_ptx_compilation_options_t {

	/**
	 * Limit the number of registers which a kernel thread may use.
	 */
	optional<ptx_register_count_t> max_num_registers_per_thread{};

	/**
	 * The minimum number of threads per block which the compiler should target
	 */
	optional<grid::block_dimension_t> min_num_threads_per_block{};

	/**
	 * Compilation optimization level (as in -O1, -O2 etc.)
	 */
	optional<optimization_level_t> optimization_level{};

	optional<device::compute_capability_t> specific_target;

	/**
	 *  Generate indications of which PTX/SASS instructions correspond to which
	 *  lines of the source code, within the compiled output
	 */
	bool generate_source_line_info {false};

	/*
	 * Generate debugging information for within the compiled output (-g)
	 */
	bool generate_debug_info {false};

	/**
	 *  Specifies which of the PTX load caching modes use by default,
	 *  when no caching mode is specified in a PTX instruction
	 */
	///@{
	optional<caching_mode_t<memory_operation_t::load>> default_load_caching_mode_;

	virtual optional<caching_mode_t<memory_operation_t::load>>& default_load_caching_mode()
	{
		return default_load_caching_mode_;
	}
	virtual optional<caching_mode_t<memory_operation_t::load>> default_load_caching_mode() const
	{
		return default_load_caching_mode_;
	}
	///@}


	/**
	 * Generate relocatable code that can be linked with other relocatable device code.
	 *
	 * @note For NVRTC, this is equivalent to specifying "--device-c" ; and if this
	 * option is not specified - that's equivalent to specifying "--device-w".
	 */
	bool generate_relocatable_device_code { false };

	// What about store caching?
};

} // namespace rtc
} // namespace cuda

#endif // CUDA_API_WRAPPERS_COMMON_PTX_COMPILATION_OPTIONS_HPP_
