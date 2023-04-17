/**
 * @file
 *
 * @brief Contains the @ref launch_configuration_t class and some auxiliary
 * functions for it.
 *
 * @note Launch configurations are  used mostly in @ref kernel_launch.hpp . 
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_LAUNCH_CONFIGURATION_CUH_
#define CUDA_API_WRAPPERS_LAUNCH_CONFIGURATION_CUH_

#include "constants.hpp"
#include "types.hpp"

#include <type_traits>
#include <utility>

namespace cuda {

struct launch_configuration_t {
	grid::composite_dimensions_t dimensions {0 , 0 };

	/**
	 * The number of bytes each grid block may use, in addition to the statically-allocated
	 * shared memory data inherent in the compiled kernel.
	 */
	memory::shared::size_t      dynamic_shared_memory_size { 0u };

	/**
	 * When true, CUDA's "cooperative launch" mechanism will be used, enabling
	 * more flexible device-wide synchronization capabilities; see CUDA Programming
	 * Guide section C.7, Grid Synchronization. (The section talks about "cooperative
	 * groups", but you should ignore those, as they are simply C++ library constructs and do not
	 * in the compiled code).
	 */
	bool                        block_cooperation { false };

	// In C++11, an inline initializer for a struct's field costs us a lot
	// of its defaulted constructors; but - we must initialize the shared
	// memory size to 0, as otherwise, people might be tempted to initialize
	// a launch configuration with { num_blocks, num_threads } - and get an
	// uninitialized shared memory size which they did not expect. So,
	// we do have the inline initializers above regardless of the language
	// standard version, and we just have to "pay the price" of spelling things out:
	launch_configuration_t() = delete;
	constexpr launch_configuration_t(const launch_configuration_t&) = default;
	constexpr launch_configuration_t(launch_configuration_t&&) = default;

	constexpr launch_configuration_t(
		grid::composite_dimensions_t grid_and_block_dimensions,
		memory::shared::size_t dynamic_shared_mem = 0u,
		bool thread_block_cooperation = false
	) :
		dimensions{grid_and_block_dimensions},
		dynamic_shared_memory_size{dynamic_shared_mem},
		block_cooperation{thread_block_cooperation}
	{ }

	constexpr launch_configuration_t(
		grid::dimensions_t grid_dims,
		grid::dimensions_t block_dims,
		memory::shared::size_t dynamic_shared_mem = 0u,
		bool thread_block_cooperation = false
	) : launch_configuration_t(
		{grid_dims, block_dims},
		dynamic_shared_mem,
		thread_block_cooperation)
	{ }

	// A "convenience" delegating ctor to avoid narrowing-conversion warnings
	constexpr launch_configuration_t(
		int grid_dims,
		int block_dims,
		memory::shared::size_t dynamic_shared_mem = 0u,
		bool thread_block_cooperation = false
	) : launch_configuration_t(
		grid::dimensions_t(grid_dims),
		grid::block_dimensions_t(block_dims),
		dynamic_shared_mem,
		thread_block_cooperation)
	{ }

   // These can be made constexpr in C++14
   launch_configuration_t& operator=(const launch_configuration_t& other) = default;
   launch_configuration_t& operator=(launch_configuration_t&&) = default;
};

/**
 * @brief a named constructor idiom for a @ref launch_config_t
 */
constexpr launch_configuration_t make_launch_config(
	grid::composite_dimensions_t grid_and_block_dimensions,
	memory::shared::size_t      dynamic_shared_memory_size = 0u,
	bool                        block_cooperation = false) noexcept
{
	return { grid_and_block_dimensions, dynamic_shared_memory_size, block_cooperation };
}

constexpr launch_configuration_t make_launch_config(
	grid::dimensions_t         grid_dimensions,
	grid::block_dimensions_t   block_dimensions,
	memory::shared::size_t     dynamic_shared_memory_size = 0u,
	bool                       block_cooperation = false) noexcept
{
	return { { grid_dimensions, block_dimensions }, dynamic_shared_memory_size, block_cooperation };
}

constexpr bool operator==(const launch_configuration_t lhs, const launch_configuration_t& rhs) noexcept
{
	return
		lhs.dimensions == rhs.dimensions and
			lhs.dynamic_shared_memory_size == rhs.dynamic_shared_memory_size and
			lhs.block_cooperation == rhs.block_cooperation;
}

constexpr bool operator!=(const launch_configuration_t lhs, const launch_configuration_t& rhs) noexcept
{
	return not (lhs == rhs);
}

namespace detail_ {

inline void validate(launch_configuration_t launch_config) noexcept(false)
{
	if (launch_config.dimensions.grid.volume() == 0) {
		throw ::std::invalid_argument("Launch config specifies a zero-volume grid-of-blocks");
	}
	if (launch_config.dimensions.block.volume() == 0) {
		throw ::std::invalid_argument("Launch config specifies a zero-volume block dimensions");
	}
	// TODO: Consider adding device-specific validations here, like checking for
	// block size limits, shared mem size limits etc - by taking an optional device
	// as a parameter
}

} // namespace detail_


} // namespace cuda

#endif // CUDA_API_WRAPPERS_LAUNCH_CONFIGURATION_CUH_
