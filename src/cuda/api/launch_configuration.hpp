/**
 * @file
 *
 * @brief Contains the class @ref cuda::launch_configuration_t and some supporting code.
 *
 * @note Launch configurations are used mostly in `kernel_launch.hpp` , and can be built
 * more easily using @ref launch_config_builer_t from `launch_config_builder.hpp`.
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_LAUNCH_CONFIGURATION_CUH_
#define CUDA_API_WRAPPERS_LAUNCH_CONFIGURATION_CUH_

#include "constants.hpp"
#include "types.hpp"

#include <type_traits>
#include <utility>

namespace cuda {

///@cond
class device_t;
class event_t;
class kernel_t;
///@endcond

namespace detail_ {

inline void validate_block_dimensions(grid::block_dimensions_t block_dims)
{
	if (block_dims.volume() == 0) {
		throw ::std::invalid_argument("Zero-volume grid-of-blocks dimensions provided");
	}
}

inline void validate_grid_dimensions(grid::dimensions_t grid_dims)
{
	if (grid_dims.volume() == 0) {
		throw ::std::invalid_argument("Zero-volume block dimensions provided");
	}
}

// Note: The reason for the verbose name is the identity of the block and grid dimension types
void validate_block_dimension_compatibility(const device_t &device, grid::block_dimensions_t block_dims);
void validate_block_dimension_compatibility(const kernel_t &kernel, grid::block_dimensions_t block_dims);

void validate_compatibility(const kernel_t &kernel, memory::shared::size_t shared_mem_size);
void validate_compatibility(const device_t &device, memory::shared::size_t shared_mem_size);

} // namespace detail_

#if CUDA_VERSION >= 12000
enum class cluster_scheduling_policy_t {
	default_ = CU_CLUSTER_SCHEDULING_POLICY_DEFAULT,
	spread = CU_CLUSTER_SCHEDULING_POLICY_SPREAD,
	load_balance = CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING
};
#endif

struct launch_configuration_t {
	grid::composite_dimensions_t dimensions { grid::dimensions_t{ 0u, 0u, 0u }, grid::block_dimensions_t{ 0u, 0u, 0u } };

	/**
	 * The number of bytes each grid block may use, in addition to the statically-allocated
	 * shared memory data inherent in the compiled kernel.
	 */
	memory::shared::size_t dynamic_shared_memory_size { 0u };

	/**
	 * When true, CUDA's "cooperative launch" mechanism will be used, enabling
	 * more flexible device-wide synchronization capabilities; see CUDA Programming
	 * Guide section C.7, Grid Synchronization. (The section talks about "cooperative
	 * groups", but you should ignore those, as they are simply C++ library constructs and do not
	 * in the compiled code).
	 */
	bool block_cooperation { false };

#if CUDA_VERSION >= 12000
	/**
	 * A kernel thus launched, will not await the completion of any previous launched kernel
	 * before its blocks begin to be scheduled. Rather, its threads will be able to use the
	 * `griddepcontrol.wait` PTX instruction (a.k.a. `cudaGridDependencySynchronize()`) -
	 * at some point during their execution - to actually wait for all previous in-flight
	 * kernels (= kernel grids) to conclude. This allows such a subsequent kernel to
	 * perform independent "preamble" tasks concurrently with the execution of its
	 * antecedents on the stream.
	 */
	bool programmatically_dependent_launch { true };

	/**
	 * If this is specified, the pointed-to event will trigger once all kernel threads
	 * have issued the `griddepcontrol.launch_dependents` instruction (a.k.a. the
	 * `cudaTriggerProgrammaticLaunchCompletion()` function).
	 *
	 * @note This is a non-owning pointer; no @ref event_t is allocated or released
	 * while using this class. Also, the actual CUDA event must be valid and not be reused
	 * or destroyed until the kernel concludes and the event fires.
	 *
	 * @note this field is independent of @ref programmatically_dependent_launch , as it
	 * regards the _conclusion_ of the launched kernel, and future kernels which may depend
	 * on it, rather than the beginning of scheduling of the launched kernel and its
	 * dependence on antecedents.
	 */
	struct {
		event_t* event { nullptr };
		// unsigned flags; WHAT ABOUT THE FLAGS?
		bool trigger_event_at_block_start { true };
	} programmatic_completion;

	/**
	 * When set to true, a GPU-scope memory synchronization will not be sufficient
	 * to establish memory activity order between this kernel and kernels in the default,
	 * or any other, memory synchronization domain - even if those kernels are launched
	 * on the same GPU.
	 */
	bool in_remote_memory_synchronization_domain { false };

	/**
	 * Dimensions of each part in the partition of the grid blocks into clusters, which
	 * can pool their shared memory together.
	 */
	struct {
		grid::dimensions_t cluster_dimensions { 1, 1, 1 };
		cluster_scheduling_policy_t scheduling_policy { cluster_scheduling_policy_t::default_ };
	} clustering;
#endif // CUDA_VERSION >= 12000

public: // non-mutators

	/**
	 * Determine whether the configuration includes launch attributes different than the default
	 * values.
	 *
	 * @note The grid dimensions, block dimensions, and dynamic shared memory size are not
	 * considered launch attributes, and their settings does not affect the result of this method.
	 */
	bool has_nondefault_attributes() const
	{
		if (block_cooperation) { return true; }
#if CUDA_VERSION >= 12000
		return  programmatically_dependent_launch or programmatic_completion.event
			or in_remote_memory_synchronization_domain or clustering.cluster_dimensions != grid::dimensions_t::point();
#else
		return false;
#endif
	}

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
		memory::shared::size_t dynamic_shared_mem = 0u
	) :
		dimensions{grid_and_block_dimensions},
		dynamic_shared_memory_size{dynamic_shared_mem}
	{ }

	constexpr launch_configuration_t(
		grid::dimensions_t grid_dims,
		grid::dimensions_t block_dims,
		memory::shared::size_t dynamic_shared_mem = 0u
	) : launch_configuration_t( {grid_dims, block_dims}, dynamic_shared_mem) { }

	// A "convenience" delegating ctor to avoid narrowing-conversion warnings
	constexpr launch_configuration_t(
		int grid_dims,
		int block_dims,
		memory::shared::size_t dynamic_shared_mem = 0u
	) : launch_configuration_t(
		grid::dimensions_t(grid_dims),
		grid::block_dimensions_t(block_dims),
		dynamic_shared_mem)
	{ }

	CPP14_CONSTEXPR launch_configuration_t& operator=(const launch_configuration_t& other) = default;
	CPP14_CONSTEXPR launch_configuration_t& operator=(launch_configuration_t&&) = default;
};

constexpr bool operator==(const launch_configuration_t lhs, const launch_configuration_t& rhs) noexcept
{
	return
		lhs.dimensions == rhs.dimensions
	    and lhs.dynamic_shared_memory_size == rhs.dynamic_shared_memory_size
		and lhs.block_cooperation == rhs.block_cooperation
#if CUDA_VERSION >= 12000
		and lhs.programmatically_dependent_launch == rhs.programmatically_dependent_launch
		and lhs.programmatic_completion.event == rhs.programmatic_completion.event
		and lhs.in_remote_memory_synchronization_domain == rhs.in_remote_memory_synchronization_domain
		and lhs.clustering.cluster_dimensions == rhs.clustering.cluster_dimensions
		and lhs.clustering.scheduling_policy == rhs.clustering.scheduling_policy
#endif // CUDA_VERSION >= 12000
		;
}

constexpr bool operator!=(const launch_configuration_t lhs, const launch_configuration_t& rhs) noexcept
{
	return not (lhs == rhs);
}

namespace detail_ {

// Note: This will not check anything related to the device or the kernel
// with which the launch configuration is to be used
inline void validate(const launch_configuration_t& launch_config) noexcept(false)
{
	validate_block_dimensions(launch_config.dimensions.block);
	validate_grid_dimensions(launch_config.dimensions.grid);
}

inline void validate_compatibility(
	const device_t& device,
	launch_configuration_t launch_config) noexcept(false)
{
	validate(launch_config);
	validate_block_dimension_compatibility(device, launch_config.dimensions.block);
	//  Uncomment if we actually get such checks
	//	validate_grid_dimension_compatibility(device, launch_config.dimensions.grid);
}

void validate_compatibility(
	const kernel_t& kernel,
	launch_configuration_t launch_config) noexcept(false);

using launch_attribute_index_t = unsigned int;

// ensure we have the same number here as the number of attribute insertions in marsha()
constexpr launch_attribute_index_t maximum_possible_kernel_launch_attributes = 7;

#if CUDA_VERSION >= 12000
// Note: The atttribute_storage must have a capacity of maximum_possible_kernel_launch_attributes+1 at least
CUlaunchConfig marshal(
	const launch_configuration_t& config,
	const stream::handle_t stream_handle,
	span<CUlaunchAttribute> attribute_storage) noexcept(true);
#endif // CUDA_VERSION >= 12000

} // namespace detail_

} // namespace cuda

#endif // CUDA_API_WRAPPERS_LAUNCH_CONFIGURATION_CUH_
