/**
 * @file
 *
 * @brief Contains the @ref launch
 *
 * @note Launch configurations are  used mostly in @ref kernel_launch.hpp . 
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_LAUNCH_CONFIG_BUILDER_CUH_
#define CUDA_API_WRAPPERS_LAUNCH_CONFIG_BUILDER_CUH_

#include "launch_configuration.hpp"
#include "kernel.hpp"
#include "device.hpp"

namespace cuda {


namespace detail_ {

struct no_value_t {};

// This is pretty unsafe; don't use this at home, kids!
template <typename T>
struct poor_mans_optional {
	static_assert(::std::is_trivially_destructible<T>::value, "Use a simpler type");
	union maybe_value_union_t {
		no_value_t no_value;
		T value;
	};

	poor_mans_optional& operator=(const T& value) {
		is_set = true;
		maybe_value.value = value;
		return *this;
	}

	poor_mans_optional& operator=(const T&& value) {
		is_set = true;
		maybe_value.value = ::std::move(value);
		return *this;
	}

	poor_mans_optional& operator=(no_value_t) {
		is_set = false;
		return *this;
	}

	poor_mans_optional& operator=(T&& value) { return *this = value; }
	poor_mans_optional() noexcept : maybe_value{ no_value_t{} } { }
	poor_mans_optional(T v) : is_set(true) {
		maybe_value.value = v;
	}
	poor_mans_optional(const poor_mans_optional& other)
	{
		if (other) {
			*this = other.value();
		}
		is_set = other.is_set;
	}
	~poor_mans_optional() noexcept { };

	T value() const { return maybe_value.value; }

	operator bool() const noexcept { return is_set; }
	void clear() noexcept { is_set = false; }
	void unset() noexcept { is_set = false; }

	bool is_set { false };
	maybe_value_union_t maybe_value;
};


} // namespace detail_

namespace grid {

namespace detail_ {

inline dimension_t div_rounding_up(overall_dimension_t dividend, block_dimension_t divisor)
{
	dimension_t quotient = (dimension_t) dividend / divisor;
		// It is up to the caller to ensure we don't overlow the dimension_t type
	return (divisor * quotient == dividend) ? quotient : quotient + 1;
}

inline dimensions_t div_rounding_up(overall_dimensions_t overall_dims, block_dimensions_t block_dims)
{
	return {
		div_rounding_up(overall_dims.x, block_dims.x),
		div_rounding_up(overall_dims.y, block_dims.y),
		div_rounding_up(overall_dims.z, block_dims.z)
	};
}

// Note: We're not implementing a grid-to-block rounding up here, since - currently -
// block_dimensions_t is the same as grid_dimensions_t.

} // namespace detail_

} // namespace grid

class launch_config_builder_t {
public:
	void resolve_dimensions()  {
		grid::composite_dimensions_t cd = get_composite_dimensions();
		dimensions_.block = cd.block;
		dimensions_.grid = cd.grid;
		if (not dimensions_.overall) {
			dimensions_.overall = cd.grid * cd.block;
		}
	}

protected:
	memory::shared::size_t  get_dynamic_shared_memory_size(grid::block_dimensions_t block_dims) const
	{
		return (memory::shared::size_t)	((dynamic_shared_memory_size_determiner_ == nullptr) ?
			dynamic_shared_memory_size_ :
			dynamic_shared_memory_size_determiner_((int) block_dims.volume()));
			// Q: Why the need for type conversion?
			// A: MSVC is being a bit finicky here for some reason
	}

#ifndef NDEBUG
	grid::composite_dimensions_t get_unvalidated_composite_dimensions() const noexcept(false)
#else
	grid::composite_dimensions_t get_composite_dimensions() const noexcept(false)
#endif
	{
		grid::composite_dimensions_t result;
		if (saturate_with_active_blocks_) {
			if (use_min_params_for_max_occupancy_) {
				throw ::std::logic_error(
					"Cannot both use the minimum grid parameters for achieving maximum occupancy, _and_ saturate "
					"the grid with fixed-size cubs.");
			}
			if (not (kernel_)) {
				throw ::std::logic_error("A kernel must be set to determine how many blocks are required to saturate the device");
			}
			if (not (dimensions_.block)) {
				throw ::std::logic_error("The block dimensions must be known to determine how many of them one needs for saturating a device");
			}
			if (dimensions_.grid or dimensions_.overall) {
				throw ::std::logic_error("Conflicting specifications: Grid or overall dimensions specified, but requested to saturate kernels with active blocks");
			}

			result.block = dimensions_.block.value();
			auto dshmem_size = get_dynamic_shared_memory_size(dimensions_.block.value());
			auto num_block_threads = (grid::block_dimension_t) dimensions_.block.value().volume();
			result.grid = kernel_->max_active_blocks_per_multiprocessor(num_block_threads, dshmem_size);
			return result;
		}
		if (use_min_params_for_max_occupancy_) {
			if (not (kernel_)) {
				throw ::std::logic_error("A kernel must be set to determine the minimum grid parameter sfor m");
			}
			if (dimensions_.block or dimensions_.grid or dimensions_.overall) {
				throw ::std::logic_error("Conflicting specifications: Grid or overall dimensions specified, but requested to saturate kernels with active blocks");
			}
			auto composite_dims = dynamic_shared_memory_size_determiner_ ?
								  kernel_->min_grid_params_for_max_occupancy(dynamic_shared_memory_size_determiner_) :
								  kernel_->min_grid_params_for_max_occupancy(dynamic_shared_memory_size_);
			result.block = composite_dims.block;
			result.grid = composite_dims.grid;
			return result;
		}
		if (dimensions_.block and dimensions_.overall) {
			result.grid = grid::detail_::div_rounding_up(dimensions_.overall.value(), dimensions_.block.value());
			result.block = dimensions_.block.value();
			return result;
		}
		if (dimensions_.grid and dimensions_.overall) {
			result.block = grid::detail_::div_rounding_up(dimensions_.overall.value(), dimensions_.grid.value());
			result.grid = dimensions_.grid.value();
			return result;
		}
		if (dimensions_.grid and dimensions_.block) {
			result.block = dimensions_.block.value();
			result.grid = dimensions_.grid.value();
			return result;
		}

		if (not dimensions_.block and not dimensions_.grid) {
			throw ::std::logic_error(
				"Neither block nor grid dimensions have been specified");
		} else if (not dimensions_.block and not dimensions_.overall) {
			throw ::std::logic_error(
				"Grid dimensions only been specified in terms of blocks, not threads, and no block dimensions specified");
		} else { // it must be the case that (not dimensions_.block and not dimensions_.overall)
			throw ::std::logic_error(
				"Only block dimensions have been specified - cannot resolve launch grid dimensions");
		}
	}

#ifndef NDEBUG
	grid::composite_dimensions_t get_composite_dimensions() const noexcept(false)
	{
		auto result = get_unvalidated_composite_dimensions();
		validate_composite_dimensions(result);
		return result;
	}
#endif

public:
	launch_configuration_t build() const
	{
		auto composite_dims = get_composite_dimensions();
		auto dynamic_shmem_size = get_dynamic_shared_memory_size(composite_dims.block);

		return launch_configuration_t{composite_dims, dynamic_shmem_size, thread_block_cooperation};
	}

protected:
	template <typename T>
	using optional = detail_::poor_mans_optional<T>;

	struct {
		optional<grid::block_dimensions_t  > block;
		optional<grid::dimensions_t        > grid;
		optional<grid::overall_dimensions_t> overall;
	} dimensions_;

	bool thread_block_cooperation { false };

	// Note: We could have used a variant between these two;
	// but the semantic is that if the determiner is not null, we use it;
	// and if you want to force a concrete apriori value, then you nullify
	// the determiner
	kernel::shared_memory_size_determiner_t dynamic_shared_memory_size_determiner_ {nullptr };
	memory::shared::size_t dynamic_shared_memory_size_ { 0 };

	const kernel_t* kernel_ { nullptr };
	optional<device::id_t> device_;
	bool saturate_with_active_blocks_ { false };
	bool use_min_params_for_max_occupancy_ { false };

	static cuda::device_t device(detail_::poor_mans_optional<device::id_t> maybe_id)
	{
		return cuda::device::get(maybe_id.value());
	}

	cuda::device_t device() const { return device(device_.value()); }

	launch_config_builder_t& operator=(launch_configuration_t config)
	{
		thread_block_cooperation = config.block_cooperation;
		dynamic_shared_memory_size_ = config.dynamic_shared_memory_size;
#ifndef NDEBUG
		block_dims_acceptable_to_kernel_or_device(config.dimensions.block);
#endif
		dimensions(config.dimensions);
		return *this;
	}

#ifndef NDEBUG
	static void compatible(
		const kernel_t*         kernel_ptr,
		memory::shared::size_t  shared_mem_size)
	{
		if (kernel_ptr == nullptr) { return; }
		if (shared_mem_size == 0) { return; }
		auto max_shared = kernel_ptr->get_maximum_dynamic_shared_memory_per_block();
		if (shared_mem_size > max_shared) {
			throw ::std::invalid_argument("Requested dynamic shared memory size "
				+ ::std::to_string(shared_mem_size) + " exceeds kernel's maximum allowed value of "
				+ ::std::to_string(max_shared));
		}
	}

	static void compatible(
		detail_::poor_mans_optional<device::id_t> maybe_device_id,
		memory::shared::size_t                    shared_mem_size)
	{
		if (not maybe_device_id) { return; }
		if (shared_mem_size == 0) { return; }
		auto max_shared = device(maybe_device_id).properties().max_shared_memory_per_block();
		if (shared_mem_size > max_shared) {
			throw ::std::invalid_argument(
				"Requested dynamic shared memory size " + ::std::to_string(shared_mem_size)
				+ " exceeds the device maximum of " + ::std::to_string(max_shared));
		}
	}

	void validate_dynamic_shared_memory_size(memory::shared::size_t size)
	{
		compatible(kernel_, size);
		compatible(device_, size);
	}

	// Note: This ignores the value of dimensions.grid an dimensions.faltatt
	static void compatible(
		const kernel_t*          kernel_ptr,
		grid::block_dimensions_t block_dims)
	{
		if (kernel_ptr == nullptr) { return; }
		auto max_block_size = kernel_ptr->maximum_threads_per_block();
		auto volume = block_dims.volume();
		if (volume > max_block_size) {
			throw ::std::invalid_argument(
				"specified block dimensions result in blocks of size " + ::std::to_string(volume)
				+ ", exceeding the maximum possible block size of " + ::std::to_string(max_block_size)
				+ " for " + kernel::detail_::identify(*kernel_ptr));
		}
	}

	static void compatible(
		detail_::poor_mans_optional<device::id_t>  maybe_device_id,
		grid::block_dimensions_t                   block_dims)
	{
		if (not maybe_device_id) { return; }
		auto dev = device(maybe_device_id);
		auto max_block_size = dev.maximum_threads_per_block();
		auto volume = block_dims.volume();
		if (volume > max_block_size) {
			throw ::std::invalid_argument(
			"specified block dimensions result in blocks of size " + ::std::to_string(volume)
			+ ", exceeding the maximum possible block size of " + ::std::to_string(max_block_size)
			+ " for " + device::detail_::identify(dev.id()));
		}
		auto dim_maxima  = grid::block_dimensions_t{
			(grid::block_dimension_t) dev.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X),
			(grid::block_dimension_t) dev.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y),
			(grid::block_dimension_t) dev.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z)
		};
		auto check =
			[dev](grid::block_dimension_t dim, grid::block_dimension_t max, const char* axis) {
				if (max > dim) {
					throw ::std::invalid_argument(
						::std::string("specified block ") + axis + "-axis dimension " + ::std::to_string(dim)
						+ " exceeds the maximum supported " + axis + " dimension of " + ::std::to_string(max)
						+ " for " + device::detail_::identify(dev.id()));
				}
			};
		check(block_dims.x, dim_maxima.x, "X");
		check(block_dims.y, dim_maxima.y, "Y");
		check(block_dims.z, dim_maxima.z, "Z");
	}

	void block_dims_acceptable_to_kernel_or_device(grid::block_dimensions_t block_dims) const
	{
		compatible(kernel_, block_dims);
		compatible(device_, block_dims);
	}

	static void dimensions_compatible(
		grid::block_dimensions_t   block,
		grid::dimensions_t         grid,
		grid::overall_dimensions_t overall)
	{
		if (grid * block != overall) {
			throw ::std::invalid_argument("specified block, grid and overall dimensions do not agree");
		}
	}

	void validate_block_dimensions(grid::block_dimensions_t block_dims) const
	{
		if (dimensions_.grid and dimensions_.overall) {
			dimensions_compatible(block_dims, dimensions_.grid.value(), dimensions_.overall.value());
		}
		block_dims_acceptable_to_kernel_or_device(block_dims);
	}

	void validate_grid_dimensions(grid::dimensions_t grid_dims) const
	{
		if (dimensions_.block and dimensions_.overall) {
			if (grid_dims * dimensions_.block.value() != dimensions_.overall.value()) {
				throw ::std::invalid_argument(
				"specified grid dimensions conflict with the already-specified "
				"block and overall dimensions");
			}
		}
	}

	void validate_overall_dimensions(grid::overall_dimensions_t overall_dims) const
	{
		if (dimensions_.block and dimensions_.grid) {
			if (dimensions_.grid.value() * dimensions_.block.value() != overall_dims) {
				throw ::std::invalid_argument(
				"specified overall dimensions conflict with the already-specified "
				"block and grid dimensions");
			}
		}
	}

	void validate_kernel(const kernel_t* kernel_ptr) const
	{
		if (dimensions_.block or (dimensions_.grid and dimensions_.overall)) {
			auto block_dims = dimensions_.block ?
						dimensions_.block.value() :
						get_composite_dimensions().block;
			compatible(kernel_ptr, block_dims);
		}
		compatible(kernel_ptr, dynamic_shared_memory_size_);
	}

	void validate_device(device::id_t device_id) const
	{
		if (dimensions_.block or (dimensions_.grid and dimensions_.overall)) {
			auto block_dims = dimensions_.block ?
				dimensions_.block.value() :
				get_composite_dimensions().block;
			compatible(device_id, block_dims);
		}
		compatible(device_id, dynamic_shared_memory_size_);
	}

	void validate_composite_dimensions(grid::composite_dimensions_t composite_dims) const
	{
		compatible(kernel_, composite_dims.block);
		compatible(device_, composite_dims.block);

		// Is there anything to validate regarding the grid dims?
	}
#endif // ifndef NDEBUG

public:
	launch_config_builder_t& dimensions(grid::composite_dimensions_t composite_dims)
	{
#ifndef NDEBUG
		validate_composite_dimensions(composite_dims);
#endif
		dimensions_.overall.unset();
		dimensions_.grid = composite_dims.grid;
		dimensions_.block = composite_dims.block;
		return *this;
	}

	launch_config_builder_t& block_dimensions(grid::block_dimensions_t dims)
	{
#ifndef NDEBUG
		validate_block_dimensions(dims);
#endif
		dimensions_.block = dims;
		if (dimensions_.grid) {
			dimensions_.overall.unset();
		}
		return *this;

	}

	launch_config_builder_t& block_dimensions(
		grid::block_dimension_t x,
		grid::block_dimension_t y = 1,
		grid::block_dimension_t z = 1)
	{
		return block_dimensions(grid::block_dimensions_t{x, y, z});
	}

	launch_config_builder_t& block_size(grid::block_dimension_t size) { return block_dimensions(size, 1, 1); }

	launch_config_builder_t& use_maximum_linear_block()
	{
		grid::block_dimension_t max_size;
		if (kernel_) {
			max_size = kernel_->maximum_threads_per_block();
		}
		else if (device_) {
			max_size = device().maximum_threads_per_block();
		}
		else {
			throw ::std::logic_error("Request to use the maximum-size linear block, with no device or kernel specified");
		}
		auto block_dims = grid::block_dimensions_t { max_size, 1, 1 };

		if (dimensions_.grid and dimensions_.overall) {
			dimensions_.overall.unset();
		}
		dimensions_.block = block_dims;
		return *this;
	}

	launch_config_builder_t& grid_dimensions(grid::dimensions_t dims)
	{
#ifndef NDEBUG
		validate_grid_dimensions(dims);
#endif
		if (dimensions_.block) {
			dimensions_.overall.unset();
		}
		dimensions_.grid = dims;
		saturate_with_active_blocks_ = false;
		return *this;
	}

	launch_config_builder_t& grid_dimensions(
		grid::dimension_t x,
		grid::dimension_t y = 1,
		grid::dimension_t z = 1)
	{
		return grid_dimensions(grid::dimensions_t{x, y, z});
	}

	launch_config_builder_t& grid_size(grid::dimension_t size) {return grid_dimensions(size, 1, 1);	}

	launch_config_builder_t& overall_dimensions(grid::overall_dimensions_t dims)
	{
#ifndef NDEBUG
		validate_overall_dimensions(dims);
#endif
		dimensions_.overall = dims;
		saturate_with_active_blocks_ = false;
		return *this;
	}
	launch_config_builder_t& overall_dimensions(
		grid::dimension_t x,
		grid::dimension_t y = 1,
		grid::dimension_t z = 1)
	{
		return overall_dimensions(grid::overall_dimensions_t{x, y, z});
	}

	launch_config_builder_t& overall_size(grid::dimension_t size) { return overall_dimensions(size, 1, 1); }

	launch_config_builder_t& block_cooperation(bool cooperation)
	{
		thread_block_cooperation = cooperation;
		return *this;
	}

	launch_config_builder_t& blocks_may_cooperate() { return block_cooperation(true); }
	launch_config_builder_t& blocks_dont_cooperate() { return block_cooperation(false); }

	launch_config_builder_t& dynamic_shared_memory_size(
		kernel::shared_memory_size_determiner_t shared_mem_size_determiner)
	{
		dynamic_shared_memory_size_determiner_ = shared_mem_size_determiner;
		return *this;
	}

	launch_config_builder_t& dynamic_shared_memory_size(memory::shared::size_t size)
	{
#ifndef NDEBUG
		validate_dynamic_shared_memory_size(size);
#endif
		dynamic_shared_memory_size_ = size;
		dynamic_shared_memory_size_determiner_ = nullptr;
		return *this;
	}

	launch_config_builder_t& dynamic_shared_memory(memory::shared::size_t size)
	{
		return dynamic_shared_memory_size(size);
	}

	launch_config_builder_t& dynamic_shared_memory(
		kernel::shared_memory_size_determiner_t shared_mem_size_determiner)
	{
		return dynamic_shared_memory_size(shared_mem_size_determiner);
	}

	launch_config_builder_t& kernel(const kernel_t* wrapped_kernel_ptr)
	{
#ifndef NDEBUG
		validate_kernel(wrapped_kernel_ptr);
#endif
		kernel_ = wrapped_kernel_ptr;
		return *this;
	}

	launch_config_builder_t& kernel_independent()
	{
		kernel_ = nullptr;
		return *this;
	}
	launch_config_builder_t& no_kernel()
	{
		kernel_ = nullptr;
		return *this;
	}

	/**
	 * @brief THis will use information about the kernel, the already-set block size,
	 * and the device to create a unidimensional grid of blocks to exactly saturate
	 * the CUDA device's capacity for simultaneous active blocks.
	 *
	 * @note This will _not_ set the block size - unlike
	 */
	launch_config_builder_t& saturate_with_active_blocks()
	{
		if (not (kernel_)) {
			throw ::std::logic_error("A kernel must be set to determine how many blocks are required to saturate the device");
		}
		if (not (dimensions_.block)) {
			throw ::std::logic_error("The block dimensions must be known to determine how many of them one needs for saturating a device");
		}
		dimensions_.grid.clear();
		dimensions_.overall.clear();
		use_min_params_for_max_occupancy_ = false;
		saturate_with_active_blocks_ = true;
		return *this;
	}

	launch_config_builder_t& min_params_for_max_occupancy()
	{
		if (not (kernel_)) {
			throw ::std::logic_error("A kernel must be set to determine how many blocks are required to saturate the device");
		}
		dimensions_.block.clear();
		dimensions_.grid.clear();
		dimensions_.overall.clear();
		use_min_params_for_max_occupancy_ = true;
		saturate_with_active_blocks_ = false;
		return *this;
	}


}; // launch_config_builder_t

inline launch_config_builder_t launch_config_builder() { return {}; }

} // namespace cuda

#endif // CUDA_API_WRAPPERS_LAUNCH_CONFIG_BUILDER_CUH_
