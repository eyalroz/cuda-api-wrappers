/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * of kernel-launch-related functions.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_LAUNCH_HPP_
#define MULTI_WRAPPER_IMPLS_LAUNCH_HPP_

#include "kernel.hpp"
#include "../types.hpp"
#include "../memory.hpp"
#include "../stream.hpp"
#include "../kernel_launch.hpp"
#include "../device.hpp"
#include "../identify.hpp"

// The following is needed for occupancy-related calculation convenience functions
#include <cuda_runtime.h>

namespace cuda_ {

template<typename Kernel, typename... KernelParameters>
void enqueue_launch(
	Kernel&&                kernel,
	stream_t const&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   arguments)
{
	static_assert(
		detail::all_true<is_valid_kernel_argument<detail::kernel_parameter_decay_t<KernelParameters>>::value...>::value,
		"All kernel parameter types must fulfill the CUDA kernel argument requirements. "
		"Refer to the documentation of 'cuda_::traits::is_valid_kernel_argument' for more details."
	);
	static constexpr bool wrapped_contextual_kernel = std::is_base_of<kernel_t, typename std::decay<Kernel>::type>::value;
#if CUDA_VERSION >= 12000
	static constexpr bool library_kernel = cuda_::detail::is_library_kernel<Kernel>::value;
#else
	static constexpr bool library_kernel = false;
#endif // CUDA_VERSION >= 12000
#ifndef NDEBUG
	// wrapped kernel and library kernel compatibility with the launch configuration
	// will be validated further inside, when we differentiate them from raw kernels
	detail::validate(launch_configuration);
#endif

	// We would have liked an "if constexpr" here, but that is unsupported by C++11, so we have to
	// use tagged dispatch for the separate behavior for raw and wrapped kernels - although the enqueue_launch
	// function for each of them will basically be just a one-liner :-(
	detail::enqueue_launch<Kernel, KernelParameters...>(
		detail::bool_constant<wrapped_contextual_kernel>{},
		detail::bool_constant<library_kernel>{},
		std::forward<Kernel>(kernel), stream, launch_configuration,
		std::forward<KernelParameters>(arguments)...);
}

namespace detail {

inline void validate_shared_mem_compatibility(
	device_t const& device,
	memory::shared::size_t shared_mem_size) noexcept(false)
{
	if (shared_mem_size == 0) { return; }
	memory::shared::size_t max_shared = device.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN);

	// Note: A single kernel may not be able to access this shared memory capacity without opting-in to
	// it using	kernel_t::set_maximum_dynamic_shared_memory_per_block. See @ref kernel_t

	if (shared_mem_size > max_shared) {
		throw std::invalid_argument(
			"A dynamic shared memory size of " + std::to_string(shared_mem_size)
			+ " bytes exceeds the device maximum of " + std::to_string(max_shared));
	}
}

inline void validate_compatibility(
	device::id_t                  device_id,
	memory::shared::size_t        shared_mem_size,
	bool                          cooperative_launch,
	optional<grid::dimensions_t>  block_cluster_dimensions) noexcept(false)
{
	auto device = device::get(device_id);
	if (not cooperative_launch or device.supports_block_cooperation()) {
		throw std::runtime_error(device::detail::identify(device_id)
			+ " cannot launch kernels with inter-block cooperation");
	}
	validate_shared_mem_compatibility(device, shared_mem_size);
	if (block_cluster_dimensions) {
#if CUDA_VERSION >= 12000
		if (not device.supports_block_clustering()) {
			throw std::runtime_error(device::detail::identify(device_id)
				+ " cannot launch kernels with inter-block cooperation");
			// TODO: Uncomment this once the CUDA driver offers info on the maximum
			// cluster size...
			//
			// auto max_cluster_size = ???;
			// auto cluster_size = block_cluster_dimensions.value().volume();
			// if (cluster_size > max_cluster_size) {
			// 	throw std::runtime_error(device::detail::identify(device_id)
			// 		+ " only supports as many as " + std::to_string(max_cluster_size)
			// 		+ "blocks per block-cluster, but " + std::to_string(cluster_size));
		}
#else
		throw std::runtime_error("Block clusters are not supported with CUDA versions earlier than 12.0");
#endif // CUDA_VERSION >= 12000
	}

	// The CUDA driver does not offer us information with which we could check the validity
	// of trying a programmatically dependent launch, or a programmatic completion event,
	// or the use of a "remote" memory synchronization domain. So, assuming that's all valid
}

template <typename Dims>
void validate_any_dimensions_compatibility(
	device_t const& device, Dims dims,
	Dims maxima,
	char const* kind) noexcept(false)
{
	auto device_id = device.id();
	auto check =
		[device_id, kind](grid::dimension_t dim, grid::dimension_t max, char const *axis) {
			if (max < dim) {
				throw std::invalid_argument(
					std::string("specified ") + kind + " " + axis + "-axis dimension " + std::to_string(dim)
					+ " exceeds the maximum supported " + axis + " dimension of " + std::to_string(max)
					+ " for " + device::detail::identify(device_id));
			}
		};
	check(dims.x, maxima.x, "X");
	check(dims.y, maxima.y, "Y");
	check(dims.z, maxima.z, "Z");
}

inline void validate_block_dimension_compatibility(
	device_t const& device,
	grid::block_dimensions_t block_dims) noexcept(false)
{
	auto max_block_size = device.maximum_threads_per_block();
	auto volume = block_dims.volume();
	if (volume > max_block_size) {
		throw std::invalid_argument(
			"Specified block dimensions result in blocks of size " + std::to_string(volume)
			+ ", exceeding the maximum possible block size of " + std::to_string(max_block_size)
			+ " for " + device::detail::identify(device.id()));
	}
	auto maxima = grid::block_dimensions_t{
		static_cast<grid::block_dimension_t>(device.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X)),
		static_cast<grid::block_dimension_t>(device.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y)),
		static_cast<grid::block_dimension_t>(device.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z))
	};
	validate_any_dimensions_compatibility(device, block_dims, maxima, "block");
}

inline void validate_grid_dimension_compatibility(
	device_t const& device,
	grid::block_dimensions_t block_dims) noexcept(false)
{
	auto maxima = grid::dimensions_t{
		static_cast<grid::dimension_t>(device.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X)),
		static_cast<grid::dimension_t>(device.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y)),
		static_cast<grid::dimension_t>(device.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z))
	};
	validate_any_dimensions_compatibility(device, block_dims, maxima, "grid");
}


inline void validate_shared_mem_size_compatibility(
	kernel_t const& kernel_ptr,
	memory::shared::size_t shared_mem_size)
{
	if (shared_mem_size == 0) { return; }
	auto max_shared = kernel_ptr.get_maximum_dynamic_shared_memory_per_block();
	if (shared_mem_size > max_shared) {
		throw std::invalid_argument(
			"Requested dynamic shared memory size "
			+ std::to_string(shared_mem_size) + " exceeds kernel's maximum allowed value of "
			+ std::to_string(max_shared));
	}
}

inline void validate_block_dimension_compatibility(
	kernel_t const&          kernel,
	grid::block_dimensions_t block_dims)
{
	auto max_block_size = kernel.maximum_threads_per_block();
	auto volume = block_dims.volume();
	if (volume > max_block_size) {
		throw std::invalid_argument(
			"specified block dimensions result in blocks of size " + std::to_string(volume)
			+ ", exceeding the maximum possible block size of " + std::to_string(max_block_size)
			+ " for " + identify(kernel));
	}
}

inline void validate_dyanmic_shared_memory_size(
	kernel_t const&          kernel,
	memory::shared::size_t   dynamic_shared_memory_size)
{
	memory::shared::size_t max_dyn_shmem = kernel.get_attribute(
		kernel::attribute_t::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES);
	if (dynamic_shared_memory_size > max_dyn_shmem) {
		throw std::invalid_argument(
			"specified size of dynamic shared memory, " + std::to_string(dynamic_shared_memory_size)
			+ "bytes, exceeds the maximum supported by  " + identify(kernel)
			+ ", " + std::to_string(max_dyn_shmem) + " bytes");
	}
}

template<typename... KernelParameters>
void enqueue_launch_helper<kernel::apriori_compiled_t, KernelParameters...>::operator()(
	kernel::apriori_compiled_t const&  wrapped_kernel,
	stream_t const &                  stream,
	launch_configuration_t            launch_configuration,
	KernelParameters &&...            arguments) const
{
	using raw_kernel_t = typename kernel::detail::raw_kernel_typegen<KernelParameters ...>::type;
	auto unwrapped_kernel_function = reinterpret_cast<raw_kernel_t>(const_cast<void *>(wrapped_kernel.ptr()));
	// Notes:
	// 1. The inner cast here is because we store the pointer as void const* - as an extra
	//    precaution against anybody trying to write through it. Now, function pointers
	//    can't get written through, but are still for some reason not considered const.
	// 2. We rely on the caller providing us with more-or-less the correct parameters -
	//    corresponding to the compiled kernel function's. I say "more or less" because the
	//    `KernelParameter` pack may contain some references, arrays and so on - which CUDA
	//    kernels cannot accept; so we massage those a bit.

	// It is assumed arguments were already been validated

	detail::enqueue_raw_kernel_launch_in_current_context(
		unwrapped_kernel_function,
		stream.device_id(),
		stream.context_handle(),
		stream.handle(),
		launch_configuration,
		std::forward<KernelParameters>(arguments)...);
}

template<typename... KernelParameters>
std::array<void const*, sizeof...(KernelParameters)>
marshal_dynamic_kernel_arguments(KernelParameters&&... arguments)
{
	return std::array<void const*, sizeof...(KernelParameters)> { &arguments... };
}

// Note: The last (valid) element of marshalled_arguments must be null
inline void enqueue_kernel_launch_by_handle_in_current_context(
	kernel::handle_t        kernel_function_handle,
	device::id_t            device_id,
	context::handle_t       context_handle,
	stream::handle_t        stream_handle,
	launch_configuration_t  launch_config,
	void const**            marshalled_arguments)
{
	// It is assumed arguments were already been validated

	status_t status;
	auto const&lc = launch_config; // alias for brevity
#if CUDA_VERSION >= 12000
	CUlaunchAttribute launch_attributes[detail::maximum_possible_kernel_launch_attributes+1];
	auto launch_attributes_span = span<CUlaunchAttribute>{
		launch_attributes, sizeof(launch_attributes)/sizeof(launch_attributes[0])
	};
	CUlaunchConfig full_launch_config = detail::marshal(lc, stream_handle, launch_attributes_span);
	status = cuLaunchKernelEx(
		&full_launch_config,
		kernel_function_handle,
		const_cast<void**>(marshalled_arguments),
		nullptr);
#else
	if (launch_config.has_nondefault_attributes())
		status = cuLaunchCooperativeKernel(
			kernel_function_handle,
			lc.dimensions.grid.x,  lc.dimensions.grid.y,  lc.dimensions.grid.z,
			lc.dimensions.block.x, lc.dimensions.block.y, lc.dimensions.block.z,
			lc.dynamic_shared_memory_size,
			stream_handle,
			const_cast<void**>(marshalled_arguments)
		);
	else {
		static constexpr auto no_arguments_in_alternative_format = nullptr;
		// TODO: Consider passing marshalled_arguments in the alternative format
		status = cuLaunchKernel(
			kernel_function_handle,
			lc.dimensions.grid.x,  lc.dimensions.grid.y,  lc.dimensions.grid.z,
			lc.dimensions.block.x, lc.dimensions.block.y, lc.dimensions.block.z,
			lc.dynamic_shared_memory_size,
			stream_handle,
			const_cast<void**>(marshalled_arguments),
			no_arguments_in_alternative_format
		);
	}
#endif // CUDA_VERSION >= 12000
	throw_if_error_lazy(status,
		std::string(" kernel launch failed for ") + kernel::detail::identify(kernel_function_handle)
		+ " on " + stream::detail::identify(stream_handle, context_handle, device_id));
}


template<typename... KernelParameters>
struct enqueue_launch_helper<kernel_t, KernelParameters...> {

	void operator()(
	kernel_t const&                       wrapped_kernel,
	stream_t const&                       stream,
	launch_configuration_t                launch_config,
	KernelParameters&&...                 arguments) const
	{
	// It is assumed arguments were already been validated

#ifndef NDEBUG
		if (wrapped_kernel.context() != stream.context()) {
			throw std::invalid_argument{"Attempt to launch " + identify(wrapped_kernel)
				+ " on " + identify(stream) + ": Different contexts"};
		}
		validate_compatibility(wrapped_kernel, launch_config);
#endif
		auto marshalled_arguments { marshal_dynamic_kernel_arguments(std::forward<KernelParameters>(arguments)...) };
		auto function_handle = wrapped_kernel.handle();
		CAW_SET_SCOPE_CONTEXT(stream.context_handle());

		enqueue_kernel_launch_by_handle_in_current_context(
			function_handle, stream.device_id(), stream.context_handle(),
			stream.handle(), launch_config, marshalled_arguments.data());
	} // operator()
};

template<typename RawKernelFunction, typename... KernelParameters>
void enqueue_launch(
	bool_constant<false>, // Not a wrapped contextual kernel,
	bool_constant<false>, // and not a library kernel, so it must be a raw kernel function
	RawKernelFunction&&       kernel_function,
	stream_t const&           stream,
	launch_configuration_t    launch_configuration,
	KernelParameters&&...     arguments)
{
	// It is assumed arguments were already been validated

	// Note: Unfortunately, even though CUDA should be aware of which context a stream belongs to,
	// and not have trouble enqueueing into a stream in another context - it balks at doing so under
	// certain conditions, so we must place ourselves in the stream's context.
	CAW_SET_SCOPE_CONTEXT(stream.context_handle());
	detail::enqueue_raw_kernel_launch_in_current_context<RawKernelFunction, KernelParameters...>(
		kernel_function, stream.device_id(), stream.context_handle(), stream.handle(), launch_configuration,
		std::forward<KernelParameters>(arguments)...);
}

template<typename Kernel, typename... KernelParameters>
void enqueue_launch(
	bool_constant<true>,  // a kernel wrapped in a kernel_t (sub)class
	bool_constant<false>, // Not a library kernel
	Kernel&&                kernel,
	stream_t const&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   arguments)
{
	// It is assumed arguments were already been validated - except for:
#ifndef NDEBUG
	if (kernel.context() != stream.context()) {
		throw std::invalid_argument{"Attempt to launch " + detail::identify(kernel)
			+ " on " + identify(stream) + ": Different contexts"};
	}
	detail::validate_compatibility(kernel, launch_configuration);
#if CUDA_VERSION >= 13020
	detail::validate_num_parameters(kernel, sizeof...(KernelParameters));
#endif
#endif // #ifndef NDEBUG

	enqueue_launch_helper<typename std::decay<Kernel>::type, KernelParameters...>{}(
		std::forward<Kernel>(kernel), stream, launch_configuration,
		std::forward<KernelParameters>(arguments)...);
}

#if CUDA_VERSION >= 12000
template<typename Kernel, typename... KernelParameters>
void enqueue_launch(
	bool_constant<false>, // Not a wrapped contextual kernel,
	bool_constant<true>,  // but a library kernel
	Kernel&&                kernel,
	stream_t const&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   arguments)
{
	// Launch configuration is assumed to have been validated separately
	// from the kernel, and their compatibility will be validated further
	// inside, against the contextualized kernel

	kernel_t contextualized = cuda_::contextualize(kernel, stream.context());
	enqueue_launch_helper<kernel_t, KernelParameters...> {}(
		contextualized, stream, launch_configuration,
		std::forward<KernelParameters>(arguments)...);
}
#endif // CUDA_VERSION >= 12000

} // namespace detail

template<typename Kernel, typename... KernelParameters>
void launch(
	Kernel&&                kernel,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   arguments)
{
	// Argument validation will occur within call to enqueue_launch

	auto primary_context = detail::get_implicit_primary_context(kernel);
	auto stream = primary_context.default_stream();

	// Note: If Kernel is a kernel_t, and its associated device is different
	// than the current device, the next call will fail:

	enqueue_launch(kernel, stream, launch_configuration, std::forward<KernelParameters>(arguments)...);
}

template<typename Kernel, typename... KernelParameters>
void launch(
	Kernel&&                kernel,
	device_t const&         device,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   arguments)
{
	// Argument validation will occur within call to enqueue_launch
	static_assert(not std::is_base_of<kernel_t, Kernel>::value,
		"A wrapped kernel is already associated with a particular GPU device; drop the gratuitous device parameter");
	auto primary_context = device.primary_context();
	auto stream = primary_context.default_stream();
	enqueue_launch(kernel, stream, launch_configuration, std::forward<KernelParameters>(arguments)...);
}

template <typename SpanOfConstVoidPtrLike>
void launch_type_erased(
	kernel_t const&         kernel,
	stream_t const&         stream,
	launch_configuration_t  launch_configuration,
	SpanOfConstVoidPtrLike  marshalled_arguments)
{
	// Note: We assume that kernel, stream and launch_configuration have already been validated.
	static_assert(
		std::is_same<typename SpanOfConstVoidPtrLike::value_type, void*>::value or
		std::is_same<typename SpanOfConstVoidPtrLike::value_type, void const*>::value,
		"The element type of the marshalled arguments container type must be either void* or void const*");
#ifndef NDEBUG
	if (kernel.context() != stream.context()) {
		throw std::invalid_argument{"Attempt to launch " + detail::identify(kernel)
			+ " on " + cuda_::detail::identify(stream) + ": Different contexts"};
	}
	detail::validate_compatibility(kernel, launch_configuration);
	detail::validate(launch_configuration);
	if (*(marshalled_arguments.end() - 1) != nullptr) {
		throw std::invalid_argument("marshalled arguments for a kernel launch must end with a nullptr element");
	}
#endif
	CAW_SET_SCOPE_CONTEXT(stream.context_handle());
	return detail::enqueue_kernel_launch_by_handle_in_current_context(
		kernel.handle(),
		stream.device_id(),
		stream.context_handle(),
		stream.handle(),
		launch_configuration,
		static_cast<void const**>(marshalled_arguments.data()));
}

#if CUDA_VERSION >= 12000
template <typename SpanOfConstVoidPtrLike>
void launch_type_erased(
	library::kernel_t const&  kernel,
	stream_t const&           stream,
	launch_configuration_t    launch_configuration,
	SpanOfConstVoidPtrLike    marshalled_arguments)
{
	// Argument validation will occur inside the call to launch_type_erased
	auto contextualized = contextualize(kernel, stream.context());
	launch_type_erased(contextualized, stream, launch_configuration, marshalled_arguments);
}
#endif // CUDA_VERSION >= 12000

#if ! CAW_CAN_GET_APRIORI_KERNEL_HANDLE

#if defined(__CUDACC__)

// Unfortunately, the CUDA runtime API does not allow for computation of the grid parameters for maximum occupancy
// from code compiled with a host-side-only compiler! See cuda_runtime.h for details

#if CUDA_VERSION >= 10000
namespace detail {

template <typename UnaryFunction>
grid::composite_dimensions_t min_grid_params_for_max_occupancy(
	void const *             ptr,
	device::id_t             device_id,
	UnaryFunction            block_size_to_dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
	int min_grid_size_in_blocks { 0 };
	int block_size { 0 };
		// Note: only initializing the values her because of a
		// spurious (?) compiler warning about potential uninitialized use.
	auto result = cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags(
		&min_grid_size_in_blocks, &block_size,
		ptr,
		block_size_to_dynamic_shared_mem_size,
		static_cast<int>(block_size_limit),
		disable_caching_override ? cudaOccupancyDisableCachingOverride : cudaOccupancyDefault
	);
	throw_if_error_lazy(result,
		"Failed obtaining parameters for a minimum-size grid for kernel " + detail::ptr_as_hex(ptr) +
			" on device " + std::to_string(device_id) + ".");
	return { (grid::dimension_t) min_grid_size_in_blocks, (grid::block_dimension_t) block_size };
}

inline grid::composite_dimensions_t min_grid_params_for_max_occupancy(
	void const *             ptr,
	device::id_t             device_id,
	memory::shared::size_t   dynamic_shared_mem_size,
	grid::block_dimension_t  block_size_limit,
	bool                     disable_caching_override)
{
	auto always_need_same_shared_mem_size =
		[dynamic_shared_mem_size](::size_t) { return dynamic_shared_mem_size; };
	return min_grid_params_for_max_occupancy(
		ptr, device_id, always_need_same_shared_mem_size, block_size_limit, disable_caching_override);
}

} // namespace detail

inline grid::composite_dimensions_t min_grid_params_for_max_occupancy(
	kernel::apriori_compiled_t const&  kernel,
	memory::shared::size_t            dynamic_shared_memory_size,
	grid::block_dimension_t           block_size_limit,
	bool                              disable_caching_override)
{
	return detail::min_grid_params_for_max_occupancy(
		kernel.ptr(), kernel.device().id(), dynamic_shared_memory_size, block_size_limit, disable_caching_override);
}

template <typename UnaryFunction>
grid::composite_dimensions_t min_grid_params_for_max_occupancy(
	kernel::apriori_compiled_t const&  kernel,
	UnaryFunction                     block_size_to_dynamic_shared_mem_size,
	grid::block_dimension_t           block_size_limit,
	bool                              disable_caching_override)
{
	return detail::min_grid_params_for_max_occupancy(
		kernel.ptr(), kernel.device_id(), block_size_to_dynamic_shared_mem_size, block_size_limit, disable_caching_override);
}
#endif // CUDA_VERSION >= 10000

#endif // defined(__CUDACC__)
#endif // ! CAW_CAN_GET_APRIORI_KERNEL_HANDLE

} // namespace cuda_

#endif // MULTI_WRAPPER_IMPLS_LAUNCH_HPP_

