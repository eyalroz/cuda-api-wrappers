/**
 * @file
 *
 * @brief Variadic, chevron-less wrappers for the CUDA kernel launch mechanism.
 *
 * This file has two stand-alone functions used for launching kernels - by
 * application code directly and by other API wrappers (e.g. @ref cuda::device_t
 * and @ref cuda::stream_t ).
 *
 * <p>The wrapper functions have two goals:
 *
 * <ul>
 * <li>Avoiding the annoying triple-chevron syntax, e.g.
 *
 *   my_kernel<<<launch, config, stuff>>>(real, args)
 *
 * and sticking to proper C++; in other words, the wrappers are "ugly"
 * instead of client code having to be.
 * <li>Avoiding some of the "parameter soup" of launching a kernel: It's
 * not so difficult to mix up shared memory sizes with stream handles; grid and
 * block dimensions with each other; and even grid/block dimensions with
 * the scalar parameters - since a `dim3` is constructible from
 * integral values. Instead, we enforce a launch configuration structure:
 * @ref cuda::launch_configuration_t .
 * </ul>
 *
 * @note You'd probably better avoid launching kernels using these
 * function directly, and go through the @ref cuda::stream_t or @ref cuda::device_t
 * proxy classes' launch mechanism (e.g.
 * `my_stream.enqueue.kernel_launch(...)`).
 *
 * @note Even though when you use this wrapper, your code will not have the silly
 * chevron, you can't use it from regular `.cpp` files compiled with your host
 * compiler. Hence the `.cuh` extension. You _can_, however, safely include this
 * file from your `.cpp` for other definitions.
 *
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_KERNEL_LAUNCH_CUH_
#define CUDA_API_WRAPPERS_KERNEL_LAUNCH_CUH_

#include "launch_configuration.hpp"
#include "kernel.hpp"
#include "kernels/apriori_compiled.hpp"
#if CUDA_VERSION >= 12000
#include "kernels/in_library.hpp"
#endif


#if CUDA_VERSION >= 9000
// The following is necessary for cudaLaunchCooperativeKernel
#include <cuda_runtime.h>
#endif // CUDA_VERSION >= 9000

#include <type_traits>
#include <utility>

namespace cuda {

///@cond
class stream_t;
///@endcond

/**
 * A named constructor idiom for @ref grid::dimensions_t,
 * which, when used, will result in a grid with a single block.
 */
constexpr grid::dimensions_t single_block() { return 1; }
/**
 * A named constructor idiom for @ref grid::dimensions_t,
 * which, when used, will result in a grid whose blocks have
 * a single thread
 */
constexpr grid::block_dimensions_t single_thread_per_block() { return 1; }

namespace detail_ {

/**
 * @brief adapt a type to be usable as a kernel parameter.
 *
 * CUDA kernels don't accept just any parameter type a C++ function may accept.
 * Specifically: No references, arrays decay (IIANM) and functions pass by address.
 * However - not all "decaying" of `::std::decay` is necessary. Such transformation
 * can be effected by this type-trait struct.
 */
template<typename P>
struct kernel_parameter_decay {
private:
	using U = typename ::std::remove_reference<P>::type;
public:
	using type = typename ::std::conditional<
		::std::is_array<U>::value,
		typename ::std::remove_extent<U>::type*,
		typename ::std::conditional<
			::std::is_function<U>::value,
			typename ::std::add_pointer<U>::type,
			U
		>::type
	>::type;
};

template<typename P>
using kernel_parameter_decay_t = typename kernel_parameter_decay<P>::type;

template<typename Fun>
struct is_function_ptr: bool_constant<
	::std::is_pointer<Fun>::value and ::std::is_function<typename ::std::remove_pointer<Fun>::type>::value> { };

inline void collect_argument_addresses(void**) { }

template <typename Arg, typename... Args>
inline void collect_argument_addresses(void** collected_addresses, Arg&& arg, Args&&... args)
{
	collected_addresses[0] = const_cast<void*>(static_cast<const void*>(&arg));
	collect_argument_addresses(collected_addresses + 1, ::std::forward<Args>(args)...);
}

// For partial template specialization on WrappedKernel...
template<typename Kernel, typename... KernelParameters>
struct enqueue_launch_helper {
	void operator()(
		Kernel&&                kernel_function,
		const stream_t &        stream,
		launch_configuration_t  launch_configuration,
		KernelParameters &&...  parameters) const;
};

template<typename Kernel, typename... KernelParameters>
void enqueue_launch(
	bool_constant<false>,
	bool_constant<false>,
	Kernel&&                kernel_function,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters);

template<typename Kernel, typename... KernelParameters>
void enqueue_launch(
	bool_constant<true>,
	bool_constant<false>,
	Kernel&&                kernel,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters);

template<typename Kernel, typename... KernelParameters>
void enqueue_launch(
	bool_constant<false>,
	bool_constant<true>,
	Kernel&&                kernel,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters);

inline void enqueue_kernel_launch_by_handle_in_current_context(
	kernel::handle_t        kernel_function_handle,
	device::id_t            device_id,
	context::handle_t       context_handle,
	stream::handle_t        stream_handle,
	launch_configuration_t  launch_config,
	const void**            marshalled_arguments);

template<typename KernelFunction, typename... KernelParameters>
void enqueue_raw_kernel_launch_in_current_context(
	KernelFunction&&        kernel_function,
	device::id_t            device_id,
	context::handle_t       context_handle,
	stream::handle_t        stream_handle,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
#ifndef __CUDACC__
// If we're not in CUDA's NVCC, this can't run properly anyway, so either we throw some
// compilation error, or we just do nothing. For now it's option 2.
;
#else
{
	using decayed_kf_type = typename ::std::decay<KernelFunction>::type;
	static_assert(::std::is_function<decayed_kf_type>::value or is_function_ptr<decayed_kf_type>::value,
		"Only a bona fide function can be launched as a CUDA kernel");
#ifndef NDEBUG
	validate(launch_configuration);
#endif
	if (not launch_configuration.has_nondefault_attributes()) {
		// regular plain vanilla launch
		kernel_function <<<
			launch_configuration.dimensions.grid,
			launch_configuration.dimensions.block,
			launch_configuration.dynamic_shared_memory_size,
			stream_handle
		>>>(::std::forward<KernelParameters>(parameters)...);
		cuda::outstanding_error::ensure_none("Kernel launch failed");
	}
	else {
#if CUDA_VERSION < 9000
		throw cuda::runtime_error(status::not_supported,
			"Only CUDA versions 9.0 and later support launching kernels with additional"
			"arguments, e.g block cooperation");
#else
		// The following hack is due to C++ not supporting arrays of length 0 -
		// but such an array being necessary for collect_argument_addresses with
		// multiple parameters. Other workarounds are possible, but would be
		// more cumbersome, except perhaps with C++17 or later.
		static constexpr const auto non_zero_num_params =
			sizeof...(KernelParameters) == 0 ? 1 : sizeof...(KernelParameters);
		void* argument_ptrs[non_zero_num_params];
		// fill the argument array with our parameters. Yes, the use
		// of the two terms is confusing here and depends on how you
		// look at things.
		detail_::collect_argument_addresses(argument_ptrs, ::std::forward<KernelParameters>(parameters)...);
#if CUDA_VERSION >= 11000
		kernel::handle_t kernel_function_handle = kernel::apriori_compiled::detail_::get_handle( (const void*) kernel_function);
		enqueue_kernel_launch_by_handle_in_current_context(
			kernel_function_handle,
			device_id,
			context_handle,
			stream_handle,
			launch_configuration,
			const_cast<const void**>(argument_ptrs));

#else // CUDA_VERSION is at least 9000 but under 11000
		(void) device_id;
		(void) context_handle;
		auto status = cudaLaunchCooperativeKernel(
			(const void *) kernel_function,
			(dim3)(uint3)launch_configuration.dimensions.grid,
			(dim3)(uint3)launch_configuration.dimensions.block,
			&argument_ptrs[0],
			(size_t)launch_configuration.dynamic_shared_memory_size,
			cudaStream_t(stream_handle));
		throw_if_error_lazy(status, "Kernel launch failed");
#endif // CUDA_VERSION >= 11000
#endif // CUDA_VERSION >= 9000
	}
}
#endif

} // namespace detail_


namespace kernel {

namespace detail_ {

// The helper code here is intended for re-imbuing kernel-related classes with the types
// of the kernel parameters. This is necessary since kernel wrappers may be type-erased
// (which makes it much easier to work with them and avoids a bunch of code duplication).
//
// Note: The type-unerased kernel must be a non-const function pointer. Why? Not sure.
// even though function pointers can't get written through, for some reason they are
// expected not to be const.


template<typename... KernelParameters>
struct raw_kernel_typegen {
	// You should be careful to only instantiate this class with nice simple types we can pass to CUDA kernels.
//	static_assert(
//		all_true<
//		    ::std::is_same<
//		    	KernelParameters,
//		    	::cuda::detail_::kernel_parameter_decay_t<KernelParameters>>::value...
//		    >::value,
//		"All kernel parameter types must be decay-invariant" );
	using type = void(*)(cuda::detail_::kernel_parameter_decay_t<KernelParameters>...);
};

} // namespace detail_

template<typename... KernelParameters>
typename detail_::raw_kernel_typegen<KernelParameters...>::type
unwrap(const kernel::apriori_compiled_t& kernel)
{
	using raw_kernel_t = typename detail_::raw_kernel_typegen<KernelParameters ...>::type;
	return reinterpret_cast<raw_kernel_t>(const_cast<void *>(kernel.ptr()));
}

} // namespace kernel

namespace detail_ {

template<typename... KernelParameters>
struct enqueue_launch_helper<kernel::apriori_compiled_t, KernelParameters...> {
	void operator()(
		const kernel::apriori_compiled_t&  wrapped_kernel,
		const stream_t &                  stream,
		launch_configuration_t            launch_configuration,
		KernelParameters &&...            parameters) const;
};

} // namespace detail_


/**
 * @brief Enqueues a kernel on a stream (=queue) on the current CUDA device.
 *
 * CUDA's 'chevron' kernel launch syntax cannot be compiled in proper C++. Thus, every kernel launch must
 * at some point reach code compiled with CUDA's nvcc. Naively, every single different kernel (perhaps up
 * to template specialization) would require writing its own wrapper C++ function, launching it. This
 * function, however, constitutes a single minimal wrapper around the CUDA kernel launch, which may be
 * called from proper C++ code (across translation unit boundaries - the caller is compiled with a C++
 * compiler, the callee compiled by nvcc).
 *
 * <p>This function is similar to C++17's `::std::apply`, or to a a beta-reduction in Lambda calculus:
 * It applies a function to its arguments; the difference is in the nature of the function (a CUDA kernel)
 * and in that the function application requires setting additional CUDA-related launch parameters,
 * additional to the function's own.
 *
 * <p>As kernels do not return values, neither does this function. It also contains no hooks, logging
 * commands etc. - if you want those, write an additional wrapper (perhaps calling this one in turn).
 *
 * @param kernel the kernel to apply. Pass it just as-it-is, as though it were any other function. Note:
 * If the kernel is templated, you must pass it fully-instantiated. Alternatively, you can pass a
 * @param stream the CUDA hardware command queue on which to place the command to launch the kernel (affects
 * the scheduling of the launch and the execution)
 * @param launch_configuration not all launches of the same kernel are identical: The launch may be configured
 * to use more of less blocks in the grid, to allow blocks dynamic memory, to control the block's dimensions
 * etc; this parameter defines that extra configuration outside the kernels' actual source. See also
 * @ref cuda::launch_configuration_t.
 * @param parameters whatever parameters @p kernel_function takes
 */
template<typename Kernel, typename... KernelParameters>
void enqueue_launch(
	Kernel&&                kernel,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters)
{
	static_assert(
		detail_::all_true<::std::is_trivially_copy_constructible<detail_::kernel_parameter_decay_t<KernelParameters>>::value...>::value,
		"All kernel parameter types must be of a trivially copy-constructible (decayed) type." );
	static constexpr const bool wrapped_contextual_kernel = ::std::is_base_of<kernel_t, typename ::std::decay<Kernel>::type>::value;
#if CUDA_VERSION >= 12000
	static constexpr const bool library_kernel = cuda::detail_::is_library_kernel<Kernel>::value;
#else
	static constexpr const bool library_kernel = false;
#endif // CUDA_VERSION >= 12000
	// We would have liked an "if constexpr" here, but that is unsupported by C++11, so we have to
	// use tagged dispatch for the separate behavior for raw and wrapped kernels - although the enqueue_launch
	// function for each of them will basically be just a one-liner :-(
	detail_::enqueue_launch<Kernel, KernelParameters...>(
		detail_::bool_constant<wrapped_contextual_kernel>{},
		detail_::bool_constant<library_kernel>{},
		::std::forward<Kernel>(kernel), stream, launch_configuration,
		::std::forward<KernelParameters>(parameters)...);
}

/**
 * Variant of @ref enqueue_launch for use with the default stream in the current context.
 *
 * @note This isn't called `enqueue` since the default stream is synchronous.
 */
template<typename Kernel, typename... KernelParameters>
void launch(
	Kernel&&                kernel,
	launch_configuration_t  launch_configuration,
	KernelParameters&&...   parameters);

/**
 * Launch a kernel with the arguments pre-marshalled into the (main) form
 * which @ref cuLaunchKernel accepts variables in: A null-terminated sequence
 * of (possibly const) `void *`'s to the argument values.
 *
 * @tparam SpanOfConstVoidPtrLike
 *     Type of the container for the marshalled arguments; typically, this
 *     would be `span<const void*>` - but it can be an `::std::vector`, or
 *     have non-const `void*` elements etc.
 * @param kernel
 *     A wrapped GPU kernel
 * @param stream
 *     Proxy for the stream on which to enqueue the kernel launch; may be the
 *     default stream of a context.
 * @param marshalled_arguments
 *     A container of `void` or `const void` pointers to the argument values
 */
///@{
template <typename SpanOfConstVoidPtrLike>
void launch_type_erased(
	const kernel_t&         kernel,
	const stream_t&         stream,
	launch_configuration_t  launch_configuration,
	SpanOfConstVoidPtrLike  marshalled_arguments);

#if CUDA_VERSION >= 12000
template <typename SpanOfConstVoidPtrLike>
void launch_type_erased(
	const library::kernel_t&  kernel,
	const stream_t&           stream,
	launch_configuration_t    launch_configuration,
	SpanOfConstVoidPtrLike    marshalled_arguments);
///@}
#endif // CUDA_VERSION >= 12000

} // namespace cuda

#endif // CUDA_API_WRAPPERS_KERNEL_LAUNCH_CUH_
