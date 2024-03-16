/**
 * @file
 *
 * @brief Contains a proxy class for CUDA arrays - GPU memory
 * with 2-D or 3-D locality and hardware support for interpolated value
 * retrieval); see also @ref texture_view.hpp .
 *
 * @note Not all kinds of arrays are supported: Only non-layered, non-cubemap
 * arrays of 2 or 3 dimensions.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_ARRAY_HPP_
#define CUDA_API_WRAPPERS_ARRAY_HPP_

#include "context.hpp"
#include "error.hpp"

#ifndef CUDA_NO_HALF
#include <cuda_fp16.h>
#endif

namespace cuda {

///@cond
class device_t;
///@endcond

template <typename T, dimensionality_t NumDimensions>
class array_t;

namespace array {

/// Raw CUDA driver handle for arrays (of any dimension)
using handle_t = CUarray;

/// Raw CUDA driver descriptor structure for an array of dimension @tparam NumDimensions
template <dimensionality_t NumDimensions>
using descriptor_t = typename ::std::conditional<NumDimensions == 2, CUDA_ARRAY_DESCRIPTOR, CUDA_ARRAY3D_DESCRIPTOR>::type;

/// Wrap an existing CUDA array in an @ref array_t instance.
template <typename T, dimensionality_t NumDimensions>
array_t<T, NumDimensions> wrap(
	device::id_t                 device_id,
	context::handle_t            context_handle,
	handle_t                     handle,
	dimensions_t<NumDimensions>  dimensions) noexcept;

namespace detail_ {

template <typename T> struct format_specifier {};

template <> struct format_specifier<uint8_t > { static constexpr const CUarray_format value = CU_AD_FORMAT_UNSIGNED_INT8;  };
template <> struct format_specifier<uint16_t> { static constexpr const CUarray_format value = CU_AD_FORMAT_UNSIGNED_INT16; };
template <> struct format_specifier<uint32_t> { static constexpr const CUarray_format value = CU_AD_FORMAT_UNSIGNED_INT32; };
template <> struct format_specifier<int8_t  > { static constexpr const CUarray_format value = CU_AD_FORMAT_SIGNED_INT8;    };
template <> struct format_specifier<int16_t > { static constexpr const CUarray_format value = CU_AD_FORMAT_SIGNED_INT16;   };
template <> struct format_specifier<int32_t > { static constexpr const CUarray_format value = CU_AD_FORMAT_SIGNED_INT32;   };
#ifndef CUDA_NO_HALF
template <> struct format_specifier<half    > { static constexpr const CUarray_format value = CU_AD_FORMAT_HALF;           };
#endif
template <> struct format_specifier<float   > { static constexpr const CUarray_format value = CU_AD_FORMAT_FLOAT;          };

template<typename T>
handle_t create_in_current_context(dimensions_t<3> dimensions)
{
	handle_t handle;
	CUDA_ARRAY3D_DESCRIPTOR descriptor;
	descriptor.Width = dimensions.width;
	descriptor.Height = dimensions.height;
	descriptor.Depth = dimensions.depth;
	descriptor.Format = format_specifier<T>::value;
	descriptor.NumChannels = 1;
		// We don't currently support an array of packed pairs or quadruplets; if you want this,
		// file an issue.
	descriptor.Flags = 0;

	auto status = cuArray3DCreate(&handle, &descriptor);
	throw_if_error_lazy(status, "failed allocating 3D CUDA array");
	return handle;
}

template<typename T>
handle_t create_in_current_context(dimensions_t<2> dimensions)
{
	CUDA_ARRAY_DESCRIPTOR descriptor;
	descriptor.Width = dimensions.width;
	descriptor.Height = dimensions.height;
	descriptor.Format = format_specifier<T>::value;
	descriptor.NumChannels = 1;
	handle_t handle;
	auto status = cuArrayCreate(&handle, &descriptor);
	throw_if_error_lazy(status, "failed allocating 2D CUDA array");
	return handle;
}

template <typename T, dimensionality_t NumDimensions>
handle_t create(context::handle_t context_handle, dimensions_t<NumDimensions> dimensions)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	return create_in_current_context<T>(dimensions);
}

template <typename T, dimensionality_t NumDimensions>
handle_t create(const context_t& context, dimensions_t<NumDimensions> dimensions);

template <dimensionality_t NumDimensions>
descriptor_t<NumDimensions> get_descriptor_in_current_context(handle_t handle);

template <>
inline descriptor_t<2> get_descriptor_in_current_context<2>(handle_t handle)
{
	descriptor_t<2> result;
	auto status = cuArrayGetDescriptor(&result, handle);
	throw_if_error_lazy(status,
		::std::string("Failed obtaining the descriptor of the CUDA 2D array at ")
		+ cuda::detail_::ptr_as_hex(handle));
	return result;
}

template <>
inline descriptor_t<4> get_descriptor_in_current_context<3>(handle_t handle)
{
	descriptor_t<3> result;
	auto status = cuArray3DGetDescriptor(&result, handle);
	throw_if_error_lazy(status,
		::std::string("Failed obtaining the descriptor of the CUDA 3D array at ")
		+ cuda::detail_::ptr_as_hex(handle));
	return result;
}

template <dimensionality_t NumDimensions>
descriptor_t<NumDimensions> get_descriptor(context::handle_t context_handle, handle_t handle)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	return get_descriptor_in_current_context<NumDimensions>(handle);
}

template <dimensionality_t NumDimensions>
dimensions_t<NumDimensions> dimensions_of(const descriptor_t<NumDimensions>& descriptor);

template <>
inline dimensions_t<3> dimensions_of(const descriptor_t<3>& descriptor)
{
	return { descriptor.Width, descriptor.Height, descriptor.Depth };
}

template <>
inline dimensions_t<2> dimensions_of(const descriptor_t<2>& descriptor)
{
	return { descriptor.Width, descriptor.Height };
}

template <dimensionality_t NumDimensions>
dimensions_t<NumDimensions> dimensions_of_in_current_context(handle_t handle_in_current_context)
{
	auto descriptor = get_descriptor_in_current_context<NumDimensions>(handle_in_current_context);
	return dimensions_of<NumDimensions>(descriptor);
}

template <dimensionality_t NumDimensions>
dimensions_t<NumDimensions> dimensions_of(context::handle_t context_handle, handle_t handle)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	return dimensions_of_in_current_context<NumDimensions>(handle);
}

} // namespace detail_

} // namespace array

/**
 * @brief Owning wrapper for CUDA 2D and 3D arrays
 *
 * A CUDA array is a multi-dimensional structure on CUDA GPUs with specific
 * GPU hardware support. CUDA arrays are _not_ equivalent to `::std::array`s,
 * nor to C/C++ arrays! Please read the relevant sections of the CUDA
 * programming guide for information regarding the uses of these arrays:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arrays
 *
 * Arrays are particularly useful in combination with texture memory (see
 * @ref cuda::texture_view): One can access elements in a multi dimensional
 * array with the index, e.g. `array[i][j][k]`. In CUDA it is possible to
 * create a texture on an array, allowing for different kind of access to array
 * elements, e.g. `texture[u][v][w]` with `u, v, w` in `[0, 1]`, with
 * normalized coordinates. Depending on the texture configuration you obtain a
 * value, which is interpolated between the nearest corresponding array elements.
 *
 * @note CUDA only supports arrays of 2 or 3 dimensions.
 *
 * @note Instances of this class do _not_ keep devices' primary contexts
 * alive/active - just like memory allocations (but unlike events and streams).
 *
 * @tparam T array element type
 * @tparam NumDimensions number of array dimensions - either 2 or 3
 */
template <typename T, dimensionality_t NumDimensions>
class array_t {
	static_assert(NumDimensions == 2 or NumDimensions == 3, "CUDA only supports 2D and 3D arrays");

public:
	/// See @ref array::handle_t
	using handle_type = array::handle_t;
	/// See @ref array::descriptor_t
	using descriptor_type = array::descriptor_t<NumDimensions>;
	/// See @ref array::dimensions_t
	using dimensions_type = array::dimensions_t<NumDimensions>;

	/**
	 * Constructs a CUDA array wrapper from the raw type used by the CUDA
	 * Runtime API - and takes ownership of the array
	 */
	array_t(device::id_t device_id, context::handle_t context_handle, handle_type handle, dimensions_type dimensions) :
		dimensions_(dimensions), device_id_(device_id), context_handle_(context_handle), handle_(handle)
	{
		assert(handle != nullptr);
	}

	array_t(const array_t& other) = delete;
	array_t(array_t&& other) noexcept : array_t(other.device_id_, other.context_handle_, other.handle_, other.dimensions_)
	{
		other.handle_ = nullptr;
	}

	~array_t() noexcept(false)
	{
		CAW_SET_SCOPE_CONTEXT(context_handle_);
		if (handle_) {
			auto status = cuArrayDestroy(handle_);
			// Note: Throwing in a noexcept destructor; if the free'ing fails, the program
			// will likely terminate
			throw_if_error_lazy(status, "Failed destroying CUDA array " + cuda::detail_::ptr_as_hex(handle_));
		}
	}

	friend array_t array::wrap<T, NumDimensions>(device::id_t, context::handle_t, handle_type, dimensions_type) noexcept;

 	handle_type get() const noexcept { return handle_; }
	device::id_t device_id() const noexcept { return device_id_; }
	context::handle_t context_handle() const noexcept { return context_handle_; }
	dimensions_type dimensions() const noexcept { return dimensions_; }
	device_t device() const noexcept;
	context_t context() const;

	/// Overall number of elements in the array, over all dimensions
	::std::size_t size() const noexcept { return dimensions().size(); }

	/// Overall size in bytes of the elements of the array, over all dimensions
	::std::size_t size_bytes() const noexcept { return size() * sizeof(T); }

	/// Get the full set of features of this array in a single structure,
	/// recognizable by the CUDA driver (e.g. for creating additional arrays)
	descriptor_type descriptor() const	{ return array::detail_::get_descriptor<NumDimensions>(context_handle_, handle_); }

protected:
	dimensions_type    dimensions_;
	device::id_t       device_id_;
	context::handle_t  context_handle_;
	handle_type        handle_;
};

namespace array {

template <typename T, dimensionality_t NumDimensions>
array_t<T, NumDimensions> wrap(
	device::id_t                 device_id,
	context::handle_t            context_handle,
	handle_t                     handle,
	dimensions_t<NumDimensions>  dimensions) noexcept
{
	return { device_id, context_handle, handle, dimensions };
}

template <typename T, dimensionality_t NumDimensions>
array_t<T,NumDimensions> create(
	const context_t&             context,
	dimensions_t<NumDimensions>  dimensions);

template <typename T, dimensionality_t NumDimensions>
array_t<T,NumDimensions> create(
	const device_t&              device,
	dimensions_t<NumDimensions>  dimensions);


} // namespace array

} // namespace cuda

#endif // CUDA_API_WRAPPERS_ARRAY_HPP_
