/**
 * @file array.hpp
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

#include <cuda/api/current_device.hpp>
#include <cuda/api/error.hpp>

#include <cuda_runtime.h>

namespace cuda {

class device_t;

template <typename T, dimensionality_t NumDimensions>
class array_t;

namespace array {

using handle_t = cudaArray*;

/**
 * @brief Wrap an existing CUDA array in an @ref array_t instance.
 */
template <typename T, dimensionality_t NumDimensions>
array_t<T, NumDimensions> wrap(
    device::id_t                 device_id,
	handle_t                     handle,
	dimensions_t<NumDimensions>  dimensions) noexcept;

namespace detail_ {

template<typename T>
handle_t create_on_current_device(dimensions_t<3> dimensions)
{
	auto channel_descriptor = cudaCreateChannelDesc<T>();
	cudaExtent extent = dimensions;
	handle_t handle;
	auto status = cudaMalloc3DArray(&handle, &channel_descriptor, extent);
	throw_if_error(status, "Failed allocating 3D CUDA array");
	return handle;
}

template<typename T>
handle_t create_on_current_device(dimensions_t<2> dimensions)
{
	auto channel_desc = cudaCreateChannelDesc<T>();
	handle_t handle;
	auto status = cudaMallocArray(&handle, &channel_desc, dimensions.width, dimensions.height);
	throw_if_error(status, "Failed allocating 2D CUDA array");
	return handle;
}

template <typename T, dimensionality_t NumDimensions>
handle_t create(const device_t& device, dimensions_t<NumDimensions> dimensions);

template <typename T, dimensionality_t NumDimensions>
handle_t create(device::id_t device_id, dimensions_t<NumDimensions> dimensions)
{
	device::current::detail_::scoped_override_t set_device_for_this_scope(device_id);
	return create_on_current_device<T>(dimensions);
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
 */
template <typename T, dimensionality_t NumDimensions>
class array_t {
	static_assert(NumDimensions == 2 or NumDimensions == 3, "CUDA only supports 2D and 3D arrays");

public:
	using handle_type = array::handle_t;
	using dimensions_type = array::dimensions_t<NumDimensions>;

	/**
	 * Constructs a CUDA array wrapper from the raw type used by the CUDA
	 * Runtime API - and takes ownership of the array
	 */
	array_t(device::id_t device_id, handle_type handle, dimensions_type dimensions) :
	    device_id_(device_id), dimensions_(dimensions), handle_(handle)
	{
		assert(handle != nullptr);
	}

	array_t(const array_t& other) = delete;
	array_t(array_t&& other) noexcept : array_t(other.device_id_, other.handle_, other.dimensions_)
	{
		other.handle_ = nullptr;
	}

	~array_t() noexcept
	{
		if (handle_) {
			auto status = cudaFreeArray(handle_);
			// Note: Throwing in a noexcept destructor; if the free'ing fails, the program
			// will likely terminate
			throw_if_error(status, "Failed freeing CUDA array");
		}
	}

	friend array_t array::wrap<T, NumDimensions>(device::id_t, handle_type, dimensions_type) noexcept;

    handle_type get() const noexcept { return handle_; }
    device_t device() const noexcept;
	dimensions_type dimensions() const noexcept { return dimensions_; }
	::std::size_t size() const noexcept { return dimensions().size(); }
	::std::size_t size_bytes() const noexcept { return size() * sizeof(T); }

protected:
	dimensions_type  dimensions_;
	handle_type      handle_;
	device::id_t     device_id_;
};

namespace array {

template <typename T, dimensionality_t NumDimensions>
inline array_t<T, NumDimensions> wrap(
    device::id_t                 device_id,
	handle_t                     handle,
	dimensions_t<NumDimensions>  dimensions) noexcept
{
	return array_t<T, NumDimensions> { device_id, handle, dimensions };
}

template <typename T, dimensionality_t NumDimensions>
array_t<T, NumDimensions> create(
	const device_t&              device,
	dimensions_t<NumDimensions>  dimensions);

} // namespace array

} // namespace cuda

#endif // CUDA_API_WRAPPERS_ARRAY_HPP_
