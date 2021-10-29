/**
 * @file array.hpp
 *
 * @brief Contains a proxy class for CUDA arrays - GPU memory
 * with 2-D or 3-D locality and hardware support for interpolated value
 * retrieval); see also @ref texture_view.hpp .
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_ARRAY_HPP_
#define CUDA_API_WRAPPERS_ARRAY_HPP_

#include <cuda/api/current_device.hpp>
#include <cuda/api/error.hpp>

#include <cuda_runtime.h>

namespace cuda {

class device_t;

namespace array {

using handle_t = cudaArray*;

namespace detail_ {

template<typename T>
handle_t allocate_on_current_device(array::dimensions_t<3> dimensions)
{
	auto channel_descriptor = cudaCreateChannelDesc<T>();
	cudaExtent extent = dimensions;
	handle_t handle;
	auto status = cudaMalloc3DArray(&handle, &channel_descriptor, extent);
	throw_if_error(status, "failed allocating 3D CUDA array");
	return handle;
}

template<typename T>
handle_t allocate_on_current_device(array::dimensions_t<2> dimensions)
{
	auto channel_desc = cudaCreateChannelDesc<T>();
	handle_t handle;
	auto status = cudaMallocArray(&handle, &channel_desc, dimensions.width, dimensions.height);
	throw_if_error(status, "failed allocating 2D CUDA array");
	return handle;
}

template<typename T>
handle_t allocate(device_t& device, array::dimensions_t<3> dimensions);

template<typename T>
handle_t allocate(device_t& device, array::dimensions_t<2> dimensions);

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
	/**
	 * Constructs a CUDA array wrapper from the raw type used by the CUDA
	 * Runtime API - and takes ownership of the array
	 */
	array_t(handle_type handle, array::dimensions_t<NumDimensions> dimensions) :
	    dimensions_(dimensions), handle_(handle)
	{
		assert(handle != nullptr);
	}

	/**
	 * Creates and wraps a new CUDA array.
	 */
	array_t(device_t& device, array::dimensions_t<NumDimensions> dimensions)
		: array_t(array::detail_::allocate<T>(device, dimensions), dimensions) {}
	array_t(const array_t& other) = delete;
	array_t(array_t&& other) noexcept : array_t(other.handle_, other.dimensions_)
	{
		other.handle_ = nullptr;
	}

	~array_t() noexcept
	{
		if (handle_) {
			auto status = cudaFreeArray(handle_);
			// Note: Throwing in a noexcept destructor; if the free'ing fails, the program
			// will likely terminate
			throw_if_error(status, "failed freeing CUDA array");
		}
	}

	handle_type get() const noexcept { return handle_; }
	array::dimensions_t<NumDimensions> dimensions() const noexcept { return dimensions_; }
	::std::size_t size() const noexcept { return dimensions().size(); }
	::std::size_t size_bytes() const noexcept { return size() * sizeof(T); }

protected:
	array::dimensions_t<NumDimensions> dimensions_;
	handle_type                        handle_;
};

} // namespace cuda

#endif // CUDA_API_WRAPPERS_ARRAY_HPP_
