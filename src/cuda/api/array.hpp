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

namespace detail_ {

template<typename T>
cudaArray* allocate_on_current_device(array::dimensions_t<3> dimensions)
{
	auto channel_descriptor = cudaCreateChannelDesc<T>();
	cudaExtent extent = dimensions;
	cudaArray* raw_cuda_array;
	auto status = cudaMalloc3DArray(&raw_cuda_array, &channel_descriptor, extent);
	throw_if_error(status, "failed allocating 3D CUDA array");
	return raw_cuda_array;
}

template<typename T>
cudaArray* allocate_on_current_device(array::dimensions_t<2> dimensions)
{
	auto channel_desc = cudaCreateChannelDesc<T>();
	cudaArray* raw_cuda_array;
	auto status = cudaMallocArray(&raw_cuda_array, &channel_desc, dimensions.width, dimensions.height);
	throw_if_error(status, "failed allocating 2D CUDA array");
	return raw_cuda_array;
}

template<typename T>
cudaArray* allocate(device_t& device, array::dimensions_t<3> dimensions);

template<typename T>
cudaArray* allocate(device_t& device, array::dimensions_t<2> dimensions);

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
	/**
	 * Constructs a CUDA array wrapper from the raw type used by the CUDA
	 * Runtime API - and takes ownership of the array
	 */
	array_t(cudaArray* raw_cuda_array, array::dimensions_t<NumDimensions> dimensions) :
	    dimensions_(dimensions), raw_array_(raw_cuda_array)
	{
		assert(raw_cuda_array != nullptr);
	}

	/**
	 * Creates and wraps a new CUDA array.
	 */
	array_t(device_t& device, array::dimensions_t<NumDimensions> dimensions)
		: array_t(array::detail_::allocate<T>(device, dimensions), dimensions) {}
	array_t(const array_t& other) = delete;
	array_t(array_t&& other) noexcept : array_t(other.raw_array_, other.dimensions_)
	{
		other.raw_array_ = nullptr;
	}

	~array_t() noexcept
	{
		if (raw_array_) {
			auto status = cudaFreeArray(raw_array_);
			// Note: Throwing in a noexcept destructor; if the free'ing fails, the program
			// will likely terminate
			throw_if_error(status, "failed freeing CUDA array");
		}
	}

	cudaArray* get() const noexcept { return raw_array_; }
	array::dimensions_t<NumDimensions> dimensions() const noexcept { return dimensions_; }
	::std::size_t size() const noexcept { return dimensions().size(); }
	::std::size_t size_bytes() const noexcept { return size() * sizeof(T); }

protected:
	array::dimensions_t<NumDimensions> dimensions_;
	cudaArray*                         raw_array_;
};

} // namespace cuda

#endif // CUDA_API_WRAPPERS_ARRAY_HPP_
