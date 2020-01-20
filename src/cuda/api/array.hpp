#pragma once
#ifndef CUDA_API_WRAPPERS_ARRAY_HPP_
#define CUDA_API_WRAPPERS_ARRAY_HPP_

#include <cuda/api/current_device.hpp>
#include <cuda/api/error.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <array>

namespace cuda {

template<bool AssumedCurrent> class device_t;

namespace array {

namespace detail {

template <typename T, size_t NumDimensions>
class array_base {

protected:
	array_base(dimensions_t<NumDimensions> dimensions, cudaArray* raw_cuda_array) :
	    dimensions_(dimensions),
	    raw_cuda_array_(raw_cuda_array) {}

public:
	array_base()                        = delete;
	array_base(const array_base& other) = delete;

	array_base(array_base&& other) :
	    raw_cuda_array_(other.raw_cuda_array_),
	    dimensions_(other.dimensions_)
	{
		other.raw_cuda_array_ = nullptr;
		other.dimensions_      = {0, 0, 0};
	}

	~array_base() noexcept
	{
		if (raw_cuda_array_) {
			auto status = cudaFreeArray(raw_cuda_array_);
			throw_if_error(status, "failed freeing CUDA array");
		}
	}

	cudaArray* get() const noexcept { return raw_cuda_array_; }

	dimensions_t<NumDimensions> dimensions() const noexcept { return dimensions_; }

	size_t size() const noexcept
		{
		size_t s = 1;
		for (size_t dimension_id = 0; dimension_id < NumDimensions; ++dimension_id) {
			s *= dimensions_[dimension_id];
		}
		return s;
	}

	size_t size_bytes() const noexcept { return size() * sizeof(T); }

protected:
	dimensions_t<NumDimensions> dimensions_;
	cudaArray*          raw_cuda_array_;
};

} // namespace detail

/**
 * @brief Wrapper for 2D and 3D arrays
 *
 * A CUDA array is a memory owning, multi dimensional, multi array and **not**
 * comparible in its functionality with a `std::array`.
 *
 * See also the documentation of CUDA arrays in the programming guide:
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-arrays
 *
 * Useful in combination with texture memory (\ref cuda::texture_view).
 * One can access elements in a multi dimensional array with the index, e.g.
 * `array[i][j][k]`. In CUDA it is possible to create a texture on an array and thus
 * use more advanced ways of accessing the elements of the array, e.g.
 * `texture[u][v][w]` with `u, v, w` in `[0, 1]`, with normalized coordinates.
 * Depending on the texture configuration you obtain a value, which is interpolated
 * between the nearest corresponding array elements.
 *
 * @note CUDA only supports arrays of 2 or 3 dimensions; array_t's cannot be
 * instantiated with other values of `NumDimensions`
 */
template <typename T, size_t NumDimensions>
class array_t;

template <typename T>
class array_t<T, 3> : public detail::array_base<T, 3> {

private:

	template<bool AssumedCurrent>
	static cudaArray*
	malloc_3d_cuda_array(const device_t<AssumedCurrent>& device, dimensions_t<3> dimensions)
	{
		typename device_t<AssumedCurrent>::scoped_setter_t set_device_for_this_scope(device.id());
		// Up to now: stick to the defaults for channel description
		auto       channel_desc = cudaCreateChannelDesc<T>();
		cudaExtent ext          = make_cudaExtent(dimensions[0], dimensions[1], dimensions[2]);
		cudaArray* raw_cuda_array;
		auto       status = cudaMalloc3DArray(&raw_cuda_array, &channel_desc, ext);
		throw_if_error(status, "failed allocating 3D CUDA array");
		return raw_cuda_array;
	}

public:

	template<bool AssumedCurrent>
	array_t(const device_t<AssumedCurrent>& device, dimensions_t<3> dimensions) :
	    detail::array_base<float, 3>(
	        dimensions,
	        malloc_3d_cuda_array(device, dimensions)) {}

};

template <typename T>
class array_t<T, 2> : public detail::array_base<T, 2> {

private:
	template<bool AssumedCurrent>
	static cudaArray*
	malloc_2d_cuda_array(const device_t<AssumedCurrent>& device, dimensions_t<2> dimensions)
	{
		typename device_t<AssumedCurrent>::scoped_setter_t set_device_for_this_scope(device.id());
		// Up to now: stick to the defaults for channel description
		auto       channel_desc = cudaCreateChannelDesc<T>();
		cudaArray* raw_cuda_array;
		auto       status =
		    cudaMallocArray(&raw_cuda_array, &channel_desc, dimensions[0], dimensions[1]);
		throw_if_error(status, "failed allocating 2D CUDA array");
		return raw_cuda_array;
	}

public:
	template<bool AssumedCurrent>
	array_t(const device_t<AssumedCurrent>& device, dimensions_t<2> dimensions) :
	    detail::array_base<T, 2>(
	        dimensions,
	        malloc_2d_cuda_array(device, dimensions)) {}
};

} // namespace array

} // namespace cuda

#endif // CUDA_API_WRAPPERS_ARRAY_HPP_
