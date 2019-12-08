/**
 * @file array.hpp
 * 
 * @brief A CUDA array wrapper class. Especially useful
 * if used in combination with CUDA textures.
 *
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_ARRAY_HPP
#define CUDA_API_WRAPPERS_ARRAY_HPP

#include <cuda/api/current_device.hpp>
#include <cuda/api/error.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <numeric>

namespace cuda {

template<bool AssumedCurrent> class device_t;

namespace array {

namespace detail {

/**
 * @brief Do not use directly
 *
 */
template <typename T, size_t NDIMS>
class array_base {

protected:
	array_base(dimensions_t<NDIMS> dims, cudaArray* array_ptr) :
	    dims_(dims),
	    array_ptr_(array_ptr) {}

public:
	array_base()                        = delete;
	array_base(const array_base& other) = delete;

	array_base(array_base&& other) :
	    array_ptr_(other.array_ptr_),
	    dims_(other.dims_) {
		other.array_ptr_ = nullptr;
		other.dims_      = {0, 0, 0};
	}

	~array_base() {
		if (array_ptr_) {
			auto status = cudaFreeArray(array_ptr_);
			throw_if_error(status, "failed freeing CUDA array");
		}
	}

	cudaArray* get() const noexcept { return array_ptr_; }

	dimensions_t<NDIMS> dims() const noexcept { return dims_; }

	size_t size() const noexcept {
		using dimensions_value_type = typename dimensions_t<NDIMS>::value_type;
		return std::accumulate(
		    dims_.begin(), dims_.end(), static_cast<dimensions_value_type>(1),
		    [](const dimensions_value_type& a, const dimensions_value_type& b) { return a * b; });
	}

	size_t size_bytes() const noexcept { return size() * sizeof(T); }

protected:
	dimensions_t<NDIMS> dims_;
	cudaArray*          array_ptr_;
};

} // namespace detail

/**
 * @brief Wrapper for 2D and 3D arrays
 * 
 * Useful in combination with texture memory.
 * 
 * @note Only template specializations of this class can be created, because
 * CUDA does not support any number of dimensions for a CUDA array.
 */
template <typename T, size_t NDIMS>
class array_t {
public:
	array_t() = delete;
};

template <typename T>
class array_t<T, 3> : public detail::array_base<T, 3> {

private:

	template<bool AssumedCurrent>
	static cudaArray*
	malloc_3d_cuda_array(const device_t<AssumedCurrent>& device, dimensions_t<3> dims) {
		typename device_t<AssumedCurrent>::scoped_setter_t set_device_for_this_scope(device.id());
		// Up to now: stick to the defaults for channel description
		auto       channel_desc = cudaCreateChannelDesc<T>();
		cudaExtent ext          = make_cudaExtent(dims[0], dims[1], dims[2]);
		cudaArray* array_ptr;
		auto       status = cudaMalloc3DArray(&array_ptr, &channel_desc, ext);
		throw_if_error(status, "failed allocating 3D CUDA array");
		return array_ptr;
	}

public:

	template<bool AssumedCurrent>
	array_t(const device_t<AssumedCurrent>& device, dimensions_t<3> dims) :
	    detail::array_base<float, 3>(
	        dims,
	        malloc_3d_cuda_array(device, dims)) {}

};

template <typename T>
class array_t<T, 2> : public detail::array_base<T, 2> {

private:
	template<bool AssumedCurrent>
	static cudaArray*
	malloc_2d_cuda_array(const device_t<AssumedCurrent>& device, dimensions_t<2> dims) {
		typename device_t<AssumedCurrent>::scoped_setter_t set_device_for_this_scope(device.id());
		// Up to now: stick to the defaults for channel description
		auto       channel_desc = cudaCreateChannelDesc<T>();
		cudaArray* array_ptr;
		auto       status =
		    cudaMallocArray(&array_ptr, &channel_desc, dims[0], dims[1]);
		throw_if_error(status, "failed allocating 2D CUDA array");
		return array_ptr;
	}

public:
	template<bool AssumedCurrent>
	array_t(const device_t<AssumedCurrent>& device, dimensions_t<2> dims) :
	    detail::array_base<T, 2>(
	        dims,
	        malloc_2d_cuda_array(device, dims)) {}
};


} // namespace array

} // namespace cuda

#endif // CUDA_API_WRAPPERS_ARRAY_HPP
