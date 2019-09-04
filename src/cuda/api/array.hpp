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

#include <cuda/api/error.hpp>
#include <cuda/api/current_device.hpp>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <array>
#include <algorithm>

namespace cuda {

namespace array {
template<typename T, size_t DIMS> class array_t;
}

namespace memory {

template<typename T>
void copy(const array::array_t<T, 3>& destination, const void *source);

}

namespace array {

template<typename T, size_t DIMS>
class array_t {
	//
	// TODO: throw an error at compile time if user wants to
	// create an instance, which is not a template specialization. This is
	// necessary because different CUDA API functions must be used to handle
	// e.g. 2D or 3D arrays.
	// 
};

template<>
class array_t<float, 3> {
	public: 

	array_t(cuda::device::id_t device_id, std::array<size_t, 3> dims) : dims_(dims) {
		cuda::device::current::scoped_override_t<> set_device_for_this_scope(device_id);
		// Up to now: stick to the defaults for channel description
		auto channel_desc = cudaCreateChannelDesc<float>();
		cudaExtent ext = make_cudaExtent(dims[0], dims[1], dims[2]);
		auto status = cudaMalloc3DArray(&array_ptr_, &channel_desc, ext);
		throw_if_error(status, "failed allocating float 3D CUDA array");
	}

	array_t() = delete;
	array_t(const array_t& other) = delete;

	~array_t() {
		auto status = cudaFreeArray(array_ptr_);
		throw_if_error(status, "failed freeing float 3D CUDA array");
	}

	size_t size() const noexcept {
		return dims_[0] * dims_[1] * dims_[2];
	}

	std::array<size_t, 3> dims() const noexcept {
		return dims_;
	}

	cudaArray* get() const noexcept {
		return array_ptr_;
	}

	
	friend void cuda::memory::copy<float>(const array_t<float, 3>& destination, const void *source);

	private:
		std::array<size_t, 3> dims_;
		cudaArray* array_ptr_;
};


}

}

#endif // CUDA_API_WRAPPERS_ARRAY_HPP
