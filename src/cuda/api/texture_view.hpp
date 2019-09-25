/**
 * @file texture_view.hpp
 *
 * @brief A CUDA texture object wrapper. It is declared as a
 * 'view' because this object only describes how a certain
 * memory resource is accessed. It is non-owning.
 *
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP
#define CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP

#include <cuda/api/array.hpp>
#include <cuda/api/error.hpp>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace cuda {

namespace texture {

struct descriptor_t : public cudaTextureDesc {
	inline descriptor_t() {
		memset(static_cast<cudaTextureDesc*>(this), 0, sizeof(cudaTextureDesc));
		this->addressMode[0] = cudaAddressModeBorder;
		this->addressMode[1] = cudaAddressModeBorder;
		this->addressMode[2] = cudaAddressModeBorder;
		this->filterMode = cudaFilterModePoint;
		this->readMode = cudaReadModeElementType;
		this->normalizedCoords = 0;
	}
};

/**
 * @brief Use texture memory for optimized read only cache access
 *
 * This represents a view on the memory owned by a CUDA array.
 */
class texture_view {
	public:
	template <typename T, size_t DIMS>
	texture_view(const cuda::array::array_t<T, DIMS>& arr, descriptor_t desc = descriptor_t()) {
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = arr.get();

		// // Create texture object
		auto status = cudaCreateTextureObject(&tobj_, &resDesc, &desc, NULL);
		throw_if_error(status, "failed creating a CUDA texture object");
    }

	~texture_view() {
		auto status = cudaDestroyTextureObject(tobj_);
		throw_if_error(status, "failed destroying texture object");
	}

	inline cudaTextureObject_t get() const noexcept {
		return tobj_;
	}

	private:
		cudaTextureObject_t tobj_ = 0;
};

}  // namespace texture

}  // namespace cuda

#endif  // CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP
