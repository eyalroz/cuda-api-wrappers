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

class texture_view {
	public:
	template <typename T, size_t DIMS>
	texture_view(const cuda::array::array_t<T, DIMS>& arr) {
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = arr.get();

		// // Specify texture object parameters
		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(texDesc));
		texDesc.addressMode[0] = cudaAddressModeBorder;
		texDesc.addressMode[1] = cudaAddressModeBorder;
		texDesc.addressMode[2] = cudaAddressModeBorder;
		texDesc.filterMode = cudaFilterModePoint;
		texDesc.readMode = cudaReadModeElementType;
		texDesc.normalizedCoords = 0;

		// // Create texture object
		auto status = cudaCreateTextureObject(&tobj_, &resDesc, &texDesc, NULL);
		throw_if_error(status, "failed creating a CUDA texture object");
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
