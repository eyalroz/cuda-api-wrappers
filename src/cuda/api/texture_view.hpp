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

}  // namespace texture

/**
 * @brief Use texture memory for optimized read only cache access
 *
 * This represents a view on the memory owned by a CUDA array. Thus you can
 * first create a CUDA array (\ref cuda::array::array_t) and subsequently
 * create a `texture_view` from it. In CUDA kernels elements of the array
 * can be accessed with e.g. `float val = tex3D<float>(tex_obj, x, y, z);`,
 * where `tex_obj` can be obtained by the member function `get()` of this
 * class.
 *
 * See also the sections in the programming guide:
 *
 * - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory
 * - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching
 */
class texture_view {
	public:
	template <typename T, size_t DIMS>
	texture_view(const cuda::array::array_t<T, DIMS>& arr, texture::descriptor_t desc = texture::descriptor_t()) {
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


}  // namespace cuda

#endif  // CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP
