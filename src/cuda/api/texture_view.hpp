#pragma once
#ifndef CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP
#define CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP

#include <cuda/api/array.hpp>
#include <cuda/api/error.hpp>

#include <cuda_runtime.h>

namespace cuda {

namespace texture {

struct descriptor_t : public cudaTextureDesc {
	inline descriptor_t()
	{
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
 * first create a CUDA array (\ref cuda::array_t) and subsequently
 * create a `texture_view` from it. In CUDA kernels elements of the array
 * can be accessed with e.g. `float val = tex3D<float>(tex_obj, x, y, z);`,
 * where `tex_obj` can be obtained by the member function `get()` of this
 * class.
 *
 * See also the following sections in the CUDA programming guide:
 *
 * - @url https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory
 * - @url https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching
 */
class texture_view {
	public:
	template <typename T, size_t NumDimensions>
	texture_view(
		const cuda::array_t<T, NumDimensions>& arr,
		texture::descriptor_t desc = texture::descriptor_t())
	{
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(resDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = arr.get();

		auto status = cudaCreateTextureObject(&texture_object, &resDesc, &desc, NULL);
		throw_if_error(status, "failed creating a CUDA texture object");
    }

	// texture_object is basically an owning pointer, therefor no copies.
	texture_view(const texture_view&) = delete;
	texture_view& operator=(const texture_view&) = delete;
	// Without copies we would like to be able to move a texture_view.
	texture_view(texture_view&& other)
		:texture_object(other.texture_object)
	{
		other.texture_object = 0;
	}
	texture_view& operator=(texture_view&& other)
	{
		std::swap(other.texture_object,texture_object);
		return *this;
	}
	~texture_view()
	{
		auto status = cudaDestroyTextureObject(texture_object);
		throw_if_error(status, "failed destroying texture object");
	}

	operator cudaTextureObject_t() const noexcept { return texture_object; }

	private:
		cudaTextureObject_t texture_object { } ;
};


}  // namespace cuda

#endif  // CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP
