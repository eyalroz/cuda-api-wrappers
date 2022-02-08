/**
 * @file texture_view.hpp
 *
 * @brief Contains a "texture view" class, for hardware-accelerated
 * access to CUDA arrays, and some related standalone functions and
 * definitions.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP
#define CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP

#include <cuda/api/array.hpp>
#include <cuda/api/error.hpp>
#include <cuda_runtime.h>

namespace cuda {

class texture_view;

namespace texture {

using raw_handle_t = cudaTextureObject_t;

/**
 * A simplifying rudimentary wrapper wrapper for the CUDA runtime API's internal
 * "texture descriptor" object, allowing the creating of such descriptors without
 * having to give it too much thought.
 *
 * @todo Could be expanded into a richer wrapper class allowing actual settings
 * of the various fields.
 */
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

/**
 * Obtain a proxy object for an already-existing CUDA texture view
 *
 * @note This is a named constructor idiom, existing of direct access to the ctor
 * of the same signature, to emphasize that a new texture view is _not_ created.
 *
 * @param device_id Device on which the texture is located
 * @param handle raw CUDA API handle for the texture view
 * @param take_ownership when true, the wrapper will have the CUDA Runtime API destroy
 * the texture view when it destructs (making an "owning" texture view wrapper;
 * otherwise, it is assume that some other code "owns" the texture view and will
 * destroy it when necessary (and not while the wrapper is being used!)
 * @return a  wrapper object associated with the specified texture view
 */
 inline texture_view wrap(device::id_t device_id, texture::raw_handle_t handle, bool take_ownership) noexcept;

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
 * - <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-and-surface-memory">texturre and surface memory</a>
 * - <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-fetching">texture fetching</a>
 *
 * @note texture_view's are essentially _owning_ - the view is a resource the CUDA
 * runtime creates for you, which then needs to be freed.
 */
class texture_view {
	using raw_handle_type = texture::raw_handle_t;

public:
	bool is_owning() const noexcept { return owning; }
    raw_handle_type raw_handle() const noexcept { return raw_handle_; }
    device_t associated_device() const noexcept;

public: // constructors and destructors

	texture_view(const texture_view& other) = delete;

	texture_view(texture_view&& other) noexcept :
		device_id_(other.device_id_),
		raw_handle_(other.raw_handle_),
		owning(other.raw_handle_)
	{
		other.owning = false;
	};


	template <typename T, dimensionality_t NumDimensions>
	texture_view(
		const cuda::array_t<T, NumDimensions>& arr,
		texture::descriptor_t descriptor = texture::descriptor_t());

public: // operators

	~texture_view()
	{
		if (owning) {
			auto status = cudaDestroyTextureObject(raw_handle_);
			throw_if_error(status, "failed destroying texture object");
		}
	}

	texture_view& operator=(const texture_view& other) = delete;
	texture_view& operator=(texture_view& other) = delete;

protected: // constructor

	// Usable by the wrap function
	texture_view(device::id_t device_id, raw_handle_type handle , bool take_ownership) noexcept
	:   device_id_(device_id), raw_handle_(handle), owning(take_ownership) { }

public: // friendship

	friend texture_view texture::wrap(device::id_t, raw_handle_type, bool) noexcept;

protected:
    device::id_t device_id_;
	raw_handle_type raw_handle_ { } ;
	bool owning;
};


inline bool operator==(const texture_view& lhs, const texture_view& rhs) noexcept
{
	return lhs.raw_handle() == rhs.raw_handle();
}

inline bool operator!=(const texture_view& lhs, const texture_view& rhs) noexcept
{
	return not (lhs.raw_handle() == rhs.raw_handle());
}

namespace texture {

inline texture_view wrap(device::id_t device_id, texture::raw_handle_t handle, bool take_ownership) noexcept
{
	return texture_view { device_id, handle, take_ownership };
}

} // namespace texture

}  // namespace cuda

#endif  // CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP
