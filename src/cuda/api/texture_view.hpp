/**
 * @file
 *
 * @brief Contains a "texture view" class, for hardware-accelerated
 * access to CUDA arrays, and some related standalone functions and
 * definitions.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP
#define CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP

#include "array.hpp"
#include "error.hpp"
#include "memory.hpp"

namespace cuda_ {

///@cond
class texture_view;
///@endcond

namespace texture {

namespace detail {

inline void destroy_view(handle_t handle)
{
	auto status = cuTexObjectDestroy(handle);
	throw_if_error_lazy(status, "Failed destroying texture object " + cuda_::detail::identify(handle));
}

}
/**
 * A simplifying rudimentary wrapper wrapper for the CUDA runtime API's internal
 * "texture descriptor" object, allowing the creating of such descriptors without
 * having to give it too much thought.
 *
 * @todo Could be expanded into a richer wrapper class allowing actual settings
 * of the various fields.
 */
struct descriptor_t : public CUDA_TEXTURE_DESC {
	inline descriptor_t()
	{
		using parent = CUDA_TEXTURE_DESC;
		memset(static_cast<parent*>(this), 0, sizeof(parent));
		// Note: This should set the fields directly listed in the CUDA Runtime API
		// version of this structure to 0.
		this->addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
		this->addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
		this->addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
		this->filterMode = CU_TR_FILTER_MODE_POINT;
	}
};

/**
 * Obtain a proxy object for an already-existing CUDA texture view
 *
 * @note This is a named constructor idiom, existing of direct access to the ctor
 * of the same signature, to emphasize that a new texture view is _not_ created.
 *
 * @param context_handle handle of the context in which the texture_view was created
 * @param handle raw CUDA API handle for the texture view
 * @param take_ownership when true, the wrapper will have the CUDA Runtime API destroy
 * the texture view when it destructs (making an "owning" texture view wrapper;
 * otherwise, it is assume that some other code "owns" the texture view and will
 * destroy it when necessary (and not while the wrapper is being used!)
 * @return a proxy object associated with the specified texture view
 */
inline texture_view wrap(
	device::id_t           device_id,
	context::handle_t      context_handle,
	texture::handle_t  handle,
	bool                   take_ownership) noexcept;

}  // namespace texture

/**
 * @brief Use texture memory for optimized read only cache access
 *
 * This represents a view on the memory owned by a CUDA array. Thus you can
 * first create a CUDA array (\ref cuda_::array_t) and subsequently
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
public: // types
	using handle_type = texture::handle_t;

protected: // types
	using scoped_context_setter = cuda_::context::current::detail::scoped_override_t;

public: // getters
	/// Getters for this object's raw fields
	///@{
	device::id_t device_id() const noexcept { return device_id_; }
	context::handle_t context_handle() const noexcept { return context_handle_; }
	handle_type raw_handle() const noexcept { return handle; }
	bool is_owning() const noexcept { return ownership_.has_token(); }
	///@}

public: // constructors and destructors
	template <typename T, dimensionality_t NumDimensions>
	texture_view(
		const cuda_::array_t<T, NumDimensions>& arr,
		texture::descriptor_t descriptor = texture::descriptor_t()) :
		device_id_(arr.device_id()),
		context_handle_(arr.context_handle())
	{
		scoped_context_setter set_context(context_handle_);
		CUDA_RESOURCE_DESC resource_descriptor;
		memset(&resource_descriptor, 0, sizeof(resource_descriptor));
		resource_descriptor.resType = CU_RESOURCE_TYPE_ARRAY;
		resource_descriptor.res.array.hArray = arr.get();

		auto status = cuTexObjectCreate(&handle, &resource_descriptor, &descriptor, nullptr);
		throw_if_error_lazy(status, "failed creating a CUDA texture object");
		ownership_ = { do_take_ownership, { context_handle_, handle } };
	}

protected: // constructor

	// Usable by the wrap function
	texture_view(
		device::id_t       device_id,
		context::handle_t  context_handle,
		handle_type        handle,
		bool               take_ownership) noexcept
	:
		device_id_(device_id),
		context_handle_(context_handle),
		handle(handle),
		ownership_(take_ownership,  { context_handle_, handle }) { }

public: // constructors & operators
	texture_view(const texture_view&) = delete;
	texture_view(texture_view&&) noexcept = default;
	texture_view& operator=(const texture_view&) = delete;
	texture_view& operator=(texture_view&&) noexcept = default;

public: // non-mutating getters

	/// @returns A non-owning proxy object for the CUDA context in which this texture is defined
	context_t context() const;

	/// @returns A non-owning proxy object for the CUDA device on which this texture is defined
	device_t device() const;

public: // friendship

	friend texture_view texture::wrap(device::id_t, context::handle_t, handle_type, bool) noexcept;

protected:
	device::id_t device_id_;
	context::handle_t context_handle_;
	texture::handle_t handle;
	detail::handle_ownership_t<texture_view> ownership_;
}; // texture_view

///@cond
inline bool operator==(const texture_view& lhs, const texture_view& rhs) noexcept
{
	return lhs.raw_handle() == rhs.raw_handle();
}

inline bool operator!=(const texture_view& lhs, const texture_view& rhs) noexcept
{
	return lhs.raw_handle() != rhs.raw_handle();
}
///@endcond
namespace texture {

inline texture_view wrap(
	device::id_t           device_id,
	context::handle_t      context_handle,
	texture::handle_t  handle,
	bool                   take_ownership) noexcept
{
	return { device_id, context_handle, handle, take_ownership };
}

} // namespace texture

CAW_DEFINE_HANDLE_TRAITS(texture::handle_t, is_contextual, cuTexObjectDestroy, cuTexObjectDestroy, identify);
// texture::detail::destroy_view

} // namespace cuda_

#endif // CUDA_API_WRAPPERS_TEXTURE_VIEW_HPP
