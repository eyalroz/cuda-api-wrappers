/**
 * @file
 *
 * @brief The @ref copy_parameters class template and related definitions.
 */
#ifndef CUDA_API_WRAPPERS_COPY_PARAMETERS_HPP
#define CUDA_API_WRAPPERS_COPY_PARAMETERS_HPP

#include "array.hpp"
#include "pointer.hpp"
#include "constants.hpp"
#include "error.hpp"

namespace cuda {

namespace memory {

enum class endpoint_t {
	source, destination
};

template<dimensionality_t NumDimensions>
struct copy_parameters_t;

namespace detail_ {

template<dimensionality_t NumDimensions>
struct base_copy_params;

template<>
struct base_copy_params<2> {
	using intra_context_type = CUDA_MEMCPY2D;
	using type = intra_context_type; // Why is there no inter-context type, CUDA_MEMCPY2D_PEER ?
};

template<>
struct base_copy_params<3> {
	using type = CUDA_MEMCPY3D_PEER;
	using intra_context_type = CUDA_MEMCPY3D;
};

// Note these, by default, support inter-context
template<dimensionality_t NumDimensions>
using base_copy_params_t = typename base_copy_params<NumDimensions>::type;

template<size_t NumDimensions>
array::dimensions_t<NumDimensions>
non_array_endpoint_dimensions(endpoint_t endpoint, const copy_parameters_t<NumDimensions>& params);

} //namespace detail_


/**
 * @brief A builder-ish subclass template around the basic 2D or 3D copy
 * parameters which CUDA's complex copying API actually takes.
 *
 * {@note This class is not "safe", in the sense that there is currently no
 * checks to ensure you've actively set all fields properly before passing
 * it on to the CUDA driver.}
 *
 * {@note this class cannot hold reference units to any contexts or allocated
 * memory, so one must ensure every resource relevant to the source and the
 * destination remains alive until the copy operation is both scheduled _and
 * executed_.}
 */
template<dimensionality_t NumDimensions>
struct copy_parameters_t : detail_::base_copy_params_t<NumDimensions> {
	using parent = detail_::base_copy_params_t<NumDimensions>;
	using this_type = copy_parameters_t<NumDimensions>;
	// TODO: Perhaps use proxies?

	using intra_context_type = typename detail_::base_copy_params<NumDimensions>::intra_context_type;

	using dimensions_type = array::dimensions_t<NumDimensions>;
	using dimension_type = array::dimension_t;

	bool is_intra_context() const noexcept { return parent::srcContext == parent::dstContext; }

	this_type& set_context(endpoint_t endpoint, const context_t& context) noexcept;

	this_type& set_single_context(const context_t& context) noexcept
	{
		set_context(endpoint_t::source, context);
		set_context(endpoint_t::destination, context);
		return *this;
	}

	// Note: This assumes default pitch
	template<typename T>
	this_type& set_endpoint(endpoint_t endpoint, const cuda::array_t<T, NumDimensions> &array) noexcept;

	// Note: This assumes default pitch
	this_type& set_endpoint_untyped(
		endpoint_t endpoint,
		context::handle_t context_handle,
		void *ptr,
		dimensions_type dimensions);

	// Note: This assumes default pitch
	template<typename T>
	this_type& set_endpoint(endpoint_t endpoint, T *ptr, dimensions_type dimensions);

	// Note: This assumes default pitch
	template<typename T>
	this_type& set_endpoint(endpoint_t endpoint, span <T> span) noexcept
	{
		return set_endpoint(endpoint, span.data(), dimensions_type(span.size()));
	}

	// Note: This assumes default pitch
	// TODO: Perhaps we should have a dimensioned offset type?
	template<typename T>
	this_type& set_endpoint(
		endpoint_t endpoint,
		context::handle_t context_handle,
		T *ptr,
		dimensions_type dimensions) noexcept;

	template<typename T>
	this_type& set_source(const cuda::array_t<T, NumDimensions> &array) noexcept
	{
		return set_endpoint(endpoint_t::source, array);
	}

	this_type& set_source_untyped(context::handle_t context_handle, void *ptr, dimensions_type dimensions)
	{
		return set_endpoint_untyped(endpoint_t::source, context_handle, ptr, dimensions);
	}

	template<typename T>
	this_type& set_source(T *ptr, dimensions_type dimensions) noexcept
	{
		return set_endpoint(endpoint_t::source, ptr, dimensions);
	}

	template<typename T>
	this_type& set_source(span <T> span) noexcept
	{
		return set_source(span.data(), dimensions_type{span.size()});
	}

	template<typename T>
	this_type& set_source(context::handle_t context_handle, T *ptr, dimensions_type dimensions) noexcept
	{
		return set_endpoint(endpoint_t::source, context_handle, ptr, dimensions);
	}

	template<typename T>
	this_type& set_destination(const cuda::array_t<T, NumDimensions> &array) noexcept
	{
		return set_endpoint(endpoint_t::destination, array);
	}

	void set_destination_untyped(
		context::handle_t context_handle,
		void *ptr,
		dimensions_type dimensions) noexcept
	{
		set_endpoint_untyped(endpoint_t::destination, context_handle, ptr, dimensions);
	}

	template<typename T>
	this_type& set_destination(T *ptr, dimensions_type dimensions) noexcept
	{
		return set_endpoint(endpoint_t::destination, ptr, dimensions);
	}

	template<typename T>
	this_type& set_destination(span<T> span) noexcept
	{
		return set_destination(span.data(), {span.size(), 1, 1});
	}


	template<typename T>
	this_type& set_destination(context::handle_t context_handle, T *ptr, dimensions_type dimensions) noexcept
	{
		return set_endpoint(endpoint_t::destination, context_handle, ptr, dimensions);
	}

	this_type& set_bytes_offset(endpoint_t endpoint, dimensions_type offset) noexcept;

	// TODO: Perhaps we should have an dimensioned offset type?
	template<typename T>
	this_type& set_offset(endpoint_t endpoint, dimensions_type offset) noexcept;

	this_type& clear_offset(endpoint_t endpoint) noexcept
	{
		return set_bytes_offset(endpoint, dimensions_type::zero());
	}

	this_type& clear_offsets() noexcept
	{
		clear_offset(endpoint_t::source);
		clear_offset(endpoint_t::destination);
		return *this;
	}

	this_type& set_bytes_pitch(endpoint_t endpoint, dimension_type pitch_in_bytes) noexcept
	{
		(endpoint == endpoint_t::source ? parent::srcPitch : parent::dstPitch) = pitch_in_bytes;
		return *this;
	}

	template<typename T>
	this_type& set_pitch(endpoint_t endpoint, dimension_type pitch_in_elements) noexcept
	{
		return set_bytes_pitch(endpoint, pitch_in_elements * sizeof(T));
	}

	template<typename T>
	this_type& set_pitches(dimension_type uniform_pitch_in_elements) noexcept
	{
		auto uniform_pitch_in_bytes { uniform_pitch_in_elements * sizeof(T) };
		set_pitch<T>(endpoint_t::source, uniform_pitch_in_bytes);
		set_pitch<T>(endpoint_t::destination, uniform_pitch_in_bytes);
		return *this;
	}

	// Note: Must be called after copy extents have been set
	this_type& set_default_pitch(endpoint_t endpoint) noexcept
	{
		return set_bytes_pitch(endpoint, parent::WidthInBytes);
	}

	// Note: Must be called after copy extents have been set
	this_type& set_default_pitches() noexcept
	{
		set_default_pitch(endpoint_t::source);
		set_default_pitch(endpoint_t::destination);
		return *this;
	}

	this_type& set_bytes_extent(dimensions_type extent_in_bytes) noexcept;

	template<typename T>
	this_type& set_extent(dimensions_type extent_in_elements) noexcept;
	// Sets how much is being copies, as opposed to the sizes of the endpoints which may be larger

	dimensions_type bytes_extent() const noexcept;

	template <typename T>
	dimensions_type extent() const noexcept
	{
		auto extent_ = bytes_extent();
#ifndef NDEBUG
		if (extent_.width % sizeof(T) != 0) {
			throw ::std::invalid_argument(
				"Attempt to get the copy extent with assumed type of size "
				+ ::std::to_string(sizeof(T)) + " while the byte extent's "
				+ "minor dimension is not a multiple of this size");
		}
#endif
		extent_.width /= sizeof(T);
		return extent_;
	}

	this_type& set_pitches(array::dimension_t uniform_pitch_in_bytes) noexcept
	{
		set_pitch(endpoint_t::source, uniform_pitch_in_bytes);
		set_pitch(endpoint_t::destination, uniform_pitch_in_bytes);
		return *this;
	}

	this_type& clear_rest() noexcept;
	// Clear any dummy fields which are required to be set to 0. Note that important fields,
	// which you have not set explicitly, will _not_ be cleared by this method.

};

template<>
inline copy_parameters_t<2>& copy_parameters_t<2>::set_endpoint_untyped(
	endpoint_t              endpoint,
	context::handle_t,
	void *                  ptr,
	array::dimensions_t<2>  dimensions);

template<>
inline copy_parameters_t<3>& copy_parameters_t<3>::set_endpoint_untyped(
	endpoint_t              endpoint,
	context::handle_t,
	void *                  ptr,
	array::dimensions_t<3>  dimensions);

template<>
template<typename T>
inline copy_parameters_t<2>& copy_parameters_t<2>::set_endpoint(
	endpoint_t endpoint,
	context::handle_t context_handle,
	T *ptr,
	array::dimensions_t<2> dimensions) noexcept
{
	array::dimensions_t<2> untyped_dims = {dimensions.width * sizeof(T), dimensions.height};
	return set_endpoint_untyped(endpoint, context_handle, ptr, untyped_dims);
}

template<>
template<typename T>
inline copy_parameters_t<2>& copy_parameters_t<2>::set_endpoint(
	endpoint_t endpoint,
	T *ptr,
	array::dimensions_t<2> dimensions)
{
	// We would have _liked_ to say:
	// auto context_handle = context::current::detail_::get_handle();
	// ... here, but alas, 2D copy structures don't support contexts, so...
	auto context_handle = context::detail_::none;
	return set_endpoint<T>(endpoint, context_handle, ptr, dimensions);
}

template<>
template<typename T>
copy_parameters_t<2> &copy_parameters_t<2>::set_endpoint(endpoint_t endpoint, const cuda::array_t<T, 2> &array) noexcept
{
	(endpoint == endpoint_t::source ? srcMemoryType : dstMemoryType) = CU_MEMORYTYPE_ARRAY;
	(endpoint == endpoint_t::source ? srcArray : dstArray) = array.get();
	(endpoint == endpoint_t::source ? srcDevice : dstDevice) = array.device_id();
	// Can't set the endpoint context - the basic data structure doesn't support that!
	return *this;
}

namespace detail_ {

template<>
inline array::dimensions_t<2> non_array_endpoint_dimensions<2>(endpoint_t endpoint, const copy_parameters_t<2>& params)
{
	using dims_type = copy_parameters_t<2>::dimensions_type;
	return (endpoint == endpoint_t::source) ?
		   dims_type{ params.WidthInBytes, params.Height } :
		   dims_type{ params.WidthInBytes, params.Height };
}

template<>
inline array::dimensions_t<3> non_array_endpoint_dimensions<3>(endpoint_t endpoint, const copy_parameters_t<3>& params)
{
	using dims_type = copy_parameters_t<3>::dimensions_type;
	return (endpoint == endpoint_t::source) ?
		   dims_type{ params.srcPitch, params.Height, params.Depth } :
		   dims_type{ params.WidthInBytes, params.Height, params.Depth };
}

} //

template<>
template<typename T>
copy_parameters_t<3>& copy_parameters_t<3>::set_endpoint(endpoint_t endpoint, const cuda::array_t<T, 3> &array) noexcept
{
	(endpoint == endpoint_t::source ? srcMemoryType : dstMemoryType) = CU_MEMORYTYPE_ARRAY;
	(endpoint == endpoint_t::source ? srcArray : dstArray) = array.get();
	(endpoint == endpoint_t::source ? srcContext : dstContext) = array.context_handle();
	return *this;
}

// 2D copy parameters only have an intra-context variant; should we silently assume the context
// is the same for both ends?
template<>
inline copy_parameters_t<2>& copy_parameters_t<2>::set_context(endpoint_t endpoint, const context_t& context) noexcept = delete;

template<>
inline copy_parameters_t<3>& copy_parameters_t<3>::set_context(endpoint_t endpoint, const context_t& context) noexcept
{
	(endpoint == endpoint_t::source ? srcContext : dstContext) = context.handle();
	return *this;
}

template<>
template<typename T>
inline copy_parameters_t<3>& copy_parameters_t<3>::set_endpoint(
	endpoint_t endpoint,
	context::handle_t context_handle,
	T *ptr,
	array::dimensions_t<3> dimensions) noexcept
{
	array::dimensions_t<3> untyped_dims = {dimensions.width * sizeof(T), dimensions.height, dimensions.depth};
	return set_endpoint_untyped(endpoint, context_handle, ptr, untyped_dims);
}

template<>
template<typename T>
inline copy_parameters_t<3>& copy_parameters_t<3>::set_endpoint(
	endpoint_t endpoint,
	T *ptr,
	array::dimensions_t<3> dimensions)
{
	return set_endpoint<T>(endpoint, context::current::detail_::get_handle(), ptr, dimensions);
}

template<>
inline copy_parameters_t<2> &copy_parameters_t<2>::clear_rest() noexcept
{
	return *this;
}

template<>
inline copy_parameters_t<3>& copy_parameters_t<3>::clear_rest() noexcept
{
	srcLOD = 0;
	dstLOD = 0;
	return *this;
}

template<>
template<typename T>
inline copy_parameters_t<2> &copy_parameters_t<2>::set_extent(dimensions_type extent_in_elements) noexcept
{
	WidthInBytes = extent_in_elements.width * sizeof(T);
	Height = extent_in_elements.height;
	return *this;
}

template<>
inline copy_parameters_t<2>& copy_parameters_t<2>::set_bytes_extent(dimensions_type extent_in_elements) noexcept
{
	WidthInBytes = extent_in_elements.width;
	Height = extent_in_elements.height;
	return *this;
}

template<>
inline copy_parameters_t<3>& copy_parameters_t<3>::set_bytes_extent(dimensions_type extent_in_elements) noexcept
{
	WidthInBytes = extent_in_elements.width;
	Height = extent_in_elements.height;
	Depth = extent_in_elements.depth;
	return *this;
}

template<>
inline copy_parameters_t<2>::dimensions_type copy_parameters_t<2>::bytes_extent() const noexcept
{
	return copy_parameters_t<2>::dimensions_type { WidthInBytes, Height };
}

template<>
inline copy_parameters_t<3>::dimensions_type copy_parameters_t<3>::bytes_extent() const noexcept
{
	return copy_parameters_t<3>::dimensions_type { WidthInBytes, Height, Depth };
}

template<>
inline copy_parameters_t<2>& copy_parameters_t<2>::set_endpoint_untyped(
	endpoint_t              endpoint,
	context::handle_t,
	void *                  ptr,
	array::dimensions_t<2>  dimensions)
{
	auto memory_type = memory::type_of(ptr);
	if (memory_type == memory::type_t::array) {
		throw ::std::invalid_argument("Attempt to use the non-array endpoint setter with array memory at " + cuda::detail_::ptr_as_hex(ptr));
	}
	if (memory_type == memory::type_t::unified_ or memory_type == type_t::device_)
	{
		(endpoint == endpoint_t::source ? srcDevice : dstDevice) = device::address(ptr);
	}
	else {
		// Either memory::type_t::host or memory::type_t::non_cuda
		if (endpoint == endpoint_t::source) { srcHost = ptr; }
		else { dstHost = ptr; }
	}
	set_bytes_pitch(endpoint, dimensions.width);
	(endpoint == endpoint_t::source ? srcMemoryType : dstMemoryType) = static_cast<CUmemorytype>
	(memory_type == memory::type_t::non_cuda ? memory::type_t::host_ : memory_type);
	// Can't set the endpoint context - the basic data structure doesn't support that!
	// (endpoint == endpoint_t::source ? srcContext : dstContext) = context_handle;

	if (bytes_extent().area() == 0) {
		set_bytes_extent(dimensions);
	}
	return *this;
}

template<>
inline copy_parameters_t<3>& copy_parameters_t<3>::set_endpoint_untyped(
	endpoint_t              endpoint,
	context::handle_t       context_handle,
	void *                  ptr,
	array::dimensions_t<3>  dimensions)
{
	auto memory_type = memory::type_of(ptr);
	if (memory_type == memory::type_t::array) {
		throw ::std::invalid_argument("Attempt to use the non-array endpoint setter with array memory at " + cuda::detail_::ptr_as_hex(ptr));
	}
	if (memory_type == memory::type_t::unified_ or memory_type == type_t::device_)
	{
		(endpoint == endpoint_t::source ? srcDevice : dstDevice) = device::address(ptr);
	}
	else {
		// Either memory::type_t::host or memory::type_t::non_cuda
		if (endpoint == endpoint_t::source) { srcHost = ptr; }
		else { dstHost = ptr; }
	}
	set_bytes_pitch(endpoint, dimensions.width);
	(endpoint == endpoint_t::source ? srcHeight : dstHeight) = dimensions.height;
	(endpoint == endpoint_t::source ? srcMemoryType : dstMemoryType) = static_cast<CUmemorytype>
	(memory_type == memory::type_t::non_cuda ? memory::type_t::host_ : memory_type);
	(endpoint == endpoint_t::source ? srcContext : dstContext) = context_handle;

	if (bytes_extent().volume() == 0) {
		set_bytes_extent(dimensions);
	}

	return *this;
}


template<>
template<typename T>
copy_parameters_t<3>& copy_parameters_t<3>::set_extent(dimensions_type extent_in_elements) noexcept
{
	dimensions_type extent_in_bytes{
	        extent_in_elements.width * sizeof(T),
	        extent_in_elements.height,
	        extent_in_elements.depth
	};
	return set_bytes_extent(extent_in_bytes);
}

template<>
inline copy_parameters_t<3>&
copy_parameters_t<3>::set_bytes_offset(endpoint_t endpoint, dimensions_type offset) noexcept
{
	(endpoint == endpoint_t::source ? srcXInBytes : dstXInBytes) = offset.width;
	(endpoint == endpoint_t::source ? srcY : dstY) = offset.height;
	(endpoint == endpoint_t::source ? srcZ : dstZ) = offset.depth;
	return *this;
}

template<>
inline copy_parameters_t<2> &
copy_parameters_t<2>::set_bytes_offset(endpoint_t endpoint, dimensions_type offset) noexcept
{
	(endpoint == endpoint_t::source ? srcXInBytes : dstXInBytes) = offset.width;
	(endpoint == endpoint_t::source ? srcY : dstY) = offset.height;
	return *this;
}

template<>
template<typename T>
copy_parameters_t<3>& copy_parameters_t<3>::set_offset(endpoint_t endpoint, dimensions_type offset) noexcept
{
	dimensions_type offset_in_bytes{offset.width * sizeof(T), offset.height, offset.depth};
	return set_bytes_offset(endpoint, offset_in_bytes);
}

template<>
template<typename T>
copy_parameters_t<2> &copy_parameters_t<2>::set_offset(endpoint_t endpoint, dimensions_type offset) noexcept
{
	dimensions_type offset_in_bytes{offset.width * sizeof(T), offset.height};
	return set_bytes_offset(endpoint, offset_in_bytes);
}

copy_parameters_t<3>::intra_context_type
inline as_intra_context_parameters(const copy_parameters_t<3>& params)
{
	if (params.srcDevice != params.dstDevice) {
		throw ::std::invalid_argument("Attempt to use inter-device copy parameters for an intra-context copy");
	}
	if (params.srcContext != params.dstContext) {
		throw ::std::invalid_argument("Attempt to use inter-context copy parameters for an intra-context copy");
	}

	// TODO: Use designated initializers in C++20
	copy_parameters_t<3>::intra_context_type result;

	result.srcXInBytes = params.srcXInBytes;
	result.srcY = params.srcY;
	result.srcZ = params.srcZ;
	result.srcLOD = params.srcLOD;
	result.srcMemoryType = params.srcMemoryType;
	result.srcHost = params.srcHost;
	result.srcDevice = params.srcDevice;
	result.srcArray = params.srcArray;
	result.reserved0 = nullptr; // srcContext
	result.srcPitch = params.srcPitch;
	result.srcHeight = params.srcHeight;

	result.dstXInBytes = params.dstXInBytes;
	result.dstY = params.dstY;
	result.dstZ = params.dstZ;
	result.dstLOD = params.dstLOD;
	result.dstMemoryType = params.dstMemoryType;
	result.dstHost = params.dstHost;
	result.dstDevice = params.dstDevice;
	result.dstArray = params.dstArray;
	result.reserved1 = nullptr;
	result.dstPitch = params.dstPitch;
	result.dstHeight = params.dstHeight;

	result.WidthInBytes = params.WidthInBytes;
	result.Height = params.Height;
	result.Depth = params.Depth;
	return result;
}

} //namespace memory

} // namespace cuda


#endif //CUDA_API_WRAPPERS_COPY_PARAMETERS_HPP
