/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard the copy parameters structure template.
 */
#ifndef CAW_MULTI_WRAPPER_IMPLS_COPY_PARAMETERS_HPP_
#define CAW_MULTI_WRAPPER_IMPLS_COPY_PARAMETERS_HPP_

#include "../copy_parameters.hpp"
#include "../memory.hpp"
#include <type_traits>

namespace cuda {

namespace detail_ {

template <typename T, dimensionality_t Dimensionality>
array::dimensions_t<Dimensionality> get_byte_dimensions(const array_t<T, Dimensionality>& arr)
{
	auto dims = arr.dimensions();
	dims.width *= sizeof(T);
	return dims;
}

template <typename T, typename = cuda::detail_::enable_if_t<is_regionish<T>::value> >
size_t get_byte_dimensions(T&& regionish)
{
	auto region = memory::const_region_t{regionish};
	return region.size(); // in bytes
}

template <typename T, typename = cuda::detail_::enable_if_t<is_ptrish<T>::value> >
size_t get_byte_dimensions(const T* ptr) { return 0; }

} // namespace detail_

template <typename Destination, typename Source>
memory::copy_parameters_t<detail_::copy_params_dimensionality<Destination, Source>::value>
make_copy_parameters(Destination&& destination, Source&& source)
{
	enum {
		dimensionality = detail_::copy_params_dimensionality<Destination, Source>::value
	};
	using cuda::memory::endpoint_t;
	memory::copy_parameters_t<dimensionality> params;
	params.set_endpoint(endpoint_t::destination, ::std::forward<Destination>(destination));
	params.set_endpoint(endpoint_t::source, ::std::forward<Source>(source));
	if (cuda::detail_::is_array<typename ::std::remove_reference<Source>::type>::value) {
		auto byte_dims = detail_::get_byte_dimensions(source);
		params.set_bytes_extent(byte_dims);
	}
	else if (cuda::detail_::is_array<typename ::std::remove_reference<Destination>::type>::value) {
		auto byte_dims = detail_::get_byte_dimensions(destination);
		params.set_bytes_extent(byte_dims);
		// The source is _not_ an array, and there's nothing else which sets the pitch properly for us;
		// but - TODO: Do we even need this?
		params.set_bytes_pitch(endpoint_t::source, params.bytes_extent().width);
	}
	else {
		throw ::std::invalid_argument("Cannot determine the copy extent using two pointer-like endpoints");
	}
	params.clear_offset(endpoint_t::source);
	params.clear_offset(endpoint_t::destination);
	// TODO: What about contexts?
	params.clear_rest();
	return params;
}

} // namespace cuda

#endif //CAW_MULTI_WRAPPER_IMPLS_COPY_PARAMETERS_HPP_
