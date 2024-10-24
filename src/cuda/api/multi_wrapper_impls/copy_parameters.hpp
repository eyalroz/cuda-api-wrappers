/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard the copy parameters structure template.
 */
#ifndef MULTI_WRAPPER_IMPLS_COPY_PARAMETERS_HPP_
#define MULTI_WRAPPER_IMPLS_COPY_PARAMETERS_HPP_

#include "../copy_parameters.hpp"
#include "../memory.hpp"
#include <type_traits>

namespace cuda {

namespace detail_ {

template <typename T>
struct array_dimensionality { };

template <typename T, dimensionality_t Dimensionality>
struct array_dimensionality<array_t<T, Dimensionality>>{ enum { value = Dimensionality }; };

//template <typename T, typename cuda::detail_::enable_if_t<cuda::detail_::is_kinda_like_contiguous_container<T>::value, void>>
//struct dimensionality<T>{ static enum { value = 1 }; };

template <typename D, typename S>
constexpr dimensionality_t common_array_dimensionality()
{
	enum {
		d_dim = detail_::array_dimensionality<D>::value,
		s_dim = detail_::array_dimensionality<S>::value
	};
	return (d_dim > s_dim) ? d_dim : s_dim;
}

} // namespace detail_

template <typename Destination, typename Source>
memory::copy_parameters_t<detail_::array_dimensionality_for_pair_t<Destination,Source>::value>
make_copy_parameters(Destination&& destination, Source&& source);

template <typename Destination, typename Source>
memory::copy_parameters_t<detail_::array_dimensionality_for_pair_t<Destination,Source>::value>
make_copy_parameters(Destination&& destination, Source&& source)
{
	enum {
		dest_type = cuda::memory::detail_::copy_endpoint_type<Destination>::value,
		src_type = cuda::memory::detail_::copy_endpoint_type<Source>::value,
		arrayish = cuda::memory::detail_::copy_endpoint_kind_t::array
	};

	enum { dimensionality = detail_::array_dimensionality_for_pair_t<Destination,Source>::value };
	memory::copy_parameters_t<dimensionality> params;
	// TODO: May need to write specialized functions for this stuff
	params.set_endpoint(params, cuda::memory::endpoint_t::destination, ::std::forward<Destination>(destination));
	params.set_endpoint(params, cuda::memory::endpoint_t::source, ::std::forward<Source>(source));
	// Note: Nothing about contents, extents etc.
}

} // namespace cuda

#endif //MULTI_WRAPPER_IMPLS_COPY_PARAMETERS_HPP_
