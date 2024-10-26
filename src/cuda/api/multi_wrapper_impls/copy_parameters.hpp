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

template <typename Destination, typename Source>
memory::copy_parameters_t<detail_::copy_params_dimensionality<Destination, Source>::value>
make_copy_parameters(Destination&& destination, Source&& source)
{
	enum {
		dest_type = cuda::memory::detail_::copy_endpoint_kind<Destination>::value,
		src_type = cuda::memory::detail_::copy_endpoint_kind<Source>::value,

		arrayish = cuda::memory::detail_::copy_endpoint_kind_t::array,

		dimensionality = detail_::copy_params_dimensionality<Destination, Source>::value
	};
	using cuda::memory::endpoint_t;
	memory::copy_parameters_t<dimensionality> params;
	params.set_endpoint(endpoint_t::destination, ::std::forward<Destination>(destination));
	params.set_endpoint(endpoint_t::source, ::std::forward<Source>(source));
	params.clear_offset(endpoint_t::source);
	params.clear_offset(endpoint_t::destination);
	// TODO: What about contexts?
	// TODO: What about extents? Is setting the endpoints sufficient?
	return params;
}

} // namespace cuda

#endif //CAW_MULTI_WRAPPER_IMPLS_COPY_PARAMETERS_HPP_
