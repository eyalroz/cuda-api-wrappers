#ifndef CUDA_API_WRAPPERS_SAFE_NARROW_CAST_HPP
#define CUDA_API_WRAPPERS_SAFE_NARROW_CAST_HPP

namespace cuda_ {
namespace detail {

template <typename Target, typename Source,
	typename = typename std::enable_if<std::is_arithmetic<Target>::value and std::is_arithmetic<Source>::value>::type>
Target safe_arithmetic_cast(Source const& x)
{
	auto x_as_target = static_cast<Target>(x);
	if (static_cast<Source>(x_as_target) != x) {
		throw std::runtime_error("narrowing cast alters value");
	}
	return x_as_target;
}

} // namespace detail
} // namespace cuda_

#endif //CUDA_API_WRAPPERS_SAFE_NARROW_CAST_HPP
