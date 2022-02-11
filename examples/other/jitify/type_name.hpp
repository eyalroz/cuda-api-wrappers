#ifndef CUDA_API_WRAPPERS_TYPE_NAME_HPP
#define CUDA_API_WRAPPERS_TYPE_NAME_HPP

#if __cplusplus >= 201703L
#include <string_view>
using std::string_view;
#define CONSTEXPR_SINCE_2014 constexpr
#else
#include "string_view.hpp"
using nonstd::string_view;
#define CONSTEXPR_SINCE_2014 nssv_constexpr14
#endif

// Note: Since some functionality here can only be constexpr if C++14 is used,
// we use the nssv_constexpr14 macro

template <typename T> CONSTEXPR_SINCE_2014 string_view type_name();

template <>
CONSTEXPR_SINCE_2014 string_view type_name<void>()
{ return "void"; }

namespace detail_ {

using type_name_prober = void;

template <typename T>
constexpr string_view wrapped_type_name()
{
#ifdef __clang__
	return __PRETTY_FUNCTION__;
#elif defined(__GNUC__)
	return __PRETTY_FUNCTION__;
#elif defined(_MSC_VER)
	return __FUNCSIG__;
#else
#error "Unsupported compiler"
#endif
}

CONSTEXPR_SINCE_2014 std::size_t wrapped_type_name_prefix_length() {
	return wrapped_type_name<type_name_prober>().find(type_name<type_name_prober>());
}

CONSTEXPR_SINCE_2014 std::size_t wrapped_type_name_suffix_length() {
	return wrapped_type_name<type_name_prober>().length()
		- wrapped_type_name_prefix_length()
		- type_name<type_name_prober>().length();
}

} // namespace detail

template <typename T>
CONSTEXPR_SINCE_2014 string_view type_name() {
	constexpr auto wrapped_name = detail_::wrapped_type_name<T>();
	CONSTEXPR_SINCE_2014 auto prefix_length = detail_::wrapped_type_name_prefix_length();
	CONSTEXPR_SINCE_2014 auto suffix_length = detail_::wrapped_type_name_suffix_length();
	CONSTEXPR_SINCE_2014 auto type_name_length = wrapped_name.length() - prefix_length - suffix_length;
	return wrapped_name.substr(prefix_length, type_name_length);
}

#endif //CUDA_API_WRAPPERS_TYPE_NAME_HPP
