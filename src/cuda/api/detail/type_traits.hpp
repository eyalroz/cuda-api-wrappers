/**
 * @file
 *
 * @brief Non-CUDA-specific type-related utility code, especially type traits.
 *
 * @note prefer to include `types.hpp` if you're actually doing any CUDA work.
 */
#ifndef CUDA_API_WRAPPERS_TYPE_TRAITS_HPP
#define CUDA_API_WRAPPERS_TYPE_TRAITS_HPP

#include <type_traits>

#ifdef _MSC_VER
/*
 * Microsoft Visual C++ (upto v2017) does not support the C++
 * keywords `and`, `or` and `not`. Apparently, the following
 * include is a work-around.
 */
#include <ciso646>
#endif

#ifndef CPP14_CONSTEXPR
#if __cplusplus >= 201402L
#define CPP14_CONSTEXPR constexpr
#else
#define CPP14_CONSTEXPR
#endif
#endif

#ifndef CAW_MAYBE_UNUSED
#if __cplusplus >= 201703L
#define CAW_MAYBE_UNUSED [[maybe_unused]]
#else
#if __GNUC__
#define CAW_MAYBE_UNUSED __attribute__((unused))
#else
#define CAW_MAYBE_UNUSED
#endif // __GNUC__
#endif // __cplusplus >= 201703L
#endif // ifndef CAW_MAYBE_UNUSED

#ifndef NOEXCEPT_IF_NDEBUG
#ifdef NDEBUG
#define NOEXCEPT_IF_NDEBUG noexcept(true)
#else
#define NOEXCEPT_IF_NDEBUG noexcept(false)
#endif
#endif // NOEXCEPT_IF_NDEBUG

namespace cuda {

namespace detail_ {

template <bool B>
using bool_constant = ::std::integral_constant<bool, B>;

using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template<bool...> struct bool_pack;

template<bool... bs>
using all_true = ::std::is_same<bool_pack<bs..., true>, bool_pack<true, bs...>>;

// This is available in C++17 as ::std::void_t, but we're only assuming C++11
template<typename...>
using void_t = void;

// This is available in C++14
template<bool B, typename T = void>
using enable_if_t = typename ::std::enable_if<B, T>::type;

template<typename T>
using remove_reference_t = typename ::std::remove_reference<T>::type;

// primary template handles types that have no nested ::type member:
template <typename, typename = void>
struct has_data_method : ::std::false_type { };

// specialization recognizes types that do have a nested ::type member:
template <typename T>
struct has_data_method<T, cuda::detail_::void_t<decltype(::std::declval<T>().data())>> : ::std::true_type { };

template <typename, typename = void>
struct has_value_type_member : ::std::false_type { };

template <typename T>
struct has_value_type_member<T, cuda::detail_::void_t<typename T::value_type>> : ::std::true_type { };

// TODO: Consider either beefing up this type trait or ditching it in favor of something simpler, or
// in the standard library
template <typename T>
struct is_kinda_like_contiguous_container :
	::std::integral_constant<bool,
		has_data_method<typename ::std::remove_reference<T>::type>::value
		and has_value_type_member<typename ::std::remove_reference<T>::type>::value
	> {};

} // namespace detail_

} // namespace cuda


#endif //CUDA_API_WRAPPERS_TYPE_TRAITS_HPP
