/**
* @file
 *
 * @brief preprocessor-focused definitions and compiler compatibility code,
 * to (preferably) be included before anything else in the library 
 */
#ifndef CUDA_API_WRAPPERS_PREAMBLE_HPP_
#define CUDA_API_WRAPPERS_PREAMBLE_HPP_

#if (__cplusplus < 201103L && (!defined(_MSVC_LANG) || _MSVC_LANG < 201103L))
#error "The CUDA API wrappers can only be compiled with C++11 or a later version of the C++ language standard"
#endif

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

#endif //CUDA_API_WRAPPERS_PREAMBLE_HPP_
