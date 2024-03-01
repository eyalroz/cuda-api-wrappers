/**
 * @file
 *
 * @brief Contains an implementation of an std::span-like class, @ref cuda::span
 *
 * @note When compiling with C++20 or later, the actual std::span is used instead
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_SPAN_HPP_
#define CUDA_API_WRAPPERS_SPAN_HPP_

#if __cplusplus >= 202002L
#include <span>
#else
#include <type_traits>
#include <cstdlib>
#endif

/**
 * @brief All definitions and functionality wrapping the CUDA Runtime API.
 */
namespace cuda {

#if __cplusplus >= 202002L
using ::std::span;
#else
/**
 * @brief A "poor man's" span class
 *
 * @todo: Replace this with a more proper implementation.
 *
 * @note Remember a span is a reference type. That means that changes to the
 * pointed-to data are _not_considered changes to the span, hence you can get
 * to that data with const methods.
 */
template<typename T>
struct span {
	using value_type = T;
	using element_type = T;
	using size_type = ::std::size_t;
	using difference_type = ::std::ptrdiff_t;
	using pointer = T*;
	using const_pointer = T const *;
	using reference = T&;
	using const_reference = const T&;

	pointer data_;
	size_type size_;

	pointer       data() const noexcept { return data_; }
	size_type     size() const noexcept { return size_; }

	// About cbegin() and cend() for spans, see:
	// https://stackoverflow.com/q/62757700/1593077
	// (for which reason, they're not implemented here)
	pointer       begin()  const noexcept { return data(); }
	pointer       end()    const noexcept { return data() + size_; }

	reference operator[](size_type idx) const noexcept { return data_[idx]; }

	// Allows a non-const-element span to be used as its const-element equivalent. With pointers,
	// we get a T* to const T* casting for free, but the span has to take care of this for itself.
	template<
		typename U = value_type,
		typename = typename ::std::enable_if<! ::std::is_const<U>::value>::type
	>
	operator span<const U>()
	{
		static_assert(::std::is_same<U,T>::value, "Invalid type specified");
		return { data_, size_ };
	}
};
#endif

} // namespace cuda

#endif // CUDA_API_WRAPPERS_SPAN_HPP_
