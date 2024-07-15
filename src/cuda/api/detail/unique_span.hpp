/**
 * @file
 *
 * @brief Contains the class @ref cuda::unique_span
 *
 * @note There is no CUDA-specific code in this file; the class is usable entirely independently
 * of the CUDA APIs and GPUs in general
 *
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_UNIQUE_SPAN_HPP_
#define CUDA_API_WRAPPERS_UNIQUE_SPAN_HPP_

#include "span.hpp"
#include "region.hpp"

#include <type_traits>
#include <memory>

namespace cuda {

/**
 * An `std::dynarry`-inspired class: Contiguous storage; size always equal to the capacity, and both
 * set at construction time; and dynamic storage, allocated _separately_ from this class itself.
 *
 * @note in owning standard-library containers, allocation is tied up with the container class itself,
 * via an Allocator template parameter. This class forgoes that "pleasure" - which is more feasible
 * considering how re-allocation is never necessary - and simply takes the allocated memory on
 * construction.
 *
 * @note unique_span is similar to the `dynarray` container, which was proposed for, but not finally
 * included in, C++14. It can be though of as a variation on std::array, with the the size and capacity
 * set dynamically, at construction time, rather than statically.
 *
 * @note unique_span = unique_span+typing or span+ownership+non_null
 *
 * @tparam T the type of individual elements in the unique_span
 */
template<typename T, typename Deleter = ::std::default_delete<T[]>>
class unique_span : public ::cuda::span<T> {
public: // span types
	using span_type = span<T>;

	// Exposing some span type definitions, strictly for terseness
	// (they're all visible on the outside anyway)
	using size_type = typename span<T>::size_type;
	using pointer = typename span<T>::pointer;
	using reference = typename span<T>::reference;
	using deleter_type = Deleter;

public: // exposing span data members
	using span<T>::data;
	using span<T>::size;

public: // constructors and destructor

	constexpr unique_span() noexcept = default;

	/// Take ownership of an existing region or span
	///
	/// These ctors are all explicit to prevent accidentally relinquishing ownership
	/// when passing to a function.
	///@{
	explicit unique_span(span_type span) noexcept : span_type{span} { }
	explicit unique_span(pointer data, size_type size) noexcept : unique_span{span_type{data, size}} { }
	explicit unique_span(memory::region_t region) noexcept : span_type{region.as_span<T>()} { }
	///@}

	// Note: No constructor which also takes a deleter. We do not hold a deleter
	// member - unlike unique_ptr's. If we wanted a general-purpose unique region
	// that's not just GPU allocation-oriented, we might have had one of those.

	/** A move constructor.
	 *
	 * @note Moving is the only way a unique_span may have its data_ field become null;
	 * the user is strongly assumed not to use the unique_span after moving from it.
	 */
	unique_span(unique_span&& other) noexcept : unique_span{ other.release() } { }
	// Disable copy construction - we do not allocate
	unique_span(const unique_span&) = delete;

	// Note: No conversion from "another type" like with ::std::unique_pointer, since
	// this class is not variant with the element type; and there's not much sense in
	// supporting conversion of memory between different deleters (/ allocators).

	~unique_span() noexcept
	{
		if (data() != nullptr) {
			deleter_type{}(data());
		}
#ifndef NDEBUG
		static_cast<span_type&>(*this) = span_type{static_cast<T*>(nullptr), 0};
#endif
	}

public: // operators

	/// No copy-assignment - that would break our ownership guarantee
	unique_span& operator=(const unique_span&) = delete;

	/// A Move-assignment operator, which takes ownership of the other region
	unique_span& operator=(unique_span&& other) noexcept
	{
		span_type released = other.release();
		if (data() != nullptr) {
			deleter_type{}(data());
		}
		static_cast<span_type&>(*this) = released;
		return *this;
	}

	/// No plain dereferencing - as there is no guarantee that any object has been
	/// initialized at those locations, nor do we know its type

	constexpr operator memory::const_region_t() const noexcept { return { data(), size() * sizeof(T) }; }

	template<typename = typename ::std::enable_if<! ::std::is_const<T>::value>::type>
	constexpr operator memory::region_t() const noexcept { return { data(), size() * sizeof(T) }; }

	constexpr span_type get() const noexcept { return { data(), size() }; }

	/// Exchange the pointer and deleter with another object.
	void swap(unique_span& other) noexcept
	{
		::std::swap<span_type >(*this, other);
	}

protected: // mutators
	/// Release ownership of any stored pointer.
	span_type release() noexcept
	{
		span_type released { data(), size() };
		static_cast<span_type &>(*this) = span_type{static_cast<T*>(nullptr), 0};
		return released;
	}
}; // class unique_span

/**
 * A parallel of ::std::make_unique_for_overwrite, for @ref unique_span<T>'s, i.e. which maintains
 * the number of elements allocated.
 *
 * @tparam T the type of elements in the allocated @ref unique_span.
 *
 * @param size The number of @tparam T elements to allocate
 */
template <typename T>
unique_span<T> make_unique_span(size_t size)
{
	return unique_span<T>{ new T[size], size };
}

} // namespace cuda

#endif // CUDA_API_WRAPPERS_UNIQUE_SPAN_HPP_
