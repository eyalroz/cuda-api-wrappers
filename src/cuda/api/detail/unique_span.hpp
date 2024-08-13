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

	// Disable copy construction - as this class never allocates;
	unique_span(const unique_span&) = delete;
	// ... and also match other kinds of unique_span's, which may get converted into
	// a span and thus leak memory on construction!
	template<typename U, typename UDeleter>
	unique_span(const unique_span<U, UDeleter>&) = delete;

	// Note: This template provides constructibility of unique_span<const T> from unique_span<const T>
	template<typename U, typename UDeleter>
	unique_span(unique_span<U,UDeleter>&& other)
		: unique_span{ other.release() }
	{
		static_assert(
			::std::is_assignable<span_type, span<U>>::value and
			::std::is_assignable<Deleter, UDeleter>::value,
			"Invalid unique_span initializer");
	}

	/// Take ownership of an existing span
	///
	/// @note These ctors are all explicit to prevent accidentally assuming ownership
	/// of a non-owned span when passing to a function, then trying to release that
	/// memory returning from it.
	///@{
	explicit unique_span(span_type span) noexcept : span_type{span} { }
	explicit unique_span(pointer data, size_type size) noexcept : unique_span{span_type{data, size}} { }
	///@}

	// Note: No constructor which also takes a deleter. We do not hold a deleter
	// member - unlike unique_ptr's. Perhaps we should?

	/** A move constructor.
	 *
	 * @note Moving is the only way a unique_span may have its @ref data_ field become
	 * null; the user is strongly assumed not to use the `unique_span` after moving from
	 * it.
	 */
	unique_span(unique_span&& other) noexcept : unique_span{ other.release() } { }

	~unique_span() noexcept
	{
		if (data() != nullptr) {
			deleter_type{}(data());
		}
#ifndef NDEBUG
		span_type::operator=(span_type{static_cast<T*>(nullptr), 0});
#endif
	}

public: // operators

	/// No copy-assignment - that would break our ownership guarantee
	unique_span& operator=(const unique_span&) = delete;

	/// A Move-assignment operator, which takes ownership of the other region
	unique_span& operator=(unique_span&& other) noexcept
	{
		swap(other);
		return *this;
		// other will be destructed, and our previous pointer - released if necessary
	}

	/// No plain dereferencing - as there is no guarantee that any object has been
	/// initialized at those locations, nor do we know its type

	constexpr operator memory::const_region_t() const noexcept { return { data(), size() * sizeof(T) }; }

	template<typename = typename ::std::enable_if<! ::std::is_const<T>::value>::type>
	constexpr operator memory::region_t() const noexcept { return { data(), size() * sizeof(T) }; }

public: // non-mutators
	constexpr span_type get() const noexcept { return { data(), size() }; }

protected: // mutators
	/// Exchange the pointer and deleter with another object.
	void swap(unique_span& other) noexcept
	{
		::std::swap<span_type>(*this, other);
	}
	/**
	 * Release ownership of the stored span
	 *
	 * @note This is not marked nodiscard by the same argument as for std::unique_ptr;
	 * see also @url https://stackoverflow.com/q/60535399/1593077 and
	 * @url http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0600r1.pdf
	 */
	span_type release() noexcept
	{
		span_type released { data(), size() };
		span_type::operator=(span_type{ static_cast<T*>(nullptr), 0 });
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
