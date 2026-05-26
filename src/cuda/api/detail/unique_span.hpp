/**
 * @file
 *
 * @brief Contains the class @ref cuda_::unique_span
 *
 * @note There is no CUDA-specific code in this file; the class is usable
 * entirely independently of the CUDA APIs and GPUs in general. The only
 * mention of CUDA is the namespace.
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_UNIQUE_SPAN_HPP_
#define CUDA_API_WRAPPERS_UNIQUE_SPAN_HPP_

#include "span.hpp"
#include "region.hpp"

#include <type_traits>
#include <memory>

namespace cuda_ {

/**
 * An `std::dynarry`-inspired class, but un-copyable: Contiguous storage; size always equal to the
 * capacity, and both set at construction time; and dynamic storage, allocated _separately_ from
 * this class itself.
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
 * @note unique_span = span + unique ownership + non_null . Well, sort of, because this
 * class supports complex construction-allocation and deletion patterns, through deleter objects.
 *
 * @tparam T an individual element in the unique_span
 */
template<typename T>
class unique_span : public span<T> {
public: // span types
	using span_type = span<T>;

	// Exposing some span type definitions, strictly for terseness
	// (they're all visible on the outside anyway)
	using size_type = typename span_type::size_type;
	using pointer = typename span_type::pointer;
	using reference = typename span_type::reference;
	using deleter_type = void (*)(span_type);

public: // exposing span data members & adding our own
	using span_type::data;
	using span_type::size;
	deleter_type deleter_;

public: // constructors and destructor

	// Note: span_type's default ctor will create a {nullptr, 0} empty span.
	constexpr unique_span() noexcept : span_type(), deleter_{nullptr} {}

	// Disable copy construction - as this class never allocates;
	unique_span(const unique_span&) = delete;
	// ... and also match other kinds of unique_span's, which may get converted into
	// a span and thus leak memory on construction!
	template<typename U>
	unique_span(const unique_span<U>&) = delete;

	// Note: This template provides constructibility of unique_span<const T> from unique_span<const T>
	template<typename U>
	unique_span(unique_span<U>&& other) noexcept : unique_span{ other.release(), other.deleter_ }
	{
		static_assert(
			std::is_assignable<span_type, span<U>>::value,
			"Invalid unique_span initializer");
	}

	/// Take ownership of an existing span
	///
	/// @note These ctors are all explicit to prevent accidentally assuming ownership
	/// of a non-owned span when passing to a function, then trying to release that
	/// memory returning from it.
	///@{
	explicit unique_span(span_type span, deleter_type deleter) noexcept
	: span_type{span}, deleter_(deleter) { }
	explicit unique_span(pointer data, size_type size, deleter_type deleter) noexcept
	: unique_span(span_type{data, size}, deleter) { }
	explicit unique_span(memory::region_t region, deleter_type deleter) NOEXCEPT_IF_NDEBUG
		: unique_span(span_type{region.start(), region.size() / sizeof(T)}, deleter)
	{
#ifndef NDEBUG
		if (sizeof(T) * size != region.size()) {
			throw std::invalid_argument("Attempt to create a unique_span with a memory region which"
				"does not comprise an integral number of areas of the element type size");
		}
#endif
	}

	///@}


	/// A move constructor.
	///
	/// @TODO Can we drop this one in favor of the general move ctor?
	unique_span(unique_span&& other) noexcept : unique_span(other.release(), other.deleter_) { }

	~unique_span()
	{
		if (data() != nullptr) {
			deleter_(*this);
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
		swap(*this, other);
		return *this;
		// other will be destructed, and our previous pointer - released if necessary
	}

	/// No plain dereferencing - as there is no guarantee that any object has been
	/// initialized at those locations, nor do we know its type

	// Note: No need for a conversion operator into region_t and const_region_t, as such
	// conversions are taken care of by the constructors of region_t and const_region_t
	// from arbitrary contiguous containers

public: // non-mutators
	constexpr span_type get() const noexcept { return { data(), size() }; }

public: // friends
	/// Exchange the members of two unique_span's
	friend void swap(unique_span& a, unique_span& b) noexcept
	{
		std::swap<span_type>(a, b);
		std::swap<deleter_type>(a.deleter_, b.deleter_);
	}

protected: // mutators
	/**
	 * Release ownership of the stored span
	 *
	 * @note This is not marked nodiscard by the same argument as for std::unique_ptr;
	 * see also @url https://stackoverflow.com/q/60535399/1593077 and
	 * @url http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0600r1.pdf
	 *
	 * @note It is the caller's responsibility to ensure it has a copy of the deleter
	 * for the released span. It is also the caller's responsibility to free the
	 * allocated memory.
	 */
	span_type release() noexcept
	{
		span_type released { data(), size() };
		span_type::operator=(span_type{ static_cast<T*>(nullptr), 0 });
		// Note that we are _not_ replacing deleter.
		return released;
	}
}; // class unique_span

namespace detail_ {

// @note if a nullptr happens to be deleted - that's not a problem;
// it is supported both by C++ delete operators and C's free(), and
// the behavior is guaranteed to be no-op.
///@{
template <typename T> void operator_delete_array(span<T> sp) { delete[] sp.data(); }
template <typename T> void operator_delete      (span<T> sp) { ::operator delete(sp.data()); };
template <typename T> void c_free               (span<T> sp) { std::free(sp.data()); }
///@}

template <typename T>
void elementwise_destruct(span<T> sp)
{
	for (auto& element : sp) { element.~T(); }
}

template <typename T>
void elementwise_destruct_then_delete(span<T> sp)
{
	elementwise_destruct(sp);
	operator_delete(sp);
}


} // namespace detail_

/**
 * A parallel of std::make_unique_for_overwrite, for @ref unique_span<T>'s, i.e. which maintains
 * the number of elements allocated.
 *
 * @param size the number of elements in the unique_span to be created. It may legitimately be 0.
 *
 * @tparam T the type of elements in the allocated @ref unique_span.
 *
 * @param size The number of @tparam T elements to allocate
 */
template <typename T>
unique_span<T> make_unique_span(size_t size) noexcept(false)
{
	// Note: It is acceptable pass 0 as the size here.
	// See https://stackoverflow.com/q/1087042/1593077
	return unique_span<T>(new T[size], size, detail_::operator_delete_array<T>);
}

namespace detail_ {

/**
 * Construct the elements of a unique span in a pre-allocated storage area
 *
 * @param[inout] allocated_space
 *     data already allocated - but with no constructed objects - in which
 *     to store the unique span
 *
 * @note
 * Q: Why are we assuming non-array, untyped allocation of the space, rather
 *    than `new T[size]`?
 * A: That's because there is no guarantee T is default-constructible; and
 *    even if it were - we have a _generator_ function which does the
 *    construction, so we can't very well have the new operator do it for us.
 *    The function which _does_ use the array construction operator is
 *    @ref make_dynarray(size_t size)
 *
 */
template <typename T, typename Generator>
unique_span <T> generate_unique_span(
    T* allocated_space,
    size_t size,
    Generator generator_by_index) noexcept(noexcept(generator_by_index(0)))
{
    // Q: Do I need to check the alignment here? Perhaps allocate more to ensure alignment?
    for (size_t i = 0; i < size; i++) {
        new(&allocated_space[i]) T(generator_by_index(i));
    }
    return unique_span<T>(allocated_space, size, operator_delete<T>);
}

} // namespace detail_

/**
 * The alternative to `std::generate` and similar functions, for @ref unique_span, seeing
 * how its elements must be constructed as it is constructed.
 *
 * @param[in] size
 *     the number of elements in the unique_span to be created. It may legitimately be 0.
 * @param[in] generator_by_index
 *     a function for generating new values for move-construction into the new unique_span
 *
 * @tparam T the type of elements in the allocated @ref unique_span.
 * @tparam Generator A type invokable with the element index, to produce a T-constructor-argument
 *
 * @param size The number of @tparam T elements to allocate
 */
///@{
template <typename T, typename Generator>
unique_span <T> generate_unique_span(
	memory::region_t region,
	Generator&& generator_by_index) noexcept(noexcept(generator_by_index(0)))
{
	auto data_ = static_cast<T*>(region.data());
	auto size_in_elements = region.size() / sizeof(T);
	return detail_::generate_unique_span<T>(data_, size_in_elements,
		std::forward<Generator>(generator_by_index));
}

template <typename T, typename Generator>
unique_span<T> generate_unique_span(size_t size, Generator&& generator_by_index)  noexcept(false)
{
	// Q: Do I need to check the alignment here? Perhaps allocate more to ensure alignment?
	auto data = static_cast<T*>(::operator new(size * sizeof(T)));
	return detail_::generate_unique_span<T, Generator>(
		data, size, std::forward<Generator>(generator_by_index));
}

template <typename Generator>
unique_span<decltype(std::declval<Generator>()(0))>
generate_unique_span(size_t size, Generator&& generator_by_index) noexcept(false)
{
	using T = decltype(std::declval<Generator>()(0));
	return generate_unique_span<T, Generator>(size, std::forward<Generator>(generator_by_index));
}
///@}

} // namespace cuda_

#endif // CUDA_API_WRAPPERS_UNIQUE_SPAN_HPP_
