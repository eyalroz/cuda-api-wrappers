/**
 * @file
 *
 * @brief An implementation of a simplistic optional-reference class
 * (as optional<T&> is problematic semantically, and also probably not
 * supported by this library's simplistic @ref optional implementation)
 *
 */
#ifndef CUDA_API_WRAPPERS_OPTIONAL_REF_HPP_
#define CUDA_API_WRAPPERS_OPTIONAL_REF_HPP_

#include "optional.hpp"

namespace cuda_ {

/**
 * A non-owning reference type which also has a disengaged state, i.e.
 * the non-owning equivalent of a @ref optional .
 *
 * @tparam T the reference type, without the reference itself. This means
 * that if T is const, it's the equivalent of a `const&`. The const-ness of
 * an optional_ref value itself only regards whether one can set it to refer
 * to a different value.
 */
template<typename T>
class optional_ref {
public:
	using value_type = T;
	using reference = T&;
	using pointer = T*;

	constexpr optional_ref() noexcept: ptr_(nullptr) { }
	constexpr optional_ref(T& v) noexcept : ptr_(&v) { }
	constexpr optional_ref(optional_ref const& other) = default;
	constexpr optional_ref(optional_ref&& other) noexcept : ptr_(other.ptr_) { }

	template<typename> friend class optional_ref;

	template<typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
	constexpr optional_ref(optional_ref<U> const& other) noexcept : ptr_(other.ptr_) { }

	optional_ref(nullopt_t) noexcept : ptr_(nullptr) { }

	template<typename U, typename = typename std::enable_if<std::is_convertible<U*, T*>::value>::type>
	constexpr optional_ref(optional<U> const& other) noexcept : ptr_(other ? std::addressof(*other) : nullptr) { }

	// Unlike regular references, this type cannot extend the life of the value held by an rvalue optional; so we
	// offer no constructor of the form: optional_ref(optional<U> && other)

	CONSTEXPR_CPP20 ~optional_ref() noexcept = default;

	constexpr bool has_value() const noexcept { return ptr_ != nullptr; }
	void reset() noexcept { ptr_ = nullptr; }
	void rebind(T& v) noexcept { ptr_ = std::addressof(v); }
	// Note: No emplace() and no swap()

	// The following becomes non-constant-expression when there's no value
	CONSTEXPR_CPP14 T& value() const
	{
		if (has_value()) { return *ptr_; }
		throw bad_optional_access{};
	}
	CONSTEXPR_CPP14 T& value_or(T& fallback_ref) const noexcept { return has_value() ? *ptr_ : fallback_ref; }
	CONSTEXPR_CPP14 T& get() const { return value(); }

	// Assignment with optional-references creates ambiguity, since it's not obvious whether we assign
	// the referred-to-value, or make one optional refer to the same value as another optional. So,
	// instead of confusing - we preclude it altogether. You can use @ref rebind() instead
	optional_ref& operator=(optional_ref const& other) noexcept = delete;
	optional_ref& operator=(optional_ref&& other) noexcept = delete;
	optional_ref& operator=(T const& value) = delete;
	optional_ref& operator=(T&& value) = delete;

	CONSTEXPR_CPP14 T& operator*() { return value(); }
	CONSTEXPR_CPP14 const T& operator*() const { return value(); }
	CONSTEXPR_CPP14 T* operator->()
	{
		if (has_value()) { return ptr_; }
		throw bad_optional_access{};
	}
	CONSTEXPR_CPP14 T const* operator->() const
	{
		if (has_value()) { return ptr_; }
		throw bad_optional_access{};
	}
	explicit constexpr operator bool() const noexcept { return has_value(); }

	// TODO:
	// 1. Consider adding an iterator class and begin() and end() methods, like in C++26
	// 2. Consider adding and_then, transform and or_else monadic methods, like in C++23
	// 3. Should this class have a swap method? A standalone swap function?

protected:
	T* ptr_;
}; // class optional_ref

namespace detail {

// TODO: Is the enabling criterion here for U appropriate? Enough to forego
// ensuring that T and U are three-way comparable?
template <typename T, typename U, typename = typename std::enable_if<std::is_convertible<U*,T*>::value>> constexpr int
three_way_compare(optional_ref<T> const& lhs, U const& rhs )
{
	return (not lhs.has_value()) ? -1 :
		(*lhs < rhs) ? -1 :
		(*lhs > *rhs) ? 1 :	0;
}

template <typename T, typename U> constexpr int
three_way_compare(optional_ref<T> const& lhs, optional_ref<U> const& rhs )
{
	return rhs.has_value() ?
		three_way_compare(lhs, *rhs) :
		lhs.has_value() ? 1 : 0;
}

} // namespace detail

template <typename T, typename U> constexpr bool
operator==(optional_ref<T> const& lhs, U const& rhs) { return detail::three_way_compare(lhs, rhs) == 0; }
template <typename T, typename U> constexpr bool
operator!=(optional_ref<T> const& lhs, U const& rhs) { return detail::three_way_compare(lhs, rhs) != 0; }
template <typename T, typename U> constexpr bool
operator< (optional_ref<T> const& lhs, U const& rhs) { return detail::three_way_compare(lhs, rhs) <  0; }
template <typename T, typename U> constexpr bool
operator<=(optional_ref<T> const& lhs, U const& rhs) { return detail::three_way_compare(lhs, rhs) <= 0; }
template <typename T, typename U> constexpr bool
operator> (optional_ref<T> const& lhs, U const& rhs) { return detail::three_way_compare(lhs, rhs) >  0; }
template <typename T, typename U> constexpr bool
operator>=(optional_ref<T> const& lhs, U const& rhs) { return detail::three_way_compare(lhs, rhs) >= 0; }

#if __cplusplus < 202002L
// With C++20, the compiler _should_ try reversing the order of parameters for a comparison
template <typename T, typename U> constexpr bool operator==(T const& lhs, optional_ref<U> const& rhs) { return lhs == rhs; }
template <typename T, typename U> constexpr bool operator!=(T const& lhs, optional_ref<U> const& rhs) { return lhs != rhs; }
template <typename T, typename U> constexpr bool operator< (T const& lhs, optional_ref<U> const& rhs) { return lhs <  rhs; }
template <typename T, typename U> constexpr bool operator<=(T const& lhs, optional_ref<U> const& rhs) { return lhs <= rhs; }
template <typename T, typename U> constexpr bool operator> (T const& lhs, optional_ref<U> const& rhs) { return lhs >  rhs; }
template <typename T, typename U> constexpr bool operator>=(T const& lhs, optional_ref<U> const& rhs) { return lhs >= rhs; }
#endif // __cplusplus < 202002L

// TODO: Should we also define equality operators between an optional_ref and types which are comparable with T,
// outside of an optional?

#if __cplusplus >= 202002L
template <typename T, std::three_way_comparable_with<T> U>
constexpr std::compare_three_way_result_t<T, U>
operator<=>(optional_ref<T> const& lhs, optional_ref<U> const& rhs) noexcept
{
	auto engagement_comparison = (lhs.has_value() <=> rhs.has_value());
	if (engagement_comparison != 0) { return engagement_comparison; }
	return *lhs <=> *rhs;
}
template <typename T>
constexpr std::strong_ordering operator<=>(optional_ref<T> const& opt, std::nullopt_t) noexcept
{
	return opt ? std::strong_ordering::greater : std::strong_ordering::equal;
}

template <typename T>
constexpr std::strong_ordering operator<=>(std::nullopt_t, optional_ref<T> const& opt) noexcept
{
	return opt ? std::strong_ordering::less : std::strong_ordering::equal;
}
#endif //  __cplusplus >= 202002L

} // namespace cuda_

#endif //CUDA_API_WRAPPERS_OPTIONAL_REF_HPP_

