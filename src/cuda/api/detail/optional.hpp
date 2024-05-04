/**
 * @file
 *
 * @brief An implementation or an importation of a @ref cuda::optional class and related definitions.
 *
 */
#ifndef CUDA_API_WRAPPERS_OPTIONAL_HPP
#define CUDA_API_WRAPPERS_OPTIONAL_HPP

#include <type_traits>

#if __cplusplus >= 201703L
#include <optional>
#include <any>
namespace cuda {
using ::std::optional;
using ::std::nullopt_t;
using ::std::nullopt;
} // namespace cuda
#elif __cplusplus >= 201402L
#include <experimental/optional>
#include <experimental/any>
namespace cuda {
using ::std::experimental::optional;
using ::std::experimental::nullopt;
using ::std::experimental::nullopt_t;
} // namespace cuda
#else

#include <type_traits>
#include <utility>

namespace cuda {


namespace detail_ {

struct no_value_t { };

} // namespace detail_

using nullopt_t = detail_::no_value_t;
constexpr nullopt_t nullopt{};

namespace detail_ {

template<typename T>
struct poor_mans_optional {
	static_assert(::std::is_trivially_destructible<T>::value, "Use a simpler type");
	union maybe_value_union_t {
		no_value_t no_value;
		T value;
	};

	poor_mans_optional &operator=(const poor_mans_optional &other) noexcept = default;

	poor_mans_optional &operator=(poor_mans_optional &&other) noexcept = default;

	poor_mans_optional &operator=(const T &value) noexcept(::std::is_nothrow_assignable<T,T>::value)
	{
		has_value_ = true;
		maybe_value.value = value;
		return *this;
	}

	poor_mans_optional &operator=(const T &&value) noexcept(::std::is_nothrow_move_assignable<T>::value)
	{
		has_value_ = true;
		maybe_value.value = ::std::move(value);
		return *this;
	}

	poor_mans_optional &operator=(no_value_t)
	{
		has_value_ = false;
		return *this;
	}

	poor_mans_optional &operator=(T &&value) noexcept(::std::is_nothrow_move_assignable<T>::value)
	{ return *this = value; }

	poor_mans_optional() noexcept: has_value_(false)
	{}

	poor_mans_optional(T v) noexcept(::std::is_nothrow_assignable<T,T>::value) : has_value_(true)
	{
		maybe_value.value = v;
	}

	poor_mans_optional(const poor_mans_optional &other) noexcept
	: has_value_(other.has_value_), maybe_value(other.maybe_value)
	{ }

	poor_mans_optional(poor_mans_optional &&other) noexcept
	: has_value_(other.has_value_), maybe_value(other.maybe_value)
	{
		other.has_value_ = false;
	}

	poor_mans_optional(nullopt_t) noexcept : has_value_(false) { }

	~poor_mans_optional() noexcept = default;

	T value() const
	{ return maybe_value.value; }

	template<typename U>
	T value_or(U&& fallback_value)
	{
		has_value_ ? maybe_value.value : static_cast<T>(::std::forward<U>(fallback_value));
	}

	T& operator*() noexcept { return maybe_value.value; }
	const T& operator*() const noexcept { return maybe_value.value; }

	operator bool() const noexcept
	{ return has_value_; }

	bool has_value() const noexcept
	{ return has_value_; }

	void reset() noexcept
	{ has_value_ = false; }

protected:
	bool has_value_;
	maybe_value_union_t maybe_value { no_value_t{} };
};

} // namespace detail_

template<typename T>
using optional = cuda::detail_::poor_mans_optional<T>;

} // namespace cuda

#endif // __cplusplus >= 201402L

#endif //CUDA_API_WRAPPERS_OPTIONAL_HPP
