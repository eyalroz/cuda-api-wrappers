/**
 * @file
 *
 * @brief An implementation or an importation of a @ref cuda::optional class and related definitions.
 *
 */
#ifndef CUDA_API_WRAPPERS_OPTIONAL_HPP
#define CUDA_API_WRAPPERS_OPTIONAL_HPP

#if __cplusplus >= 201703L
#include <optional>
#include <any>
namespace cuda {
using ::std::optional;
using ::std::nullopt;
} // namespace cuda
#elif __cplusplus >= 201402L
namespace cuda {
#include <experimental/optional>
#include <experimental/any>
using ::std::experimental::optional;
using ::std::experimental::nullopt;
} // namespace cuda
#else

#include <type_traits>
#include <utility>

namespace cuda {

namespace detail_ {

struct no_value_t { };

template<typename T>
struct poor_mans_optional {
	static_assert(::std::is_trivially_destructible<T>::value, "Use a simpler type");
	union maybe_value_union_t {
		no_value_t no_value;
		T value;
	};

	poor_mans_optional &operator=(const poor_mans_optional &other) = default;

	poor_mans_optional &operator=(poor_mans_optional &&other) = default;

	poor_mans_optional &operator=(const T &value)
	{
		has_value_ = true;
		maybe_value.value = value;
		return *this;
	}

	poor_mans_optional &operator=(const T &&value)
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

	poor_mans_optional &operator=(T &&value)
	{ return *this = value; }

	poor_mans_optional() noexcept: has_value_(false), maybe_value{no_value_t{}}
	{}

	poor_mans_optional(T v) : has_value_(true)
	{
		maybe_value.value = v;
	}

	poor_mans_optional(const poor_mans_optional &other)
	{
		if (other) {
			*this = other.value();
		}
		has_value_ = other.has_value_;
	}

	~poor_mans_optional() noexcept
	{};

	T value() const
	{ return maybe_value.value; }

	operator bool() const noexcept
	{ return has_value_; }

	bool has_value() const noexcept
	{ return has_value_; }

	void clear() noexcept
	{ has_value_ = false; }

	void unset() noexcept
	{ has_value_ = false; }

protected:
	bool has_value_{false};
	maybe_value_union_t maybe_value;
};

} // namespace detail_

template<typename T>
using optional = cuda::detail_::poor_mans_optional<T>;
using nullopt = detail_::no_value_t;

} // namespace cuda

#endif // __cplusplus >= 201402L

#endif //CUDA_API_WRAPPERS_OPTIONAL_HPP
