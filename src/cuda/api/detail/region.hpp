/**
 * @file
 *
 * @brief A memory region class (@ref cuda::memory::region_t and @ref
 * cuda::memory::const_region_t) and related functionality.
 *
 * @note There is no CUDA-specific functionality here, and this class could be
 * used irrespective of the CUDA APIs and GPUs in general.
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_REGION_HPP_
#define CUDA_API_WRAPPERS_REGION_HPP_

#include "type_traits.hpp"
#include <stdexcept>

#ifndef CPP14_CONSTEXPR
#if __cplusplus >= 201402L
#define CPP14_CONSTEXPR constexpr
#else
#define CPP14_CONSTEXPR
#endif
#endif

#ifndef NOEXCEPT_IF_NDEBUG
#ifdef NDEBUG
#define NOEXCEPT_IF_NDEBUG noexcept(true)
#else
#define NOEXCEPT_IF_NDEBUG noexcept(false)
#endif
#endif // NOEXCEPT_IF_NDEBUG


namespace cuda {

namespace memory {

namespace detail_ {

// Note: T should be either void or void const, nothing else
template <class T>
class base_region_t {
public: // types
	using pointer = T*;
	using const_pointer = const T*;
	using size_type = size_t;
	// No pointer difference type, as the pointers here are void-typed and
	// there's no sense in subtracting them except internally
	// difference_type = std::ptrdiff_t

private:
	T* start_ = nullptr;
	size_type size_in_bytes_ = 0;

	// If we were using C++17 or later, we could forget about this and use `std::byte`
	using char_type = typename ::std::conditional<::std::is_const<T>::value, const char *, char *>::type;
public:
	constexpr base_region_t() noexcept = default;
	constexpr base_region_t(pointer start, size_type size_in_bytes) noexcept
		: start_(start), size_in_bytes_(size_in_bytes) {}

	template <typename E, size_t N>
	constexpr base_region_t(E (&arr)[N]) noexcept
		: start_(arr), size_in_bytes_(N * sizeof(E)) {}

	/**
	 * A constructor from types such as `::std::span`'s or `::std::vector`'s, whose data is in
	 * a contiguous region of memory
	 */
	template <typename ContiguousContainer, typename = cuda::detail_::enable_if_t<
		cuda::detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, void>>
	constexpr base_region_t(ContiguousContainer&& contiguous_container) noexcept
	: start_(contiguous_container.data()), size_in_bytes_(contiguous_container.size() * sizeof(*(contiguous_container.data())))
	{
		static_assert(::std::is_const<T>::value or not ::std::is_const<decltype(*(contiguous_container.data()))>::value,
			"Attempt to construct a non-const memory region from a container of const data");
	}

	template <typename U>
	CPP14_CONSTEXPR span<U> as_span() const NOEXCEPT_IF_NDEBUG
	{
		static_assert(
			::std::is_const<U>::value or not ::std::is_const<typename ::std::remove_pointer<T>::type>::value,
			"Attempt to create a non-const span referencing a const memory region");
#ifndef NDEBUG
		if (size() == 0) {
			throw ::std::logic_error("Attempt to use a span of size 0 as a sequence of typed elements");
		}
		if (size() % sizeof(U) != 0) {
			throw ::std::logic_error("Attempt to use a region of size not an integral multiple of the size of a type, "
				"as a span of elements of that type");
		}
#endif
		return span<U> { static_cast<U*>(data()), size() / sizeof(U) };
	}

	template <typename U>
	CPP14_CONSTEXPR operator span<U>() const NOEXCEPT_IF_NDEBUG { return as_span<U>(); }

	constexpr pointer start() const noexcept { return start_; }
	constexpr size_type size() const noexcept { return size_in_bytes_; }
	constexpr pointer data() const noexcept { return start(); }
	constexpr pointer get() const noexcept { return start(); }

protected:
	constexpr base_region_t subregion(size_type offset_in_bytes, size_type size_in_bytes) const
#ifdef NDEBUG
		noexcept
#endif
	{
#if ! defined(NDEBUG) && __cplusplus >= 201402L
		if (offset_in_bytes >= size_in_bytes_) {
			throw ::std::invalid_argument("subregion begins past region end");
		}
		else if (offset_in_bytes + size_in_bytes > size_in_bytes_) {
			throw ::std::invalid_argument("subregion exceeds original region bounds");
		}
#endif
		return { static_cast<char_type>(start_) + offset_in_bytes, size_in_bytes };
	}
};

template <typename T>
constexpr bool operator==(const base_region_t<T>& lhs, const base_region_t<T>& rhs)
{
	return lhs.start() == rhs.start()
		and lhs.size() == rhs.size();
}

template <typename T>
constexpr bool operator!=(const base_region_t<T>& lhs, const base_region_t<T>& rhs)
{
	return not (lhs == rhs);
}

}  // namespace detail_

/**
 * An untyped, but sized, region in some memory space
 */
struct region_t : public detail_::base_region_t<void> {
	using base_region_t<void>::base_region_t;
	constexpr region_t subregion(size_t offset_in_bytes, size_t size_in_bytes) const
	{
		return { base_region_t<void>::subregion(offset_in_bytes, size_in_bytes).data(), size_in_bytes };
	}
};

/**
 * An untyped, but sized, region with const-constrained data in some memory space
 */
struct const_region_t : public detail_::base_region_t<void const> {
	using base_region_t<void const>::base_region_t;
	const_region_t(region_t r) : base_region_t(r.start(), r.size()) {}
	const_region_t subregion(size_t offset_in_bytes, size_t size_in_bytes) const
	{
		return {
			base_region_t<void const>::subregion(offset_in_bytes, size_in_bytes).data(),
			size_in_bytes
		};
	}
};

} // namespace memory

} // namespace cuda

#endif // CUDA_API_WRAPPERS_REGION_HPP_
