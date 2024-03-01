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

#include <type_traits>
#include <stdexcept>

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
	base_region_t() noexcept = default;
	base_region_t(pointer start, size_type size_in_bytes) noexcept
		: start_(start), size_in_bytes_(size_in_bytes) {}

	template <typename U>
	base_region_t(span<U> span) noexcept : start_(span.data()), size_in_bytes_(span.size() * sizeof(U))
	{
		static_assert(::std::is_const<T>::value or not ::std::is_const<U>::value,
			"Attempt to construct a non-const memory region from a const span");
	}

	template <typename U>
	span<U> as_span() const NOEXCEPT_IF_NDEBUG
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
	operator span<U>() const NOEXCEPT_IF_NDEBUG { return as_span<U>(); }

	pointer& start() noexcept { return start_; }
	size_type& size() noexcept { return size_in_bytes_; }

	size_type size() const noexcept { return size_in_bytes_; }
	pointer start() const noexcept { return start_; }
	pointer data() const noexcept { return start(); }
	pointer get() const noexcept { return start(); }

protected:
	base_region_t subregion(size_type offset_in_bytes, size_type size_in_bytes) const
#ifdef NDEBUG
		noexcept
#endif
	{
#ifndef NDEBUG
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
bool operator==(const base_region_t<T>& lhs, const base_region_t<T>& rhs)
{
	return lhs.start() == rhs.start()
		and lhs.size() == rhs.size();
}

template <typename T>
bool operator!=(const base_region_t<T>& lhs, const base_region_t<T>& rhs)
{
	return not (lhs == rhs);
}

}  // namespace detail_

/**
 * An untyped, but sized, region in some memory space
 */
struct region_t : public detail_::base_region_t<void> {
	using base_region_t<void>::base_region_t;
	region_t subregion(size_t offset_in_bytes, size_t size_in_bytes) const
	{
		auto parent_class_subregion = base_region_t<void>::subregion(offset_in_bytes, size_in_bytes);
		return { parent_class_subregion.data(), parent_class_subregion.size() };
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
		auto parent_class_subregion = base_region_t<void const>::subregion(offset_in_bytes, size_in_bytes);
		return { parent_class_subregion.data(), parent_class_subregion.size() };
	}
};

} // namespace memory

} // namespace cuda

#endif // CUDA_API_WRAPPERS_REGION_HPP_
