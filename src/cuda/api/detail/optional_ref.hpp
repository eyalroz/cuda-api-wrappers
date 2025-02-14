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

namespace cuda {

namespace detail_ {

template<typename T>
struct optional_ref {
	optional_ref &operator=(const optional_ref &other) = default;

	optional_ref &operator=(optional_ref &&other) = default;

	optional_ref &operator=(const T &value) = delete;

	optional_ref &operator=(const T &&value) = delete;

	optional_ref() noexcept: ptr_(nullptr)
	{ }

	optional_ref(T& v) noexcept : ptr_(&v) 
	{ }

	optional_ref(const optional_ref &other) = default;

	optional_ref(nullopt_t) noexcept : ptr_(nullptr) { }

	~optional_ref() noexcept = default;

	T& value() const
	{ return *ptr_; }

	T& value_or(T& fallback_ref) const
	{
		return has_value() ? value() : fallback_ref;
	}

	T& operator*() noexcept { return *ptr_; }
	const T& operator*() const noexcept { return *ptr_; }
	T* operator->() noexcept { return ptr_; }
	const T* operator->() const noexcept { return ptr_; }

	bool has_value() const noexcept
	{ return ptr_ != nullptr; }

	operator bool() const noexcept
	{ return has_value(); }

	void reset() noexcept
	{ ptr_ = nullptr; }

protected:
	T* ptr_;
};

} // namespace detail_

template<typename T>
using optional_ref = cuda::detail_::optional_ref<T>;

} // namespace cuda

#endif //CUDA_API_WRAPPERS_OPTIONAL_REF_HPP_
