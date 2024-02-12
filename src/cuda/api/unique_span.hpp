/**
 * @file
 *
 * @brief Contains an implementation of an std::unique_span-like class, @ref `cuda::unique_span`
 *
 */

#pragma once
#ifndef CUDA_API_WRAPPERS_UNIQUE_SPAN_HPP_
#define CUDA_API_WRAPPERS_UNIQUE_SPAN_HPP_

#include "current_context.hpp"
#include "memory.hpp"
#include "detail/span.hpp"
#include "types.hpp"

#include <type_traits>

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
 * @note unique_span is a container which was proposed for, but not finally included in, C++14.
 * It can be though of as a variation on std::array, with the size=capacity set dynamically rather
 * than statically.
 *
 * @note unique_span = unique_span+typing or span+ownership+non_null
 *
 * @tparam T the element type-
 */

template<typename T, typename Deleter>
class unique_span : public ::cuda::span<T> {
public: // span types
	using span_type = span<T>;

	// Exposing some span type definitions, strictly for terseness
	// (they're all visible on the outside anyway)
	using size_type = typename span<T>::size_type;
	using pointer = typename span<T>::pointer;
	using reference = typename span<T>::reference;
	using deleter_type = Deleter;

protected: // exposing span data members
	using span<T>::data_;
	using span<T>::size_;

public: // constructors and destructor

	constexpr unique_span() noexcept = delete;

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
	// that's not just GPU allcoation-oriented, we might have had one of those.

	/** A move constructor.
	 *
	 * @note Moving is the only way a dynarray may have its data_ field become null;
	 * the user is strongly assumed not to use the dynarray after moving from it.
	 */
	unique_span(unique_span&& other) noexcept : unique_span{ other.release() } { }
	// Disable copy construction - we do not allocate
	unique_span(const unique_span&) = delete;

	// Note: No conversion from "another type" like with ::std::unique_pointer, since
	// this class is not variant with the element type; and there's not much sense in
	// supporting conversion of memory between different deleters (/ allocators).

	~unique_span() noexcept
	{
		if (data_ != nullptr) {
			deleter_type{}(data_);
		}
#ifndef NDEBUG
		data_ = nullptr;
		size_ = 0;
#endif
	}

public: // operators

	/// No copy-assignment - that would break our ownership guarantee
	unique_span& operator=(const unique_span&) = delete;

	/// A Move-assignment operator, which takes ownership of the other region
	unique_span& operator=(unique_span&& other) noexcept
	{
		span_type released = other.release();
		if (data_ != nullptr) {
			deleter_type{}(data_);
		}
		data_ = released.data_;
		size_ = released.size_;
	}

	/// No plain dereferencing - as there is no guarantee that any object has been
	/// initialized at those locations, nor do we know its type

	constexpr operator memory::const_region_t() const noexcept { return { data_, size_ * sizeof(T) }; }

	template<typename = typename ::std::enable_if<! ::std::is_const<T>::value>::type>
	constexpr operator memory::region_t() const noexcept { return { data_, size_ * sizeof(T) }; }

	constexpr span_type get() const noexcept { return { data_, size_ }; }

	/// Exchange the pointer and deleter with another object.
	void swap(unique_span& other) noexcept
	{
		::std::swap<span_type >(*this, other);
	}

protected: // mutators
	/// Release ownership of any stored pointer.
	span_type release() noexcept
	{
		span_type released { data_, size_ };
		data_ = nullptr;
#ifndef NDEBUG
		size_ = 0;
#endif
		return released;
	}
}; // class unique_span

namespace memory {

namespace device {

template <typename T>
using unique_span = cuda::unique_span<T, detail_::deleter>;

namespace detail_ {

template <typename T>
unique_span<T> make_unique_span(const context::handle_t context_handle, size_t size)
{
	// Note: _Not_ asserting trivial-copy-constructibility here; so if you want to copy data
	// to/from the device using this object - it's your own repsonsibility to ensure that's
	// a valid thing to do.
	CAW_SET_SCOPE_CONTEXT(context_handle);
	return unique_span<T>{ allocate_in_current_context(size * sizeof(T)) };
}

} // namespace detail_

/**
 * @brief Create a variant of ::std::unique_pointer for an array in
 * device-global memory.
 *
 * @note CUDA's runtime API always has a current device; but -
 * there is not necessary a current context; so a primary context
 * for a device may be created through this call.
 *
 * @tparam T  an array type; _not_ the type of individual elements
 *
 * @param context       The CUDA device context in which to make the
 *                      allocation.
 * @param num_elements  the number of elements to allocate
 *
 * @return an ::std::unique_ptr pointing to the constructed T array
*/
template <typename T>
unique_span<T> make_unique_span(const context_t& context, size_t size);
template <typename T>
unique_span<T> make_unique_span(const device_t& device, size_t size);
template <typename T>
unique_span<T> make_unique_span(size_t size);

} // namespace device

/// See @ref `device::make_unique_span(const context_t& context, size_t num_elements)`
template <typename T> 
inline device::unique_span<T> make_unique_span(const context_t& context, size_t num_elements)
{
	return device::make_unique_span<T>(context, num_elements);
}

/// See @ref `device::make_unique_span(const device_t& device, size_t num_elements)`
template <typename T>
inline device::unique_span<T> make_unique_span(const device_t& device, size_t num_elements)
{
	return device::make_unique_span<T>(device, num_elements);
}

namespace host {

template <typename T>
using unique_span = cuda::unique_span<T, detail_::deleter>;

template <typename T>
unique_span<T> make_unique_span(const context_t& context, size_t size, allocation_options options = {});
template <typename T>
unique_span<T> make_unique_span(const device_t& device, size_t size);
template <typename T>
unique_span<T> make_unique_span(size_t size);

} // namespace host

namespace managed {

template <typename T>
using unique_span = cuda::unique_span<T, detail_::deleter>;

namespace detail_ {

template <typename T>
unique_span<T> make_unique_span(
	const context::handle_t  context_handle,
	size_t                   size,
	initial_visibility_t     initial_visibility = initial_visibility_t::to_all_devices)
{
	CAW_SET_SCOPE_CONTEXT(context_handle);
	return unique_span<T>{ allocate_in_current_context(size * sizeof(T), initial_visibility) };
}

} // namespace detail_

template <typename T>
unique_span<T> make_unique_span(
	const context_t&      context,
	size_t                size,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices);
template <typename T>
unique_span<T> make_unique_span(
	const device_t&       device,
	size_t                size,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices);
template <typename T>
unique_span<T> make_unique_span(
	size_t                size,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices);

} // namespace managed

} // namespace memory

} // namespace cuda

#endif // CUDA_API_WRAPPERS_UNIQUE_SPAN_HPP_
