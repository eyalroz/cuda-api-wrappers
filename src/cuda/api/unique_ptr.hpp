/**
 * @file unique_ptr.hpp
 *
 * @brief A smart pointer for CUDA device- and host-side memory, similar
 * to the standard library's <a href="http://en.cppreference.com/w/cpp/memory/unique_ptr">::std::unique_ptr</a>.
 *
 */
#ifndef CUDA_API_WRAPPERS_UNIQUE_PTR_HPP_
#define CUDA_API_WRAPPERS_UNIQUE_PTR_HPP_

#include <cuda/api/current_device.hpp>
#include <cuda/api/memory.hpp>

namespace cuda {
namespace memory {
namespace detail_ {


template<typename T, typename Deleter>
struct make_unique_selector { using non_array = ::std::unique_ptr<T, Deleter>; };
template<typename U, typename Deleter> struct make_unique_selector<U[], Deleter> { using unbounded_array = ::std::unique_ptr<U[], Deleter>; };
template<typename T, size_t N, typename Deleter> struct make_unique_selector<T[N], Deleter> { struct bounded_array { }; };


/**
 * A CUDA equivalent of the ::std::make_unique, using cuda::memory::unique_ptr
 * rather than ::std::unique_ptr (i.e. using cuda::memory::free() for freeing
 *
 * @note Only trivially-constructible types are supported
 */
template<typename T, typename Allocator, typename Deleter>
inline typename detail_::make_unique_selector<T, Deleter>::non_array make_unique()
{
	static_assert(::std::is_trivially_constructible<T>::value,
		"Allocating with non-trivial construction on the device is not supported.");
	auto space_ptr = Allocator()(sizeof(T));
	return typename detail_::make_unique_selector<T, Deleter>::non_array(static_cast<T*>(space_ptr));
}
template<typename T, typename Allocator, typename Deleter>
inline typename detail_::make_unique_selector<T, Deleter>::unbounded_array make_unique(size_t num_elements)
{
	// If this function is instantiated, T is of the form "element_type[]"
	using element_type = typename ::std::remove_extent<T>::type;
	static_assert(sizeof(element_type) % alignof(element_type) == 0,
		"Alignment handling unsupported for now");
	static_assert(::std::is_trivially_constructible<element_type>::value,
		"Allocating with non-trivial construction on the device is not supported.");
	void* space_ptr = Allocator()(sizeof(element_type) * num_elements);
	return typename detail_::make_unique_selector<T, Deleter>::unbounded_array(static_cast<element_type*>(space_ptr));
}
template<typename T, typename Allocator, typename Deleter, typename... Args>
inline typename detail_::make_unique_selector<T, Deleter>::bounded_array make_unique(Args&&...) = delete;

using deleter = device::detail_::deleter;

template<typename T>
inline ::std::unique_ptr<T, deleter>
make_unique(cuda::device::id_t device_id, size_t n)
{
	cuda::device::current::detail_::scoped_override_t set_device_for_this_scope(device_id);
	return memory::detail_::make_unique<T, device::detail_::allocator, deleter>(n);
}

template<typename T>
inline ::std::unique_ptr<T, deleter>
make_unique(cuda::device::id_t device_id)
{

	cuda::device::current::detail_::scoped_override_t set_device_for_this_scope(device_id);

	return memory::detail_::make_unique<T, device::detail_::allocator, deleter>();
}

} // namespace detail_

namespace device {

template<typename T>
using unique_ptr = ::std::unique_ptr<T, detail_::deleter>;

template<typename T>
inline unique_ptr<T> make_unique(cuda::device_t device, size_t n);

template<typename T>
inline unique_ptr<T> make_unique(cuda::device_t device);

template<typename T>
inline unique_ptr<T> make_unique(T* raw_ptr)
{
	// We should not have to care about single-elements vs arrays here. I think
	return unique_ptr<T>(raw_ptr);
}

} // namespace device

namespace host {

template<typename T>
using unique_ptr = ::std::unique_ptr<T, detail_::deleter>;

template<typename T>
inline unique_ptr<T> make_unique(size_t n)
{
	return cuda::memory::detail_::make_unique<T, detail_::allocator, detail_::deleter>(n);
}

template<typename T>
inline unique_ptr<T> make_unique()
{
	return cuda::memory::detail_::make_unique<T, detail_::allocator, detail_::deleter>();
}

} // namespace host

namespace managed {

template<typename T>
using unique_ptr = ::std::unique_ptr<T, detail_::deleter>;

template<typename T>
inline unique_ptr<T> make_unique(
	size_t                n,
	initial_visibility_t  initial_visibility = initial_visibility_t::to_all_devices)
{
	return (initial_visibility == initial_visibility_t::to_all_devices) ?
		cuda::memory::detail_::make_unique<T, detail_::allocator<
			initial_visibility_t::to_all_devices>, detail_::deleter
		>(n) :
		cuda::memory::detail_::make_unique<T, detail_::allocator<
			initial_visibility_t::to_supporters_of_concurrent_managed_access>, detail_::deleter
		>(n);
}

template<typename T>
inline unique_ptr<T> make_unique(
	initial_visibility_t initial_visibility = initial_visibility_t::to_all_devices)
{
	return (initial_visibility == initial_visibility_t::to_all_devices) ?
		cuda::memory::detail_::make_unique<T, detail_::allocator<
			initial_visibility_t::to_all_devices>, detail_::deleter
		>() :
		cuda::memory::detail_::make_unique<T, detail_::allocator<
			initial_visibility_t::to_supporters_of_concurrent_managed_access>, detail_::deleter
		>();

}
} // namespace managed

} // namespace memory
} // namespace cuda

#endif // CUDA_API_WRAPPERS_UNIQUE_PTR_HPP_
