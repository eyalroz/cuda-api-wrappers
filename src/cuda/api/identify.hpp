/**
 * @file
 *
 * @brief String-generating `identify()` functions for the various
 * classes and other types in this library - used mostly for reporting
 * errors and throwing exceptions
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_IDENTIFY_HPP_
#define CUDA_API_WRAPPERS_IDENTIFY_HPP_

#include "types.hpp"

#include <string>

namespace cuda_ {

namespace detail {

template <typename I, bool UpperCase = false>
std::string as_hex(I x)
{
	static_assert(std::is_unsigned<I>::value, "only signed representations are supported");
	unsigned num_hex_digits = 2*sizeof(I);
	if (x == 0) return "0x0";

	enum { bits_per_hex_digit = 4 }; // = log_2 of 16
	static const char* digit_characters =
		UpperCase ? "0123456789ABCDEF" : "0123456789abcdef" ;

	std::string result(num_hex_digits,'0');
	for (unsigned digit_index = 0; digit_index < num_hex_digits ; digit_index++)
	{
		size_t bit_offset = (num_hex_digits - 1 - digit_index) * bits_per_hex_digit;
		auto hexadecimal_digit = (x >> bit_offset) & 0xF;
		result[digit_index] = digit_characters[hexadecimal_digit];
	}
	return "0x0" + result.substr(result.find_first_not_of('0'), std::string::npos);
}

// TODO: Perhaps find a way to avoid the extra function, so that as_hex() can
// be called for pointer types as well? Would be easier with boost's uint<T>...
template <typename I, bool UpperCase = false>
std::string ptr_as_hex(const I* ptr)
{
	return as_hex(reinterpret_cast<uintptr_t>(ptr));
}

} // namespace detail

namespace device {
namespace detail {
inline std::string identify(device::id_t device_id)
{
	return std::string("device ") + std::to_string(device_id);
}
} // namespace detail
} // namespace device

namespace context {
namespace detail {

inline std::string identify(handle_t handle)
{
	return "context " + cuda_::detail::ptr_as_hex(handle);
}

inline std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail::identify(device_id);
}

} // namespace detail

namespace current {
namespace detail {
inline std::string identify(context::handle_t handle)
{
	return "current context: " + context::detail::identify(handle);
}
inline std::string identify(context::handle_t handle, device::id_t device_id)
{
	return "current context: " + context::detail::identify(handle, device_id);
}
} // namespace detail
} // namespace current

} // namespace context

namespace device {
namespace primary_context {
namespace detail {

inline std::string identify(handle_t handle, device::id_t device_id)
{
	return "context " + context::detail::identify(handle, device_id);
}
inline std::string identify(handle_t handle)
{
	return "context " + context::detail::identify(handle);
}
} // namespace detail
} // namespace primary_context
} // namespace device

namespace stream {
namespace detail {

inline std::string identify(handle_t handle)
{
	return (handle == nullptr) ? "default/null stream" :
		"stream at " + cuda_::detail::ptr_as_hex(handle);
}

inline std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail::identify(device_id);
}
inline std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " in " + context::detail::identify(context_handle);
}
inline std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " in " + context::detail::identify(context_handle, device_id);
}
} // namespace detail
} // namespace stream

namespace event {
namespace detail {
inline std::string identify(handle_t handle)
{
	return "event " + cuda_::detail::ptr_as_hex(handle);
}
inline std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail::identify(device_id);
}
inline std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " on " + context::detail::identify(context_handle);
}
inline std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " on " + context::detail::identify(context_handle, device_id);
}
} // namespace detail
} // namespace event

namespace array {
namespace detail {
inline std::string identify(handle_t handle)
{
	return "array at " + cuda_::detail::ptr_as_hex(handle);
}
} // namespace detail
} // namespace array

namespace kernel {
namespace detail {

inline std::string identify(const void* ptr)
{
	return "kernel " + cuda_::detail::ptr_as_hex(ptr);
}
inline std::string identify(const void* ptr, device::id_t device_id)
{
	return identify(ptr) + " on " + device::detail::identify(device_id);
}
inline std::string identify(const void* ptr, context::handle_t context_handle)
{
	return identify(ptr) + " in " + context::detail::identify(context_handle);
}
inline std::string identify(const void* ptr, context::handle_t context_handle, device::id_t device_id)
{
	return identify(ptr) + " in " + context::detail::identify(context_handle, device_id);
}
inline std::string identify(handle_t handle)
{
	return "kernel at " + cuda_::detail::ptr_as_hex(handle);
}
inline std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " in " + context::detail::identify(context_handle);
}
inline std::string identify(handle_t handle,  device::id_t device_id)
{
	return identify(handle) + " on " + device::detail::identify(device_id);
}
inline std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " in " + context::detail::identify(context_handle, device_id);
}

} // namespace detail
} // namespace kernel

namespace memory {
namespace detail {

inline std::string identify(region_t region)
{
	return std::string("memory region at ") + cuda_::detail::ptr_as_hex(region.data())
		+ " of size " + std::to_string(region.size());
}

inline std::string identify(location_t location)
{
	switch (location.type) {
	case CU_MEM_LOCATION_TYPE_DEVICE:
		if (location.id != CU_DEVICE_CPU) {
			return "global memory of " + cuda_::device::detail::identify(location.id);
		}
		// fallthrough
	case CU_MEM_LOCATION_TYPE_HOST:
		return "host (system) memory";
	case CU_MEM_LOCATION_TYPE_HOST_NUMA:
		return "host (system) NUMA node " + std::to_string(location.id);
	case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
		return "current host (system) NUMA node ";
	default:
		return "(invalid)";
	}
}

} // namespace detail

namespace ipc {
namespace detail {

inline std::string identify(const void* ptr)
{
	return "IPC-imported pointer " + cuda_::detail::ptr_as_hex(ptr);
}

} // namespace detail
} // namespace ipc

} // namespace memory

namespace link {
namespace detail {

inline std::string identify(handle_t handle)
{
	return "link" + cuda_::detail::ptr_as_hex(handle);
}

} // namespace detail
} // namespace link

namespace texture {
namespace detail {

inline std::string identify(handle_t handle)
{
	return "texture " + std::to_string(handle);
}

} // namespace detail
} // namespace texture

namespace memory {
namespace virtual_ {

namespace reservation {

namespace detail {
inline std::string identify(cuda_::detail::tagged<reserved_address_range_t, region_t> region_)
{
	return identify(region_.value);
}
} // namespace detail
} // namespace reservation

namespace mapping {

namespace detail {

inline std::string identify(region_t address_range) {
	return std::string("mapping of ") + memory::detail::identify(address_range);
}

} // namespace detail

} // namespace mapping

} // namespace virtual_
} // namespace memory

} // namespace cuda_

#endif // CUDA_API_WRAPPERS_IDENTIFY_HPP_
