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

class context_t;
class stream_t;
class event_t;
class module_t;
#if CUDA_VERSION >= 12000
class library_t;
#endif
class kernel_t;
namespace graph { class node_t; }
namespace memory {
class pool_t;
class physical_allocation_t;
namespace virtual_ { class mapping_t; }
} // namespace memory
namespace kernel { class apriori_compiled_t; }
#if CUDA_VERSION >= 12000
namespace library { class kernel_t; }
#endif

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

std::string identify(const context_t& context);

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

std::string identify(const stream_t& stream);

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

std::string identify(const event_t& event);

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

std::string identify(const kernel_t& kernel);

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

namespace apriori_compiled {

#if ! CAW_CAN_GET_APRIORI_KERNEL_HANDLE
namespace detail {
inline std::string identify(const apriori_compiled_t& kernel);
} // namespace detail
#endif // ! CAW_CAN_GET_APRIORI_KERNEL_HANDLE

} // namespace apriori_compiled

} // namespace kernel

namespace memory {
namespace detail {

inline std::string identify(region_t region)
{
	return std::string("memory region at ") + cuda_::detail::ptr_as_hex(region.data())
		+ " of size " + std::to_string(region.size());
}

#if CUDA_VERSION >= 10020
inline std::string identify(location_t location)
{
	switch (location.type) {
	case CU_MEM_LOCATION_TYPE_DEVICE:
		if (location.id != CU_DEVICE_CPU) {
			return "global memory of " + cuda_::device::detail::identify(location.id);
		}
		// fallthrough
#if CUDA_VERSION >= 12020
	case CU_MEM_LOCATION_TYPE_HOST:
		return "host (system) memory";
	case CU_MEM_LOCATION_TYPE_HOST_NUMA:
		return "host (system) NUMA node " + std::to_string(location.id);
	case CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT:
		return "current host (system) NUMA node ";
#endif // CUDA_VERSION >= 12020
	default:
		return "(invalid)";
	}
}
#endif // CUDA_VERSION >= 10020
} // namespace detail

namespace ipc {
namespace detail {

inline std::string identify(const void* ptr)
{
	return "IPC-imported pointer " + cuda_::detail::ptr_as_hex(ptr);
}

} // namespace detail
} // namespace ipc

#if CUDA_VERSION >= 10000
namespace external {
namespace detail {

inline std::string identify(subregion_spec_t subregion_spec)
{
	return "subregion of size " + std::to_string(subregion_spec.size)
		   + " at offset " + std::to_string(subregion_spec.offset);
}

inline std::string identify(handle_t handle)
{
	return "external memory resource at " + cuda_::detail::ptr_as_hex(handle);
}

std::string identify(descriptor_t descriptor);
std::string identify(handle_t handle, descriptor_t descriptor);

} // namespace detail

} // namespace external
#endif // CUDA_VERSION >= 10000

#if CUDA_VERSION >= 11020
namespace pool {

namespace detail {

inline std::string identify(pool::handle_t handle)
{
	return "memory pool at " + cuda_::detail::ptr_as_hex(handle);
}

inline std::string identify(pool::handle_t handle, cuda_::device::id_t device_id)
{
	return identify(handle) + " on " + cuda_::device::detail::identify(device_id);
}

std::string identify(const pool_t &pool);

} // namespace detail

} // namespace pool
#endif // CUDA_VERSION >= 11020

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

namespace physical_allocation {
namespace detail {

std::string identify(physical_allocation_t const& physical_allocation);

} // namespace detail
} // namespace physical_allocation

namespace virtual_ {

#if CUDA_VERSION >= 10020
namespace reservation {

namespace detail {
inline std::string identify(cuda_::detail::tagged<reserved_address_range_t, region_t> region_)
{
	return identify(region_.value);
}
} // namespace detail
} // namespace reservation
#endif // CUDA_VERSION >= 10020


namespace mapping {

namespace detail {

inline std::string identify(region_t address_range) {
	return std::string("mapping of ") + memory::detail::identify(address_range);
}

} // namespace detail

} // namespace mapping

} // namespace virtual_

#if CUDA_VERSION >= 10020
namespace physical_allocation {

namespace detail {

inline std::string identify(handle_t handle, size_t size) {
	return std::string("physical allocation with handle ") + std::to_string(handle)
		+ " of size " + std::to_string(size);
}

} // namespace detail

} // namespace physical_allocation
#endif // CUDA_VERSION >= 10020

namespace virtual_ {
namespace detail {

std::string identify(mapping_t const& mapping);

} // namespace detail
} // namespace virtual_

} // namespace memory

namespace graph {


namespace node {

namespace detail {

std::string identify(const node_t &node);

} // namespace detail
} // namespace node
} // namespace graph

namespace module {

namespace detail {

inline std::string identify(module::handle_t handle)
{
	return std::string("module ") + cuda_::detail::ptr_as_hex(handle);
}

inline std::string identify(module::handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " in " + context::detail::identify(context_handle);
}

inline std::string identify(module::handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " in " + context::detail::identify(context_handle, device_id);
}

std::string identify(const module_t &module);

} // namespace detail

} // namespace module

#if CUDA_VERSION >= 12000
namespace library {

namespace detail {

inline std::string identify(const handle_t &handle)
{
	return std::string("library ") + cuda_::detail::ptr_as_hex(handle);
}

std::string identify(const library_t& library);

} // namespace detail

namespace kernel {

namespace detail {

inline std::string identify(kernel::handle_t handle)
{
	return "library kernel at " + cuda_::detail::ptr_as_hex(handle);
}

inline std::string identify(library::handle_t library_handle, kernel::handle_t handle)
{
	return identify(handle) + " within " + library::detail::identify(library_handle);
}

std::string identify(const library::kernel_t &kernel);

} // namespace detail

} // namespace kernel

} // namespace library
#endif // CUDA_VERSION >= 12000

namespace kernel {

inline std::string identify(const kernel_t& kernel);

} // namespace kernel


} // namespace cuda_

#endif // CUDA_API_WRAPPERS_IDENTIFY_HPP_
