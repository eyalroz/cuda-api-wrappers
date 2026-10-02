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

// class device_t;
namespace device { class primary_context_t; }
class context_t;
class stream_t;
class event_t;
class module_t;
#if CUDA_VERSION >= 12000
class library_t;
#endif
class kernel_t;
namespace graph {
class node_t;
class template_t;
}
namespace memory {
class pool_t;
class physical_allocation_t;
namespace ipc { class imported_ptr_t; }
namespace virtual_ { class mapping_t; }
} // namespace memory
namespace kernel { class apriori_compiled_t; }
#if CUDA_VERSION >= 12000
namespace library { class kernel_t; }
#endif

namespace detail {
std::string identify(memory::region_t region);
std::string identify(const context_t& context);
std::string identify(const stream_t& stream);
std::string identify(const event_t& event);
std::string identify(const kernel_t& kernel);
#if CUDA_VERSION >= 12000
std::string identify(const library::kernel_t& library_kernel);
std::string identify(const library_t& library);
#endif
std::string identify(const module_t &module);
#if CUDA_VERSION >= 10000
std::string identify(const graph::node_t &node);
std::string identify(const graph::template_t& graph_template);
#endif
#if CUDA_VERSION >= 11020
std::string identify(const memory::pool_t &pool);
#endif
#if CUDA_VERSION >= 10020
std::string identify(const memory::physical_allocation_t& physical_allocation);
std::string identify(memory::virtual_::mapping_t const& mapping);
#endif
#if CAW_CAN_GET_APRIORI_KERNEL_HANDLE
std::string identify(const kernel::apriori_compiled_t& kernel);
#endif // ! CAW_CAN_GET_APRIORI_KERNEL_HANDLE
} // namespace detail

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

namespace detail {

inline std::string identify(context::handle_t handle) {	return "context at" + ptr_as_hex(handle); }
inline std::string identify(stream::handle_t handle) { return (handle) ? "stream at " + ptr_as_hex(handle) : "default/null stream"; }
inline std::string identify(event::handle_t handle) { return "event at " + ptr_as_hex(handle); }
inline std::string identify(array::handle_t handle) { return "array at " + ptr_as_hex(handle); }
inline std::string identify(kernel::handle_t handle) { return "kernel at " + ptr_as_hex(handle); }
#if CUDA_VERSION >= 10000
inline std::string identify(memory::external::handle_t handle) { return "external memory resource at " + ptr_as_hex(handle); }
#endif
#if CUDA_VERSION >= 12000
inline std::string identify(library::kernel::handle_t handle) { return "library kernel at " + ptr_as_hex(handle); }
inline std::string identify(library::handle_t handle) { return "library kernel at " + ptr_as_hex(handle); }
#endif
#if CUDA_VERSION >= 11020
inline std::string identify(memory::pool::handle_t handle) { return "memory pool at " + ptr_as_hex(handle); }
#endif
inline std::string identify(link::handle_t handle) { return "link" + ptr_as_hex(handle); }
inline std::string identify(texture::handle_t handle) { return "texture " + std::to_string(handle); }
#if CUDA_VERSION >= 10000
inline std::string identify(graph::template_::handle_t handle) { return "execution graph template " + ptr_as_hex(handle); }
inline std::string identify(graph::instance::handle_t handle) { return "execution graph instance " + ptr_as_hex(handle); }
inline std::string identify(graph::node::handle_t handle) { return std::string("node with handle ") + ptr_as_hex(handle); }
#endif // CUDA_VERSION >= 10000
inline std::string identify(module::handle_t handle) { return std::string("module ") + ptr_as_hex(handle); }
inline std::string identify(tagged<context_t, context::handle_t> handle) { return identify(handle.untag()); }
inline std::string identify(tagged<device::primary_context_t, context::handle_t> handle) { return "primary " + identify(handle.untag()); }
inline std::string identify(tagged<memory::virtual_::mapping_t, memory::region_t> handle) {	return "mapping of " + identify(handle.value); }
#if CUDA_VERSION >= 10200
inline std::string identify(tagged<memory::virtual_::reserved_address_range_t, memory::region_t> region_) { return "reserved " + identify(region_.value); }
#endif
inline std::string identify(tagged<memory::ipc::imported_ptr_t, void*> ptr_) { return "imported pointer " + ptr_as_hex(ptr_.value); }

} // namespace detail

namespace device {
namespace detail {
inline std::string identify(device::id_t device_id)
{
	return std::string("device ") + std::to_string(device_id);
}
} // namespace detail
} // namespace device

namespace detail {
inline std::string identify(memory::region_t region)
{
	return std::string("memory region at ") + ptr_as_hex(region.data())
		+ " of size " + std::to_string(region.size());
}

} // namespace detail

namespace context {

namespace detail {

inline std::string identify(handle_t handle, device::id_t device_id)
{
	return cuda_::detail::identify(handle) + " on " + device::detail::identify(device_id);
}

} // namespace detail

namespace current {
namespace detail {
inline std::string identify(context::handle_t handle)
{
	return "current context: " + cuda_::detail::identify(handle);
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
	return "primary " + cuda_::detail::identify(handle);
}
} // namespace detail
} // namespace primary_context
} // namespace device

namespace stream {
namespace detail {

inline std::string identify(handle_t handle, device::id_t device_id)
{
	return cuda_::detail::identify(handle) + " on " + device::detail::identify(device_id);
}
inline std::string identify(handle_t handle, context::handle_t context_handle)
{
	return cuda_::detail::identify(handle) + " in " + cuda_::detail::identify(context_handle);
}
inline std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return cuda_::detail::identify(handle) + " in " + context::detail::identify(context_handle, device_id);
}
} // namespace detail
} // namespace stream

namespace event {
namespace detail {

inline std::string identify(handle_t handle, device::id_t device_id)
{
	return cuda_::detail::identify(handle) + " on " + device::detail::identify(device_id);
}
inline std::string identify(handle_t handle, context::handle_t context_handle)
{
	return cuda_::detail::identify(handle) + " on " + cuda_::detail::identify(context_handle);
}
inline std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return cuda_::detail::identify(handle) + " on " + context::detail::identify(context_handle, device_id);
}
} // namespace detail
} // namespace event

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
	return identify(ptr) + " in " + cuda_::detail::identify(context_handle);
}
inline std::string identify(const void* ptr, context::handle_t context_handle, device::id_t device_id)
{
	return identify(ptr) + " in " + context::detail::identify(context_handle, device_id);
}
inline std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " in " + cuda_::detail::identify(context_handle);
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

std::string identify(descriptor_t descriptor);
std::string identify(handle_t handle, descriptor_t descriptor);

} // namespace detail

} // namespace external
#endif // CUDA_VERSION >= 10000

#if CUDA_VERSION >= 11020
namespace pool {

namespace detail {

inline std::string identify(pool::handle_t handle, cuda_::device::id_t device_id)
{
	return cuda_::detail::identify(handle) + " on " + cuda_::device::detail::identify(device_id);
}

} // namespace detail
} // namespace pool
#endif // CUDA_VERSION >= 11020

} // namespace memory

namespace texture {
namespace detail {

} // namespace detail
} // namespace texture

namespace memory {

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

} // namespace memory

namespace module {

namespace detail {

inline std::string identify(module::handle_t handle, context::handle_t context_handle)
{
	return cuda_::detail::identify(handle) + " in " + cuda_::detail::identify(context_handle);
}

inline std::string identify(module::handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return cuda_::detail::identify(handle) + " in " + context::detail::identify(context_handle, device_id);
}

} // namespace detail

} // namespace module

#if CUDA_VERSION >= 12000
namespace library {

namespace detail {

inline std::string identify(const handle_t &handle)
{
	return std::string("library ") + cuda_::detail::ptr_as_hex(handle);
}

} // namespace detail

namespace kernel {

namespace detail {

inline std::string identify(library::handle_t library_handle, kernel::handle_t handle)
{
	return cuda_::detail::identify(handle) + " within " + library::detail::identify(library_handle);
}

std::string identify(const library::kernel_t &kernel);

} // namespace detail

} // namespace kernel

} // namespace library
#endif // CUDA_VERSION >= 12000

} // namespace cuda_
#endif // CUDA_API_WRAPPERS_IDENTIFY_HPP_
