#ifndef CUDA_API_WRAPPERS_DETAIL_IDENTIFY_HPP_
#define CUDA_API_WRAPPERS_DETAIL_IDENTIFY_HPP_

#include "../identify.hpp"
#include "../context.hpp"
#include "../graph/node.hpp"
#include "../event.hpp"
#include "../memory_pool.hpp"
#include "../module.hpp"
#include "../virtual_memory.hpp"
#include "../library.hpp"
#include "../kernels/in_library.hpp"

namespace cuda_ {
namespace context {
namespace detail {

inline std::string identify(const context_t& context)
{
	return identify(context.handle(), context.device_id());
}

} // namespace detail
} // namespace context

namespace stream {
namespace detail {
inline std::string identify(const stream_t& stream)
{
	return identify(stream.handle(), stream.context().handle(), stream.device().id());
}
} // namespace detail
} // namespace stream

namespace event {
namespace detail {

inline std::string identify(const event_t& event)
{
	return identify(event.handle(), event.context_handle(), event.device_id());
}

} // namespace detail
} // namespace event

namespace graph {

namespace template_ {

namespace detail {

inline std::string identify(const template_t& graph_template)
{
	return identify(graph_template.handle());
}

} // namespace detail

} // namespace template_

namespace node {

namespace detail {

inline std::string identify(const node_t &node)
{
	return identify(node.handle(), node.containing_graph_handle());
}

} // namespace detail
} // namespace node
} // namespace graph

namespace memory { namespace external { namespace detail {

inline std::string identify(descriptor_t descriptor)
{
	return "external memory resource of kind " + std::to_string(descriptor.type);
}

inline std::string identify(handle_t handle, descriptor_t descriptor)
{
	return "external memory resource of kind " + std::to_string(descriptor.type)
		   + " at " + cuda_::detail::ptr_as_hex(handle);
}

} // namespace detail
} // namespace external

namespace pool {
namespace detail {

inline std::string identify(const pool_t& pool)
{
	return identify(pool.handle(), pool.device_id());
}

} // namespace detail
} // namespace pool

namespace physical_allocation {
namespace detail {

inline std::string identify(physical_allocation_t const& physical_allocation)
{
	return physical_allocation::detail::identify(physical_allocation.handle(), physical_allocation.size());
}

} // namespace detail
} // namespace physical_allocation

namespace virtual_ {
namespace detail {

inline std::string identify(mapping_t const& mapping)
{
	return mapping::detail::identify(mapping.address_range());
}

} // namespace detail
} // namespace virtual_

} // namespace memory

namespace module {

namespace detail {

inline std::string identify(const module_t& module)
{
	return identify(module.handle(), module.context_handle(), module.device_id());
}

} // namespace detail

} // namespace module

namespace library {

namespace detail {

inline std::string identify(const library_t& library)
{
	return identify(library.handle());
}

} // namespace detail

namespace kernel {

namespace detail {

inline std::string identify(const kernel_t& library_kernel)
{
	return identify(library_kernel.library_handle(), library_kernel.handle());
}

} // namespace detail

} // namespace kernel

} // namespace library

namespace kernel {

namespace apriori_compiled {

#if ! CAW_CAN_GET_APRIORI_KERNEL_HANDLE
namespace detail {
inline std::string identify(const apriori_compiled_t& kernel)
{
	return "apriori-compiled kernel " + cuda_::detail::ptr_as_hex(kernel.ptr())
		+ " in " + context::detail::identify(kernel.context());
}
} // namespace detail
#endif // ! CAW_CAN_GET_APRIORI_KERNEL_HANDLE

} // namespace apriori_compiled

namespace detail {

inline std::string identify(const kernel_t& kernel)
{
	return identify(kernel.handle()) + " in " + context::detail::identify(kernel.context());
}

} // namespace detail

} // namespace kernel


} // namespace cuda_

#endif //CUDA_API_WRAPPERS_DETAIL_IDENTIFY_HPP_
