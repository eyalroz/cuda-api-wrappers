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
namespace detail {

inline std::string identify(const context_t& context)
{
	return context::detail::identify(context.handle(), context.device_id());
}

inline std::string identify(const stream_t& stream)
{
	return stream::detail::identify(stream.handle(), stream.context().handle(), stream.device().id());
}

inline std::string identify(const event_t& event)
{
	return event::detail::identify(event.handle(), event.context_handle(), event.device_id());
}

inline std::string identify(const kernel_t& kernel)
{
	return kernel::detail::identify(kernel.handle()) + " in " + identify(kernel.context());
}

inline std::string identify(const library::kernel_t& library_kernel)
{
	return library::kernel::detail::identify(library_kernel.library_handle(), library_kernel.handle());
}

inline std::string identify(const library_t& library)
{
	return library::detail::identify(library.handle());
}

inline std::string identify(const module_t& module)
{
	return module::detail::identify(module.handle(), module.context_handle(), module.device_id());
}

inline std::string identify(const graph::node_t &node)
{
	return graph::node::detail::identify(node.handle(), node.containing_graph_handle());
}

inline std::string identify(const graph::template_t& graph_template)
{
	return cuda_::detail::identify(graph_template.handle());
}

inline std::string identify(const memory::pool_t& pool)
{
	return memory::pool::detail::identify(pool.handle(), pool.device_id());
}

inline std::string identify(memory::physical_allocation_t const& physical_allocation)
{
	return memory::physical_allocation::detail::identify(physical_allocation.handle(), physical_allocation.size());
}

inline std::string identify(memory::virtual_::mapping_t const& mapping)
{
	return detail::identify(mapping.address_range());
}

#if ! CAW_CAN_GET_APRIORI_KERNEL_HANDLE
inline std::string identify(const kernel::apriori_compiled_t& kernel)
{
	return "apriori-compiled kernel " + cuda_::detail::ptr_as_hex(kernel.ptr())
		+ " in " + cuda_::detail::identify(kernel.context());
}
#endif // ! CAW_CAN_GET_APRIORI_KERNEL_HANDLE

} // namespace detail


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

} // namespace memory

} // namespace cuda_

#endif //CUDA_API_WRAPPERS_DETAIL_IDENTIFY_HPP_
