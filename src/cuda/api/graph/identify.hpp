/**
 * @file
 *
 */
#pragma once
#ifndef CUDA_GRAPH_API_WRAPPERS_ERROR_HPP_
#define CUDA_GRAPH_API_WRAPPERS_ERROR_HPP_

#include <cuda/api/types.hpp>
#include <cuda/api/error.hpp>

namespace cuda {

namespace graph {

namespace template_ {

namespace detail_ {

inline ::std::string identify(handle_t handle)
{
	return "execution graph template " + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}
/*

inline ::std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " on " + context::detail_::identify(context_handle);
}

inline ::std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " on " + context::detail_::identify(context_handle, device_id);
}
*/

} // namespace detail_

} // namespace template_

namespace instance {

namespace detail_ {

inline ::std::string identify(handle_t handle)
{
	return "execution graph instance " + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail_::identify(device_id);
}

inline ::std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " on " + context::detail_::identify(context_handle);
}

inline ::std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " on " + context::detail_::identify(context_handle, device_id);
}

} // namespace detail_

} // namespace instance

namespace node {

namespace detail_ {

inline ::std::string identify(handle_t handle)
{
	return ::std::string("node with handle ") + ::cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(handle_t node_handle, template_::handle_t graph_template_handle)
{
	return identify(node_handle) + " in " + template_::detail_::identify(graph_template_handle);
}

} // namespace detail_

} // namespace node

} // namespace graph

} // namespace cuda

#endif // CUDA_GRAPH_API_WRAPPERS_ERROR_HPP_
