/**
 * @file
 *
 */
#pragma once
#ifndef CUDA_GRAPH_API_WRAPPERS_ERROR_HPP_
#define CUDA_GRAPH_API_WRAPPERS_ERROR_HPP_

#if CUDA_VERSION >= 10000

#include "../types.hpp"

namespace cuda_ {

namespace graph {

namespace template_ {

namespace detail {

inline std::string identify(handle_t handle)
{
	return "execution graph template " + cuda_::detail::ptr_as_hex(handle);
}

inline std::string identify(handle_t handle, device::id_t device_id)
{
	return identify(handle) + " on " + device::detail::identify(device_id);
}
/*

inline std::string identify(handle_t handle, context::handle_t context_handle)
{
	return identify(handle) + " on " + context::detail::identify(context_handle);
}

inline std::string identify(handle_t handle, context::handle_t context_handle, device::id_t device_id)
{
	return identify(handle) + " on " + context::detail::identify(context_handle, device_id);
}
*/

} // namespace detail

} // namespace template_

namespace instance {

namespace detail {

inline std::string identify(handle_t handle)
{
	return "execution graph instance " + cuda_::detail::ptr_as_hex(handle);
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

} // namespace instance

namespace node {

namespace detail {

inline std::string identify(handle_t handle)
{
	return std::string("node with handle ") + ::cuda_::detail::ptr_as_hex(handle);
}

inline std::string identify(handle_t node_handle, template_::handle_t graph_template_handle)
{
	return identify(node_handle) + " in " + template_::detail::identify(graph_template_handle);
}

} // namespace detail

} // namespace node

} // namespace graph

} // namespace cuda_

#endif // CUDA_VERSION >= 10000

#endif // CUDA_GRAPH_API_WRAPPERS_ERROR_HPP_
