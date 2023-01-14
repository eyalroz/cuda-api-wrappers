/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * which regard CUDA (execution) graphs.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_GRAPHS_HPP_
#define MULTI_WRAPPER_IMPLS_GRAPHS_HPP_

#include "../device.hpp"
#include "../pointer.hpp"
#include "../memory.hpp"
#include "../primary_context.hpp"
#include "../stream.hpp"
#include "../virtual_memory.hpp"
#include "../kernel.hpp"
#include "../event.hpp"
#include "../apriori_compiled_kernel.hpp"
#include "../current_context.hpp"
#include "../graph/node.hpp"
#include "../graph/template.hpp"
#include "../graph/instance.hpp"

namespace cuda {

namespace graph {

#if CUDA_VERSION >= 11000

inline bool is_enabled_in(const node_t& node, const instance_t& instance)
{
	unsigned result;
	auto status = cuGraphNodeGetEnabled(instance.handle(), node.handle(), &result);
	throw_if_error_lazy(status, "Determining whether " + node::detail_::identify(node) + " is active in " + instance::detail_::identify(instance));
	return (result == 1);
}

inline void set_enabled_in(const node_t& node, const instance_t& instance, bool enabled)
{
	auto status = cuGraphNodeSetEnabled(instance.handle(), node.handle(), enabled);
	throw_if_error_lazy(status, "Enabling " + node::detail_::identify(node) + " in " + instance::detail_::identify(instance));
}
#endif // CUDA_VERSION >= 11000

inline void launch(const instance_t& instance, const stream_t& stream)
{
	context::current::detail_::scoped_override_t set_context_for_current_scope(stream.context_handle());
	auto status = cuGraphLaunch(instance.handle(), stream.handle());
	throw_if_error_lazy(status, "Launching " + instance::detail_::identify(instance) + " on " + stream::detail_::identify(stream));
}

namespace instance {

#if CUDA_VERSION >= 11000
inline void upload(const instance_t& instance, const stream_t& stream)
{
	context::current::detail_::scoped_override_t set_context_for_current_scope(stream.context_handle());
	auto status = cuGraphUpload(instance.handle(), stream.handle());
	throw_if_error_lazy(status, "Uploading " + instance::detail_::identify(instance) + " on " + stream::detail_::identify(stream));
}
#endif // CUDA_VERSION >= 11000

inline void update(const instance_t& destination, const template_t& source)
{
#if CUDA_VERSION < 12000
	node::handle_t impermissible_node_handle{};
	instance::update_status_t update_status;
	auto status = cuGraphExecUpdate(destination.handle(), source.handle(), &impermissible_node_handle, &update_status);
	if (is_failure(status)) {
		throw instance::update_failure(update_status, node::wrap(source.handle(), impermissible_node_handle));
	}
#else
	CUgraphExecUpdateResultInfo update_result_info;
	auto status = cuGraphExecUpdate(destination.handle(), source.handle(), &update_result_info);
	if (is_failure(status)) {
		// TODO: Add support for reporting errors involving edges, not just single nodes
		throw instance::update_failure(update_result_info.result, node::wrap(source.handle(), update_result_info.errorNode));
	}
#endif
}

} // namespace instance

//inline void instance_t::launch(const stream_t& stream) const
//{
//	instance::launch(*this, stream);
//}

inline void instance_t::upload(const stream_t& stream) const
{
	instance::upload(*this, stream);
}

inline instance_t template_t::instantiate(
#if CUDA_VERSION >= 11000
	bool free_previous_allocations_before_relaunch
#endif
#if CUDA_VERSION >= 11700
	, bool use_per_node_priorities
#endif
#if CUDA_VERSION >= 12000
	, bool upload_on_instantiation
	, bool make_device_launchable
#endif
)
{
	return graph::instantiate(
		*this,
#if CUDA_VERSION >= 11000
		free_previous_allocations_before_relaunch
#endif
#if CUDA_VERSION >= 11700
		, use_per_node_priorities
#endif
#if CUDA_VERSION >= 12000
		, upload_on_instantiation
		, make_device_launchable
#endif
	);
}

namespace node {

namespace detail_ {

inline ::std::string identify(const node_t &node)
{
	return identify(node.handle(), node.containing_graph_handle());
}

} // namespace detail_

} // namespace node

inline template_t node_t::containing_graph() const noexcept
{
	static constexpr const bool dont_take_ownership { false };
	return template_::wrap(containing_graph_handle(), dont_take_ownership);
}

namespace instance {

namespace detail_ {

inline ::std::string describe(
	instance::update_status_t  update_status,
	node::handle_t             node_handle,
	template_::handle_t        graph_template_handle)
{
	::std::string result = describe(update_status);
	if (node_handle != node::no_handle) {
		result += node::detail_::identify(node_handle, graph_template_handle);
	}
	return result;
}

/*
inline ::std::string identify(instance::handle_t handle)
{
	return "execution graph instance at " + cuda::detail_::ptr_as_hex(handle);
}

inline ::std::string identify(instance::handle_t handle, template_::handle_t template_handle)
{
	return identify(handle) + " within " + graph::template_::detail_::identify(template_handle);
}

inline ::std::string identify(const instance_t& instance)
{
	return identify(instance.handle(), instance.template_handle());
}
*/

} // namespace detail_

} // namespace instance

namespace template_ {

namespace detail_ {

inline ::std::string identify(const template_t& graph_template)
{
	return identify(graph_template.handle());
}

} // namespace detail_

} // namespace template_

} // namespace graph

inline ::std::string describe(graph::instance::update_status_t update_status, optional<graph::node_t> node)
{
	return node ?
		   describe(update_status) :
		   graph::instance::detail_::describe(update_status, node.value().handle(), node.value().containing_graph_handle());
}

inline graph::template_t stream_t::end_capture() const
{
	graph::template_::handle_t new_graph;
	auto status = cuStreamEndCapture(handle_, &new_graph);
	throw_if_error_lazy(status, "Completing the capture of operations into a graph on " + stream::detail_::identify(*this));
	return graph::template_::wrap(new_graph, do_take_ownership);
}

inline void stream_t::enqueue_t::graph_launch(const graph::instance_t& graph_instance) const
{
	graph::launch(graph_instance, associated_stream);
}


} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_GRAPHS_HPP_

