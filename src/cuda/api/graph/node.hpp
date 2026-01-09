/**
 * @file
 *
 * @brief Graph template node proxy (base-)class base-class @ref node_t and supporting code.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NODE_HPP
#define CUDA_API_WRAPPERS_NODE_HPP

#if CUDA_VERSION >= 10000

#include "../types.hpp"
#include "identify.hpp"

namespace cuda {

namespace graph {

///@endcond
class node_t;
class template_t;
class instance_t;
///@endcond

namespace node {

node_t wrap(template_::handle_t graph_handle, handle_t handle) noexcept;

namespace detail_ {

::std::string identify(const node_t &node);

inline status_t get_dependencies(
	handle_t                     handle,
	handle_t      * __restrict__ dependency_handles,
	::std::size_t * __restrict__ num_dependencies)
{
#if CUDA_VERSION >= 13000
	static constexpr auto no_edge_data { nullptr };
	return cuGraphNodeGetDependencies(handle, dependency_handles, no_edge_data, num_dependencies);
#else
	return cuGraphNodeGetDependencies(handle, dependency_handles, num_dependencies);
#endif
}

inline status_t get_dependents(
	handle_t                     handle,
	handle_t      * __restrict__ dependent_handles,
	::std::size_t * __restrict__ num_dependencies)
{
#if CUDA_VERSION >= 13000
	static constexpr auto no_edge_data { nullptr };
	return cuGraphNodeGetDependentNodes(handle, dependent_handles, no_edge_data, num_dependencies);
#else
	return cuGraphNodeGetDependentNodes(handle, dependent_handles, num_dependencies);
#endif
}

#if CUDA_VERSION >= 13010
inline template_::handle_t graph_handle_of(handle_t handle)
{
	template_::handle_t graph_template_handle;
	auto status = cuGraphNodeGetContainingGraph(handle, &graph_template_handle);
	throw_if_error_lazy(status, "Failed obtaining the graph template containing " + graph::node::detail_::identify(handle));
	return graph_template_handle;
}
#endif // CUDA_VERSION >= 13010

} // namespace detail_

using type_t = CUgraphNodeType;

} // namespace node

/**
 * @brief Wrapper class for a CUDA execution graph node
 *
 * Use this class to pass and receive nodes from/to other CUDA API wrapper functions
 * and objects (in particular, `cuda::graph::template_t` and `cuda::graph::instance_t`).
 *
 * @note { This is a reference-type; it does not own the node, and will not remove
 * the node from its graph-template when destroyed. You may therefore safely make copies
 * of it. }
 *
 * @note { A node is always tied to a specific graph-template; it cannot be added to
 * another graph-template or used independently. }
 */
class node_t {
public: // data types
	using handle_type = node::handle_t;
	using dependencies_type = ::std::vector<node_t>;
	using dependents_type = ::std::vector<node_t>;
	using type_type = node::type_t;
	using size_type = size_t;

// TODO: WRITEME
public:
	handle_type handle() const noexcept  { return handle_; }
	template_t containing_graph() const noexcept;
	template_::handle_t containing_graph_handle() const noexcept { return graph_template_handle_; }
	type_type type_() const
	{
		type_type result;
		auto status = cuGraphNodeGetType(handle_, &result);
		throw_if_error_lazy(status, "Obtaining the type of " + node::detail_::identify(*this));
		return result;
	}

	size_t num_dependencies() const
	{
		size_t num_dependencies_;
		static constexpr auto no_returned_handles = nullptr;
		auto status = node::detail_::get_dependencies(handle_, no_returned_handles, &num_dependencies_);
		throw_if_error_lazy(status, "Obtaining the number of nodes on which " + node::detail_::identify(*this) + " is dependent");
		return num_dependencies_;
	}

	size_t num_dependents() const
	{
		size_t num_dependents_;
		static constexpr auto no_returned_handles = nullptr;
		auto status = node::detail_::get_dependents(handle_, no_returned_handles, &num_dependents_);
		throw_if_error_lazy(status, "Obtaining the number of nodes dependent on " + node::detail_::identify(*this));
		return num_dependents_;
	}

	dependencies_type dependencies() const
	{
		size_type num_dependencies_ { num_dependencies() } ;
		::std::vector<node::handle_t> node_handles {num_dependencies_ };
		auto status = node::detail_::get_dependencies(handle_, node_handles.data(), &num_dependencies_);
		throw_if_error_lazy(status, "Obtaining the set nodes on which " + node::detail_::identify(*this) + " is dependent");
		dependencies_type result;
		for (const auto& node_handle : node_handles) {
			result.emplace_back(node::wrap(graph_template_handle_, node_handle));
		}
		return result;
	}

	dependencies_type dependents() const
	{
		size_type num_dependents_ { num_dependents() } ;
		::std::vector<node::handle_t> node_handles {num_dependents_ };
		auto status = node::detail_::get_dependents(handle_, node_handles.data(), &num_dependents_);
		throw_if_error_lazy(status, "Obtaining the set nodes dependent on " + node::detail_::identify(*this));
		dependencies_type result;
		for (const auto& node_handle : node_handles) {
			result.emplace_back(node::wrap(graph_template_handle_, node_handle));
		}
		return result;
	}

protected: // constructors and destructors
	node_t(template_::handle_t graph_template_handle, handle_type handle) noexcept
	: graph_template_handle_(graph_template_handle), handle_(handle) { }

public: // friendship
	friend node_t node::wrap(template_::handle_t graph_handle, node::handle_t handle) noexcept;

public:  // constructors and destructors
	node_t(const node_t&) noexcept = default; // It's a reference type, so copying is not a problem
	node_t(node_t&&) noexcept = default; // It's a reference type, so copying is not a problem

	node_t& operator=(node_t other) noexcept
	{
		graph_template_handle_ = other.graph_template_handle_;
		handle_ = other.handle_;
		return *this;
	}

protected:
	template_::handle_t graph_template_handle_;
	handle_type handle_;
};

namespace node {

inline node_t wrap(template_::handle_t graph_handle, handle_t handle) noexcept
{
	return { graph_handle, handle };
}

#if CUDA_VERSION >= 13010
namespace detail_ {

/// Returns a node proxy class given only a node handle
inline node_t from_handle(handle_t handle)
{
	auto graph_template_handle = graph_handle_of(handle);
	return wrap(graph_template_handle, handle);
}

} // namespace detail_
#endif // CUDA_VERSION >= 13010

} // namespace node

} // namespace graph

} // namespace cuda

#endif // CUDA_VERSION >= 10000

#endif //CUDA_API_WRAPPERS_NODE_HPP
