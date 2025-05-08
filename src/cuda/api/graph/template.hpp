/**
 * @file
 *
 * @brief An implementation of a subclass of @ref `kernel_t` for kernels
 * compiled together with the host-side program.
 *
 * @todo Implement batch mem op insertion and param setting:
 * cuGraphAddBatchMemOpNode, cuGraphBatchMemOpNodeGetParams, cuGraphBatchMemOpNodeSetParams
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_GRAPH_TEMPLATE_HPP
#define CUDA_API_WRAPPERS_GRAPH_TEMPLATE_HPP

#if CUDA_VERSION >= 10000

#include <cuda/api/graph/typed_node.hpp>
#include <cuda/api/graph/node.hpp>
#include <cuda/api/types.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/graph/identify.hpp>

#include <vector>
#include <cassert>
#include <algorithm>

namespace cuda {

///@cond
class device_t;
class stream_t;
///@endcond

namespace graph {

///@endcond
class template_t;
class instance_t;
///@endcond

namespace node {

namespace detail_ {

// I'm not so sure about this...
using edge_t = ::std::pair<node_t, node_t>;

inline ::std::string identify(const edge_t &edge)
{
	return ::std::string("edge from " + node::detail_::identify(edge.first)
					   + " to " + node::detail_::identify(edge.second));
}

template <typename NodeOrHandle>
handle_t as_handle(const NodeOrHandle& node_or_handle) noexcept
{
	return node_or_handle.handle();
}

template <> inline handle_t as_handle(const handle_t& handle) noexcept { return handle; }

template <template <typename> class Container, typename NodeOrHandle>
struct as_handles_partial_specialization_helper{
	static typename ::std::conditional<
	    ::std::is_same<NodeOrHandle, handle_t>::value,
		span<handle_t>,
		::std::vector<handle_t>
	>::type
	as_handles(Container<NodeOrHandle>&& nodes_or_handles)
	{
		static_assert(
			::std::is_same<typename ::std::remove_const<NodeOrHandle>::type, node::handle_t>::value or
			::std::is_same<typename ::std::remove_const<NodeOrHandle>::type, node::handle_t>::value,
			"Unsupported graph node dependency specifier type. Use either cuda::graph::node_t or cuda::graph::node::handle_t");
		::std::vector<handle_t> handles;
		handles.reserve(nodes_or_handles.size());
		::std::transform(
			nodes_or_handles.begin(),
			nodes_or_handles.end(),
			::std::back_inserter(handles),
			as_handle<NodeOrHandle> );
		return handles;
	}
};

template <template <typename> class Container>
struct as_handles_partial_specialization_helper<Container, handle_t> {
	Container<handle_t> as_handles(Container<handle_t>&& node_handles)
	{
		return node_handles;
	}
};

template <template <typename> class Container, typename NodeOrHandle>
static typename ::std::conditional<
	::std::is_same<NodeOrHandle, handle_t>::value,
	span<handle_t>,
	::std::vector<handle_t>
>::type
as_handles(Container<NodeOrHandle>&& nodes_or_handles)
{
	return as_handles_partial_specialization_helper<Container, handle_t>::as_handles(
		::std::forward<Container<NodeOrHandle>>(nodes_or_handles));
}


} // namespace detail_

} // namespace node

namespace template_ {

template_t wrap(handle_t handle, bool take_ownership = false) noexcept;

namespace detail_ {

::std::string identify(const template_t& template_);

inline status_t delete_edges(
	template_::handle_t         template_handle,
	span<const node::handle_t>  edge_source_handles,
	span<const node::handle_t>  edge_destination_handles)
{
	auto num_edges = edge_source_handles.size();
	assert(edge_source_handles.size() == num_edges && "Mismatched sizes of sources and destinations");

	auto result = cuGraphRemoveDependencies(
		template_handle,edge_source_handles.data(), edge_destination_handles.data(), num_edges);
	return result;
}

inline status_t delete_edges(
	template_::handle_t  template_handle,
	span<const node_t>   edge_sources,
	span<const node_t>   edge_destinations)
{
	auto num_edges = edge_sources.size();
	assert(edge_destinations.size() == num_edges && "Mismatched sizes of sources and destinations");

	// TODO: With C++14, consider make_unique here and no container
	auto handles_buffer = ::std::vector<node::handle_t>{num_edges * 2};
	{
		auto handles_iter = handles_buffer;
		::std::transform(edge_sources.begin(), edge_sources.end(), handles_buffer.data(),
			[](const node_t &node) { return node.handle(); });
		::std::transform(edge_destinations.begin(), edge_destinations.end(), handles_buffer.data() + num_edges,
			[](const node_t &node) { return node.handle(); });
	}
	span<const node::handle_t> edge_source_handles { handles_buffer.data(), num_edges };
	span<const node::handle_t> edge_destination_handles { handles_buffer.data() + num_edges, num_edges };
	return delete_edges(template_handle, edge_source_handles, edge_destination_handles);
}

// Note: duplication of code with delete_edges
inline status_t insert_edges(
	template_::handle_t  template_handle,
	span<const node_t>   edge_sources,
	span<const node_t>   edge_destinations)
{
	auto num_edges = edge_sources.size();
	assert(edge_destinations.size() == num_edges && "Mismatched sizes of sources and destinations");

	// TODO: With C++14, consider make_unique here and no container
	auto handles_buffer = ::std::vector<node::handle_t>{num_edges * 2};
	{
		auto handles_iter = handles_buffer;
		::std::transform(edge_sources.begin(), edge_sources.end(), handles_buffer.data(),
			[](const node_t &node) { return node.handle(); });
		::std::transform(edge_destinations.begin(), edge_destinations.end(), handles_buffer.data() + num_edges,
			[](const node_t &node) { return node.handle(); });
	}
	const node::handle_t* sources_handles = handles_buffer.data();
	const node::handle_t* destinations_handles = handles_buffer.data() + num_edges;
	auto result = cuGraphAddDependencies(
		template_handle,sources_handles, destinations_handles, edge_sources.size());
	return result;
}

inline status_t delete_edges(
	template_::handle_t template_handle,
	span<const node::detail_::edge_t> edges)
{
	// TODO: With C++14, consider make_unique here
	auto handles_buffer = ::std::vector<node::handle_t>{edges.size() * 2};
	auto sources_iterator = handles_buffer.begin();
	auto  destinations_iterator = handles_buffer.begin() + edges.size();
	for(const auto& edge : edges) {
		*(sources_iterator++)      = edge.first.handle();
		*(destinations_iterator++) = edge.second.handle();
	}
	const node::handle_t* sources_handles = handles_buffer.data();
	const node::handle_t* destinations_handles = handles_buffer.data() + edges.size();
	auto result = cuGraphRemoveDependencies(
		template_handle,sources_handles, destinations_handles, edges.size());
	return result;
}

// Note: duplication of code with delete_edges
inline status_t insert_edges(
	template_::handle_t               template_handle,
	span<const node::detail_::edge_t> edges)
{
	// TODO: With C++14, consider make_unique here
	auto handles_buffer = ::std::vector<node::handle_t>{edges.size() * 2};
	auto sources_iterator = handles_buffer.begin();
	auto  destinations_iterator = handles_buffer.begin() + edges.size();
	for(const auto& edge : edges) {
		*(sources_iterator++)      = edge.first.handle();
		*(destinations_iterator++) = edge.second.handle();
	}
	const node::handle_t* sources_handles = handles_buffer.data();
	const node::handle_t* destinations_handles = handles_buffer.data() + edges.size();
	auto result = cuGraphAddDependencies(
		template_handle,sources_handles, destinations_handles, edges.size());
	return result;
}

template <node::kind_t Kind>
status_t invoke_inserter_possibly_with_context(
	cuda::detail_::bool_constant<false>,
	node::handle_t&      new_node_handle,
	template_::handle_t  graph_template_handle,
	CUgraphNode*         dependency_handles,
	size_t               num_dependency_handles,
	typename node::detail_::kind_traits<Kind>::raw_parameters_type&
	                     raw_params,
	context::handle_t)
{
	auto raw_params_maybe_ptr = node::detail_::maybe_add_ptr<Kind>(raw_params);
	return node::detail_::kind_traits<Kind>::inserter(
		&new_node_handle,
		graph_template_handle,
		dependency_handles,
		num_dependency_handles,
		raw_params_maybe_ptr);
}

template <node::kind_t Kind>
status_t invoke_inserter_possibly_with_context(
	cuda::detail_::bool_constant<true>,
	node::handle_t&      new_node_handle,
	template_::handle_t  graph_template_handle,
	CUgraphNode*         dependency_handles,
	size_t               num_dependency_handles,
	typename node::detail_::kind_traits<Kind>::raw_parameters_type&
	                     raw_params,
	context::handle_t    context_handle)
{
	auto raw_params_maybe_ptr = node::detail_::maybe_add_ptr<Kind>(raw_params);
	return node::detail_::kind_traits<Kind>::inserter(
		&new_node_handle,
		graph_template_handle,
		dependency_handles,
		num_dependency_handles,
		raw_params_maybe_ptr,
		context_handle);
}

template <node::kind_t Kind>
node::handle_t insert_node(
	template_::handle_t graph_template_handle,
	context::handle_t context_handle,
	typename node::detail_::kind_traits<Kind>::raw_parameters_type raw_params)
{
	using traits_type = typename node::detail_::kind_traits<Kind>;

	// Defining a useless bool here to circumvent gratuitous warnings from MSVC
	const bool context_needed_but_missing =
		traits_type::inserter_takes_context and context_handle == context::detail_::none;
	if (context_needed_but_missing) {
		throw ::std::invalid_argument(
			"Attempt to insert a CUDA graph template " + ::std::string(traits_type::name)
			+ " node without specifying an execution context");
	}

	node::handle_t new_node_handle;
	auto no_dependency_handles = nullptr;
	size_t no_dependencies_size = 0;
	auto status = invoke_inserter_possibly_with_context<Kind>(
		cuda::detail_::bool_constant<traits_type::inserter_takes_context>{},
		new_node_handle,
		graph_template_handle,
		no_dependency_handles,
		no_dependencies_size,
		raw_params,
		context_handle);
	throw_if_error_lazy(status, "Inserting a " + ::std::string(traits_type::name) + " node into "
								+ template_::detail_::identify(graph_template_handle));
	return new_node_handle;
}

template <node::kind_t Kind, typename... Ts>
node::typed_node_t<Kind> build_params_and_insert_node(
	template_::handle_t  graph_template_handle,
	context::handle_t    context_handle,
	Ts&&...              params_ctor_args)
{

	using traits_type = typename node::detail_::kind_traits<Kind>;
	using parameters_t = typename traits_type::parameters_type;

	// TODO: Why won't this work?
	// static_assert(::std::is_constructible<parameters_t, Ts...>::value,
	//	"Node parameters are not constructible from the arguments passed");

	parameters_t params { ::std::forward<Ts>(params_ctor_args)... };
	typename traits_type::raw_parameters_type raw_params = traits_type::marshal(params);
	auto node_handle = insert_node<Kind>(graph_template_handle, context_handle, raw_params);
	return node::wrap<Kind>(graph_template_handle, node_handle, ::std::move(params));
}

template <node::kind_t Kind, typename... Ts>
node::typed_node_t<Kind> get_context_handle_build_params_and_insert_node(
	cuda::detail_::true_type, // we've been given a context
	template_::handle_t graph_template_handle,
	const context_t& context,
	Ts&&... params_ctor_args)
{
	return build_params_and_insert_node<Kind>(graph_template_handle, context.handle(), ::std::forward<Ts>(params_ctor_args)...);
}

template <node::kind_t Kind, typename... Ts>
node::typed_node_t<Kind> get_context_handle_build_params_and_insert_node(
	cuda::detail_::false_type, // We've not been given a context
	template_::handle_t graph_template_handle,
	Ts&&... params_ctor_args)
{
	auto current_context_handle = context::current::detail_::get_handle();
	// TODO: Consider handling the case of no current context, e.g. by using the default device' primary context
	return build_params_and_insert_node<Kind>(
		graph_template_handle, current_context_handle, ::std::forward<Ts>(params_ctor_args)...);
}


template <node::kind_t Kind, typename... Ts>
node::typed_node_t<Kind> build_params_and_insert_node_wrapper(
	cuda::detail_::false_type , // inserter doesn't takes a context
	template_::handle_t graph_template_handle,
	Ts&&... params_ctor_args)
{
	return build_params_and_insert_node<Kind>(graph_template_handle, context::detail_::none, ::std::forward<Ts>(params_ctor_args)...);
}

template <node::kind_t Kind, typename T, typename... Ts>
node::typed_node_t<Kind> build_params_and_insert_node_wrapper(
	cuda::detail_::true_type, // inserter takes a context
	template_::handle_t graph_template_handle,
	T&& first_arg, // still don't know of T is a context or something else
	Ts&&... params_ctor_args)
{
	static constexpr const bool first_arg_is_a_context =
		::std::is_same<typename cuda::detail_::remove_reference_t<T>, cuda::context_t>::value;
	return get_context_handle_build_params_and_insert_node<Kind>(
//return blah<Kind>(
		cuda::detail_::bool_constant<first_arg_is_a_context>{},
		graph_template_handle, ::std::forward<T>(first_arg), ::std::forward<Ts>(params_ctor_args)...);
}

} // namespace detail_

} // namespace template_

/**
 * @brief A template for a CUDA (execution) graph, which may be instantiated
 * (repeatedly) for actual execution on a CUDA stream.
 *
 * @todo Should this class, itself, be a C++ container of the nodes? That would
 * require caching the nodes at some point.
 */
class template_t {
public: // type definitions
	using size_type = size_t;
	using handle_type = template_::handle_t;

	using node_ref_type = node_t; /// TODO: Should I use a proper ref instead? I wonder...

	/// This is essentially a reference-type
	/// TODO: Should we use a custom structure here?
	using edge_type = ::std::pair<node_ref_type, node_ref_type>;

	using node_ref_container_type = ::std::vector<node_t>;

	/// This is a structure-of-arrays, where the i'th elements of each node container
	/// constitute an edge.
	using edge_container_type = ::std::vector<edge_type>;

public: // getters

	/// The raw CUDA handle for this event
	handle_type handle() const noexcept { return handle_; }

	/// True if this wrapper is responsible for telling CUDA to destroy the event upon the wrapper's own destruction
	bool is_owning() const noexcept { return owning_; }

public: // non-mutators

#if CUDA_VERSION >= 11030
	struct dot_printing_options_t {
		bool debug_data;
		bool use_runtime_types;
		// TODO: Consider having a map/array mapping of kind_t's to bools
		struct {
			struct {
				bool kernel;
				bool host_function;
			} launch;
			struct {
				bool allocate;
				bool free;
				bool copy;
				bool set;
			} memory_ops;
			bool event;
			struct {
				bool signal;
				bool wait;
			} external_semaphore;
		} node_params;
		bool kernel_node_attributes;
		bool node_and_kernel_function_handles;

		unsigned compose() const {
			return 0u
				| (debug_data                            ? CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE : 0)
				| (use_runtime_types                     ? CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES : 0)
				| (node_params.launch.kernel             ? CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS : 0)
				| (node_params.launch.host_function      ? CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS : 0)
#if CUDA_VERSION >= 11040
				| (node_params.memory_ops.allocate       ? CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS : 0)
				| (node_params.memory_ops.free           ? CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS : 0)
#endif // CUDA_VERSION >= 11040
				| (node_params.memory_ops.copy           ? CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS : 0)
				| (node_params.memory_ops.set            ? CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS : 0)
				| (node_params.event                     ? CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS : 0)
				| (node_params.external_semaphore.signal ? CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS : 0)
				| (node_params.external_semaphore.wait   ? CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS : 0)
				| (kernel_node_attributes                ? CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES : 0)
				| (node_and_kernel_function_handles      ? CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES : 0)
				;
		}
		// TODO: Consider initializing all fields to false in the default constructor
	};
#endif // CUDA_VERSION >= 11030

	/**
	 * @brief Make a copy not just of this wrapper class instance, but of the actual graph template held by the CUDA driver
	 *
	 * @note Copying or copy-assigning a template_t object does _not_ have CUDA make a copy of its internal
	 * representation of the graph; thus, if you write `auto graph2 = graph1`, then add or remove nodes or edges
	 * via `graph2` - they will also be added or removed from the graph referred to by `graph1`.
	 */
	template_t clone() const
	{
		handle_type clone_handle;
		auto status = cuGraphClone(&clone_handle, handle_);
		throw_if_error_lazy(status, "Cloning " + template_::detail_::identify(*this));
		return template_t{ clone_handle, do_take_ownership };
	}

#if CUDA_VERSION >= 11030
	/**
	 * @brief serialize a representation of the structure of this graph template in the GraphViz DOT
	 * format, to a file.
	 *
	 * @param[in] dot_filename Path of the DOT file to serialize the graph template into.
	 * @param[in] printing_options Options passed to the CUDA driver, controlling which aspects of
	 * the graph template representation to include in the DOT and which to skip/suppress.
	 */
	void print_dot(const char* dot_filename, dot_printing_options_t printing_options = {}) const
	{
		auto status = cuGraphDebugDotPrint(handle_, dot_filename, printing_options.compose());
		throw_if_error_lazy(status, "Printing " + template_::detail_::identify(*this) + " to file " + dot_filename);
	}
#endif // CUDA_VERSION >= 11030
	/**
	 * @brief get the number of nodes  in the graph template.
	 */
	size_type num_nodes() const
	{
		::std::size_t num_nodes_;
		auto status = cuGraphGetNodes(handle_, nullptr, &num_nodes_);
		throw_if_error_lazy(status, "Obtaining the number of nodes in " + template_::detail_::identify(*this));
		return num_nodes_;
	}

	/**
	 * @brief returns a container of (wrapped) references to the nodes in the graph template.
	 *
	 * @todo Currently, this makes two heap allocations, then releases one. Instead, we could return
	 * a structure wrapping the node handles, whose operator[], at(), and iterator dereferencing
	 * build (non-owning) @ref node_t wrappers around the handles.
	 */
	node_ref_container_type nodes() const
	{
		size_type num_nodes_ { num_nodes() } ;
		::std::vector<node::handle_t> node_handles {num_nodes_ };
		auto status = cuGraphGetNodes(handle_, node_handles.data(), &num_nodes_);
		throw_if_error_lazy(status, "Obtaining the set of nodes of " + template_::detail_::identify(*this));
		node_ref_container_type node_refs;
		for (const auto& node_handle : node_handles) {
			node_refs.emplace_back(node::wrap(handle_, node_handle));
		}
		return node_refs;
	}

	/**
	 * @brief get the number of root nodes in the graph template, i.e. the nodes with no incoming edges,
	 * i.e. no dependencies on any other nodes.
	 */
	size_type num_roots() const
	{
		// Note: Code duplication with num_nodes()
		::std::size_t num_roots_;
		auto status = cuGraphGetRootNodes(handle_, nullptr, &num_roots_);
		throw_if_error_lazy(status, "Obtaining the number of root nodes in " + template_::detail_::identify(*this));
		return num_roots_;
	}

	/**
	 * @brief get the number of root nodes in the graph template, i.e. the nodes with no incoming edges,
	 * i.e. no dependencies on any other nodes.
	 */
	node_ref_container_type roots() const
	{
		// Note: Code duplication wiuth nodes()
		size_type num_roots_ {num_roots() } ;
		::std::vector<node::handle_t> root_node_handles {num_roots_ };
		auto status = cuGraphGetRootNodes(handle_, root_node_handles.data(), &num_roots_);
		throw_if_error_lazy(status, "Obtaining the set of root nodes of " + template_::detail_::identify(*this));
		node_ref_container_type root_node_refs;
		for (const auto& node_handle : root_node_handles) {
			root_node_refs.emplace_back(node::wrap(handle_, node_handle));
		}
		return root_node_refs;
	}

	/// Get the number (directed) edges, i.e. dependencies between nodes, in the execution graph.
	size_type num_edges() const
	{
		size_type num_edges;
		auto status = cuGraphGetEdges(handle_, nullptr, nullptr, &num_edges);
		throw_if_error_lazy(status, "Obtaining the number of edges in " + template_::detail_::identify(*this));
		return num_edges;
	}

	edge_container_type edges() const
	{
		size_type num_edges_ { num_edges() } ;
		::std::vector<node::handle_t> from_node_handles { num_edges_ };
		::std::vector<node::handle_t> to_node_handles { num_edges_ };
		auto status = cuGraphGetEdges(handle_, from_node_handles.data(), to_node_handles.data(), &num_edges_);
		throw_if_error_lazy(status, "Obtaining the set of edges in " + template_::detail_::identify(*this));
		edge_container_type edges;
		// TODO: Use container/range zipping, and a ranged-for loop
		{
			auto from_iter = from_node_handles.cbegin();
			auto to_iter = from_node_handles.cbegin();
			for (; from_iter != from_node_handles.cend(); from_iter++, to_iter++) {
				assert(to_iter != to_node_handles.cend());
				auto from_node_ref = node::wrap(handle_, *from_iter);
				auto to_node_ref = node::wrap(handle_, *to_iter);
				edges.emplace_back(from_node_ref, to_node_ref);
			}
		}
		return edges;
	}

	/**
	 * @brief A gadget through which nodes are inserted into the graph template
	 *
	 * @note{Consider replacing with a single method which takes a reflection-enum,
	 * e.g. `my_stream.insert(node::type_t::copy, param1, param2)`}
	 *
	 * @note{The runtime API supposedly offers to insert "copy to symbol" and "copy from symbol"
	 * nodes. These are actually just plain 1D memory copies; use
	 * @ref cuda::memory::symbol::locate() to obtain the symbol's address. }
	 */
	class insert_t {
	protected:
		const template_t& associated_template;

		template_::handle_t handle() const noexcept { return associated_template.handle(); }

	public:
		insert_t(const template_t& template_) : associated_template(template_) {}

		void edge(node_ref_type source, node_ref_type dest) const
		{
			struct {
				const node::handle_t source;
				const node::handle_t dest;
			} handles { source.handle(), dest.handle() };
			static constexpr const size_t remove_just_one = 1;
			auto status = cuGraphAddDependencies(
				handle(), &handles.source, &handles.dest, remove_just_one);

			throw_if_error_lazy(status, "Inserting " + node::detail_::identify(edge_type{source, dest})
				+ " into " + template_::detail_::identify(associated_template));
		}

		void edge(edge_type edge_) const
		{
			return edge(edge_.first, edge_.second);
		}

		void edges(span<const node_ref_type> sources, span<const node_ref_type> destinations) const
		{
			if (sources.size() != destinations.size()) {
				throw ::std::invalid_argument(
					"Differing number of source nodes and destination nodes ("
					+ ::std::to_string(sources.size()) + " != " + ::std::to_string(destinations.size())
					+ " in a request to insert edges into " + template_::detail_::identify(associated_template) );
			}
			auto status = template_::detail_::insert_edges(handle(), sources, destinations);

			throw_if_error_lazy(status, "Destroying " + ::std::to_string(sources.size()) + " edges in "
				+ template_::detail_::identify(associated_template));
		}

		void edges(span<const edge_type> edges) const
		{
			auto status = template_::detail_::insert_edges(handle(), edges);

			throw_if_error_lazy(status, "Inserting " + ::std::to_string(edges.size()) + " edges into "
				+ template_::detail_::identify(associated_template));
		}

		template <node::kind_t Kind, typename T, typename... Ts>
		typename node::typed_node_t<Kind> node(
			T&& arg, Ts&&... node_params_ctor_arguments) const
		{
			// Note: arg may be either the first parameters constructor argument, or a context passed
			// before the constructor arguments; due to the lack of C++17's if constexpr, we can only act
			// on this knowledge in another function.
			static constexpr const bool inserter_takes_context = node::detail_::kind_traits<Kind>::inserter_takes_context;
			return template_::detail_::build_params_and_insert_node_wrapper<Kind>(
				cuda::detail_::bool_constant<inserter_takes_context>{}, handle(),
				::std::forward<T>(arg), ::std::forward<Ts>(node_params_ctor_arguments)...);
		}
	}; // insert_t

	/**
	 * @brief A gadget through which nodes are enqueued into the graph template
	 *
	 * @note Consider replacing with a single method which takes a reflection-enum,
	 * e.g. `my_stream.insert(node::type_t::copy, param1, param2)`
	 */
	class delete_t {
	protected:
		const template_t &associated_template;
		handle_type handle() const noexcept { return associated_template.handle(); }

	public:
		delete_t(const template_t &template_) : associated_template(template_) {}

		void node(node_ref_type node) const
		{
			auto status = cuGraphDestroyNode(node.handle());
			throw_if_error_lazy(status, "Deleting " + node::detail_::identify(node)
				+ " in " + template_::detail_::identify(associated_template));
		}

		void edge(edge_type const& edge_) const
		{
			struct {
				const node::handle_t source;
				const node::handle_t dest;
			} handles { edge_.first.handle(), edge_.second.handle() };
			static constexpr const size_t remove_just_one = 1;
			auto status = cuGraphRemoveDependencies(
				handle(), &handles.source, &handles.dest, remove_just_one);

			throw_if_error_lazy(status, "Destroying " + node::detail_::identify(edge_)
										+ " in " + template_::detail_::identify(associated_template));
		}

		void edges(span<const node_ref_type> sources, span<const node_ref_type> destinations) const
		{
			if (sources.size() != destinations.size()) {
				throw ::std::invalid_argument(
					"Differing number of source nodes and destination nodes ("
					+ ::std::to_string(sources.size()) + " != " + ::std::to_string(destinations.size())
					+ " in a request to insert edges into " + template_::detail_::identify(associated_template) );
			}
			auto status = template_::detail_::delete_edges(handle(), sources, destinations);

			throw_if_error_lazy(status, "Destroying " + ::std::to_string(sources.size()) + " edges in "
				+ template_::detail_::identify(associated_template));
		}

		void edges(span<edge_type> edges) const
		{
			auto status = template_::detail_::delete_edges(handle(), edges);

			throw_if_error_lazy(status, "Destroying " + ::std::to_string(edges.size()) + " edges in "
				+ template_::detail_::identify(associated_template));
		}
	}; // delete_t

public: // friendship

	friend template_t template_::wrap(handle_type  handle, bool take_ownership) noexcept;

protected: // constructors
	template_t(handle_type handle, bool owning) noexcept
	: handle_(handle), owning_(owning)
	{ }

public: // ctors & dtor
	template_t(const template_t& other) noexcept = delete;
	template_t(template_t&& other) noexcept : template_t(other.handle_, other.owning_)
	{
		other.owning_ = false;
	}

	~template_t() DESTRUCTOR_EXCEPTION_SPEC
	{
		if (owning_) {
			auto status = cuGraphDestroy(handle_);
#ifdef THROW_IN_DESTRUCTORS
			throw_if_error_lazy(status, "Destroying " + template_::detail_::identify(*this));
#else
			(void) status;
#endif
		}
	}

public: // operators
	template_t& operator=(const template_t&) = delete;
	template_t& operator=(template_t&& other) noexcept
	{
		::std::swap(handle_, other.handle_);
		::std::swap(owning_, other.owning_);
		return *this;
	}

public: // non-mutators
	instance_t instantiate(
#if CUDA_VERSION >= 11040
		bool free_previous_allocations_before_relaunch = false
#endif
#if CUDA_VERSION >= 11700
		, bool use_per_node_priorities = false
#endif
#if CUDA_VERSION >= 12000
		, bool upload_on_instantiation = false
		, bool make_device_launchable = false
#endif
	);

public: // data members
	const insert_t insert { *this };
	const delete_t delete_ { *this };
private: // data members
	// Note: A CUDA graph template is not specific to a context, nor a device!
	template_::handle_t handle_;
	bool owning_;
}; // class template_t

namespace template_ {

inline template_t wrap(handle_t handle, bool take_ownership) noexcept
{
	return { handle, take_ownership };
}

inline template_t create()
{
	constexpr const unsigned flags { 0 };
	handle_t handle;
	auto status = cuGraphCreate(&handle, flags);
	throw_if_error_lazy(status, "Creating a CUDA graph");
	return wrap(handle, do_take_ownership);
}

inline ::std::string identify(const template_t& template_)
{
	return "CUDA execution graph template at " + cuda::detail_::ptr_as_hex(template_.handle());
}

constexpr const ::std::initializer_list<node_t> no_dependencies {};

template <node::kind_t Kind, template <typename> class Container, typename NodeOrHandle, typename... NodeParametersCtorParams>
node::typed_node_t<Kind> insert_node(
	const template_t& graph,
	Container<NodeOrHandle> dependencies,
	NodeParametersCtorParams... node_parameters_ctor_params)
{
	using traits_type = typename node::detail_::kind_traits<Kind>;
	node::parameters_t<Kind> params { ::std::forward<NodeParametersCtorParams>(node_parameters_ctor_params)... };
	auto raw_params = traits_type::marshal(params);
	auto untyped_node = template_::detail_::insert_node(graph.handle(), raw_params, dependencies);
	return node::wrap<Kind>(untyped_node.containing_graph(), untyped_node.handle(), params);
	// Remember: untyped_node is not an owning object, so nothing is released (nor
	// is ownership passed in the returned typed_node
}

} // namespace template_

inline template_t create()
{
	return template_::create();
}


/**
 * @brief Searches for the copy of a graph (template) node in another graph
 * (template), which had originally been cloned from the graph in which the
 * node is situated.
 *
 * @param node A node existing in some (implicit) graph template.
 * @param cloned_graph A CUDA execution graph template, originally cloned from the
 * graph containing @p node
 * @return A reference to the copy of @p node in @p clone, if one exists,
 * or a disengaged optional otherwise.
 *
 * TODO: Implement this in the multi-wrapper impls directory, or just in
 * the dire
 */
inline optional<node_t> find_in_clone(node_t node, const template_t& cloned_graph)
{
	// The find function sets the result to 0 (nullptr) if the input
	// parameters were valid, but the node was not found
	auto search_result = reinterpret_cast<node::handle_t>(0x1);
	auto status = cuGraphNodeFindInClone(&search_result, node.handle(), cloned_graph.handle());
	if (status == cuda::status::invalid_value and search_result != nullptr) {
		return nullopt;
	}
	throw_if_error_lazy(status, "Searching for a copy of " + node::detail_::identify(node) + " in " + template_::detail_::identify(cloned_graph));
	return node::wrap(cloned_graph.handle(), search_result);
}

} // namespace graph

} // namespace cuda

#endif // CUDA_VERSION >= 10000

#endif // CUDA_API_WRAPPERS_GRAPH_TEMPLATE_HPP
