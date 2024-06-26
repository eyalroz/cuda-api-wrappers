/**
 * @file
 *
 * @brief
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_TYPED_NODE_HPP
#define CUDA_API_WRAPPERS_TYPED_NODE_HPP

#if CUDA_VERSION >= 10000

#include "node.hpp"
#include "../detail/for_each_argument.hpp"
#include "../error.hpp"
#include "../device.hpp"
#include "../event.hpp"
#include "../kernel.hpp"
#include "../launch_configuration.hpp"
#include "../memory_pool.hpp"

#include <vector>
#include <iostream>
#include <utility>
#include <type_traits>


namespace cuda {

namespace graph {

/**
 * Creates a raw structure of pointers to kernel arguments, which may then be used with a
 * @ref kernel_t and @ref launch_config_t to define a CUDA execution graph kernel node
 * @param kernel_arguments references - not pointers - to the kernel arguments.
 * @return A value-type structure containing the pointers; it must be kept alive while
 * the execution graph template (or any of its instances) are alive.
 */
template<typename... Ts>
::std::vector<void*> make_kernel_argument_pointers(const Ts&... kernel_arguments)
{
	// The extra curly brackets mean the vector constructor gets an ::std::initializer_list<void*>;
	// Note that we don't add an extern terminating NULLPTR - the Graph API is different than
	// the traditional kernel launch API in that respect
	return { { const_cast<void *>(reinterpret_cast<const void*>(&kernel_arguments)) ... } };
}

namespace node {

enum class kind_t : ::std::underlying_type<CUgraphNodeType>::type {
	kernel_launch = CU_GRAPH_NODE_TYPE_KERNEL,
	memory_copy = CU_GRAPH_NODE_TYPE_MEMCPY, memcpy = memory_copy,
	memory_set = CU_GRAPH_NODE_TYPE_MEMSET, memset = memory_set,
	host_function_call = CU_GRAPH_NODE_TYPE_HOST,
	child_graph = CU_GRAPH_NODE_TYPE_GRAPH,
	empty = CU_GRAPH_NODE_TYPE_EMPTY,
#if CUDA_VERSION >= 11010
	wait = CU_GRAPH_NODE_TYPE_WAIT_EVENT, wait_on_event = wait,
	event = CU_GRAPH_NODE_TYPE_EVENT_RECORD, record_event = event,
#endif // CUDA_VERSION >= 11010
// Semaphores are not yet supported by the CUDA API wrapper library,
// neither in graphs or outside of them
//
//	signal_semaphore = CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL,
//	wait_on_semaphore = CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT,
#if CUDA_VERSION >= 11040
	memory_allocation = CU_GRAPH_NODE_TYPE_MEM_ALLOC, malloc = memory_allocation,
	memory_free = CU_GRAPH_NODE_TYPE_MEM_FREE, memfree = memory_free,
#endif // CUDA_VERSION >= 11040
#if CUDA_VERSION >= 11070
    // Note: We support just one batch op, but the kind value is for any batch op
	memory_barrier = CU_GRAPH_NODE_TYPE_BATCH_MEM_OP,
#endif
#if CUDA_VERSION >= 12030
	conditional = CU_GRAPH_NODE_TYPE_CONDITIONAL
#endif // CUDA_VERSION >= 12030
};

#if CUDA_VERSION >= 12030
namespace conditional {

using handle_t = CUgraphConditionalHandle;
using kind_t = CUgraphConditionalNodeType;
using default_value_t = unsigned;

namespace detail_ {

using flags_t = unsigned;

inline handle_t create(
	template_::handle_t  graph_template,
	context::handle_t    context_handle,
	default_value_t      default_value)
{
	handle_t result;
	auto status = cuGraphConditionalHandleCreate(
		&result, graph_template, context_handle, default_value, CU_GRAPH_COND_ASSIGN_DEFAULT);
	throw_if_error_lazy(status, "Failed creating a graph conditional");
	return result;
}

} // namespace detail_

struct parameters_t {
	kind_t kind;
	optional<handle_t> handle;
	optional<context::handle_t> context_handle; // Note this must match the context used to create the graph template handle
	optional<template_::handle_t> graph_template_handle;
	optional<default_value_t> default_value; // Ignored in the handle has already been created
};

} // namespace conditional
#endif // CUDA_VERSION >= 12030

namespace detail_ {

// Note: Eventually, when could drop support for CUDA graphs for versions
// before 12.3 - and that would let us drop most of these structures and
// just use cuGraphAddNode, with marshalling of parameters into a
// CUgraphNodeParams.
template<kind_t Kind>
struct kind_traits;

template<>
struct kind_traits<kind_t::empty> {
	static constexpr const auto name = "empty";
#if CUDA_VERSION >= 12030
	using raw_parameters_type = CUgraphNodeParams;
	static constexpr const bool inserter_takes_context = false; // The context is passed within the parameter struct
	static constexpr const bool inserter_takes_params_by_ptr = true;
	using parameters_type = cuda::nullopt_t;

	static constexpr const auto inserter = cuGraphAddNode;
	static constexpr const auto setter = cuGraphNodeSetParams;
	static constexpr const auto getter = nullptr;

	static raw_parameters_type marshal(const parameters_type&) {
		raw_parameters_type raw_params;
		::std::memset(&raw_params, 0, sizeof(raw_parameters_type));
		raw_params.type = static_cast<CUgraphNodeType>(kind_t::empty);
		return raw_params;
	}
#endif // CUDA_VERSION >= 12030
};

template<>
struct kind_traits<kind_t::child_graph> {
	static constexpr const auto name = "child graph";
	using raw_parameters_type = template_::handle_t;
	static constexpr const bool inserter_takes_params_by_ptr = false;
	static constexpr const bool inserter_takes_context = false;
	using parameters_type = template_t;
	static constexpr const auto inserter = cuGraphAddChildGraphNode; // 1 extra param
	// no param setter!
	static constexpr const auto getter = cuGraphChildGraphNodeGetGraph;
#if CUDA_VERSION >= 11010
	static constexpr const auto instance_setter = cuGraphExecChildGraphNodeSetParams;
#endif

	static raw_parameters_type marshal(const parameters_type& params);
};

#if CUDA_VERSION >= 11010
template<>
struct kind_traits<kind_t::record_event> {
	static constexpr const auto name = "record event";
	using raw_parameters_type = event::handle_t;
	static constexpr const bool inserter_takes_context = false;
	static constexpr const bool inserter_takes_params_by_ptr = false;
	using parameters_type = event_t;
	static constexpr const auto inserter = cuGraphAddEventRecordNode; // 1 extra param
	static constexpr const auto setter = cuGraphEventRecordNodeSetEvent;
	static constexpr const auto getter = cuGraphEventRecordNodeGetEvent;
	static constexpr const auto instance_setter = cuGraphExecEventRecordNodeSetEvent;

	static raw_parameters_type marshal(const parameters_type& params)
	{
		return params.handle();
	}
};

template<>
struct kind_traits<kind_t::wait_on_event> {
	static constexpr const auto name = "wait on event";
	using raw_parameters_type = event::handle_t;
	static constexpr const bool inserter_takes_context = false;
	static constexpr const bool inserter_takes_params_by_ptr = false;
	using parameters_type = event_t;
	static constexpr const auto inserter = cuGraphAddEventWaitNode; // 1 extra param
	static constexpr const auto setter = cuGraphEventWaitNodeSetEvent;
	static constexpr const auto getter = cuGraphEventWaitNodeGetEvent;
	static constexpr const auto instance_setter = cuGraphExecEventWaitNodeSetEvent;

	static raw_parameters_type marshal(const parameters_type& params)
	{
		return params.handle();

	}
};
#endif // CUDA_VERSION >= 11010

template<>
struct kind_traits<kind_t::host_function_call> {
	static constexpr const auto name = "host function call";
	using raw_parameters_type = CUDA_HOST_NODE_PARAMS;
	static constexpr const bool inserter_takes_context = false;
	static constexpr const bool inserter_takes_params_by_ptr = true;
	struct parameters_type {
		stream::callback_t function_ptr;
		void* user_data;
	};
	static constexpr const auto inserter = cuGraphAddHostNode; // 1 extra param
	static constexpr const auto setter = cuGraphHostNodeSetParams;
	static constexpr const auto getter = cuGraphHostNodeGetParams;
	static constexpr const auto instance_setter = cuGraphExecHostNodeSetParams;

	static raw_parameters_type marshal(const parameters_type& params)
	{
		return { params.function_ptr, params.user_data };
	}
	static raw_parameters_type marshal(::std::pair<stream::callback_t, void*> param_pair)
	{
		return { param_pair.first, param_pair.second };
	}
};

template<>
struct kind_traits<kind_t::kernel_launch> {
	static constexpr const auto name = "kernel launch";
	using raw_parameters_type = CUDA_KERNEL_NODE_PARAMS;
	static constexpr const bool inserter_takes_context = false;
	static constexpr const bool inserter_takes_params_by_ptr = true;
	struct parameters_type {
		kernel_t kernel;
		launch_configuration_t launch_config;
		::std::vector<void*> marshalled_arguments; // Does _not_ need a nullptr "argument" terminator
	};
	static constexpr const auto inserter = cuGraphAddKernelNode; // 1 extra param
	static constexpr const auto setter = cuGraphKernelNodeSetParams;
	static constexpr const auto getter = cuGraphKernelNodeGetParams;
	static constexpr const auto instance_setter = cuGraphExecKernelNodeSetParams;

	static raw_parameters_type marshal(const parameters_type& params)
	{
		// With C++20, use designated initializers
		raw_parameters_type raw_params;
		// Note: With CUDA versions earlier than 11.0, this next instruction
		// only supports non-apriori-compiled kernels, i.e. ones you compile
		// dynamically or load from a module. To change this,; we woulds need
		// to distinguish the inserter, raw parameters and marshal members
		// between apriori and non-apriori kernels
		raw_params.func = params.kernel.handle();

		raw_params.gridDimX = params.launch_config.dimensions.grid.x;
		raw_params.gridDimY = params.launch_config.dimensions.grid.y;
		raw_params.gridDimZ = params.launch_config.dimensions.grid.z;
		raw_params.blockDimX = params.launch_config.dimensions.block.x;
		raw_params.blockDimY = params.launch_config.dimensions.block.y;
		raw_params.blockDimZ = params.launch_config.dimensions.block.z;
		raw_params.sharedMemBytes = params.launch_config.dynamic_shared_memory_size;
		raw_params.kernelParams = const_cast<decltype(raw_params.kernelParams)>(params.marshalled_arguments.data());
		raw_params.extra = nullptr; // we don't use "extra"
		return raw_params;
	}
};

#if CUDA_VERSION >= 11040
template<>
struct kind_traits<kind_t::memory_allocation> {
	static constexpr const auto name = "memory allocation";
	using raw_parameters_type = CUDA_MEM_ALLOC_NODE_PARAMS;
	static constexpr const bool inserter_takes_context = false;
	static constexpr const bool inserter_takes_params_by_ptr = true;
	using parameters_type = ::std::pair<device_t, size_t>; // for now, assuming no peer access
	// no setter
	static constexpr const auto inserter = cuGraphAddMemAllocNode; // 1 extra param
	// static constexpr const auto setter;
	static constexpr const auto getter = cuGraphMemAllocNodeGetParams;

	static raw_parameters_type marshal(const parameters_type& params)
	{
		static constexpr const auto no_export_handle_kind = memory::pool::shared_handle_kind_t::no_export;
		raw_parameters_type raw_params;
		raw_params.poolProps = memory::pool::detail_::create_raw_properties<no_export_handle_kind>(params.first.id());
		// TODO: DO we need to specify the allocation location in an access descriptor?
		raw_params.accessDescs = nullptr; // for now, assuming no peer access
		raw_params.accessDescCount = 0; // for now, assuming no peer access
		raw_params.bytesize = params.second;
		// TODO: Is the following really necessary to set to 0? Not sure, so let's
		// do it as a precaution
		raw_params.dptr = memory::device::address(nullptr);
		return raw_params;
	}
};
#endif // CUDA_VERSION >= 11040

template<>
struct kind_traits<kind_t::memory_set> {
	// TODO: Consider support for pitched memory; we would probably need mdspans for that...
	static constexpr const auto name = "memory set";
	using raw_parameters_type = CUDA_MEMSET_NODE_PARAMS;
	static constexpr const bool inserter_takes_context = true;
	static constexpr const bool inserter_takes_params_by_ptr = true;
	struct parameters_type {
		memory::region_t region;
		size_t width_in_bytes; // cannot exceed 4
		unsigned value; // must fit inside `width_in_bytes`, i.e. not exceed 2^(width*8) -1
	};
	static constexpr const auto inserter = cuGraphAddMemsetNode; // 2 extra params incl context
	// no setter
	static constexpr const auto getter = cuGraphMemsetNodeGetParams;

	static raw_parameters_type marshal(const parameters_type& params)
	{
		static constexpr const size_t max_width = sizeof(parameters_type::value);
		if (params.width_in_bytes > max_width) {
			throw ::std::invalid_argument("Unsupported memset value width (maximum is " + ::std::to_string(max_width));
		}
		const unsigned long min_overwide_value = 1lu << (params.width_in_bytes * CHAR_BIT);
		if (static_cast<unsigned long>(params.value) >= min_overwide_value) {
			throw ::std::invalid_argument("Memset value exceeds specified width");
		}
		CUDA_MEMSET_NODE_PARAMS raw_params;
		raw_params.dst = memory::device::address(params.region.data());
		// Not using raw_params.pitch since height is 1
		raw_params.height = 1u;
		raw_params.value = params.value;
		raw_params.elementSize = static_cast<decltype(raw_params.elementSize)>(params.width_in_bytes);
		raw_params.width = params.region.size() / params.width_in_bytes;
		return raw_params;
	}
};

#if CUDA_VERSION >= 11040
template<>
struct kind_traits<kind_t::memory_free> {
	static constexpr const auto name = "memory free";
	using raw_parameters_type = CUdeviceptr;
	static constexpr const bool inserter_takes_context = false;
	static constexpr const bool inserter_takes_params_by_ptr = false; // the void* _is_ the parameter
	using parameters_type = void*;
	static constexpr const auto inserter = cuGraphAddMemFreeNode; // 1 extra param
	// no setter
	// static constexpr const auto setter = ;
	static constexpr const auto getter = cuGraphMemFreeNodeGetParams;

	static raw_parameters_type marshal(const parameters_type& params)
	{
		return memory::device::address(params);
	}
};
#endif // CUDA_VERSION >= 11040

#if CUDA_VERSION >= 11070
template<>
struct kind_traits<kind_t::memory_barrier> {
	static constexpr const auto name = "memory barrier";
	using raw_parameters_type = CUDA_BATCH_MEM_OP_NODE_PARAMS;
	static constexpr const bool inserter_takes_context = false;
	static constexpr const bool inserter_takes_params_by_ptr = true;
	using parameters_type = ::std::pair<context_t, cuda::memory::barrier_scope_t>;
	static constexpr const auto inserter = cuGraphAddBatchMemOpNode; // 1 extra param
	static constexpr const auto setter = cuGraphBatchMemOpNodeSetParams;
	static constexpr const auto getter = cuGraphBatchMemOpNodeGetParams;

	static raw_parameters_type marshal(const parameters_type& params)
	{
		auto const & context = params.first;
		raw_parameters_type raw_params;
		raw_params.count = 1;
		raw_params.ctx = context.handle();
		raw_params.flags = 0;
		CUstreamBatchMemOpParams memory_barrier_op;
		memory_barrier_op.operation = CU_STREAM_MEM_OP_BARRIER;
		memory_barrier_op.memoryBarrier.operation = CU_STREAM_MEM_OP_BARRIER;
		auto const & scope = params.second;
		memory_barrier_op.memoryBarrier.flags = static_cast<unsigned>(scope);
		raw_params.paramArray = &memory_barrier_op;
		return raw_params;
	}
};
#endif // CUDA_VERSION >= 11070

template<>
struct kind_traits<kind_t::memory_copy> {
	static constexpr const auto name = "memory copy";
	using raw_parameters_type = CUDA_MEMCPY3D;
	static constexpr const bool inserter_takes_context = true;
	static constexpr const bool inserter_takes_params_by_ptr = true;
	static constexpr const dimensionality_t num_dimensions { 3 };
	using parameters_type = memory::copy_parameters_t<3>;

	static constexpr const auto inserter = cuGraphAddMemcpyNode; // 2 extra params incl. context
	static constexpr const auto setter = cuGraphMemcpyNodeSetParams;
	static constexpr const auto getter = cuGraphMemcpyNodeGetParams;

	static raw_parameters_type marshal(const parameters_type& params) {
		auto& params_ptr = const_cast<parameters_type&>(params);
		// This is quite a nasty bit of voodoo. The thing is, CUDA_MEMCPY3D_PEER is
		// the plain struct serving as the base class of copy_parameters_t<3>; and
		// CUDA_MEMCPY3D and CUDA_MEMCPY3D_PEER are "punnable", i.e. they have the
		// same size and field layout, with the former having some "reserved" fields
		// which are not to be used, that in the latter are used for contexts.
		return reinterpret_cast<CUDA_MEMCPY3D&>(params_ptr);
	}
};

#if CUDA_VERSION >= 12040
template<>
struct kind_traits<kind_t::conditional> {
	static constexpr const auto name = "conditional";
	using raw_parameters_type = CUgraphNodeParams; // within which, CUDA_CONDITIONAL_NODE_PARAMS will be used
	static constexpr const bool inserter_takes_context = false; // The context is passed within the parameter struct
	static constexpr const bool inserter_takes_params_by_ptr = true;
	using parameters_type = conditional::parameters_t;

	static constexpr const auto inserter = cuGraphAddNode;
	static constexpr const auto setter = cuGraphNodeSetParams;
	static constexpr const auto getter = nullptr;

	static raw_parameters_type marshal(const parameters_type& params) {
		raw_parameters_type raw_params;
		::std::memset(&raw_params, 0, sizeof(raw_parameters_type));
		raw_params.type = static_cast<CUgraphNodeType>(kind_t::conditional);
		raw_params.conditional.type = params.kind;
		raw_params.conditional.ctx = params.context_handle ?
									 params.context_handle.value() : context::current::detail_::get_handle();
		if (not params.handle and
		   not (params.graph_template_handle and params.context_handle and params.default_value)) {
			throw ::std::invalid_argument(
				"Conditional node creation parameters specify neither a pre-existing conditional handle, "
				"nor the arguments required for its creation");
		}
		raw_params.conditional.handle = params.handle ? params.handle.value() :
			conditional::detail_::create(
				params.graph_template_handle.value(), params.context_handle.value(), params.default_value.value());
		raw_params.conditional.size = 1;
		return raw_params;
	}
};
#endif // CUDA_VERSION >= 12300

template <kind_t Kind, typename = typename ::std::enable_if<kind_traits<Kind>::inserter_takes_params_by_ptr>::type>
typename kind_traits<Kind>::raw_parameters_type *
maybe_add_ptr(const typename kind_traits<Kind>::raw_parameters_type& raw_params)
{
	return const_cast<typename kind_traits<Kind>::raw_parameters_type *>(&raw_params);
}

template <kind_t Kind, typename = typename ::std::enable_if<not kind_traits<Kind>::inserter_takes_params_by_ptr>::type>
const typename kind_traits<Kind>::raw_parameters_type&
maybe_add_ptr(const typename kind_traits<Kind>::raw_parameters_type& raw_params) { return raw_params; }

} // namespace detail_

template <kind_t Kind> using parameters_t = typename detail_::kind_traits<Kind>::parameters_type;

template <kind_t Kind>
class typed_node_t;

template <kind_t Kind>
typed_node_t<Kind> wrap(template_::handle_t graph_handle, handle_t handle, parameters_t<Kind> parameters) noexcept;

template <kind_t Kind>
class typed_node_t : public node_t {
	using parameters_type = parameters_t<Kind>;
protected:
	using traits = detail_::kind_traits<Kind>;
	using raw_parameters_type = typename traits::raw_parameters_type;
	static constexpr char const * const name = traits::name;

public:
	/**
	 * @return The _cached_ node parameters known by this object since its construction.
	 */
	const parameters_type& parameters() const noexcept
	{
		static_assert(Kind != kind_t::empty, "Empty CUDA graph nodes don't have parameters");
		return params_;
	}

	parameters_type requery_parameters() const
	{
		typename traits::raw_parameters_type raw_params;
		if (traits::param_getter == nullptr) {
			throw cuda::runtime_error(status::named_t::not_supported,
				"Querying parameters is not supported for this kind of node: " + node::detail_::identify(*this));
		}
		auto status = traits::param_getter(handle(), &raw_params);
		throw_if_error_lazy(status, "setting parameters for " + node::detail_::identify(*this));
		params_ = traits::unmarshal(raw_params);
		return params_;
	}

	void set_parameters(parameters_t<Kind> parameters)
	{
		static_assert(Kind != kind_t::empty, "Empty CUDA graph nodes don't have parameters");
		auto marshalled_params = traits::marshal(parameters);
		auto status = traits::param_setter(handle(), &marshalled_params);
		throw_if_error_lazy(status, "setting parameters for " + node::detail_::identify(*this));
	}

public: // friendship
	friend typed_node_t node::wrap<Kind>(template_::handle_t graph_handle, node::handle_t handle, parameters_t<Kind> parameters) noexcept;

protected: // constructors and destructors
	typed_node_t(template_::handle_t graph_template_handle, handle_type handle, parameters_type parameters) noexcept
	: node_t(graph_template_handle, handle), params_(::std::move(parameters)) { }

public:  // constructors and destructors
	typed_node_t(const typed_node_t<Kind>&) = default; // It's a reference type, so copying is not a problem
	typed_node_t(typed_node_t<Kind>&&) noexcept = default; // It's a reference type, so copying is not a problem

	typed_node_t<Kind>& operator=(typed_node_t<Kind> other) noexcept
	{
		node_t::operator=(other);
		params_ = other.params_;
	}

protected: // data members
	mutable parameters_t<Kind> params_;
};

template <kind_t Kind>
typed_node_t<Kind> wrap(template_::handle_t graph_handle, handle_t handle, parameters_t<Kind> parameters) noexcept
{
	return typed_node_t<Kind>{ graph_handle, handle, ::std::move(parameters) };
}

} // namespace node

inline node::parameters_t<node::kind_t::kernel_launch>
make_launch_primed_kernel(
	kernel_t kernel,
	launch_configuration_t launch_config,
	const ::std::vector<void*>& argument_pointers)
{
	return { ::std::move(kernel), ::std::move(launch_config), ::std::move(argument_pointers) };
}

template <typename... KernelParameters>
node::parameters_t<node::kind_t::kernel_launch>
make_launch_primed_kernel(
	kernel_t kernel,
	launch_configuration_t launch_config,
	const KernelParameters&... kernel_arguments)
{
	return {
		::std::move(kernel),
		::std::move(launch_config),
		make_kernel_arg_ptrs(kernel_arguments...)
	};
}

} // namespace graph

} // namespace cuda

#endif // CUDA_VERSION >= 10000

#endif //CUDA_API_WRAPPERS_TYPED_NODE_HPP
