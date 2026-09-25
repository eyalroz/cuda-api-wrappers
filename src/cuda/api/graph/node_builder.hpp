/**
 * @file
 *
 * @brief Convenience classes for construction execution graph nodes
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_NODE_BUILDER_HPP
#define CUDA_API_WRAPPERS_NODE_BUILDER_HPP

#if CUDA_VERSION >= 10000

#include "typed_node.hpp"

namespace cuda_ {

namespace graph {

namespace node {

namespace detail {

inline std::logic_error make_unspec_error(const char *node_type, const char *missing_arg_name)
{
	// Yes, returning it, not throwing it. This is an exception builder function
	return std::logic_error(
		std::string("Attempt to build a CUDA execution graph node of type ") + node_type +
		" without specifying its " + missing_arg_name + " argument");
}

} // namespace detail

template <kind_t Kind>
class typed_builder_t;

class builder_t
{
public:
	template <kind_t Kind>
	typed_builder_t<Kind> kind() { return typed_builder_t<Kind>{}; }
};

// Note: Can't build empty vertices for now

template <>
class typed_builder_t<kind_t::child_graph> {
public:
	static constexpr kind_t kind = kind_t::child_graph;
	using this_type = typed_builder_t<kind>;
	using built_type = typed_node_t<kind>;
	using traits = cuda_::graph::node::detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	params_type params_;

	struct {
		bool template_ { false };
	} was_set; // Yes, this is an ugly alternative to using optionals

	// This wrapper method ensures the builder-ish behavior, i.e. always returning the builder
	// for further work via method invocation.
public:
	params_type& params() noexcept { return params_; }

	this_type& template_(template_t subgraph) {
		params_ = std::move(subgraph);
		was_set.template_ = true;
		return *this;
	}

	CAW_MAYBE_UNUSED built_type build_within(const cuda_::graph::template_t& graph_template)
	{
		if (not was_set.template_) {
			throw detail::make_unspec_error("child graph", "child graph template");
		}
		return graph_template.insert.node<kind>(std::move(params_));
	}
}; // typed_builder_t<kind_t::child_graph>

#if CUDA_VERSION >= 11010

template <>
class typed_builder_t<kind_t::record_event> {
public:
	static constexpr kind_t kind = kind_t::record_event;
	using this_type = typed_builder_t<kind>;
	using built_type = typed_node_t<kind>;
	using traits = cuda_::graph::node::detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	params_type params_;

	struct {
		bool event { false };
	} was_set; // Yes, this is an ugly alternative to using optionals

	// This wrapper method ensures the builder-ish behavior, i.e. always returning the builder
	// for further work via method invocation.

public:
	params_type& params() noexcept { return params_; }

	this_type& event(event_t event) {
		params_ = std::move(event);
		was_set.event = true;
		return *this;
	}

	CAW_MAYBE_UNUSED built_type	build_within(const cuda_::graph::template_t& graph_template)
	{
		if (not was_set.event) {
			throw detail::make_unspec_error("record event", "event");
		}
		return graph_template.insert.node<kind>(std::move(params_));
	}
}; // typed_builder_t<kind_t::record_event>

template <>
class typed_builder_t<kind_t::wait_on_event> {
public:
	static constexpr kind_t kind = kind_t::wait_on_event;
	using this_type = typed_builder_t<kind>;
	using built_type = typed_node_t<kind>;
	using traits = cuda_::graph::node::detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	params_type params_;

	struct {
		bool event { false };
	} was_set; // Yes, this is an ugly alternative to using optionals

public:
	params_type& params() noexcept { return params_; }

	this_type& event(event_t event) {
		params_ = std::move(event);
		was_set.event = true;
		return *this;
	}

	CAW_MAYBE_UNUSED built_type	build_within(const cuda_::graph::template_t& graph_template)
	{
		if (not was_set.event) {
			throw detail::make_unspec_error("wait on event", "event");
		}
		return graph_template.insert.node<kind>(std::move(params_));
	}
}; // typed_builder_t<kind_t::wait_event>

#endif // CUDA_VERSION >= 11010

template <>
class typed_builder_t<kind_t::host_function_call> {
public:
	static constexpr kind_t kind = kind_t::host_function_call;
	using this_type = typed_builder_t<kind>;
	using built_type = typed_node_t<kind>;
	using traits = cuda_::graph::node::detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	params_type params_;

	struct {
		bool function_ptr_set { false };
		bool user_argument_set {false };
	} was_set; // Yes, this is an ugly alternative to using optionals

public:
	params_type& params() noexcept { return params_; }

	this_type& function(stream::callback_t host_callback_function)
	{
		params_.function_ptr = std::move(host_callback_function);
		was_set.function_ptr_set = true;
		return *this;
	}

	this_type& argument(void* callback_argument)
	{
		params_.user_data = callback_argument;
		was_set.user_argument_set = true;
		return *this;
	}

	CAW_MAYBE_UNUSED built_type	build_within(const cuda_::graph::template_t& graph_template)
	{
		if (not was_set.function_ptr_set) {
			throw detail::make_unspec_error("kernel_launch", "host callback function pointer");
		}
		if (not was_set.user_argument_set) {
			throw detail::make_unspec_error("kernel_launch", "user-specified callback function argument");
		}
		return graph_template.insert.node<kind>(params_);
	}
}; // typed_builder_t<kind_t::host_function_call>

template <>
class typed_builder_t<kind_t::kernel_launch> {
public:
	static constexpr kind_t kind = kind_t::kernel_launch;
	using this_type = typed_builder_t<kind>;
	using built_type = typed_node_t<kind>;
	using traits = cuda_::graph::node::detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	params_type params_ {
		kernel_t { kernel::wrap(cuda_::device::id_t(0), nullptr, nullptr) },
		{ 0, 0 },
		{ }
	}; // An ugly way of constructing with invalid junk; see `was_set` below. We could
	   // have possibly used some kind of optional

	struct {
		bool kernel { false };
		bool launch_config { false };
		bool marshalled_arguments { false };
	} was_set; // Yes, this is an ugly alternative to using optionals; but - have
	           // you ever looked at the implementation of optional?...

public:
	params_type& params() noexcept { return params_; }

	this_type& kernel(kernel_t kernel)
	{
		// we can't just make an assignment to the `kernel` field, we have to reassign
		// the whole structure...
		params_ = { std::move(kernel), params_.launch_config, std::move(params_.marshalled_arguments) };
		was_set.kernel = true;
		return *this;
	}

	// Note: There is _no_ member for passing an apriori compiled kernel
	// function and a device, since that would either mean leaking a primary context ref unit,
	// or actually holding on to one in this class, which doesn't make sense. The graph template
	// can't hold a ref unit...

	this_type& launch_configuration(launch_configuration_t launch_config)
	{
		params_.launch_config = std::move(launch_config);
		was_set.launch_config = true;
		return *this;
	}

	this_type& marshalled_arguments(std::vector<void*> argument_ptrs)
	{
		params_.marshalled_arguments = std::move(argument_ptrs);
		was_set.marshalled_arguments = true;
		return *this;
	}

	template <typename... Ts>
	this_type& arguments(Ts&&... args)
	{
		return marshalled_arguments(make_kernel_argument_pointers(std::forward<Ts>(args)...));
	}

	CAW_MAYBE_UNUSED built_type	build_within(const cuda_::graph::template_t& graph_template)
	{
		if (not was_set.kernel) {
			throw detail::make_unspec_error("kernel_launch", "kernel");
		}
		if (not was_set.launch_config) {
			throw detail::make_unspec_error("kernel_launch", "launch configuration");
		}
		if (not was_set.marshalled_arguments) {
			throw detail::make_unspec_error("kernel_launch", "launch arguments");
		}
		return graph_template.insert.node<kind>(params_);
	}
}; // typed_builder_t<kind_t::kernel_launch>

#if CUDA_VERSION >= 11040

template <>
class typed_builder_t<kind_t::memory_allocation> {
public:
	static constexpr kind_t kind = kind_t::memory_allocation;
	using this_type = typed_builder_t<kind>;
	using built_type = typed_node_t<kind>;
	using traits = cuda_::graph::node::detail::kind_traits<kind>;
	using params_type = traits::parameters_type;
	using endpoint_t = cuda_::memory::endpoint_t;

protected:
	params_type params_;

	struct {
		bool device { false };
		bool size_in_bytes {false };
	} was_set; // Yes, this is an ugly alternative to using optionals

public:
	params_type& params() { return params_; }

	CAW_MAYBE_UNUSED built_type	build_within(const cuda_::graph::template_t& graph_template)
	{
		if (not was_set.device) {
			throw detail::make_unspec_error("memory allocation", "device");
		}
		if (not was_set.size_in_bytes) {
			throw detail::make_unspec_error("memory allocation", "allocation size in bytes");
		}
		return graph_template.insert.node<kind>(params_);
	}

	this_type& device(device_t device) {
		params_.first = std::move(device);
		was_set.device = true;
		return *this;
	}
	this_type& size(size_t size) {
		params_.second = size;
		was_set.size_in_bytes = true;
		return *this;
	}
}; // typed_builder_t<kind_t::memory_allocation>

#endif // CUDA_VERSION >= 11040

template <>
class typed_builder_t<kind_t::memory_copy> {
public:
	static constexpr kind_t kind = kind_t::memory_copy;
	using this_type = typed_builder_t<kind>;
	using built_type = typed_node_t<kind>;
	using traits = cuda_::graph::node::detail::kind_traits<kind>;
	using params_type = traits::parameters_type;
	using dimensions_type = params_type::dimensions_type;
	using endpoint_t = cuda_::memory::endpoint_t;
//	static constexpr dimensionality_t num_dimensions = traits::num_dimensions;


protected:
	params_type params_;

public:
	params_type& params() { return params_; }
//	built_type build();
#if __cplusplus >= 201703L
	CAW_MAYBE_UNUSED
#endif
	built_type	build_within(const cuda_::graph::template_t& graph_template)
	{
		// TODO: What about the extent???!!!
		return graph_template.insert.node<kind>(params_);
	}

//	this_type& context(endpoint_t endpoint, const context_t& context) noexcept
//	{
//		params_.set_context(endpoint, context); return *this;
//	}
//
//	this_type& single_context(const context_t& context) noexcept
//	{
//		params_.set_single_context(context); return *this;
//	}

	// Note: This next variadic method should not be necessary considering
	// the one right after it which uses the forwarding idiom; and yet - if we
	// only keep the forwarding-source-method, we get errors.
//	template <typename... Ts>
//	this_type& source(const Ts&... args) {
//		params_.set_source(args...);
//		return *this;
//	}

	template <typename... Ts>
	this_type& source(Ts&&... args) {
		params_.set_source(std::forward<Ts>(args)...);
		return *this;
	}
//
//	template <typename... Ts>
//	this_type& destination(const Ts&... args) {
//      params.set_destination(args...);
//      return *this;
//	}

	template <typename... Ts>
	this_type& destination(Ts&&... args) {
		params_.set_destination(std::forward<Ts>(args)...);
		return *this;
	}

	template <typename... Ts>
	this_type& endpoint(endpoint_t endpoint, Ts&&... args) {
		params_.set_endpoint(endpoint, std::forward<Ts>(args)...);
		return *this;
	}

//	this_type& source_untyped(context::handle_t context_handle, void *ptr, dimensions_type dimensions) noexcept
//	{
//		params_.set_endpoint_untyped(endpoint_t::source, context_handle, ptr, dimensions);
//		return *this;
//	}
//
//	this_type& destination_untyped(context::handle_t context_handle, void *ptr, dimensions_type dimensions) noexcept
//	{
//		params_.set_destination_untyped(context_handle, ptr, dimensions);
//		return *this;
//	}
//
//	this_type& endpoint_untyped(endpoint_t endpoint, context::handle_t context_handle, void *ptr, dimensions_type dimensions) noexcept
//	{
//		params_.set_endpoint_untyped(endpoint_t::source, context_handle, ptr, dimensions);
//		return *this;
//	}

	// TODO: Need a proper builder for copy parameters; otherwise we'll need to implement one here, when it's
	// already half-implemented there... it will need:
	// 1. To sort out context stuff (already done in the copy parameters, but requires explicit setting atm
	// 2. deduce extent when none specified
	// 3. prevent direct manipulation of the parameters (which is currently allowed), so that we can apply logic
	//    such as "has the extent been set?"  etc.
	// 4. set defaults when relevant, e.g. w.r.t. pitches and such
}; // typed_builder_t<kind_t::memory_copy>

template <>
class typed_builder_t<kind_t::memory_set> {
	// Note: Unlike memory_copy, for which the underlying parameter type, CUDA_MEMCPY3D_PEER, is also used
	// in non-graph context - here the only builder functionality is for graph vertex construction; so we don't
	// do any forwarding to a rich parameters class or its own builder.
public:
	static constexpr kind_t kind = kind_t::memory_set;
	using this_type = typed_builder_t<kind>;
	using built_type = typed_node_t<kind>;
	using traits = cuda_::graph::node::detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	params_type params_;
	struct {
		bool region { false };
		bool value_and_width { false };
	} was_set;

	template <typename T>
	void set_width() {
	}

public:
	const params_type& params() { return params_; }

	this_type& region(memory::region_t region) noexcept
	{
		params_.region = region; was_set.region = true;
		return *this;
	}

	template <typename T>
	this_type& value(uint32_t v) noexcept
	{
		static_assert(sizeof(T) <= 4, "Type of value to set is too wide; maximum size is 4");
		static_assert(sizeof(T) != 3, "Size of type to set is not a power of 2");
		static_assert(std::is_trivially_copy_constructible<T>::value, "Only a trivially-constructible value can be used for memset'ing");
		params_.width_in_bytes = sizeof(T);
		switch(sizeof(T)) {
			// TODO: Maybe we should use uint_t<N> template? Maybe use if constexpr with C++17?
		case 1:  params_.value = reinterpret_cast<uint8_t&>(v); break;
		case 2:  params_.value = reinterpret_cast<uint16_t&>(v); break;
		case 4:
		default: params_.value = reinterpret_cast<uint32_t&>(v); break;
		}
		was_set.value_and_width = true;
		return *this;
	}

	CAW_MAYBE_UNUSED built_type	build_within(const cuda_::graph::template_t& graph_template)
	{
		if (not was_set.region) {
			throw detail::make_unspec_error("memory set", "memory region");
		}
		if (not was_set.value_and_width) {
			throw detail::make_unspec_error("memory set", "value to set");
		}
		return graph_template.insert.node<kind>(params_);
	}
}; // typed_builder_t<kind_t::memory_set>

#if CUDA_VERSION >= 11040
template <>
class typed_builder_t<kind_t::memory_free> {
public:
	static constexpr kind_t kind = kind_t::memory_free;
	using this_type = typed_builder_t<kind>;
	using built_type = typed_node_t<kind>;
	using traits = cuda_::graph::node::detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	params_type params_;
	struct {
		bool address { false };
	} was_set;

public:
	const params_type& params() { return params_; }

	this_type& region(void* address) noexcept
	{
		params_ = address; was_set.address = true;
		return *this;
	}

	this_type& region(memory::region_t allocated_region) noexcept { return this->region(allocated_region.data()); }

	CAW_MAYBE_UNUSED built_type	build_within(const cuda_::graph::template_t& graph_template)
	{
		if (not was_set.address) {
			throw detail::make_unspec_error("memory free", "allocated region starting address");
		}
		return graph_template.insert.node<kind>(params_);
	}
}; // typed_builder_t<kind_t::memory_free>

#endif // CUDA_VERSION >= 11040

#if CUDA_VERSION >= 11070
template <>
class typed_builder_t<kind_t::memory_barrier> {
public:
	static constexpr kind_t kind = kind_t::memory_barrier;
	using this_type = typed_builder_t<kind>;
	using built_type = typed_node_t<kind>;
	using traits = cuda_::graph::node::detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	params_type params_;
	struct {
		bool context { false };
		bool barrier_scope { false };
	} was_set;

public:
	const params_type& params() { return params_; }

	this_type& context(context_t context) noexcept
	{
		params_.first = std::move(context);
		was_set.context = true;
		return *this;
	}

	this_type& barrier_scope(memory::barrier_scope_t scope) noexcept
	{
		params_.second = scope; was_set.barrier_scope = true;
		return *this;
	}

	CAW_MAYBE_UNUSED built_type	build_within(const cuda_::graph::template_t& graph_template)
	{
		if (not was_set.context) {
			throw detail::make_unspec_error("memory barrier", "CUDA context");
		}
		if (not was_set.barrier_scope) {
			throw detail::make_unspec_error("memory barrier", "barrier scope");
		}
		return graph_template.insert.node<kind>(params_);
	}
}; // typed_builder_t<kind_t::memory_barrier>

#endif // CUDA_VERSION >= 11070

} // namespace node

} // namespace graph

} // namespace cuda_

#endif // CUDA_VERSION >= 10000

#endif //CUDA_API_WRAPPERS_NODE_BUILDER_HPP
