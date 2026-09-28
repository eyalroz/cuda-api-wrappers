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

// TODO: Can we add empty nodes?
// Note: Builders make (non-owning) _copies_ of wrapper classes.

template <>
class typed_builder_t<kind_t::child_graph> {
public:
	static constexpr auto kind = kind_t::child_graph;
	using this_type = typed_builder_t;
	using built_type = typed_node_t<kind>;
	using traits = detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	optional<template_t> template__;

	// This wrapper method ensures the builder-ish behavior, i.e. always returning the builder
	// for further work via method invocation.
public:
	params_type params() const noexcept { return { *template__ }; }

	this_type& template_(template_t subgraph)
	{
		template__.emplace(std::move(subgraph));
		return *this;
	}

	CAW_MAYBE_UNUSED built_type build_within(const template_t& graph_template) const
	{
		if (not template__) {
			throw detail::make_unspec_error("child graph", "child graph template");
		}
		return graph_template.insert.node<kind>(params());
	}
}; // typed_builder_t<kind_t::child_graph>

#if CUDA_VERSION >= 11010

template <>
class typed_builder_t<kind_t::record_event> {
public:
	static constexpr auto kind = kind_t::record_event;
	using this_type = typed_builder_t;
	using built_type = typed_node_t<kind>;
	using traits = detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	optional_ref<event_t const> event_ {};

	// This wrapper method ensures the builder-ish behavior, i.e. always returning the builder
	// for further work via method invocation.

public:
	params_type params() const noexcept { return { *event_ }; }

	this_type& event(const event_t& event) {
		event_.rebind(event);
		return *this;
	}

	CAW_MAYBE_UNUSED built_type	build_within(const template_t& graph_template) const
	{
		if (not event_) {
			throw detail::make_unspec_error("record event", "event");
		}
		return graph_template.insert.node<kind>(params());
	}
}; // typed_builder_t<kind_t::record_event>

template <>
class typed_builder_t<kind_t::wait_on_event> {
public:
	static constexpr auto kind = kind_t::wait_on_event;
	using this_type = typed_builder_t;
	using built_type = typed_node_t<kind>;
	using traits = detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	optional_ref<const event_t> event_;

public:
	params_type params() const noexcept { return { *event_ }; }

	this_type& event(const event_t& event) {
		event_.rebind(event);
		return *this;
	}

	CAW_MAYBE_UNUSED built_type	build_within(const template_t& graph_template) const
	{
		if (not event_) {
			throw detail::make_unspec_error("wait on event", "event");
		}
		return graph_template.insert.node<kind>(params());
	}
}; // typed_builder_t<kind_t::wait_event>

#endif // CUDA_VERSION >= 11010

template <>
class typed_builder_t<kind_t::host_function_call> {
public:
	static constexpr auto kind = kind_t::host_function_call;
	using this_type = typed_builder_t;
	using built_type = typed_node_t<kind>;
	using traits = detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	optional<stream::callback_t> function_ptr_;
	optional<void*> user_data_;

public:
	params_type params() const noexcept { return { *function_ptr_, *user_data_ }; }

	this_type& function(stream::callback_t host_callback_function)
	{
		function_ptr_ = std::move(host_callback_function);
		return *this;
	}

	this_type& argument(void* callback_argument)
	{
		user_data_ = callback_argument;
		return *this;
	}

	CAW_MAYBE_UNUSED built_type	build_within(const template_t& graph_template) const
	{
		if (not function_ptr_) {
			throw detail::make_unspec_error("kernel_launch", "host callback function pointer");
		}
		if (not user_data_) {
			throw detail::make_unspec_error("kernel_launch", "user-specified callback function argument");
		}
		return graph_template.insert.node<kind>(params());
	}
}; // typed_builder_t<kind_t::host_function_call>

template <>
class typed_builder_t<kind_t::kernel_launch> {
public:
	static constexpr auto kind = kind_t::kernel_launch;
	using this_type = typed_builder_t;
	using built_type = typed_node_t<kind>;
	using traits = detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	optional_ref<const kernel_t> kernel_;
	optional<launch_configuration_t> launch_config_;
	optional<std::vector<void*>> marshalled_arguments_;

public:
	params_type params() const noexcept { return { *kernel_, *launch_config_, *marshalled_arguments_ }; }

	this_type& kernel(const kernel_t& kernel)
	{
		kernel_.rebind(kernel);
		return *this;
	}

	// Note: There is _no_ member for passing an apriori compiled kernel
	// function and a device, since that would either mean leaking a primary context ref unit,
	// or actually holding on to one in this class, which doesn't make sense. The graph template
	// can't hold a ref unit...

	this_type& launch_configuration(launch_configuration_t launch_config)
	{
		launch_config_ = std::move(launch_config);
		return *this;
	}

	this_type& marshalled_arguments(std::vector<void*> argument_ptrs)
	{
		marshalled_arguments_ = std::move(argument_ptrs);
		return *this;
	}

	template <typename... Ts>
	this_type& arguments(Ts&&... args)
	{
		return marshalled_arguments(make_kernel_argument_pointers(std::forward<Ts>(args)...));
	}

	CAW_MAYBE_UNUSED built_type	build_within(const template_t& graph_template) const
	{
		if (not kernel_) {
			throw detail::make_unspec_error("kernel_launch", "kernel");
		}
		if (not launch_config_) {
			throw detail::make_unspec_error("kernel_launch", "launch configuration");
		}
		if (not marshalled_arguments_) {
			throw detail::make_unspec_error("kernel_launch", "launch arguments");
		}
		return graph_template.insert.node<kind>(params());
	}
}; // typed_builder_t<kind_t::kernel_launch>

#if CUDA_VERSION >= 11040

template <>
class typed_builder_t<kind_t::memory_allocation> {
public:
	static constexpr auto kind = kind_t::memory_allocation;
	using this_type = typed_builder_t;
	using built_type = typed_node_t<kind>;
	using traits = detail::kind_traits<kind>;
	using params_type = traits::parameters_type;
	using endpoint_t = memory::endpoint_t;

protected:
	optional_ref<const device_t> device_;
	optional<size_t> size_in_bytes_;

public:
	params_type params() const noexcept { return { *device_, *size_in_bytes_ }; }

	CAW_MAYBE_UNUSED built_type	build_within(const template_t& graph_template) const
	{
		if (not device_) {
			throw detail::make_unspec_error("memory allocation", "device");
		}
		if (not size_in_bytes_) {
			throw detail::make_unspec_error("memory allocation", "allocation size in bytes");
		}
		return graph_template.insert.node<kind>(params());
	}

	this_type& device(const device_t& device) {
		device_.rebind(device);
		return *this;
	}
	this_type& size(size_t size) {
		size_in_bytes_ = size;
		return *this;
	}
}; // typed_builder_t<kind_t::memory_allocation>

#endif // CUDA_VERSION >= 11040

template <>
class typed_builder_t<kind_t::memory_copy> {
public:
	static constexpr auto kind = kind_t::memory_copy;
	using this_type = typed_builder_t;
	using built_type = typed_node_t<kind>;
	using traits = detail::kind_traits<kind>;
	using params_type = traits::parameters_type;
	using dimensions_type = params_type::dimensions_type;
	using endpoint_t = memory::endpoint_t;
//	static constexpr dimensionality_t num_dimensions = traits::num_dimensions;


protected:
	memory::copy_parameters_t<3> copy_params_ {};

public:
	params_type const& params() const { return copy_params_; }

//	built_type build();
#if __cplusplus >= 201703L
	CAW_MAYBE_UNUSED
#endif
	built_type build_within(const template_t& graph_template) const
	{
		// TODO: What about the extent???!!!
		return graph_template.insert.node<kind>(params());
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
		copy_params_.set_source(std::forward<Ts>(args)...);
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
		copy_params_.set_destination(std::forward<Ts>(args)...);
		return *this;
	}

	template <typename... Ts>
	this_type& endpoint(endpoint_t endpoint, Ts&&... args) {
		copy_params_.set_endpoint(endpoint, std::forward<Ts>(args)...);
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
	static constexpr auto kind = kind_t::memory_set;
	using this_type = typed_builder_t;
	using built_type = typed_node_t<kind>;
	using traits = detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	optional<memory::region_t> region_;
	size_t width_ {};
	optional<unsigned> value_;

public:
	params_type params() const { return { *region_, width_, *value_ }; }

	this_type& region(memory::region_t region) noexcept
	{
		region_ = region;
		return *this;
	}

	template <typename T>
	this_type& value(uint32_t v) noexcept(sizeof(unsigned) < 4)
	{
		static_assert(sizeof(T) <= 4, "Type of value to set is too wide; maximum size is 4");
		static_assert(sizeof(T) != 3, "Size of type to set is not a power of 2");
		static_assert(std::is_trivially_copy_constructible<T>::value, "Only a trivially-constructible value can be used for memset'ing");
		width_ = sizeof(T);
		switch(sizeof(T)) {
			// TODO: Maybe we should use uint_t<N> template? Maybe use if constexpr with C++17?
		case 1:  value_ = v & ~uint8_t{0}; break;
		case 2:  value_ = v & ~uint16_t{0}; break;
		case 4:
		default:
			if(v > std::numeric_limits<unsigned>::max()) {
				throw std::invalid_argument("value exceeds the representation ability of unsigned");
			}
			value_ = v; break;
		}
		return *this;
	}

	CAW_MAYBE_UNUSED built_type	build_within(const template_t& graph_template) const
	{
		if (not region_) {
			throw detail::make_unspec_error("memory set", "memory region");
		}
		if (not value_) {
			throw detail::make_unspec_error("memory set", "value to set");
		}
		return graph_template.insert.node<kind>(params());
	}
}; // typed_builder_t<kind_t::memory_set>

#if CUDA_VERSION >= 11040
template <>
class typed_builder_t<kind_t::memory_free> {
public:
	static constexpr auto kind = kind_t::memory_free;
	using this_type = typed_builder_t;
	using built_type = typed_node_t<kind>;
	using traits = detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	optional<void*> ptr_;

public:
	params_type params() { return *ptr_; }

	this_type& region(void* ptr) noexcept
	{
		ptr_ = ptr;
		return *this;
	}

	this_type& region(memory::region_t allocated_region) noexcept { return this->region(allocated_region.data()); }

	CAW_MAYBE_UNUSED built_type	build_within(const template_t& graph_template)
	{
		if (not ptr_) {
			throw detail::make_unspec_error("memory free", "allocated region pointer");
		}
		return graph_template.insert.node<kind>(params());
	}
}; // typed_builder_t<kind_t::memory_free>

#endif // CUDA_VERSION >= 11040

#if CUDA_VERSION >= 11070
template <>
class typed_builder_t<kind_t::memory_barrier> {
public:
	static constexpr auto kind = kind_t::memory_barrier;
	using this_type = typed_builder_t;
	using built_type = typed_node_t<kind>;
	using traits = detail::kind_traits<kind>;
	using params_type = traits::parameters_type;

protected:
	optional_ref<const context_t> context_;
	optional<memory::barrier_scope_t> barrier_scope_;

public:
	params_type params() const { return { *context_, *barrier_scope_ }; }

	this_type& context(const context_t& context) noexcept
	{
		context_.rebind(context);
		return *this;
	}

	this_type& barrier_scope(memory::barrier_scope_t scope) noexcept
	{
		barrier_scope_ = scope;
		return *this;
	}

	CAW_MAYBE_UNUSED built_type	build_within(const template_t& graph_template) const
	{
		if (not context_) {
			throw detail::make_unspec_error("memory barrier", "CUDA context");
		}
		if (not barrier_scope_) {
			throw detail::make_unspec_error("memory barrier", "barrier scope");
		}
		return graph_template.insert.node<kind>(params());
	}
}; // typed_builder_t<kind_t::memory_barrier>

#endif // CUDA_VERSION >= 11070

} // namespace node

} // namespace graph

} // namespace cuda_

#endif // CUDA_VERSION >= 10000

#endif //CUDA_API_WRAPPERS_NODE_BUILDER_HPP
