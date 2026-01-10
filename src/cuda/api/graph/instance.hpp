/**
 * @file
 *
 * @brief A CUDA execution graph instance wrapper class and some
 * associated definitions.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_INSTANCE_HPP
#define CUDA_API_WRAPPERS_INSTANCE_HPP

#if CUDA_VERSION >= 10000

#include "node.hpp"
#include "identify.hpp"
#include "../types.hpp"

namespace cuda {

class stream_t;

namespace graph {

class instance_t;

///@endcond
class template_t;
///@endcond

namespace instance {

using update_status_t = CUgraphExecUpdateResult;

namespace update_status {

enum named_t : ::std::underlying_type<update_status_t>::type {
	success                                          = CU_GRAPH_EXEC_UPDATE_SUCCESS,
	failure_for_unexpected_reason                    = CU_GRAPH_EXEC_UPDATE_ERROR,
	topology_has_changed                             = CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED,
	node_type_has_changed                            = CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED,
	kernel_node_function_has_changed                 = CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED,
	unsupported_kind_of_parameter_change             = CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED,
	unsupported_aspect_of_node                       = CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED,
#if CUDA_VERSION >= 11020
	unsupported_kind_of_kernel_node_function_change  = CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE,
#if CUDA_VERSION >= 11060
	unsupported_kind_of_node_attributes_change       = CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED,
#endif // CUDA_VERSION >= 11060
#endif // CUDA_VERSION >= 11020
};

constexpr inline bool operator==(const update_status_t &lhs, const named_t &rhs) noexcept { return lhs == static_cast<update_status_t>(rhs); }
constexpr inline bool operator!=(const update_status_t &lhs, const named_t &rhs) noexcept { return lhs != static_cast<update_status_t>(rhs); }
constexpr inline bool operator==(const named_t &lhs, const update_status_t &rhs) noexcept { return static_cast<update_status_t>(lhs) == rhs; }
constexpr inline bool operator!=(const named_t &lhs, const update_status_t &rhs) noexcept { return static_cast<update_status_t>(lhs) != rhs; }

namespace detail_ {

constexpr const char * const descriptions[] = {
	"success",
	"failure for an unexpected reason described in the return value of the function",
	"topology has changed",
	"node type has changed",
	"kernel node function has changed",
	"parameters changed in an unsupported way",
	"something about the node is not supported",
	"unsupported kind of kernel node function change",
	"unsupported kind of node attributes change"
};

inline bool is_node_specific(update_status_t update_status)
{
	return
		update_status != success and
		update_status != failure_for_unexpected_reason and
		update_status != topology_has_changed and
		update_status != unsupported_kind_of_parameter_change;
}

} // namespace detail_

} // namespace update_status

namespace detail_ {

using flags_t = cuuint64_t;

inline const char *describe(instance::update_status_t update_status)
{
	return instance::update_status::detail_::descriptions[update_status];
}

inline ::std::string describe(
	instance::update_status_t  update_status,
	node::handle_t             node_handle,
	template_::handle_t        graph_template_handle);

#if CUDA_VERSION >= 13010
inline id_t get_id(handle_t handle)
{
	id_t id;
	auto status = cuGraphExecGetId(handle, &id);
	throw_if_error_lazy(status, "Getting the local (DOT-printing) ID of " + identify(handle));
	return id;
}
#endif // CUDA_VERSION >= 13010

} // namespace detail_

} // namespace instance


/**
 * @brief enqueues the execution of a execution-graph instance via a stream
 *
 * @note recall that graph execution is not serialized on the stream, i.e. one could say that the execution is not
 * restricted to that single stream; the relation to the stream is that no graph node will be executed before the
 * currently-enqueued work is concluded, and that no further work on the stream will proceed until the graph is fully
 * executed.
 *
 * @note Only one execution graph instance may be executing at a time. One can, however, execute the same graph
 * concurrently multiple times, but creating multiple instances of the same graph template (@ref template_t).
 * It is possible to execute/launch the same instance multiple times _sequentially_.
 *
 * @note Take care to have the graph either free any allocations it makes, or use a graph instantiated with
 * @ref CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH - or the (second) launch of the instance will fail.
 *
 * @note An execution graph instance must be "uploaded" to the GPU before it is executed - i.e. have
 * appropriate physical device resources allocated for it. If this is not done on instantiation, nor
 * explicitly with @ref instance::upload, calling this function will first upload it, then schedule
 * it.
 */
 void launch(const instance_t& instance, const stream_t& stream);

} // namespace graph

/**
 * @brief Determine whether the API call returning the specified status had succeeded
 */
inline ::std::string describe(graph::instance::update_status_t status)
{
	return graph::instance::detail_::describe(status);
}

::std::string describe(graph::instance::update_status_t update_status, optional<graph::node_t> node);

/**
 * @brief Determine whether the API call returning the specified status had succeeded
 */
inline constexpr bool is_success(graph::instance::update_status_t status)
{ 
	return status == graph::instance::update_status::success;
}

/**
 * @brief Determine whether the API call returning the specified status had failed
 */
constexpr bool is_failure(graph::instance::update_status_t status)  { return not is_success(status); }

namespace graph {

namespace instance {

instance_t wrap(template_::handle_t template_handle, handle_t handle, bool  is_owning) noexcept;

namespace detail_ {

::std::string identify(const instance_t &instance);

} // namespace detail_

// TODO: Add support for reporting errors involving edges
class update_failure : public ::std::runtime_error {
public:
	using parent = ::std::runtime_error;

	update_failure(
		update_status_t kind,
		optional<node_t>&& impermissible_node,
		::std::string&& what_arg) noexcept
		:
		parent((what_arg.empty() ? "" : what_arg + ": ") + describe(kind, impermissible_node)),
		kind_(kind),
		impermissible_node_(std::move(impermissible_node))
	{
		// TODO: Ensure the kind needs a node handle IFF a node handle has been provided
	}

	update_failure(update_status_t kind, node_t impermissible_node) noexcept
		: update_failure(kind, optional<node_t>(std::move(impermissible_node)), "")
	{ }

	update_status_t kind() const noexcept { return kind_; }
	node_t impermissible_node() const { return impermissible_node_.value(); }

private:
	update_status_t kind_;
	optional<node_t> impermissible_node_;
};

/**
 * Update the nodes of an execution graph instance with the settings in the nodes
 * of a compatible execution graph template.
 *
 * @param destination An execution graph instance whose node settings are to be updated
 * @param source An execution graph template which is either the one from which @p destination
 * was instantiated, or one that is "topologically identical" to the instance, i.e. has the
 * same types of nodes with the same edges (and no others).
 */
void update(const instance_t& destination, const template_t& source);

namespace detail_ {

template <node::kind_t Kind>
status_t set_node_parameters_nothrow(
	const instance::handle_t instance_handle,
	const node::handle_t node_handle,
	const typename node::detail_::kind_traits<Kind>::raw_parameters_type raw_params)
{
	auto raw_params_maybe_ptr = node::detail_::maybe_add_ptr<Kind>(raw_params);
	return node::detail_::kind_traits<Kind>::instance_setter(instance_handle, node_handle, raw_params_maybe_ptr);
}

} // namespace detail_


template <node::kind_t Kind>
void set_node_parameters(
	const instance_t& instance,
	const node_t& node,
	const node::parameters_t<Kind> parameters);

} // namespace instance

namespace detail_ {

inline void launch_graph_in_current_context(stream::handle_t stream_handle, instance::handle_t graph_instance_handle)
{
	auto status = cuGraphLaunch(graph_instance_handle, stream_handle);
	throw_if_error_lazy(status, "Trying to launch "
		+ instance::detail_::identify(graph_instance_handle) + " on " + stream::detail_::identify(stream_handle));
}

inline void launch(context::handle_t context_handle, stream::handle_t stream_handle, instance::handle_t graph_instance_handle)
{
	context::current::detail_::scoped_override_t set_context_for_this_scope(context_handle);
	launch_graph_in_current_context(stream_handle, graph_instance_handle);
}

} // namespace detail_

class instance_t {
public: // data types
	using handle_type = instance::handle_t;

public: // getters
	template_::handle_t template_handle() const noexcept { return template_handle_; }
	handle_type handle() const noexcept	{ return handle_; }
	bool is_owning() const noexcept { return owning_; }

protected: // constructors
	instance_t(template_::handle_t template_handle, handle_type handle, bool owning) noexcept
	: template_handle_(template_handle), handle_(handle), owning_(owning)
	{ }

public: // constructors & destructor
	instance_t(const instance_t& other) noexcept = delete;

	instance_t(instance_t&& other) noexcept : instance_t(other.template_handle_, other.handle_, other.owning_)
	{
		other.owning_ = false;
	}
	~instance_t() DESTRUCTOR_EXCEPTION_SPEC
	{
		if (owning_) {
			auto status = cuGraphExecDestroy(handle_);
#if THROW_IN_DESTRUCTORS
			throw_if_error_lazy(status, "Destroying " + instance::detail_::identify(*this));
#else
			(void) status;
#endif
		}
	}

public: // operators
	instance_t& operator=(const instance_t&) = delete;
	instance_t& operator=(instance_t&& other) noexcept
	{
		::std::swap(template_handle_, other.template_handle_);
		::std::swap(handle_, other.handle_);
		::std::swap(owning_, other.owning_);
		return *this;
	}


public: // friends
	friend instance_t instance::wrap(template_::handle_t template_handle, handle_type handle, bool  is_owning) noexcept;

public: // non-mutators
	void update(const template_t& update_source) const
	{
		instance::update(*this, update_source);
	}

	void launch(const stream_t& stream) const
	{
		graph::launch(*this, stream);
	}
#if CUDA_VERSION >= 11010
	void upload(const stream_t& stream) const;
#endif // CUDA_VERSION >= 11010

#if CUDA_VERSION >= 12000
	bool frees_allocations_before_relaunch() const
	{
		instance::detail_::flags_t flags;
		auto status = cuGraphExecGetFlags (handle_, &flags);
		throw_if_error_lazy(status, "Obtaining execution graph instance flags");
		return flags & CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
	}

	bool uses_node_priorities() const
	{
		instance::detail_::flags_t flags;
		auto status = cuGraphExecGetFlags (handle_, &flags);
		throw_if_error_lazy(status, "Obtaining execution graph instance flags");
		return flags & CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY;
	}

#endif

#if CUDA_VERSION >= 13010
	/// Get the 'local' ID of this graph instance, corresponding to the
	/// ID one would find in the output of DOT printing the instance.
	id_t get_id() const
	{
		return instance::detail_::get_id(handle_);
	}
#endif // CUDA_VERSION >= 13010

	template <node::kind_t Kind>
	void set_node_parameters(const node_t& node, node::parameters_t<Kind> new_parameters)
	{
		instance::set_node_parameters<Kind>(*this, node, ::std::move(new_parameters));
	}

	template <node::kind_t Kind>
	void set_node_parameters(const node::typed_node_t<Kind>& node)
	{
		instance::set_node_parameters<Kind>(*this, node);
	}

protected:
	template_::handle_t template_handle_;
	handle_type handle_;
	bool owning_;
};

/**
 * @brief Have a GPU reserve resource and maintain a "copy" of an execution graph instance, allowing
 * it to actually be schedule for execution.
 *
 * @note The upload can happen in one of three forms:
 *
 *   1. on creating of the graph instance (@ref instance_t);
 *   2. Explicitly with this function
 *   3. Implicitly, if @ref launch is called before the graph instance was otherwise uploaded -
 *      before the actual scheduling of the instance for execution via the stream
 *
 */
void upload(const instance_t& instance, const stream_t& stream);

namespace instance {

inline instance_t wrap(template_::handle_t template_handle, handle_t handle, bool  is_owning) noexcept
{
	return instance_t{template_handle, handle, is_owning};
}

enum : bool {
	do_free_previous_allocations_before_relaunch    = true,
	auto_free                                       = true,
	dont_free_previous_allocations_before_relaunch  = false,
	no_auto_free                                    = false,
#if CUDA_VERSION >= 12000

	do_upload_on_instantiation                      = true,
	dont_upload_on_instantiation                    = false,
	auto_upload                                     = true,
	no_auto_upload                                  = false,
	manual_upload                                   = false,

	make_launchable_from_device_code                = true,
	dont_make_launchable_from_device_code           = true,
	do_make_device_launchable                       = true,
	dont_make_device_launchable                     = false,
#endif // CUDA_VERSION >= 12000
#if CUDA_VERSION >= 11700

	do_use_per_node_priorities                      = true,
	do_use_per_node_priority                        = true,
	dont_use_per_node_priorities                    = false,
	dont_use_per_node_priority                      = true,
	use_stream_priority                             = false
#endif // CUDA_VERSION >= 11700
};

namespace detail_ {

#if CUDA_VERSION >= 11040
inline flags_t build_flags(
	bool free_previous_allocations_before_relaunch
#if CUDA_VERSION >= 12000
	, bool upload_on_instantiation
	, bool make_device_launchable
#endif // CUDA_VERSION >= 12000
#if CUDA_VERSION >= 11700
	, bool use_per_node_priorities
#endif
	)
{
	return
		(free_previous_allocations_before_relaunch ? CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH : 0)
#if CUDA_VERSION >= 12000
		| (upload_on_instantiation ? CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD : 0)
		| (make_device_launchable  ? CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH : 0)
#endif
#if CUDA_VERSION >= 11700
		| (use_per_node_priorities ? CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY : 0)
#endif
		;
}
#endif // CUDA_VERSION >= 11040

inline ::std::string identify(const instance_t& instance)
{
	return identify(instance.handle()) + " instantiated from "
		+ template_::detail_::identify(instance.template_handle());
}

inline ::std::string identify(const instance_t& instance, const template_t& template_)
{
	return identify(instance.handle()) + " instantiated from "
	   + template_::detail_::identify(template_);
}

} // namespace detail_

template <node::kind_t Kind>
void set_node_parameters(
	const instance_t&          instance,
	const node_t&              node,
	node::parameters_t<Kind>   parameters)
{
	auto status = detail_::set_node_parameters_nothrow<Kind>(
		instance.handle(), node.handle(), node::detail_::kind_traits<Kind>::marshal(parameters));
	throw_if_error_lazy(status, "Setting parameters of " + node::detail_::identify(node)
		+ " in " + instance::detail_::identify(instance));
}


template <node::kind_t Kind>
void set_node_parameters(
	const instance_t&                instance,
	const node::typed_node_t<Kind>&  node_with_new_params)
{
	return set_node_parameters<Kind>(
		instance, static_cast<node_t&>(node_with_new_params), node_with_new_params.parameters());
}


} // namespace instance

inline instance_t instantiate(
	const template_t& template_
#if CUDA_VERSION >= 11040
	, bool free_previous_allocations_before_relaunch = false
#endif
#if CUDA_VERSION >= 12000
	, bool upload_on_instantiation = false
	, bool make_device_launchable = false
#endif
#if CUDA_VERSION >= 11700
	, bool use_per_node_priorities = false
#endif
)
{
#if CUDA_VERSION >= 11040
	instance::detail_::flags_t flags = instance::detail_::build_flags(
		free_previous_allocations_before_relaunch
#if CUDA_VERSION >= 12000
		, upload_on_instantiation, make_device_launchable
#endif
#if CUDA_VERSION >= 11700
		, use_per_node_priorities
#endif
	);
#endif // CUDA_VERSION >= 11040
	instance::handle_t instance_handle;
#if CUDA_VERSION >= 11040
	auto status = cuGraphInstantiateWithFlags(&instance_handle, template_.handle(), flags);
	throw_if_error_lazy(status, "Instantiating " + template_::detail_::identify(template_) );
#else
	static constexpr const size_t log_buffer_size { 2048 };
	auto log_buffer = make_unique_span<char>(log_buffer_size);
	node::handle_t error_node;
	auto status = cuGraphInstantiate(&instance_handle, template_.handle(), &error_node, log_buffer.data(), log_buffer_size);
	throw_if_error_lazy(status, "Instantiating " + template_::detail_::identify(template_) + ": error at "
		+ node::detail_::identify(error_node) + " ; log buffer contents:\n" + log_buffer.data());
#endif // CUDA_VERSION >= 11000
	static constexpr const bool is_owning { true };
	return instance::wrap(template_.handle(), instance_handle, is_owning);
}

void launch(const cuda::stream_t& stream, const instance_t& instance);

} // namespace graph

} // namespace cuda

#endif // CUDA_VERSION >= 10000

#endif //CUDA_API_WRAPPERS_INSTANCE_HPP
