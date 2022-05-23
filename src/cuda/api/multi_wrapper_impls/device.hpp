/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard devices, including primary contexts. Specifically:
 *
 * 1. Functions in the `cuda::device` namespace.
 * 2. Methods of @ref `cuda::device_t` and possibly some relates classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_DEVICE_HPP_
#define MULTI_WRAPPER_IMPLS_DEVICE_HPP_

#include "../device.hpp"
#include "../event.hpp"
#include "../kernel_launch.hpp"
#include "../stream.hpp"
#include "../primary_context.hpp"
#include "../kernel.hpp"
#include "../apriori_compiled_kernel.hpp"
#include "../current_context.hpp"
#include "../current_device.hpp"
#include "../peer_to_peer.hpp"

namespace cuda {

namespace device {

namespace primary_context {

inline bool is_active(const device_t& device)
{
	return detail_::is_active(device.id());
}

inline void destroy(const device_t& device)
{
	auto status = cuDevicePrimaryCtxReset(device.id());
	throw_if_error(status, "Failed destroying/resetting the primary context of device " + ::std::to_string(device.id()));
}

inline primary_context_t get(const device_t& device)
{
	auto pc_handle = detail_::get_handle(device.id(), true);
	return detail_::wrap( device.id(), pc_handle, true);
}

} // namespace primary_context

namespace peer_to_peer {

inline bool can_access(device_t accessor, device_t peer)
{
	return detail_::can_access(accessor.id(), peer.id());
}

inline void enable_access(device_t accessor, device_t peer)
{
	return context::peer_to_peer::enable_access(accessor.primary_context(), peer.primary_context());
}

inline void disable_access(device_t accessor, device_t peer)
{
#ifndef NDEBUG
	if (accessor == peer) {
		throw std::invalid_argument("A device cannot be used as its own peer");
	}
#endif
	context::peer_to_peer::disable_access(accessor.primary_context(), peer.primary_context());
}

inline bool can_access_each_other(device_t first, device_t second)
{
	return can_access(first, second) and can_access(second, first);
}

inline void enable_bidirectional_access(device_t first, device_t second)
{
#ifndef NDEBUG
	if (first == second) {
		throw std::invalid_argument("A device cannot be used as its own peer");
	}
#endif
	context::peer_to_peer::enable_bidirectional_access(first.primary_context(), second.primary_context());
}

inline void disable_bidirectional_access(device_t first, device_t second)
{
#ifndef NDEBUG
	if (first == second) {
		throw std::invalid_argument("A device cannot be used as its own peer");
	}
#endif
	context::peer_to_peer::disable_bidirectional_access(first.primary_context(), second.primary_context());
}

inline attribute_value_t get_attribute(attribute_t attribute, device_t first, device_t second)
{
#ifndef NDEBUG
	if (first == second) {
		throw std::invalid_argument("A device cannot be used as its own peer");
	}
#endif
	return detail_::get_attribute(attribute, first.id(), second.id());
}

} // namespace peer_to_peer

inline stream_t primary_context_t::default_stream() const noexcept
{
	return stream::wrap(device_id_, handle_, stream::default_stream_handle);
}

} // namespace device

// device_t methods

inline stream_t device_t::default_stream(bool hold_primary_context_refcount_unit) const
{
	auto pc = primary_context();
	if (hold_primary_context_refcount_unit) {
		device::primary_context::detail_::increase_refcount(id_);
	}
	return stream::wrap(
		id(), pc.handle(), stream::default_stream_handle,
		do_not_take_ownership, hold_primary_context_refcount_unit);
}

inline stream_t device_t::create_stream(
	bool                will_synchronize_with_default_stream,
	stream::priority_t  priority) const
{
	return stream::create(*this, will_synchronize_with_default_stream, priority);
}

inline device::primary_context_t device_t::primary_context(bool hold_pc_refcount_unit) const
{
	auto pc_handle = primary_context_handle();
	if (hold_pc_refcount_unit) {
		device::primary_context::detail_::increase_refcount(id_);
		// Q: Why increase the refcount here, when `primary_context_handle()`
		//    ensured this has already happened for this object?
		// A: Because an unscoped primary_context_t needs its own refcount
		//    unit (e.g. in case this object gets destructed but the
		//    primary_context_t is still alive.
	}
	return device::primary_context::detail_::wrap(id_, pc_handle, hold_pc_refcount_unit);
}

inline void synchronize(const device_t& device)
{
	auto pc = device.primary_context();
	context::current::detail_::scoped_override_t set_device_for_this_scope(pc.handle());
	context::current::detail_::synchronize(device.id(), pc.handle());
}

template <typename KernelFunction, typename ... KernelParameters>
void device_t::launch(
KernelFunction kernel_function, launch_configuration_t launch_configuration,
KernelParameters ... parameters) const
{
	auto pc = primary_context();
	pc.default_stream().enqueue.kernel_launch(
	kernel_function, launch_configuration, parameters...);
}

inline context_t device_t::create_context(
	context::host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
	bool                                            keep_larger_local_mem_after_resize) const
{
	return context::create(*this, synch_scheduling_policy, keep_larger_local_mem_after_resize);
}

inline event_t device_t::create_event(
	bool          uses_blocking_sync,
	bool          records_timing,
	bool          interprocess)
{
	// The current implementation of event::create is not super-smart,
	// but it's probably not worth it trying to improve just this function
	return event::create(*this, uses_blocking_sync, records_timing, interprocess);
}

namespace detail_ {

template<typename Kernel>
device::primary_context_t get_implicit_primary_context(Kernel)
{
	return device::current::get().primary_context();
}

template<>
inline device::primary_context_t get_implicit_primary_context<kernel_t>(kernel_t kernel)
{
	auto context = kernel.context();
	auto device = context.device();
	auto primary_context = device.primary_context();
	if (context != primary_context) {
		throw std::logic_error("Attempt to launch a kernel associated with a non-primary context without specifying a stream associated with that context.");
	}
	return primary_context;
}

template<>
inline device::primary_context_t get_implicit_primary_context<apriori_compiled_kernel_t>(apriori_compiled_kernel_t kernel)
{
	const kernel_t& kernel_ = kernel;
	return get_implicit_primary_context(kernel_);
}

} // namespace detail_

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_DEVICE_HPP_

