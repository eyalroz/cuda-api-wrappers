/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard events. Specifically:
 *
 * 1. Functions in the `cuda::event` namespace.
 * 2. Methods of @ref `cuda::event_t` and possibly some relates classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_EVENT_HPP_
#define MULTI_WRAPPER_IMPLS_EVENT_HPP_

#include "../device.hpp"
#include "../event.hpp"
#include "../stream.hpp"
#include "../primary_context.hpp"
#include "../virtual_memory.hpp"
#include "../current_context.hpp"
#include "../current_device.hpp"

#include <type_traits>
#include <vector>
#include <algorithm>

namespace cuda {

namespace event {

inline event_t create(
	const context_t&  context,
	bool              uses_blocking_sync,
	bool              records_timing,
	bool              interprocess)
{
	// Yes, we need the ID explicitly even on the current device,
	// because event_t's don't have an implicit device ID.
	return event::detail_::create(context.device_id(), context.handle(), uses_blocking_sync, records_timing, interprocess);
}

inline event_t create(
	device_t&  device,
	bool       uses_blocking_sync,
	bool       records_timing,
	bool       interprocess)
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device.id());
	return event::detail_::create_in_current_context(
		device.id(),
		context::current::detail_::get_handle(),
		uses_blocking_sync, records_timing, interprocess);
}

namespace ipc {

inline handle_t export_(const event_t& event)
{
	return detail_::export_(event.handle());
}

inline event_t import(const context_t& context, const handle_t& event_ipc_handle)
{
	bool do_not_take_ownership { false };
	return event::wrap(context.device_id(), context.handle(), detail_::import(event_ipc_handle), do_not_take_ownership);
}


inline event_t import(const device_t& device, const handle_t& event_ipc_handle)
{
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device.id());
	auto handle = detail_::import(event_ipc_handle);
	return event::wrap(device.id(), context::current::detail_::get_handle(), handle, do_not_take_ownership);
}

} // namespace ipc

} // namespace event

inline device_t event_t::device() const
{
	return cuda::device::get(device_id());
}

inline context_t event_t::context() const
{
	constexpr const bool dont_take_ownership { false };
	return context::wrap(device_id(), context_handle_, dont_take_ownership);
}

inline void event_t::record(const stream_t& stream) const
{
#ifndef NDEBUG
	if (stream.context_handle() != context_handle_) {
		throw std::invalid_argument("Attempt to record an event on a stream in a different context");
	}
#endif
	event::detail_::enqueue(context_handle_, stream.handle(), handle_);
}

inline void event_t::fire(const stream_t& stream) const
{
	record(stream);
	stream.synchronize();
}

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_EVENT_HPP_

