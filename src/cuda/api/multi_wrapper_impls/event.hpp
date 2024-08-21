/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard events. Specifically:
 *
 * 1. Functions in the `cuda::event` namespace.
 * 2. Methods of @ref cuda::event_t and possibly some relates classes.
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
	return event::detail_::create(
		context.device_id(),
		context.handle(),
		do_not_hold_primary_context_refcount_unit,
		uses_blocking_sync,
		records_timing,
		interprocess);
}

inline event_t create(
	const device_t&  device,
	bool             uses_blocking_sync,
	bool             records_timing,
	bool             interprocess)
{
	// While it's possible that the device's primary context is
	// currently active, we have no guarantee that it will not soon
	// become inactive. So, we increase the PC refcount "on behalf"
	// of the stream, to make sure the PC does not de-activate
	//
	// todo: consider having the event wrapper take care of the primary
	//  context refcount.
	//
	auto pc = device.primary_context(do_not_hold_primary_context_refcount_unit);
	CAW_SET_SCOPE_CONTEXT(pc.handle());
	device::primary_context::detail_::increase_refcount(device.id());
	return event::detail_::create_in_current_context(
		device.id(),
		context::current::detail_::get_handle(),
		do_hold_primary_context_refcount_unit,
		uses_blocking_sync, records_timing, interprocess);
}

namespace ipc {

inline handle_t export_(const event_t& event)
{
	return detail_::export_(event.handle());
}

inline event_t import(const context_t& context, const handle_t& event_ipc_handle)
{
	static constexpr const bool do_not_take_ownership { false };
	static constexpr const bool do_not_own_pc_refcount_unit { false };
	return event::wrap(
		context.device_id(),
		context.handle(),
		detail_::import(event_ipc_handle),
		do_not_take_ownership,
		do_not_own_pc_refcount_unit);
}


inline event_t import(const device_t& device, const handle_t& event_ipc_handle)
{
	auto pc = device.primary_context();
	device::primary_context::detail_::increase_refcount(device.id());
	auto handle = detail_::import(event_ipc_handle);
	return event::wrap(
		device.id(), context::current::detail_::get_handle(), handle,
		do_not_take_ownership, do_hold_primary_context_refcount_unit);
}

} // namespace ipc

} // namespace event

inline device_t device_of(const event_t& event)
{
	return cuda::device::get(event.device_id());
}

inline context_t context_of(const event_t& event)
{
	static constexpr const bool dont_take_ownership { false };
	return context::wrap(event.device_id(), event.context_handle(), dont_take_ownership);
}

void record(const event_t& event, const stream_t& stream)
{
#ifndef NDEBUG
    if (stream.context_handle() != event.context_handle()) {
        throw ::std::invalid_argument("Attempt to record an event on a stream in a different context");
    }
#endif
    event::detail_::enqueue(event.context_handle(), stream.handle(), event.handle());
}

inline void fire(const event_t& event, const stream_t& stream)
{
	record(event, stream);
    wait(event);
}

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_EVENT_HPP_

