/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard streams. Specifically:
 *
 * 1. Functions in the `cuda::stream` namespace.
 * 2. Methods of @ref `cuda::stream_t` and possibly some relates classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_STREAM_HPP_
#define MULTI_WRAPPER_IMPLS_STREAM_HPP_

#include "../array.hpp"
#include "../device.hpp"
#include "../event.hpp"
#include "../kernel_launch.hpp"
#include "../pointer.hpp"
#include "../stream.hpp"
#include "../primary_context.hpp"
#include "../kernel.hpp"
#include "../current_context.hpp"
#include "../current_device.hpp"

namespace cuda {

namespace stream {

namespace detail_ {

inline ::std::string identify(const stream_t& stream)
{
	return identify(stream.handle(), stream.context().handle(), stream.device().id());
}

#if CUDA_VERSION >= 9020
inline device::id_t device_id_of(stream::handle_t stream_handle)
{
	return context::detail_::get_device_id(context_handle_of(stream_handle));
}
#endif // CUDA_VERSION >= 9020

inline void record_event_in_current_context(
device::id_t       current_device_id,
context::handle_t  current_context_handle_,
stream::handle_t   stream_handle,
event::handle_t    event_handle)
{
	auto status = cuEventRecord(event_handle, stream_handle);
	throw_if_error(status, "Failed scheduling " + event::detail_::identify(event_handle)
						   + " on " + stream::detail_::identify(stream_handle, current_context_handle_, current_device_id));
}

} // namespace detail_

inline stream_t create(
const device_t&  device,
bool             synchronizes_with_default_stream,
priority_t       priority)
{
	cuda::device::current::detail_::scoped_context_override_t set_device_for_this_scope{device.id()};
	auto stream_handle = detail_::create_in_current_context(synchronizes_with_default_stream, priority);
	return stream::wrap(device.id(), context::current::detail_::get_handle(), stream_handle);
}

inline stream_t create(
const context_t&  context,
bool              synchronizes_with_default_stream,
priority_t        priority)
{
	return detail_::create(context.device_id(), context.handle(), synchronizes_with_default_stream, priority);
}

} // namespace stream

inline void stream_t::enqueue_t::wait(const event_t& event_)
{
	auto device_id = associated_stream.device_id_;
	device::current::detail_::scoped_context_override_t set_device_for_this_scope(device_id);

	// Required by the CUDA runtime API; the flags value is currently unused
	constexpr const unsigned int flags = 0;

	auto status = cuStreamWaitEvent(associated_stream.handle_, event_.handle(), flags);
	throw_if_error(status, "Failed scheduling a wait for " + event::detail_::identify(event_.handle())
						   + " on " + stream::detail_::identify(associated_stream));

}

inline event_t& stream_t::enqueue_t::event(event_t& existing_event)
{
	auto device_id = associated_stream.device_id_;
	auto context_handle = associated_stream.context_handle_;
	auto stream_context_handle_ = associated_stream.context_handle_;
	if (existing_event.context_handle() != stream_context_handle_) {
		throw ::std::invalid_argument(
			"Attempt to enqueue " + event::detail_::identify(existing_event)
			+ " on a stream in a different context: " + stream::detail_::identify(associated_stream));
	}
	context::current::detail_::scoped_ensurer_t ensure_a_context{context_handle};
	stream::detail_::record_event_in_current_context(
		device_id, context_handle, associated_stream.handle_,existing_event.handle());
	return existing_event;
}

inline event_t stream_t::enqueue_t::event(
	bool          uses_blocking_sync,
	bool          records_timing,
	bool          interprocess)
{
	auto context_handle = associated_stream.context_handle_;
	context::current::detail_::scoped_override_t set_device_for_this_scope(context_handle);

	event_t ev { event::detail_::create_in_current_context(
	associated_stream.device_id_, context_handle,
	uses_blocking_sync, records_timing, interprocess) };
	// Note that, at this point, the event is not associated with this enqueue object's stream.
	this->event(ev);
	return ev;
}

inline device_t stream_t::device() const noexcept
{
	return cuda::device::wrap(device_id_);
}

inline context_t stream_t::context() const noexcept
{
	constexpr const bool dont_take_ownership { false };
	return context::wrap(device_id_, context_handle_, dont_take_ownership);
}

#if CUDA_VERSION >= 11000

inline void copy_attributes(const stream_t &dest, const stream_t &src)
{
#ifndef NDEBUG
	if (dest.device() != src.device()) {
		throw ::std::invalid_argument("Attempt to copy attributes between streams on different devices");
	}
	if (dest.context() != src.context()) {
		throw ::std::invalid_argument("Attempt to copy attributes between streams on different contexts");
	}
#endif
	context::current::detail_::scoped_override_t set_device_for_this_scope(dest.context_handle());
	auto status = cuStreamCopyAttributes(dest.handle(), src.handle());
	throw_if_error(status);
}

#endif // CUDA_VERSION >= 11000

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_STREAM_HPP_

