/**
 * @file
 *
 * @brief Implementations requiring the definitions of multiple CUDA entity proxy classes,
 * and which regard contexts. Specifically:
 *
 * 1. Functions in the `cuda::context` namespace.
 * 2. Methods of @ref `cuda::context_t` and possibly some relates classes.
 */
#pragma once
#ifndef MULTI_WRAPPER_IMPLS_CONTEXT_HPP_
#define MULTI_WRAPPER_IMPLS_CONTEXT_HPP_

#include "../device.hpp"
#include "../stream.hpp"
#include "../kernel.hpp"
#include "../module.hpp"
#include "../virtual_memory.hpp"
#include "../current_context.hpp"
#include "../current_device.hpp"
#include "../peer_to_peer.hpp"

namespace cuda {

namespace context {

namespace detail_ {

inline handle_t get_primary_for_same_device(handle_t handle, bool increase_refcount)
{
	auto device_id = get_device_id(handle);
	return device::primary_context::detail_::get_handle(device_id, increase_refcount);
}

inline bool is_primary_for_device(handle_t handle, device::id_t device_id)
{
	auto context_device_id = context::detail_::get_device_id(handle);
	if (context_device_id != device_id) {
		return false;
	}
	constexpr const bool dont_increase_refcount { false };
	auto pc_handle = device::primary_context::detail_::get_handle(device_id, dont_increase_refcount);
	return handle == pc_handle;
}

} // namespace detail

inline bool is_primary(const context_t& context)
{
	return context::detail_::is_primary_for_device(context.handle(), context.device_id());
}

inline void synchronize(const context_t& context)
{
	return detail_::synchronize(context.device_id(), context.handle());
}

namespace current {

namespace detail_ {

/**
 * @todo This function is a bit shady, consider dropping it.s
 */
inline handle_t push_default_if_missing()
{
	auto handle = detail_::get_handle();
	if (handle != context::detail_::none) {
		return handle;
	}
	// TODO: consider using cudaSetDevice here instead
	auto current_device_id = device::current::detail_::get_id();
	auto pc_handle = device::primary_context::detail_::obtain_and_increase_refcount(current_device_id);
	push(pc_handle);
	return pc_handle;
}

/**
 * @note This specialized scope setter is used in API calls which aren't provided a context
 * as a parameter, and when there is no context that's current. Such API calls are necessarily
 * device-related (i.e. runtime-API-ish), and since there is always a current device, we can
 * (and in fact must) fall back on that device's primary context as what the user assumes we
 * would use. In these situations, we must also "leak" that device's primary context, in the
 * sense of adding to its reference count without ever decreasing it again - since the only
 * juncture at which we can decrease it is the scoped context setter's fallback; and if we
 * do that, we will actually trigger the destruction of that primary context. As a consequence,
 * if one _ever_ uses an API wrapper call which relies on this scoped context setter, the only
 * way for them to destroy the primary context is either via @ref device_t::reset() (or
 * manually decreasing the reference count to zero, which supposedly they will not do).
 *
 * @note not sure about how appropriate it is to pop the created primary context off
 */
class scoped_current_device_fallback_t {
public:
	device::id_t device_id_;
	context::handle_t pc_handle_ { context::detail_::none };

	explicit scoped_current_device_fallback_t()
	{
		auto current_context_handle = get_handle();
		if (current_context_handle  == context::detail_::none) {
			device_id_ = device::current::detail_::get_id();
			pc_handle_ = device::primary_context::detail_::obtain_and_increase_refcount(device_id_);
			context::current::detail_::push(pc_handle_);
		}
	}

	~scoped_current_device_fallback_t()
	{
//	    if (pc_handle_ != context::detail_::none) {
//            context::current::detail_::pop();
//	        device::primary_context::detail_::decrease_refcount(device_id_);
//	    }
	}
};

} // namespace detail_

inline scoped_override_t::scoped_override_t(const context_t &context) : parent(context.handle())
{}

inline scoped_override_t::scoped_override_t(context_t &&context) : parent(context.handle())
{}

} // namespace current

inline context_t create_and_push(
device_t                               device,
host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
bool                                   keep_larger_local_mem_after_resize)
{
	auto handle = detail_::create_and_push(device.id(), synch_scheduling_policy, keep_larger_local_mem_after_resize);
	bool take_ownership = true;
	return context::wrap(device.id(), handle, take_ownership);
}

inline context_t create(
device_t                               device,
host_thread_synch_scheduling_policy_t  synch_scheduling_policy,
bool                                   keep_larger_local_mem_after_resize)
{
	auto created = create_and_push(device, synch_scheduling_policy, keep_larger_local_mem_after_resize);
	current::pop();
	return created;
}

namespace peer_to_peer {

inline bool can_access(context_t accessor, context_t peer)
{
	return device::peer_to_peer::detail_::can_access(accessor.device_id(), peer.device_id());
}

inline void enable_access(context_t accessor, context_t peer)
{
	detail_::enable_access(accessor.handle(), peer.handle());
}

inline void disable_access(context_t accessor, context_t peer)
{
	detail_::disable_access(accessor.handle(), peer.handle());
}

inline void enable_bidirectional_access(context_t first, context_t second)
{
	// Note: What happens when first and second are the same context? Or on the same device?
	enable_access(first,  second);
	enable_access(second, first );
}

inline void disable_bidirectional_access(context_t first, context_t second)
{
	// Note: What happens when first and second are the same context? Or on the same device?
	disable_access(first,  second);
	disable_access(second, first );
}


} // namespace peer_to_peer

namespace current {

namespace peer_to_peer {

inline void enable_access_to(const context_t &peer_context)
{
	context::peer_to_peer::detail_::enable_access_to(peer_context.handle());
}

inline void disable_access_to(const context_t &peer_context)
{
	context::peer_to_peer::detail_::disable_access_to(peer_context.handle());
}

} // namespace peer_to_peer

} // namespace current

} // namespace context

inline memory::region_t context_t::global_memory_type::allocate(size_t size_in_bytes)
{
	return cuda::memory::device::detail_::allocate(context_handle_, size_in_bytes);
}

inline device_t context_t::global_memory_type::associated_device() const
{
	return cuda::device::get(device_id_);
}

inline context_t context_t::global_memory_type::associated_context() const
{
	constexpr const bool non_owning { false };
	return cuda::context::wrap(device_id_, context_handle_, non_owning);
}

inline bool context_t::is_primary() const
{
	auto pc_handle = device::primary_context::detail_::obtain_and_increase_refcount(device_id_);
	device::primary_context::detail_::decrease_refcount(device_id_);
	return handle_ == pc_handle;
}

template <typename ContiguousContainer,
cuda::detail_::enable_if_t<detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool>>
module_t context_t::create_module(ContiguousContainer module_data) const
{
	return module::create<ContiguousContainer>(*this, module_data);
}

template <typename ContiguousContainer,
cuda::detail_::enable_if_t<detail_::is_kinda_like_contiguous_container<ContiguousContainer>::value, bool>>
module_t context_t::create_module(ContiguousContainer module_data, link::options_t link_options) const
{
	return module::create<ContiguousContainer>(*this, module_data, link_options);
}

inline void context_t::enable_access_to(const context_t& peer) const
{
	context::peer_to_peer::enable_access(*this, peer);
}

inline void context_t::disable_access_to(const context_t& peer) const
{
	context::peer_to_peer::disable_access(*this, peer);
}

inline device_t context_t::device() const
{
	return device::wrap(device_id_);
}

inline stream_t context_t::create_stream(
bool                will_synchronize_with_default_stream,
stream::priority_t  priority)
{
	return stream::detail_::create(device_id_, handle_, will_synchronize_with_default_stream, priority);
}

} // namespace cuda

#endif // MULTI_WRAPPER_IMPLS_CONTEXT_HPP_

