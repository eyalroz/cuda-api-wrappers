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

inline bool is_primary(handle_t cc_handle, device::id_t current_context_device_id)
{
	// Note we assume current_context_device_id really is the device ID for cc_handle;
	// otherwise we could just use is_primary_for_device()
	return cc_handle == device::primary_context::detail_::get_handle(current_context_device_id);
}

} // namespace detail_

inline bool is_primary()
{
	auto current_context = get();
	return detail_::is_primary(current_context.handle(), current_context.device_id());
}

namespace detail_ {

inline scoped_override_t::scoped_override_t(bool hold_primary_context_ref_unit, device::id_t device_id, handle_t context_handle)
: hold_primary_context_ref_unit_(hold_primary_context_ref_unit), device_id_or_0_(device_id)
{
	if (hold_primary_context_ref_unit) { device::primary_context::detail_::increase_refcount(device_id); }
	push(context_handle);
}

inline scoped_override_t::~scoped_override_t()
{
	if (hold_primary_context_ref_unit_) { device::primary_context::detail_::decrease_refcount(device_id_or_0_); }
	pop();
}


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
 * as a parameter, and when it may be the case that no context is current. Such API calls
 * are generally supposed to be independent of a specific context; but - CUDA still often
 * expects some context to exist and be current to perform whatever it is we want it to do.
 * It would be unreasonable to create new contexts for the purposes of such calls - as then,
 * the caller would often need to maintain these contexts after the call. Instead, we fall
 * back on a primary context of one of the devices - and since no particular device is
 * specified, we choose that to be the default device. When we do want the caller to keep
 * a context alive - we increase the primary context's refererence count, keeping it alive
 * automatically. In these situations, the ref unit "leaks" past the scope of the ensurer
 * object - but the instantiator would be aware of this, having asked for such behavior
 * explicitly; and would itself carry the onus of decreasing the ref unit at some point.
 *
 * @note See also the simpler @ref cuda::context::current::scoped_ensurer_t ,
 * which takes the context handle to push in the first place.
 */
class scoped_existence_ensurer_t {
public:
	context::handle_t context_handle;
	device::id_t device_id_;
	bool decrease_pc_refcount_on_destruct_;

	explicit scoped_existence_ensurer_t(bool avoid_pc_refcount_increase = true)
		: context_handle(get_handle()),
		  decrease_pc_refcount_on_destruct_(avoid_pc_refcount_increase)
	{
		if (context_handle == context::detail_::none) {
			device_id_ = device::current::detail_::get_id();
			context_handle = device::primary_context::detail_::obtain_and_increase_refcount(device_id_);
			context::current::detail_::push(context_handle);
		}
		else { decrease_pc_refcount_on_destruct_ = false; }
	}

	~scoped_existence_ensurer_t()
	{
	    if (context_handle != context::detail_::none and decrease_pc_refcount_on_destruct_) {
            context::current::detail_::pop();
	        device::primary_context::detail_::decrease_refcount(device_id_);
	    }
	}
};

} // namespace detail_

class scoped_override_t : private detail_::scoped_override_t {
protected:
	using parent = detail_::scoped_override_t;
public:

	explicit scoped_override_t(device::primary_context_t&& primary_context)
		: parent(primary_context.is_owning(), primary_context.device_id(), primary_context.handle()) {}
	explicit scoped_override_t(const context_t& context) : parent(context.handle()) {}
	explicit scoped_override_t(context_t&& context) : parent(context.handle()) {}
	~scoped_override_t() = default;
};


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
	return context::current::detail_::is_primary(handle(), device_id());
}

// Note: The context_t::create_module() member functions are defined in module.hpp,
// for better separation of runtime-origination and driver-originating headers; see
// issue #320 on the issue tracker.

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

