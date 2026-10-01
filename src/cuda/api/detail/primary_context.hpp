/**
 * @file
 *
 * @brief Definitions regrading devices' primary context _not_ including
 * the @ref device::primary_context_t class - separated into this file
 * to prevent circular inclusion dependencies.
 *
 * @note contains the @ref pc_refcount_unit_t class.
 */
#ifndef CUDA_API_WRAPPERS_DETAIL_PRIMARY_CONTEXT_HPP_
#define CUDA_API_WRAPPERS_DETAIL_PRIMARY_CONTEXT_HPP_

#include "token_holder.hpp"

namespace cuda_ {

namespace device {

namespace primary_context {

namespace detail {

struct state_t {
	context::flags_t flags;
	int              is_active; // non-zero value means true
};

inline state_t raw_state(device::id_t device_id)
{
	state_t result;
	auto status = cuDevicePrimaryCtxGetState(device_id, &result.flags, &result.is_active);
	throw_if_error(status, "Failed obtaining the state of the primary context for "
		+ device::detail::identify(device_id));
	// Note: Not sanitizing the flags from having CU_CTX_MAP_HOST set
	return result;
}

inline context::flags_t flags(device::id_t device_id)
{
	return raw_state(device_id).flags & ~CU_CTX_MAP_HOST;
}

inline bool is_active(device::id_t device_id)
{
	return raw_state(device_id).is_active;
}

// We used this wrapper for a one-linear to track PC releases
inline status_t decrease_refcount_nothrow(device::id_t device_id) noexcept
{
	return cuDevicePrimaryCtxRelease(device_id);
}

inline void decrease_refcount(device::id_t device_id)
{
	auto status = decrease_refcount_nothrow(device_id);
	throw_if_error_lazy(status, "Failed releasing the reference to the primary context for " + device::detail::identify(device_id));
}

inline handle_t obtain_and_increase_refcount(device::id_t device_id)
{
	handle_t primary_context_handle;
	auto status = cuDevicePrimaryCtxRetain(&primary_context_handle, device_id);
	throw_if_error_lazy(status,
		"Failed obtaining (and possibly creating, and adding a reference count to) the primary context for "
		+ device::detail::identify(device_id));
	return primary_context_handle;
}

inline void increase_refcount(device::id_t device_id)
{
	obtain_and_increase_refcount(device_id);
}

// Note the refcount semantics here, they're a bit tricky
inline context::handle_t get_handle(device::id_t device_id, bool with_refcount_increase = false)
{
	auto handle = obtain_and_increase_refcount(device_id);
	if (not with_refcount_increase) {
		decrease_refcount(device_id);
	}
	return handle;
}

} // namespace detail

} // namespace primary_context

} // namespace device

namespace detail {

struct release_pc_refcount_helper {
	void operator()(device::id_t device_id) const CAW_DESTRUCTOR_EXCEPTION_SPEC {
#ifdef CAW_THROW_IN_DESTRUCTORS
		device::primary_context::detail::decrease_refcount_nothrow(device_id);
#else
		device::primary_context::detail::decrease_refcount(device_id);
#endif
	}
};

class pc_refcount_unit_t : public token_holder<release_pc_refcount_helper, device::id_t> {
	using handle_type = device::id_t;
	using parent_type = token_holder;
	using parent_type::token_holder;
};

} // namespace detail

} // namespace cuda_

#endif /* CUDA_API_WRAPPERS_DETAIL_PRIMARY_CONTEXT_HPP_ */
