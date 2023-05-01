/**
 * @file
 *
 * @brief Settings and actions related to the interaction of multiple devices (adding
 * on those already in @ref device.hpp)
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_PEER_TO_PEER_HPP_
#define CUDA_API_WRAPPERS_PEER_TO_PEER_HPP_

#include "current_context.hpp"

namespace cuda {

namespace device {

namespace peer_to_peer {

// Aliases for all CUDA device attributes

constexpr const attribute_t link_performance_rank = CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK; /// A relative value indicating the performance of the link between two devices
constexpr const attribute_t	access_support = CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED; /// 1 if access is supported, 0 otherwise
constexpr const attribute_t	native_atomics_support = CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED; /// 1 if the first device can perform native atomic operations on the second device, 0 otherwise
#if CUDA_VERSION >= 10000
constexpr const attribute_t	array_access_support = CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED; /// 1 if special array iterpolatory access operations are supported across the link, 0 otherwise
#endif


namespace detail_ {
/**
 * @brief Get one of the numeric attributes for a(n ordered) pair of devices,
 * relating to their interaction
 *
 * @note This is the device-pair equivalent of @ref device_t::get_attribute()
 *
 * @param attribute identifier of the attribute of interest
 * @param source source device
 * @param destination destination device
 * @return the numeric attribute value
 */
inline attribute_value_t get_attribute(attribute_t attribute, id_t source, id_t destination)
{
	attribute_value_t value;
	auto status = cuDeviceGetP2PAttribute(&value, attribute, source, destination);
	throw_if_error_lazy(status, "Failed obtaining peer-to-peer device attribute for device pair ("
		+ ::std::to_string(source) + ", " + ::std::to_string(destination) + ')');
	return value;
}

/**
 * @brief Check whether a device can access another, peer device, subject
 * to access being enabled.
 *
 * @note A true value returned from this function does not mean access is currently
 * _enabled_, i.e. accesses might still fail when this is true if access has not been
 * enabled.
 */
inline bool can_access(const device::id_t accessor, const device::id_t peer)
{
	int result;
	auto status = cuDeviceCanAccessPeer(&result, accessor, peer);
	throw_if_error_lazy(status, "Failed determining whether " + device::detail_::identify(accessor)
		+ " can access " + device::detail_::identify(peer));
	return (result == 1);
}

} // namespace detail_

} // namespace peer_to_peer

} // namespace device

namespace context {

namespace current {

namespace peer_to_peer {

void enable_access_to(const context_t &context, const context_t &peer_context);

void disable_access_to(const context_t &context, const context_t &peer_context);

} // namespace peer_to_peer

} // namespace current

namespace peer_to_peer {

namespace detail_ {

inline void enable_access_to(context::handle_t peer_context)
{
	enum : unsigned {fixed_flags = 0 };
	// No flags are supported as of CUDA 8.0
	auto status = cuCtxEnablePeerAccess(peer_context, fixed_flags);
	throw_if_error_lazy(status, "Failed enabling access to peer " + context::detail_::identify(peer_context));
}

inline void disable_access_to(context::handle_t peer_context)
{
	auto status = cuCtxDisablePeerAccess(peer_context);
	throw_if_error_lazy(status, "Failed disabling access to peer " + context::detail_::identify(peer_context));
}

inline void enable_access(context::handle_t accessor, context::handle_t peer)
{
	context::current::detail_::scoped_override_t set_context_for_this_context(accessor);
	enable_access_to(peer);
}

inline void disable_access(context::handle_t accessor, context::handle_t peer)
{
	context::current::detail_::scoped_override_t set_context_for_this_context(accessor);
	disable_access_to(peer);
}

} // namespace detail_

/**
 * @brief Check if a CUDA context can access the global memory of another CUDA context
 */
bool can_access(context_t accessor, context_t peer);

/**
 * @brief Enable access by one CUDA device to the global memory of another
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 */
void enable_access(context_t accessor, context_t peer);

/**
 * @brief Disable access by one CUDA device to the global memory of another
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 */
void disable_access(context_t accessor, context_t peer);

/**
 * @brief Enable access both by the @p first to the @p second context and the other way around.
 */
void enable_bidirectional_access(context_t first, context_t second);

/**
 * @brief Disable access both by the @p first to the @p second context and the other way around.
 */
void disable_bidirectional_access(context_t first, context_t second);

} // namespace peer_to_peer
} // namespace context

namespace device {

namespace peer_to_peer {

/**
 * @brief Determine whether one CUDA device can access the global memory
 * of another CUDA device.
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 * @return true iff acess is possible
 */
inline bool can_access(device_t accessor, device_t peer);

/**
 * @brief Enable access by one CUDA device to the global memory of another
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 *
 * @todo Consider disabling this, given that access is context-specific
 */
inline void enable_access(device_t accessor, device_t peer);

/**
 * @brief Disable access by one CUDA device to the global memory of another
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 *
 * @todo Consider disabling this, given that access is context-specific
 */
inline void disable_access(device_t accessor, device_t peer);

/**
 * @brief Determine whether two CUDA devices can currently access each other.
 */
inline bool can_access_each_other(device_t first, device_t second);

/**
 * @brief Enable access both by the @p first to the @p second device and the other way around.
 */
inline void enable_bidirectional_access(device_t first, device_t second);

/**
 * @brief Disable access both by the @p first to the @p second device and the other way around.
 */
inline void disable_bidirectional_access(device_t first, device_t second);

/**
 * @brief Get one of the numeric attributes for a(n ordered) pair of devices,
 * relating to their interaction
 *
 * @note This is the device-pair equivalent of @ref device_t::get_attribute()
 *
 * @param attribute identifier of the attribute of interest
 * @param first the device accessing an att
 * @param second destination device
 * @return the numeric attribute value
 */
inline attribute_value_t get_attribute(attribute_t attribute, device_t first, device_t second);

} // namespace peer_to_peer
} // namespace device
} // namespace cuda

#endif // CUDA_API_WRAPPERS_PEER_TO_PEER_HPP_
