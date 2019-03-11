/**
 * @file peer_to_peer.hpp
 *
 * @brief Settings and actions related to the interaction of multiple devices
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_PEER_TO_PEER_HPP_
#define CUDA_API_WRAPPERS_PEER_TO_PEER_HPP_

#include <cuda/api/device.hpp>

namespace cuda {
namespace device {
namespace peer_to_peer {

/**
 * @brief The value of type for all CUDA device "attributes"; see also @ref attribute_t.
 */
using attribute_value_t = int;

/**
 * @brief An identifier of a integral-numeric-value attribute of a CUDA device.
 *
 * @note Somewhat annoyingly, CUDA devices have attributes, properties and flags.
 * Attributes have integral number values; properties have all sorts of values,
 * including arrays and limited-length strings (see
 * @ref cuda::device::properties_t), and flags are either binary or
 * small-finite-domain type fitting into an overall flagss value (see
 * @ref cuda::device_t::flags_t). Flags and properties are obtained all at once,
 * attributes are more one-at-a-time.
 */
using attribute_t = cudaDeviceP2PAttr;

/**
 * Aliases for all CUDA device attributes
 */
enum : std::underlying_type<attribute_t>::type {
		link_performance_rank = cudaDevP2PAttrPerformanceRank, /**< A relative value indicating the performance of the link between two devices */                       //!< link_performance_rank
		access_support = cudaDevP2PAttrAccessSupported, /**< 1 if access is supported, 0 otherwise */                                                                    //!< access_support
		native_atomics_support = cudaDevP2PAttrNativeAtomicSupported /**< 1 if the first device can perform native atomic operations on the second device, 0 otherwise *///!< native_atomics_support
};

/**
 * @brief Determine whether one CUDA device can access the global memory
 * of another CUDA device.
 *
 * @param accessor id of the device interested in making a remote access
 * @param peer id of the device which is to be accessed
 * @return true iff acesss is possible
 */
inline bool can_access(id_t accessor, id_t peer)
{
	int result;
	auto status = cudaDeviceCanAccessPeer(&result, accessor, peer);
	throw_if_error(status,
		"Failed determining whether CUDA device " + std::to_string(accessor) + " can access CUDA device "
			+ std::to_string(peer));
	return (result == 1);
}

/**
 * @brief Determine whether one CUDA device can access the global memory
 * of another CUDA device.
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 * @return true iff acess is possible
 */
template<bool FirstIsAssumedCurrent, bool SecondIsAssumedCurrent>
inline bool can_access(const device_t<FirstIsAssumedCurrent>& accessor, const device_t<SecondIsAssumedCurrent>& peer);

/**
 * @brief Enable access by one CUDA device to the global memory of another
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 */
inline void enable_access(id_t accessor_id, id_t peer_id)
{
	enum
		: unsigned {fixed_flags = 0
	};
	// No flags are supported as of CUDA 8.0
	device::current::scoped_override_t<> set_device_for_this_scope(accessor_id);
	auto status = cudaDeviceEnablePeerAccess(peer_id, fixed_flags);
	throw_if_error(status,
		"Failed enabling access of device " + std::to_string(accessor_id) + " to device " + std::to_string(peer_id));
}

/**
 * @brief Disable access by one CUDA device to the global memory of another
 *
 * @param accessor device interested in making a remote access
 * @param peer device to be accessed
 */
inline void disable_access(id_t accessor_id, id_t peer_id)
{
	device::current::scoped_override_t<> set_device_for_this_scope(accessor_id);
	auto status = cudaDeviceDisablePeerAccess(peer_id);
	throw_if_error(status,
		"Failed disabling access of device " + std::to_string(accessor_id) + " to device " + std::to_string(peer_id));
}

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
	auto status = cudaDeviceGetP2PAttribute(&value, attribute, source, destination);
	throw_if_error(status,
		"Failed obtaining peer-to-peer device attribute for device pair (" + std::to_string(source) + ", "
			+ std::to_string(destination) + ')');
	return value;
}

template<bool FirstIsAssumedCurrent, bool SecondIsAssumedCurrent>
inline attribute_value_t get_attribute(attribute_t attribute, const device_t<FirstIsAssumedCurrent>& source,
	const device_t<SecondIsAssumedCurrent>& destination);

} // namespace peer_to_peer
} // namespace device
} // namespace cuda

#endif // CUDA_API_WRAPPERS_PEER_TO_PEER_HPP_
