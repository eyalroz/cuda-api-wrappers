/**
 * @file
 *
 * @brief Definition of a wrapper class for CUDA PCI device ID information.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_PCI_ID_HPP_
#define CUDA_API_WRAPPERS_PCI_ID_HPP_

#include "types.hpp"
#include "error.hpp"

#include <string>

namespace cuda {
namespace device {

/**
 * Location "coordinates" for a CUDA device on a PCIe bus
 *
 * @note can be compiled from individual values from a device's properties;
 * see {@ref properties_t}.
 */
struct pci_location_t {
	/**
	 * The four fields of the PCI configuration space.
	 *
	 * @note Only the first three are actually used/recognized by the CUDA driver, and
	 * when querying a CUDA device for its PCI ID, function will be unused. However - we
	 * have it be able to parse the different common string notations of PCI IDs; see
	 * @url https://wiki.xenproject.org/wiki/Bus:Device.Function_(BDF)_Notation .
	 */
	///@{
	optional<int> domain;
	int bus;
	int device;
	optional<int> function;
	///@}

	operator ::std::string() const;

	/**
	 * Parse a string representation of a device's PCI location.
	 *
	 * @note This is not a ctor so as to maintain the PODness.
	 *
	 * @note There are multiple notations for PCI IDs:
	 *
	 * 	   domain::bus::device.function
	 *	   domain::bus::device
	 *     bus::device.function
	 *
	 * and any of them can be used.
	 */
	static pci_location_t parse(const ::std::string& id_str);

	/// @copydoc parse(const ::std::string& id_str)
	static pci_location_t parse(const char* id_str);
};

namespace detail_ {

/**
 * Obtain a CUDA device id for a PCIe bus device
 *
 * @param pci_id the location on (one of) the PCI bus(es) of
 * the device of interest
 */
inline id_t resolve_id(pci_location_t pci_id)
{
	::std::string as_string { pci_id };
	id_t cuda_device_id;
	auto result = cuDeviceGetByPCIBusId(&cuda_device_id, as_string.c_str());
	throw_if_error_lazy(result,
		"Failed obtaining a CUDA device ID corresponding to PCI id " + as_string);
	return cuda_device_id;
}

} // namespace detail_


} // namespace device
} // namespace cuda

#endif // CUDA_API_WRAPPERS_PCI_ID_HPP_
