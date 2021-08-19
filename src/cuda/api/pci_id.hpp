/**
 * @file pci_id.hpp
 *
 * @brief Definition of a wrapper class for CUDA PCI device ID
 * information.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_PCI_ID_CUH_
#define CUDA_API_WRAPPERS_PCI_ID_CUH_

#include <cuda/api/error.hpp>
#include <cuda/common/types.hpp>

#include <cuda_runtime_api.h>

#include <string>

namespace cuda {
namespace device {

/**
 * Location "coordinates" for a CUDA device on a PCIe bus
 *
 * @note can be compiled from individual values from a device's properties;
 * see @ref properties_t
 */
struct pci_location_t {
	// This is simply what we get in CUDA's cudaDeviceProp structure
	int domain;
	int bus;
	int device;
	int function;

	operator ::std::string() const;
	// This is not a ctor so as to maintain the PODness
	static pci_location_t parse(const ::std::string& id_str);
	static pci_location_t parse(const char* id_str);
public:
	static constexpr const int unused { -1 };
		// In lieu of making this class a variant with 3 type combinations.
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
	auto result = cudaDeviceGetByPCIBusId(&cuda_device_id, as_string.c_str());
	throw_if_error(result,
		"Failed obtaining a CUDA device ID corresponding to PCI id " + as_string);
	return cuda_device_id;
}

} // namespace detail_


} // namespace device
} // namespace cuda

#endif // CUDA_API_WRAPPERS_PCI_ID_HPP_
