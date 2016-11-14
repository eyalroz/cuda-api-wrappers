#pragma once
#ifndef CUDA_API_WRAPPERS_PCI_ID_H_
#define CUDA_API_WRAPPERS_PCI_ID_H_

#include "cuda/api/types.h"

#include <string>
#include <istream>

namespace cuda {
namespace device {

/**
 * An aggregate struct for the fields one obtains as device attributes
 * (or in the cudaDeviceProp structure) identifying a device's location
 * on the PCI bus(es) on a system
 */
struct pci_id_t {
	// This is simply what we get in CUDA's cudaDeviceProp structure
	int domain;
	int bus;
	int device;

	operator std::string() const;
	// This is not a ctor so as to maintain the plain-old-structness
	static pci_id_t parse(const std::string& id_str);
	device::id_t cuda_device_id() const;
};

} //namespace device
} // namespace cuda

// TODO: Does this really need to be outside the namespace? I wonder
std::istream& operator>>(std::istream& is, cuda::device::pci_id_t& pci_id);

#endif /* CUDA_API_WRAPPERS_PCI_ID_H_ */
