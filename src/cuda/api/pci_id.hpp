/**
 * @file pci_id.hpp
 *
 * @brief iostream-related freestanding operator functions for
 * @ref cuda::device::pci_location_t instances and iostream-related methods of
 * the @ref cuda::device::pci_location_t class.
 *
 * @note This file is split off from {@ref pci_id.h} since
 * it requires inclusions of standard library headers which most of the
 * API wrapper code doesn't.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_PCI_ID_HPP_
#define CUDA_API_WRAPPERS_PCI_ID_HPP_

#include <cuda/api/pci_id.h>
#include <cuda/api/error.hpp>

#include <string>
#include <istream>
#include <ostream>
#include <sstream>

#include <cuda_runtime_api.h>

namespace cuda {
namespace device {

// TODO: Does this really need to be outside the namespace? I wonder
inline std::istream& operator>>(std::istream& is, cuda::device::pci_location_t& pci_id)
{
	auto format_flags(is.flags());
	is >> std::hex;
	is >> pci_id.domain; is.ignore(1); // ignoring a ':'
	is >> pci_id.bus;    is.ignore(1); // ignoring a ':'
	is >> pci_id.device;
	is.flags(format_flags);
	return is;
}

inline std::ostream& operator<<(std::ostream& os, const cuda::device::pci_location_t& pci_id)
{
	auto format_flags(os.flags());
	os << std::hex << pci_id.domain << ':' << pci_id.bus << ':' << pci_id.device;
	os.flags(format_flags);
	return os;
}

inline pci_location_t::operator std::string() const
{
	std::ostringstream oss;
	oss << (*this);
	return oss.str();
}

inline pci_location_t pci_location_t::parse(const std::string& id_str)
{
	std::istringstream iss(id_str);
	pci_location_t id;
	iss >> id;
	return id;
}

/**
 * Obtain the CUDA device id for the device at a specified location on
 * a PCIe bus
 */
inline id_t resolve_id(pci_location_t pci_id)
{
	std::string as_string = pci_id;
	id_t cuda_device_id;
	auto result = cudaDeviceGetByPCIBusId(&cuda_device_id, as_string.c_str());
	throw_if_error(result,
		"Failed obtaining a CUDA device ID corresponding to PCI id " + as_string);
	return cuda_device_id;
}

} //namespace device
} // namespace cuda

#endif /* CUDA_API_WRAPPERS_PCI_ID_HPP_ */
