/**
 * @file pci_id.hpp
 *
 * @brief iostream-related freestanding operator functions for
 * @ref cuda::device::pci_location_t instances and iostream-related methods of
 * the @ref cuda::device::pci_location_t class.
 *
 * @note This file is split off from {@ref pci_id.hpp} since
 * it requires inclusions of standard library headers which most of the
 * API wrapper code doesn't.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_PCI_ID_HPP_
#define CUDA_API_WRAPPERS_PCI_ID_HPP_

#include <cuda/api/pci_id.hpp>

#include <string>
#include <istream>
#include <ostream>
#include <sstream>


namespace cuda {
namespace device {

///@cond

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

///@endcond

} //namespace device
} // namespace cuda

#endif // CUDA_API_WRAPPERS_PCI_ID_HPP_
