/**
 * @file pci_id_impl.hpp
 *
 * @brief iostream-related freestanding operator functions for
 * @ref cuda::device::pci_location_t instances and iostream-related methods of
 * the @ref cuda::device::pci_location_t class.
 *
 * @note This file is split off from {@ref pci_id.hpp} since
 * it requires inclusions of standard library headers which most of the
 * API wrapper code can do without.
 */
#pragma once
#ifndef CUDA_API_WRAPPERS_PCI_ID_HPP_IMPL_
#define CUDA_API_WRAPPERS_PCI_ID_HPP_IMPL_

#include <cuda/api/pci_id.hpp>
#include <string>
#include <istream>
#include <ostream>
#include <sstream>


namespace cuda {
namespace device {

///@cond

inline ::std::istream& operator>>(::std::istream& is, cuda::device::pci_location_t& pci_id)
{
	auto format_flags(is.flags());
	is >> ::std::hex;

	// There are 3 acceptable formats:
	//
	//   domain::bus::device.function
	//   domain::bus::device
	//   bus::device.function
	//
	// so we can't immediately know which field the values go into.
	
	int first_field;
	is >> first_field;
	auto get_colon = [&]() {
		auto c = is.get();
	//	if (c == istream::traits_type::eof() or  ) {
		if (c != ':') {
			throw ::std::invalid_argument("Invalid format of a PCI location for a CUDA device 1");
		}
	};
	get_colon();

	int second_field;
	is >> second_field;
	switch(is.get()) {
	case '.':
		// It's the third format
		pci_id.domain = pci_location_t::unused; // Is this a reasonable choice? I woudld  have liked that...
		pci_id.bus = first_field;
		pci_id.device = second_field;
		is >> pci_id.function;
		if (not is.good()) {
			throw ::std::invalid_argument("Failed parsing PCI location ID for a CUDA device 2");
		}
		break;
	case ':': {
		pci_id.domain = first_field;
		pci_id.bus = second_field;
		is >> pci_id.device;
		if (is.peek() != '.') {
			// It's the second format.
			pci_id.function = pci_location_t::unused; // Is this a reasonable choice? I woudld  have liked that...
			is.flags(format_flags);
			return is;
		}
		else {
			// It's the first format.
			is.get();
			is >> pci_id.function;
			is.flags(format_flags);
			return is;
		}
	}
	}
	is.flags(format_flags);
	throw ::std::invalid_argument("Failed parsing PCI location ID for a CUDA device");
}

inline ::std::ostream& operator<<(::std::ostream& os, const cuda::device::pci_location_t& pci_id)
{
	auto format_flags(os.flags());
	os << ::std::hex;
	if (pci_id.domain != pci_location_t::unused) { os  << pci_id.domain << ':'; }
	os << pci_id.bus << ':' << pci_id.device;
	if (pci_id.function != pci_location_t::unused) { os << '.' << pci_id.function; }
	os.flags(format_flags);
	return os;
}

inline pci_location_t::operator ::std::string() const
{
	::std::ostringstream oss;
	oss << (*this);
	return oss.str();
}

inline pci_location_t pci_location_t::parse(const ::std::string& id_str)
{
	::std::istringstream iss(id_str);
	pci_location_t id;
	iss >> id;
	return id;
}

inline pci_location_t pci_location_t::parse(const char* id_str)
{
	::std::istringstream iss(id_str);
	pci_location_t id;
	iss >> id;
	return id;
}

///@endcond

} //namespace device
} // namespace cuda

#endif // CUDA_API_WRAPPERS_PCI_ID_HPP_IMPL_
