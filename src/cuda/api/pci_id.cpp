
#include <cuda/api/pci_id.h>
#include <cuda/api/error.hpp>

#include <string>
#include <sstream>

#include <cuda_runtime_api.h>

std::istream& operator>>(std::istream& is, cuda::device::pci_id_t& pci_id)
{
	is >> pci_id.domain; is.ignore(1); // ignoring a ':'
	is >> pci_id.bus;    is.ignore(1); // ignoring a ':'
	is >> pci_id.device;
	return is;
}

namespace cuda {
namespace device {

pci_id_t::operator std::string() const
{
	std::ostringstream oss;
	oss << domain << ':' << bus << ';' << device;
	return oss.str();
}

pci_id_t pci_id_t::parse(const std::string& id_str)
{
	std::istringstream iss(id_str);
	pci_id_t id;
	iss >> id;
	return id;
}

device::id_t pci_id_t::cuda_device_id() const
{
	auto as_string = operator std::string();
	device::id_t cuda_device_id;
	auto result = cudaDeviceGetByPCIBusId(&cuda_device_id, as_string.c_str());
	throw_if_error(result,
		"Failed obtaining a CUDA device ID corresponding to PCI id " + as_string);
	return cuda_device_id;

}

} //namespace device
} // namespace cuda
