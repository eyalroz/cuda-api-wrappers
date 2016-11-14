#include "cuda/api/pci_id.hpp"
#include "cuda/api/device.hpp"

#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char **argv)
{
	// TODO: cudaChooseDevice

	cuda::device::id_t device_id = cuda::device::default_device_id;

	if (argc > 1) {
		// Being very cavalier about our command-line arguments here...
		device_id = std::stoi(argv[1]);
		std::cout << "Will work with the CUDA device having ID " << device_id << "\n";
	}
	else {
		std::cout << "Will work with the default CUDA device.\n";
	}

	auto device = cuda::device::get(device_id);

	std::cout << "Have obtained a proxy for device \"" << device.name() << "\" (ID " << device.id() << ")\n";

	std::cout
		<< "The maximum number of registers per block on this device is "
		<< device.get_attribute(cudaDevAttrMaxRegistersPerBlock) << "\n";

	auto pci_id = device.pci_id();
	std::string pci_id_str(pci_id);

	std::cout
		<< "The device's PCI bus ID is " << pci_id << "\n"
		<< "... and if we obtain it via an std::string it should be the same: " << pci_id_str << "\n";

	std::cout << "If we now resolve the device ID using the PCI ID string, we get... " << std::flush;
	auto re_obtained_device = cuda::device::get(pci_id_str);
	std::cout
		<< re_obtained_device.id()
		<< " ("<< (re_obtained_device.id() == device_id ? "" : "NOT ")
		<< "the same device)\n";

	std::string cache_preference_names[] = {
		"No preference",
		"Equal L1 and shared memory",
		"Prefer shared memory over L1",
		"Prefer L1 over shared memory",
	};

	auto cache_preference = device.cache_preference();
	std::cout
		<< "The cache preference for device " << device_id << " is: "
		<< cache_preference_names[(unsigned) cache_preference] << ".\n";

	auto new_cache_preference =
		cache_preference == cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory ?
		cuda::multiprocessor_cache_preference_t::prefer_shared_memory_over_l1 :
		cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory;
	std::cout
		<< "Changing the cache preference to "
		<< cache_preference_names[(unsigned) new_cache_preference] << "... " << std::flush;
	device.set_cache_preference(new_cache_preference);
	std::cout
		<< "done.\n"
		<< "Let's get the cache preference again (we don't cache it);\n";
	cache_preference = device.cache_preference();
	std::cout
		<< "The cache preference for device " << device_id << " is now: "
		<< cache_preference_names[(unsigned) cache_preference] << ".\n";

	if (cache_preference  != new_cache_preference) {
		throw std::logic_error("CUDA device cache preference does not agree "
			"with the value it was previously set to");
	}

	auto printf_fifo_size = device.get_resource_limit(cudaLimitPrintfFifoSize);
	std::cout
		<< "The printf/fprintf FIFO size for device " << device_id << " is "
		<< printf_fifo_size << ".\n";

	decltype(printf_fifo_size) new_printf_fifo_size =
		(printf_fifo_size <= 1024) ?  2 * printf_fifo_size : printf_fifo_size - 512;
	std::cout << "Changing the printf FIFO size to " << new_printf_fifo_size << "... ";
	device.set_resource_limit(cudaLimitPrintfFifoSize, new_printf_fifo_size);
	std::cout
		<< "done.\n"
		<< "Let's get the printf FIFO size again (we don't cache it);\n";
	printf_fifo_size = device.get_resource_limit(cudaLimitPrintfFifoSize);
	std::cout
		<< "The printf/fprintf FIFO size for device " << device_id << " is now "
		<< printf_fifo_size << ".\n";
	if (printf_fifo_size != new_printf_fifo_size) {
		throw std::logic_error("CUDA device resource limit does not agree "
			"with the value it was previously set to");
	}

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
