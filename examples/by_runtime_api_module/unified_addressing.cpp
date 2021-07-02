/**
 * An example program utilizing calls from the CUDA Runtime
 * API module:
 *
 *   Unified Addressing
 *
 * In this program, two processes will each run
 * one kernel, wait for the other process' kernel to
 * complete execution, and inspect each other's kernel's
 * output - in an output buffer that each of them learns
 * about from the other process.
 *
 */
#include <cuda/api/device.hpp>
#include <cuda/api/multi_wrapper_impls.hpp>
#include <cuda/api/pointer.hpp>
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>

[[noreturn]] bool die_(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	cuda::device::id_t device_id =  (argc > 1) ?
		std::stoi(argv[1]) : cuda::device::default_device_id;
	auto device = cuda::device::get(device_id);

	std::cout << "Using CUDA device " << device.name() << " (having device ID " << device.id() << ")" << std::endl;

	static const size_t allocation_size { 1024 };
	auto memory_region = device.memory().allocate(allocation_size);

	auto ptr = cuda::memory::pointer::wrap(memory_region.start());

	std::cout
		<< "Verifying a wrapper for raw pointer " << memory_region.start()
		<< " allocated on the CUDA device." << std::endl;

	switch (ptr.attributes().memory_type()) {
	using namespace cuda::memory;
	case host_memory:         die_("Pointer incorrectly reported to point into host memory"); break;
	case managed_memory:      die_("Pointer incorrectly reported not to point to managed memory"); break;
	case unregistered_memory: die_("Pointer incorrectly reported to point to \"unregistered\" memory"); break;
	case device_memory:       break;
	}
	{
		auto ptr_device = ptr.device();
		auto ptr_device_id = ptr_device.id();
		(ptr_device_id == device_id) or die_(
			"Pointer incorrectly reported as associated with device ID " + std::to_string(ptr_device_id) +
			" rather than " + std::to_string(device_id) + "\n");
	}
	(ptr.get() == memory_region.start()) or die_("Invalid get() output");
	if (ptr.get_for_device() != memory_region.start()) {
		std::stringstream ss;
		ss
			<< "Reported device-side address isn't the address we get from allocation: "
			<< ptr.get_for_device() << " != " << memory_region.start();
		die_(ss.str());
	}
	(ptr.get_for_host() == nullptr) or die_("Unexpected non-nullptr host-side address reported");

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
