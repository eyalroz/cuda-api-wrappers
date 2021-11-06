/**
 * An example program utilizing most/all calls from the CUDA
 * Runtime API module:
 *
 *   Device Management
 *
 * but does not include the API calls relating to IPC (inter-
 * process communication) - sharing pointers to device memory
 * and events among operating system process. That should be
 * covered by a different program.
 *
 */
#include <cuda/api/device.hpp>
#include <cuda/api/devices.hpp>
#include <cuda/api/error.hpp>
#include <cuda/api/miscellany.hpp>
#include <cuda/api/pci_id_impl.hpp>
#include <cuda/api/peer_to_peer.hpp>

#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <cstdlib>
#include <cassert>

[[noreturn]] void die_(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

namespace tests {

void basics(cuda::device::id_t device_id)
{
	// TODO: cudaChooseDevice

	// Being very cavalier about our command-line arguments here...

	if (cuda::device::count() <= device_id) {
		die_("No CUDA device with ID " + std::to_string(device_id));
	}

	auto device = cuda::device::get(device_id);

	std::cout << "Using CUDA device " << device.name() << " (having device ID " << device.id() << ")\n";

	if (device.id() != device_id) {
		die_("The device's reported ID and the ID for which we created the device differ: "
		+ std::to_string(device.id()) + " !=" +  std::to_string(device_id));
	}

	if (device.id() != device.memory().device_id()) {
		die_("The device's reported ID and the device's memory object's reported device ID differ: "
		+ std::to_string(device.id()) + " !=" +  std::to_string(device.memory().device_id()));
	}
}

void attributes_and_properties()
{
	auto device = cuda::device::current::get();

	auto max_registers_per_block = device.get_attribute(cudaDevAttrMaxRegistersPerBlock);
	std::cout
	<< "Maximum number of registers per block on this device: "
	<< max_registers_per_block << "\n";
	assert(device.properties().regsPerBlock == max_registers_per_block);
}

void pci_bus_id()
{
	auto device = cuda::device::current::get();

	auto pci_id = device.pci_id();
	std::string pci_id_str(pci_id);

	cuda::outstanding_error::ensure_none(cuda::do_clear_errors);

	auto re_obtained_device = cuda::device::get(pci_id_str);
	assert(re_obtained_device == device);

}

void global_memory()
{
	auto device = cuda::device::current::get();

	auto device_global_mem = device.memory();
	auto total_memory = device_global_mem.amount_total();
	auto free_memory = device_global_mem.amount_total();

	std::cout
	<< "Device " << std::to_string(device.id()) << " reports it has:\n"
	<< free_memory << " Bytes free out of " << total_memory << " Bytes total global memory.\n";

	assert(free_memory <= total_memory);
}

// Specific attributes and properties with their own API calls:
// L1/shared mem (CacheConfig), shared memory bank size (SharedMemConfig)
// and stream priority range
void shared_memory()
{
	auto device = cuda::device::current::get();

	std::string cache_preference_names[] = {
		"No preference",
		"Equal L1 and shared memory",
		"Prefer shared memory over L1",
		"Prefer L1 over shared memory",
		};

	auto cache_preference = device.cache_preference();
	std::cout << "The cache preference for device " << device.id() << " is: "
	<< cache_preference_names[(unsigned) cache_preference] << ".\n";

	auto new_cache_preference =
		cache_preference == cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory ?
		cuda::multiprocessor_cache_preference_t::prefer_shared_memory_over_l1 :
		cuda::multiprocessor_cache_preference_t::prefer_l1_over_shared_memory;
	device.set_cache_preference(new_cache_preference);
	cache_preference = device.cache_preference();
	assert(cache_preference == new_cache_preference);

	auto shared_mem_bank_size = device.shared_memory_bank_size();
	shared_mem_bank_size =
		(shared_mem_bank_size == cudaSharedMemBankSizeFourByte) ?
		cudaSharedMemBankSizeEightByte : cudaSharedMemBankSizeFourByte;
	device.set_shared_memory_bank_size(shared_mem_bank_size);

}

void stream_priority_range()
{
	auto device = cuda::device::current::get();

	auto stream_priority_range = device.stream_priority_range();
	if (stream_priority_range.is_trivial()) {
		std::cout << "Device " << device.id() << " does not support stream priorities. "
												 "All streams will have the same (default) priority.\n";
	}
	else {
		std::cout << "Streams on device " << device.id() << " have priorities between "
		<< std::to_string(stream_priority_range.greatest) << " (lowest value, most prioritized) and "
		<< stream_priority_range.least << " (highest value, least prioritized)\n";
		assert(stream_priority_range.least > stream_priority_range.greatest);
	}
}

void resource_limits()
{
	auto device = cuda::device::current::get();

	auto printf_fifo_size = device.get_resource_limit(cudaLimitPrintfFifoSize);
	std::cout << "The printf FIFO size for device " << device.id() << " is " << printf_fifo_size << ".\n";
	decltype(printf_fifo_size) new_printf_fifo_size =
		(printf_fifo_size <= 1024) ?  2 * printf_fifo_size : printf_fifo_size - 512;
	device.set_resource_limit(cudaLimitPrintfFifoSize, new_printf_fifo_size);
	printf_fifo_size = device.get_resource_limit(cudaLimitPrintfFifoSize);
	assert(printf_fifo_size == new_printf_fifo_size);
}

// Flags - yes, they're yet another kind of attribute/property
void flags()
{
	auto device = cuda::device::current::get() ;

	std::cout << "Device " << device.id() << " uses a"
	<< (device.synch_scheduling_policy() ? " synchronous" : "n asynchronous")
	<< " scheduling policy.\n";
	std::cout << "Device " << device.id() << " is set to "
	<< (device.keeping_larger_local_mem_after_resize() ? "keeps" : "discards")
	<< " shared memory allocation after launch.\n";
	std::cout << "Device " << device.id()
	<< " is set " << (device.can_map_host_memory() ? "to allow" : "not to allow")
	<< " pinned mapped memory.\n";
	// TODO: Change the settings as well obtaining them
}

void peer_to_peer(std::pair<cuda::device::id_t,cuda::device::id_t> peer_ids)
{
	// Assumes at least two devices are available

	auto device = cuda::device::get(peer_ids.first);
	// This makes assumptions about the valid IDs and their use
	auto peer = cuda::device::get(peer_ids.second);
	if (device.can_access(peer)) {
		auto atomics_supported_over_link = cuda::device::peer_to_peer::get_attribute(
			cudaDevP2PAttrNativeAtomicSupported, device, peer);
		std::cout
		<< "Native atomics are " << (atomics_supported_over_link ? "" : "not ")
		<< "supported over the link from device " << device.id()
		<< " to device " << peer.id() << ".\n";
		device.disable_access_to(peer);
		// TODO: Try some device-to-device access here, expect an exception
		device.enable_access_to(peer);
		// TODO: Try some device-to-device access here
	}
}

void current_device_manipulation()
{
	auto device_count = cuda::device::count();

	if (device_count > 1) {
		auto device_0 = cuda::device::get(0);
		auto device_1 = cuda::device::get(1);
		cuda::device::current::set(device_0);
		assert(cuda::device::current::get() == device_0);
		assert(cuda::device::current::detail_::get_id() == device_0.id());
		cuda::device::current::set(device_1);
		assert(cuda::device::current::get() == device_1);
		assert(cuda::device::current::detail_::get_id() == device_1.id());
	}

	try {
		cuda::device::current::detail_::set(device_count);
		die_("Should not have been able to set the current device to "
		+ std::to_string(device_count) + " since that's the device count, and "
		+ "the maximum valid ID should be " + std::to_string(device_count - 1)
		+ " (one less)");
	}
	catch(cuda::runtime_error& e) {
		(void) e; // This avoids a spurious warning in MSVC 16.11
		assert(e.code() == cuda::status::invalid_device);
		// We expected to get this exception, just clear it
		cuda::outstanding_error::clear();
	}

	// Iterate over all devices
	// ------------------------

	auto devices = cuda::devices();
	assert(devices.size() == cuda::device::count());
	std::cout << "There are " << devices.size() << " 'elements' in devices().\n";
	std::cout << "Let's count the device IDs... ";
	for(auto device : cuda::devices()) {
		std::cout << (int) device.id() << ' ';
		device.synchronize();
	}
	std::cout << '\n';
}


} // namespace tests

int main(int argc, char **argv)
{
	if (cuda::device::count() == 0) {
		die_("No CUDA devices on this system");
	}

	cuda::device::id_t device_id =  (argc > 1) ?
		std::stoi(argv[1]) : cuda::device::default_device_id;

	tests::basics(device_id);
	tests::attributes_and_properties();
	tests::pci_bus_id();
	tests::global_memory();
	tests::shared_memory();
	tests::stream_priority_range();
	tests::resource_limits();

	if (cuda::devices().size() > 1) {
		auto peer_id = (argc > 2) ?
			std::stoi(argv[2]) : (cuda::device::current::get().id() + 1) % cuda::device::count();

		tests::peer_to_peer({device_id, peer_id});
		tests::current_device_manipulation();
	}

	for (auto device : cuda::devices()) {
		device.synchronize();
		device.reset();
	}

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
