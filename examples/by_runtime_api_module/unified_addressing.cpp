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
 * about from the other process
 *
 * TODO: Mostly unimplemented for now.
 *
 */
#include "../common.hpp"

#include <sstream>
#include <cstdlib>

namespace tests {

void pointer_properties(const cuda::device_t& device)
{
	constexpr const cuda::size_t fixed_size { 123 };
	cuda::context_t contexts[2] = {
		cuda::context::create(device),
		cuda::context::create(device)
	};
	cuda::memory::device::unique_ptr<char[]> regions[2] = {
		cuda::memory::device::make_unique<char[]>(contexts[0], fixed_size),
		cuda::memory::device::make_unique<char[]>(contexts[1], fixed_size)
	};
	void* raw_pointers[2] = {
		regions[0].get(),
		regions[1].get()
	};
	cuda::memory::pointer_t<void> pointers[2] = {
		cuda::memory::pointer::wrap(raw_pointers[0]),
		cuda::memory::pointer::wrap(raw_pointers[1]),
	};
	auto primary_context = device.primary_context();
	cuda::context::current::push(primary_context); // so that we check from a different context
	for(size_t i = 0; i < 2; i++) {
		auto reported_device_id = cuda::memory::pointer::detail_::get_attribute<CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL>(raw_pointers[i]);
		assert_(reported_device_id == device.id());
		auto context_handle = cuda::memory::pointer::detail_::get_attribute<CU_POINTER_ATTRIBUTE_CONTEXT>(raw_pointers[i]);
		assert_(context_handle == contexts[i].handle());
		auto ptr_mem_type = cuda::memory::type_of(raw_pointers[i]);
		assert_(ptr_mem_type == cuda::memory::type_t::device_ or ptr_mem_type == cuda::memory::type_t::unified_);
		if (i == 0) {
			std::cout << "The memory type reported for pointers to memory allocated on the device is: " << memory_type_name(ptr_mem_type) << "\n";
		}
		assert_(pointers[i].get_for_device() == raw_pointers[i]);
		try {
			[[maybe_unused]] auto host_ptr = pointers[i].get_for_host();
			die_("Was expecting the host_ptr() method to fail for a device-side pointer");
		} catch(cuda::runtime_error& e) {
			if (e.code() != cuda::status::named_t::invalid_value) {
				throw e;
			}
		}
		auto ptr_reported_as_managed = cuda::memory::pointer::detail_::get_attribute<CU_POINTER_ATTRIBUTE_IS_MANAGED>(raw_pointers[i]);
		assert_(ptr_reported_as_managed == 0);
//		auto ptr_reported_as_mapped = cuda::memory::pointer::detail_::get_attribute<CU_POINTER_ATTRIBUTE_MAPPED>(raw_pointers[i]);
//		assert_(ptr_reported_as_mapped == 0);
#if CUDA_VERSION >= 11030
		auto mempool_handle = cuda::memory::pointer::detail_::get_attribute<CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE>(raw_pointers[i]);
		assert_(mempool_handle == nullptr);
#endif
		auto raw_offset_ptr = cuda::memory::as_pointer(cuda::memory::device::address(raw_pointers[i]) + 17);

		cuda::memory::region_t range  = pointers[i].containing_range();

		cuda::memory::pointer_t<void> offset_ptr { raw_offset_ptr };
		cuda::memory::region_t range_for_offset_ptr = offset_ptr.containing_range();
		assert_(range == range_for_offset_ptr);
		assert_(range_for_offset_ptr.start() == raw_pointers[i]);
//		std::cout << "range_for_offset_ptr.start() == " << range_for_offset_ptr.start() << '\n';
//		std::cout << "range_for_offset_ptr.size() == " << range_for_offset_ptr.size() << '\n';
//		std::cout << "offset_ptr == " << offset_ptr.get() << '\n';

		// Consider testing:
		//	CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE
		//	CU_POINTER_ATTRIBUTE_MAPPED
		//	CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES
		//	CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
		//	CU_POINTER_ATTRIBUTE_ACCESS_FLAGS

	}

}

void wrapped_pointers_and_regions(const cuda::device_t& device)
{
	static const size_t allocation_size { 1024 };
	auto memory_region = device.memory().allocate(allocation_size);

	auto ptr = cuda::memory::pointer::wrap(memory_region.start());

	std::cout
		<< "Verifying a wrapper for raw pointer " << memory_region.start()
		<< " allocated on the CUDA device." << std::endl;

	switch (cuda::memory::type_of(ptr)) {
	using namespace cuda::memory;
	case host_:         die_("Pointer incorrectly reported to point into host memory"); break;
	case array:         die_("Pointer incorrectly reported to point to array memory"); break;
//	case unregistered_memory: die_("Pointer incorrectly reported to point to \"unregistered\" memory"); break;
	case unified_:       std::cout << "Allocated global-device-memory pointer reported to be of unified memory type.";
		// die_("Pointer incorrectly reported not to point to managed memory"); break;
	case device_:       break;
	}
	{
		auto ptr_device = ptr.device();
		auto ptr_device_id = ptr_device.id();
		(ptr_device_id == device.id()) or die_(
			"Pointer incorrectly reported as associated with " + cuda::device::detail_::identify(device.id())
			+ " rather than + " + cuda::device::detail_::identify(device.id()));
	}
	(ptr.get() == memory_region.start()) or die_("Invalid get() output");
	if (ptr.get_for_device() != memory_region.start()) {
		std::stringstream ss;
		ss
			<< "Reported device-side address isn't the address we get from allocation: "
			<< ptr.get_for_device() << " != " << memory_region.start();
		die_(ss.str());
	}
	try {
		auto host_side_ptr = ptr.get_for_host();
		std::stringstream ss;
		ss << "Unexpected success getting a host-side pointer for a device-only allocation; allocated pointer: "
				<< ptr.get() << ", " << " host-side pointer: " << host_side_ptr;
	}
	catch(cuda::runtime_error& e) {
		if (e.code() != cuda::status::invalid_value) { throw e; }
	}
}

} // namespace tests

int main(int argc, char **argv)
{
	cuda::device::id_t device_id =  (argc > 1) ?
		std::stoi(argv[1]) : cuda::device::default_device_id;
	auto device = cuda::device::get(device_id);

	std::cout << "Using CUDA device " << device.name() << " (having device ID " << device.id() << ")" << std::endl;

	tests::wrapped_pointers_and_regions(device);

	tests::pointer_properties(device);

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
