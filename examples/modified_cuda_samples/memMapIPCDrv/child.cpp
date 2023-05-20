/*
 * Derived from the nVIDIA CUDA 11.4 samples by
 *
 *   Eyal Rozenberg
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 *
 * The original code is Copyright 2019 NVIDIA Corporation.
 */


/**
 * @file
 * using cuMemMap APIs and with one process per GPU for computation.
 */

#include "memMapIPC.hpp"
#include <system_error>
#include <fstream>
#include <sstream>
#include <memory>

#if __cplusplus >= 201703L
#include <filesystem>
#endif

namespace kernel {

constexpr const char *fatbin_filename = "memMapIpc_kernel.fatbin";
constexpr const char *name = "memMapIpc_kernel";

} // namespace kernel

// region will be divided up into multiple  sub-regions, one per process
std::vector<memory_mapping_t> import_and_map_allocations(
   cuda::memory::region_t mappings_region,
   std::vector<shared_allocation_handle_t> &shareable_handles,
   cuda::device_t device)
{
	namespace virtual_mem = cuda::memory::virtual_;
	namespace allocation = cuda::memory::physical_allocation;

	auto subregion_size = mappings_region.size() / shareable_handles.size();

	std::vector<memory_mapping_t> mappings;
//	mappings.reserve(shareable_handles.size()); // vector reservation, not virtual memory reservation)
	auto enumerated_shared_handles = enumerate(shareable_handles);
	std::transform(enumerated_shared_handles.cbegin(), enumerated_shared_handles.cend(), std::back_inserter(mappings),
		[&](decltype(enumerated_shared_handles)::const_value_type index_and_handle) {
			auto allocation = allocation::import<shared_mem_handle_kind>(index_and_handle.item, subregion_size);
			auto subregion = mappings_region.subregion(index_and_handle.index * subregion_size, subregion_size);
			return virtual_mem::map(subregion, allocation);
		});

	// Retain peer access and map all chunks to mapDevice
	virtual_mem::set_access_mode(mappings_region, device,
		cuda::memory::access_permissions_t::read_and_write());
	return mappings;
}


std::string get_file_contents(const char *path)
{
	std::ios::openmode open_mode = std::ios::in | std::ios::binary;
	std::ifstream ifs(path, open_mode);
	if (ifs.bad() or ifs.fail()) {
		throw std::system_error(errno, std::system_category(), std::string("opening ") + path + " in binary read mode");
	}
	std::ostringstream oss;
	oss << ifs.rdbuf();
	return oss.str();
}

shm_and_info_t open_shared_interprocess_memory()
{
   shm_and_info_t result{};
   result.shm = nullptr;

   if (sharedMemoryOpen(names::shared_memory_region, sizeof(shmStruct), &result.info) != 0) {
	   printf("Failed to create shared memory slab\n");
	   exit(EXIT_FAILURE);
   }
   result.shm = reinterpret_cast<shmStruct*>(result.info.addr);
   return result;
}

std::vector<memory_mapping_t>
get_virtual_mem_mappings(
	cuda::memory::region_t reserved_mappings_region,
	cuda::device_t device,
	int num_processes)
{
   ipcHandle *ipcChildHandle = NULL;

   checkIpcErrors(ipcOpenSocket(ipcChildHandle));

   // Receive all physical_allocation handles shared by Parent.
   std::vector<shared_allocation_handle_t> shared_allocation_handles(num_processes);
   checkIpcErrors(ipcRecvShareableHandles(ipcChildHandle, shared_allocation_handles));

   // Reserve the required contiguous VA space for the allocations

   // Import the memory allocations shared by the parent with us and map them in
   // our address space.
   auto mappings = import_and_map_allocations(reserved_mappings_region, shared_allocation_handles, device);

   // Since we have imported allocations shared by the parent with us, we can
   // close all the ShareableHandles.
   for (int i = 0; i < num_processes; i++) {
	   checkIpcErrors(ipcCloseShareableHandle(shared_allocation_handles[i]));
   }
   checkIpcErrors(ipcCloseSocket(ipcChildHandle));
   return mappings;
}

bool results_are_valid(
	int                     id_of_this_child,
	cuda::memory::region_t  mappings_region,
	const cuda::stream_t&   stream,
	int                     num_processes)
{
	std::cout << "Process " << id_of_this_child << ": verifying..." << std::endl;

	// Copy the data onto host and verify value if it matches expected value or
	// not.
	auto verification_buffer = std::unique_ptr<char[]>(new char[data_buffer_size]);
	cuda::memory::region_t subregion_for_this_process =
		mappings_region.subregion(id_of_this_child * data_buffer_size, data_buffer_size);
	stream.enqueue.copy(verification_buffer.get(), subregion_for_this_process);
	stream.synchronize();

	// The contents should have the id_of_this_child of the sibling just after me
	// (as that was the last sibling to over
	char compareId = static_cast<char>((id_of_this_child + 1) % num_processes);
	for (unsigned long long j = 0; j < data_buffer_size; j++) {
		if (verification_buffer.get()[j] != compareId) {
			std::cerr << "Process " << id_of_this_child << ": Verification mismatch at " << j << ": "
					  << static_cast<int>(verification_buffer[j]) << " != " << static_cast<int>(compareId);
			return false;
		}
	}
	return true;
}

cuda::launch_configuration_t make_launch_config(const cuda::device_t &device, const cuda::kernel_t &kernel)
{
	const int num_threads_per_block = 128;
	auto max_active_blocks_per_sm = kernel.max_active_blocks_per_multiprocessor(num_threads_per_block, 0);
	auto num_blocks = max_active_blocks_per_sm * device.multiprocessor_count();
	auto launch_config = cuda::make_launch_config(num_blocks, num_threads_per_block);
	return launch_config;
}


void childProcess(int devId, int id_of_this_child, char **)
{
	auto device{cuda::device::get(devId)};

	auto shm_and_info = open_shared_interprocess_memory();
	auto num_processes = static_cast<int>(shm_and_info.shm->nprocesses);

	barrierWait(shm_and_info.shm);

	auto reserved_region = cuda::memory::region_t{nullptr, data_buffer_size * num_processes};
	// TODO: Double-check we don't need to multiple this by num_processes; also,
	// I don't like this starting from 0/nullptr.
	auto reservation = cuda::memory::virtual_::reserve(reserved_region);

	auto mappings = get_virtual_mem_mappings(reserved_region, device, num_processes);

	auto context = cuda::context::create(device);
	auto stream = context.create_stream(cuda::stream::nonblocking);

	auto fatbin = get_file_contents(kernel::fatbin_filename);
	auto module = cuda::module::create(device, fatbin);

	auto kernel = module.get_kernel(kernel::name);
	auto launch_config = make_launch_config(device, kernel);

	for (int sibling_process_offset = 0; sibling_process_offset < num_processes; sibling_process_offset++) {
		// Interact with (cyclically) consecutive child processes after
		// this one, and their respective buffers
		auto sibling_process_id = (sibling_process_offset + id_of_this_child) % num_processes;

		auto sibling_region = reserved_region.subregion(sibling_process_id * data_buffer_size, data_buffer_size);
		auto val = static_cast<char>(id_of_this_child);

		// Push a simple kernel on th buffer.
		stream.enqueue.kernel_launch(kernel, launch_config,
			static_cast<char*>(sibling_region.data()), static_cast<int>(sibling_region.size()), val);
		stream.synchronize();

		// Wait for all my sibling processes to push this stage of their work
		// before proceeding to the next. This makes the data in the buffer
		// deterministic.
		barrierWait(shm_and_info.shm);
		if (id_of_this_child == 0) {
			std::cout << "Step " << sibling_process_offset << "done" << std::endl;
		}
	}

	if (not results_are_valid(id_of_this_child, reserved_region, stream, num_processes)) {
		std::cout << "\nFAILURE\n";
		exit(EXIT_FAILURE);
	}
	std::cout << "\nSUCCESS\n";
	exit(EXIT_SUCCESS);
}

