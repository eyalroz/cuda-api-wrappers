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


#include "memMapIPC.hpp"

std::vector<cuda::device_t> get_usable_devices()
{
	std::vector<cuda::context_t> contexts;
	std::vector<cuda::device_t> selected_devices;

	// Pick all the devices that can access each other's memory for this test
	// Keep in mind that CUDA has minimal support for fork() without a
	// corresponding exec() in the child process, but in this case our
	// spawnProcess will always exec, so no need to worry.
	for (auto device: cuda::devices())
	{
		auto compute_mode = device.get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE);
		// This sample requires two processes accessing each device, so we need
		// to ensure exclusive or prohibited mode is not set
		if (compute_mode != CU_COMPUTEMODE_DEFAULT) {
			std::cout << "Device " << device.id() << "uses an compute mode unsupported by for this sample\n";
			continue;
		}

		if (not device.supports_virtual_memory_management()) {
			std::cout << "Device " << device.id() << " doesn't support VIRTUAL ADDRESS MANAGEMENT.\n";
			continue;
		}
		auto supports_ipc_handles =
#if defined(__linux__)
			device.get_attribute(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED);
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
			// Assuming no Linux means Windows? Fishy..
			device.get_attribute(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED);
#else
#error "Only Linux or Windows platforms are supported by this example"
#endif

		if (not supports_ipc_handles) {
			std::cout << "Device " << device.id() << " does not support requested handle type for IPC.\"";
			continue;
		}

		bool can_access_all_peers = std::all_of(cuda::devices().begin(), cuda::devices().end(),
			[&](const cuda::device_t& peer_device) {
				return cuda::device::peer_to_peer::can_access_each_other(device, peer_device);
			} );

		if (can_access_all_peers) {
			auto context = cuda::context::create(device);

			// Enable peers here.  This isn't necessary for IPC, but it will
			// setup the peers for the device.  For systems that only allow 8
			// peers per GPU at a time, this acts to remove devices from CanAccessPeer
			for (const auto& peer_context : contexts) {
				// Actually this doubles work
				cuda::context::peer_to_peer::enable_bidirectional_access(context, peer_context);
			}
			contexts.emplace_back(std::move(context));

			selected_devices.push_back(device);
		} else {
			std::cout << "Device " << device.id()  << " is not peer capable with some other selected peers, skipping it.\n";
		}
	}

	if (selected_devices.empty()) {
		std::cout << "No CUDA devices meet our criteria for this IPC example.\n";
		exit(EXIT_SUCCESS);
	}

	if (selected_devices.size() > max_num_devices_to_use) {
		selected_devices.erase(selected_devices.begin() + max_num_devices_to_use, selected_devices.end());
	}
	return selected_devices;
}

shm_and_info_t create_shared_interprocess_memory()
{
	shm_and_info_t result{};
	result.shm = nullptr;

	if (sharedMemoryCreate(names::shared_memory_region, sizeof(shmStruct), &result.info) != 0) {
		throw std::system_error(errno, std::system_category(), "creating an IPC shared memory handle");
	}

	result.shm = (volatile shmStruct *) result.info.addr;
	memset((void *) result.shm, 0, sizeof(shmStruct));
	return result;
}

// Windows-specific LPSECURITYATTRIBUTES
void getDefaultSecurityDescriptor(CUmemAllocationProp *prop)
{
#if defined(__linux__)
	(void) prop;
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	static const char sddl[] = "D:P(OA;;GARCSDWDWOCCDCLCSWLODTWPRPCRFA;;;WD)";
	static OBJECT_ATTRIBUTES objAttributes;
	static bool objAttributesConfigured = false;

	if (!objAttributesConfigured) {
	  PSECURITY_DESCRIPTOR secDesc;
	  BOOL result = ConvertStringSecurityDescriptorToSecurityDescriptorA(
		  sddl, SDDL_REVISION_1, &secDesc, NULL);
	  if (result == 0) {
		printf("IPC failure: getDefaultSecurityDescriptor Failed! (%d)\n",
			   GetLastError());
	  }

	  InitializeObjectAttributes(&objAttributes, NULL, 0, NULL, secDesc);

	  objAttributesConfigured = true;
	}

	prop->win32HandleMetaData = &objAttributes;
#endif
}

allocation_t make_allocation(cuda::device_t device,	size_t single_allocation_size)
{
	auto props = cuda::memory::physical_allocation::create_properties_for<shared_mem_handle_kind>(device);

	// Get the minimum granularity supported for physical_allocation with cuMemCreate()
	auto granularity = props.minimum_granularity();
	if (single_allocation_size % granularity != 0) {
		std::ostringstream oss;
		oss << "Allocation size " << single_allocation_size << " is not a multiple of minimum supported "
			<< "granularity for device " << device.id();
		throw std::invalid_argument(oss.str());
	}

	// Windows-specific LPSECURITYATTRIBUTES is required when
	// CU_MEM_HANDLE_TYPE_WIN32 is used. The security attribute defines the scope
	// of which exported allocations may be tranferred to other processes. For all
	// other handle types, pass NULL.
	getDefaultSecurityDescriptor(&props.raw);

	return cuda::memory::physical_allocation::create(single_allocation_size, props);
}

Process spawn_child_process(const char* path_to_this_executable, std::size_t process_index, cuda::device_t device)
{
	// TODO: Avoid using NVIDIA's multiprocess helper header in favor of something less ugly.

	auto device_id_str { std::to_string(device.id()) };
	auto process_index_str { std::to_string(process_index) };

	char *const args[] = {
		const_cast<char*>(path_to_this_executable),
		const_cast<char*>(device_id_str.c_str()),
		const_cast<char*>(process_index_str.c_str()),
		nullptr
	};
	return spawnProcess(path_to_this_executable, args);
}

std::vector<Process> spawn_child_processes(const char *path_to_this_executable, std::vector<cuda::device_t>& devices)
{
	std::vector<Process> processes;
	auto enumerated_devices = enumerate(devices);
	std::transform(enumerated_devices.cbegin(), enumerated_devices.cend(), std::back_inserter(processes),
		[&](typename decltype(enumerated_devices)::const_value_type index_and_device) {
			return spawn_child_process(path_to_this_executable, index_and_device.index, index_and_device.item);
		}
	);
	return processes;
}

void parentProcess(const char *path_to_this_executable)
{
	// TODO: Use some RAII for shared memory instead of this clunky thing
	auto shm_and_info = create_shared_interprocess_memory();
	auto shared_mem = shm_and_info.shm;

	auto selected_devices = get_usable_devices();

	auto num_processes = (int) selected_devices.size();
	shared_mem->nprocesses = num_processes;

	auto first_selected_device = selected_devices[0];

	// Allocate `nprocesses` number of memory chunks and obtain a shareable handle
	// for each physical_allocation - to share with all children.

	std::vector<allocation_t> allocations;
	std::generate_n(std::back_inserter(allocations), num_processes,
		[&]() { return make_allocation(first_selected_device, data_buffer_size); }
	);
	std::vector<shared_allocation_handle_t> shared_handles;
	std::transform(allocations.cbegin(), allocations.cend(), std::back_inserter(shared_handles),
		[](const allocation_t& allocation) { return allocation.sharing_handle<shared_mem_handle_kind>(); });

	std::vector<Process> processes = spawn_child_processes(path_to_this_executable, selected_devices);

	barrierWait(&shared_mem->barrier, &shared_mem->sense, num_processes + 1);

	ipcHandle *ipcParentHandle = nullptr;
	checkIpcErrors(ipcCreateSocket(ipcParentHandle, names::interprocess_pipe, processes));
	checkIpcErrors(ipcSendShareableHandles(ipcParentHandle, shared_handles, processes));

	// Close the shareable handles as they are not needed anymore.
	for (int i = 0; i < num_processes; i++) {
		checkIpcErrors(ipcCloseShareableHandle(shared_handles[i]));
	}

	// And wait for them to finish
	for (const auto& index_and_process : enumerate(processes)) {
		if (waitProcess(&index_and_process.item) != EXIT_SUCCESS) {
			std::cerr << "The " << index_and_process.index << "'th process failed!" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	checkIpcErrors(ipcCloseSocket(ipcParentHandle));
	sharedMemoryClose(&shm_and_info.info);
}
