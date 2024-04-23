/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 * Modifications Copyright (c) 2023, Eyal Rozenberg <eyalroz1@gmx.com>
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample demonstrates Inter Process Communication
 * using one process per GPU for computation.
 */

#include <cuda/api.hpp>

//#include <stdio.h>
//#include <stdlib.h>
#include <iostream>
#include <vector>
#include "helper_multiprocess.h"

static const char shmName[] = "streamOrderedAllocationIPCshm";
static const char ipcName[] = "streamOrderedAllocationIPC_pipe";

// For direct NVLINK and PCI-E peers, at max 8 simultaneous peers are allowed
// For NVSWITCH connected peers like DGX-2, simultaneous peers are not limited
// in the same way.

#define MAX_DEVICES (32)
#define DATA_SIZE (64ULL << 20ULL)  // 64MB

#if defined(__linux__)
#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define cpu_atomic_add32(a, x) InterlockedAdd((volatile LONG *)a, x)
#else
#error Unsupported system
#endif

typedef struct shmStruct_st {
	size_t nprocesses;
	int barrier;
	int sense;
	cuda::device::id_t devices[MAX_DEVICES];
	cuda::memory::pool::ipc::ptr_handle_t exportPtrData[MAX_DEVICES];
} shmStruct;

__global__ void simpleKernel(char *ptr, int sz, char val)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	for (; idx < sz; idx += (gridDim.x * blockDim.x)) {
		ptr[idx] = val;
	}
}

static void barrierWait(volatile int *barrier, volatile int *sense,
						unsigned int n)
{
	int count;

	// Check-in
	count = cpu_atomic_add32(barrier, 1);
	if (count == (int) n)  // Last one in
		*sense = 1;
	while (!*sense);

	// Check-out
	count = cpu_atomic_add32(barrier, -1);
	if (count == 0)  // Last one out
		*sense = 0;
	while (*sense);
}

static volatile shmStruct * obtain_info_shared_by_parent()
{
	sharedMemoryInfo info;

	if (sharedMemoryOpen(shmName, sizeof(shmStruct), &info) != 0) {
		printf("Failed to create shared memory slab\n");
		exit(EXIT_FAILURE);
	}
	volatile auto shm = (volatile shmStruct *) info.addr;
	size_t procCount = shm->nprocesses;

	barrierWait(&shm->barrier, &shm->sense, (unsigned int) (procCount + 1));
	return shm;
}

static void childProcess(int index_in_shared_devices)
{
	using std::vector;

	volatile shmStruct *shm = NULL;
	int threads = 128;
	vector<cuda::memory::pool::ipc::imported_ptr_t> imported_ptrs;

	vector<char> verification_buffer(DATA_SIZE);

	ipcHandle *ipcChildHandle = NULL;
	checkIpcErrors(ipcOpenSocket(ipcChildHandle));

	shm = obtain_info_shared_by_parent();
	size_t procCount = shm->nprocesses;

	auto device = cuda::device::get(shm->devices[index_in_shared_devices]);

	// Receive all allocation handles shared by Parent.
	vector<shared_pool_handle_t>shared_pool_handles(shm->nprocesses);
	checkIpcErrors(ipcRecvShareableHandles(ipcChildHandle, shared_pool_handles));
	auto stream = device.create_stream(cuda::stream::async);
	auto wrapped_kernel = cuda::kernel::get(device, simpleKernel);
	auto launch_config = cuda::launch_config_builder()
		.block_size(threads)
		.no_dynamic_shared_memory()
		.kernel(&wrapped_kernel)
		.saturate_with_active_blocks()
		.build();

	vector<cudaMemPool_t> pools(shm->nprocesses);

	// Import mem pools from all the devices created in the master
	// process using shareable handles received via socket
	// and import the pointer to the allocated buffer using
	// exportData filled in shared memory by the master process.
	for (size_t i = 0; i < shm->nprocesses; i++) {
		auto shared_pool_handle = shared_pool_handles[i];
		auto pool_device = cuda::device::get(shm->devices[i]);
		auto pool = cuda::memory::pool::ipc::import<shared_handle_kind>(pool_device, shared_pool_handle);
		auto permissions = pool.permissions(device);
		if (not (permissions.read and permissions.write)) {
			pool.set_permissions(device, cuda::memory::permissions::read_and_write());
		}

		// Import the allocations from each memory pool
		std::transform(shm->exportPtrData, shm->exportPtrData + procCount, std::back_inserter(imported_ptrs),
			[&](const volatile cuda::memory::pool::ipc::ptr_handle_t& shared_allocation) {
				const auto nonvolatile_handle_ptr = const_cast<cuda::memory::pool::ipc::ptr_handle_t*>(&shared_allocation);
				return pool.import(*nonvolatile_handle_ptr);
			});
		// Since we have imported allocations shared by the parent with us, we can
		// close this ShareableHandle.
		checkIpcErrors(ipcCloseShareableHandle(shared_pool_handle));
	}

	// Since we have imported allocations shared by the parent with us, we can
	// close the socket.
	checkIpcErrors(ipcCloseSocket(ipcChildHandle));

	// At each iteration of the loop, each sibling process will push work on
	// their respective devices accessing the next peer mapped buffer allocated
	// by the master process (these can come from other sibling processes as
	// well). To coordinate each process' access, we force the stream to wait for
	// the work already accessing this buffer.
	for (size_t i = 0; i < procCount; i++) {
		size_t bufferId = (i + index_in_shared_devices) % procCount;

		// Push a simple kernel on it
		stream.enqueue.kernel_launch(simpleKernel, launch_config,
			static_cast<char *>(imported_ptrs[bufferId].get()), DATA_SIZE, index_in_shared_devices);
		cuda::outstanding_error::ensure_none();
		stream.synchronize();

		// Wait for all my sibling processes to push this stage of their work
		// before proceeding to the next. This prevents siblings from racing
		// ahead and clobbering the recorded event or waiting on the wrong
		// recorded event.
		barrierWait(&shm->barrier, &shm->sense, (unsigned int) procCount);
		if (index_in_shared_devices == 0) {
			std::cout << "Process " << index_in_shared_devices << ": Step " << i << " of " << procCount << " done\n";
		}
	}

	// Now wait for my buffer to be ready so I can copy it locally and verify it
	stream.enqueue.copy(verification_buffer.data(), imported_ptrs[index_in_shared_devices].get(), DATA_SIZE);

	// And wait for all the queued up work to complete
	stream.synchronize();

	std::cout << "Process " << index_in_shared_devices << " verifying...\n";

	// The contents should have the id of the sibling just after me
	char compareId = (char) ((index_in_shared_devices + 1) % procCount);
	for (unsigned long long j = 0; j < DATA_SIZE; j++) {
		if (verification_buffer[j] != compareId) {
			std::cout << "Process " << index_in_shared_devices << ": Verification mismatch at " << j
				<< ": " << (int) verification_buffer[j] << " != " << (int) compareId << '\n';
		}
	}

	stream.synchronize();

	std::cout << "Process " << index_in_shared_devices << " complete!\n";
	// Actually, real completion will only be achieved after the destructors here free everything up
}

static
void collect_relevant_device_ids(volatile shmStruct *shm)
{
	using std::vector;

	// Pick all the devices that can access each other's memory for this test
	// Keep in mind that CUDA has minimal support for fork() without a
	// corresponding exec() in the child process, but in this case our
	// spawnProcess will always exec, so no need to worry.
	for(const auto device : cuda::devices()) {
		if (not device.supports_memory_pools()) {
			std::cout << "Device " << device.id() << " does not support cuda memory pools, skipping...\n";
			continue;
		}
		if (not device.get_attribute(CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED)) {
			std::cout << "Device " << device.id() << " does not support CUDA IPC Handle, skipping...\n";
			continue;
		}
		// This sample requires two processes accessing each device, so we need
		// to ensure exclusive or prohibited mode is not set
		if (device.get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_MODE)  != CU_COMPUTEMODE_DEFAULT) {
			std::cout << "Device " << device.id() << " is in an unsupported compute mode for this sample\n";
			continue;
		}

		bool all_peers_are_bidi_accessible =
			std::all_of(shm->devices, shm->devices + shm->nprocesses,
				[&](cuda::device::id_t peer_id) {
					auto peer = cuda::device::get(peer_id);
					return cuda::device::peer_to_peer::can_access_each_other(device, peer);
				} );
		if (all_peers_are_bidi_accessible) {
			// Enable peers here.  This isn't necessary for IPC, but it will
			// setup the peers for the device.  For systems that only allow 8
			// peers per GPU at a time, this acts to remove devices from CanAccessPeer
			std::for_each(shm->devices, shm->devices + shm->nprocesses,
				[&](cuda::device::id_t peer_id) {
					auto peer = cuda::device::get(peer_id);
					cuda::device::peer_to_peer::enable_bidirectional_access(device, peer);
				});
			shm->devices[shm->nprocesses++] = device.id();
			if (shm->nprocesses >= MAX_DEVICES) break;
		}
		else {
			std::cout << "Device " << device.id() << " is not peer capable with some other selected peers, skipping\n";
		}
	}

	if (shm->nprocesses <= 1) {
		std::cout << "No pair of CUDA devices supports accessing each other's memory "
					 "(and we are cowardly avoiding running this program with a single device)\n";
		std::cout << "\nSUCCESS\n";
		exit(EXIT_SUCCESS);
	}
}

static void parentProcess(char *app)
{
	using std::vector;
	sharedMemoryInfo info;
	volatile shmStruct *shm = NULL;
	vector<cuda::memory::pool::ipc::ptr_handle_t> shareable_ptrs;
	vector<Process> processes;

	if (sharedMemoryCreate(shmName, sizeof(*shm), &info) != 0) {
		printf("Failed to create shared memory slab\n");
		exit(EXIT_FAILURE);
	}
	shm = (volatile shmStruct *) info.addr;
	memset((void *) shm, 0, sizeof(*shm));

	collect_relevant_device_ids(shm);

	vector<shared_pool_handle_t> shareable_pool_handles(shm->nprocesses);
	vector<cuda::stream_t> streams; streams.reserve(shm->nprocesses);
	vector<cuda::memory::pool_t> pools; pools.reserve(shm->nprocesses);

	// Now allocate memory for each process and fill the shared
	// memory buffer with the export data and get memPool handles to communicate
	for (size_t i = 0; i < shm->nprocesses; i++) {
		auto device = cuda::device::get(i);
		// Note that creating this stream keeps the device's primary context alive, even if the
		// device wrapper gets discarded
		streams.emplace_back(device.create_stream(cuda::stream::async));
		pools.emplace_back(device.create_memory_pool<shared_handle_kind>());
		auto region = pools[i].allocate(streams[i], DATA_SIZE);
		shareable_pool_handles[i] = cuda::memory::pool::ipc::export_<shared_handle_kind>(pools[i]);
		shareable_ptrs.emplace_back(cuda::memory::pool::ipc::export_ptr(region.data()));
	}

	// Launch the child processes!
	std::stringstream ss;
	for (size_t i = 0; i < shm->nprocesses; i++) {
		ss.str();
		ss << "%d" << i;
		char formatted_id[std::numeric_limits<int>::digits10 + 1];
		char *const args[] = {app, formatted_id, NULL};
		processes.push_back(spawnProcess(app, args));
	}

	barrierWait(&shm->barrier, &shm->sense, (unsigned int) (shm->nprocesses + 1));

	ipcHandle *ipcParentHandle = NULL;
	checkIpcErrors(ipcCreateSocket(ipcParentHandle, ipcName, processes));
	checkIpcErrors(ipcSendShareableHandles(ipcParentHandle, shareable_pool_handles, processes));

	// Close the shareable handles as they are not needed anymore.
	for(const auto& shareable_handle : shareable_pool_handles) {
		checkIpcErrors(ipcCloseShareableHandle(shareable_handle));
	}
	checkIpcErrors(ipcCloseSocket(ipcParentHandle));

	// And wait for them to finish
	for (size_t i = 0; i < processes.size(); i++) {
		if (waitProcess(&processes[i]) != EXIT_SUCCESS) {
			std::cout << "Process " << i << " failed!\n";
			exit(EXIT_FAILURE);
		}
	}

	// Clean up!
	sharedMemoryClose(&info);
}

// Host code
int main(int argc, char **argv)
{
	if (argc == 1) {
		// This is how you, the user, run this example
		parentProcess(argv[0]);
	}
	else {
		// This is how the parent process spawns its children;
		// you don't have to worry about the extra argument
		childProcess(atoi(argv[1]));
	}
	return EXIT_SUCCESS;
}
