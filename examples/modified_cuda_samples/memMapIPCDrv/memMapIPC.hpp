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
 * applicable to both the child and the parent processes.
 */

// TODO: Added a type for process indices / IDs.


#include "helper_multiprocess.h"
#include "../../enumerate.hpp"

#include <cuda/api.hpp>

#include <string>
#include <iostream>

// `ipcHandleTypeFlag` specifies the platform specific handle type this sample
// uses for importing and exporting memory physical_allocation. On Linux this sample
// specifies the type as CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR meaning that
// file descriptors will be used. On Windows this sample specifies the type as
// CU_MEM_HANDLE_TYPE_WIN32 meaning that NT HANDLEs will be used. The
// ipcHandleTypeFlag variable is a convenience variable and is passed by value
// to individual requests.
//constexpr const auto shared_mem_handle_kind = (cuda::memory::virtual_::physical_allocation::source_kind_t)
//#if defined(__linux__)
//	CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
//#else
//	CU_MEM_HANDLE_TYPE_WIN32;
//#endif

// For direct NVLINK and PCI-E peers, at max 8 simultaneous peers are allowed
// For NVSWITCH connected peers like DGX-2, simultaneous peers are not limited
// in the same way.
constexpr const std::size_t max_num_devices_to_use { 32 };

#define PROCESSES_PER_DEVICE 1
constexpr const size_t data_buffer_size { 4ULL * 1024ULL * 1024ULL };

namespace names {

const char interprocess_pipe[] = "/memmap_ipc_pipe";
const char shared_memory_region[] = "/memmap_ipc_shm";

} // namespace constants

typedef struct shmStruct_st {
	size_t nprocesses;
	int barrier;
	int sense;
} shmStruct;

#if defined(__linux__)
#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define cpu_atomic_add32(a, x) InterlockedAdd((volatile LONG *)a, x)
#else
#error Unsupported system
#endif

struct shm_and_info_t {
	volatile shmStruct *shm;
	sharedMemoryInfo info;
};


void barrierWait(volatile int *barrier, volatile int *sense, int n);
void barrierWait(volatile shmStruct* shm);
