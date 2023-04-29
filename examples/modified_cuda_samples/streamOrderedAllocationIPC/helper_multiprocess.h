/**
 * Copyright 2017-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#ifndef HELPER_MULTIPROCESS_H
#define HELPER_MULTIPROCESS_H

#include <cuda/api/types.hpp>
#include <cuda/api/ipc.hpp>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <iostream>
#include <tchar.h>
#include <strsafe.h>
#include <sddl.h>
#include <aclapi.h>
#include <winternl.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <cerrno>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <memory.h>
#include <sys/un.h>
#endif
#include <vector>
#include <system_error>

typedef struct sharedMemoryInfo_st {
    void *addr;
    size_t size;
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    HANDLE shmHandle;
#else
    int shmFd;
#endif
} sharedMemoryInfo;

constexpr const auto shared_handle_kind = (cuda::memory::pool::shared_handle_kind_t)
#if defined(__linux__)
	CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
#else
	CU_MEM_HANDLE_TYPE_WIN32;
#endif

//using shared_allocation_handle_t = cuda::memory::pool::ipc::ptr_handle_t;
using shared_pool_handle_t = cuda::memory::pool::shared_handle_t<shared_handle_kind>;
using shared_allocation_handle_t = cuda::memory::physical_allocation::shared_handle_t<shared_handle_kind>;
using ShareableHandle = shared_pool_handle_t;


int sharedMemoryCreate(const char *name, size_t sz, sharedMemoryInfo *info);

int sharedMemoryOpen(const char *name, size_t sz, sharedMemoryInfo *info);

void sharedMemoryClose(sharedMemoryInfo *info);


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
typedef PROCESS_INFORMATION Process;
#else
typedef pid_t Process;
#endif

Process spawnProcess(const char *app, char * const *args);

int waitProcess(const Process *process);

#define checkIpcErrors(ipcFuncResult) \
    if (ipcFuncResult == -1) { fprintf(stderr, "Failure at %u %s\n", __LINE__, __FILE__); exit(EXIT_FAILURE); }

#if defined(__linux__)
struct ipcHandle_st {
    int socket;
    char *socketName;
};
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
struct ipcHandle_st {
    std::vector<HANDLE> hMailslot; // 1 Handle in case of child and `num children` Handles for parent.
};
#endif

typedef struct ipcHandle_st ipcHandle;

int
ipcCreateSocket(ipcHandle *&handle, const char *name, const std::vector<Process>& processes);

int
ipcOpenSocket(ipcHandle *&handle);

int
ipcCloseSocket(ipcHandle *handle);

int
ipcRecvShareableHandles(ipcHandle *handle, std::vector<shared_pool_handle_t>& shareable_pool_handles);

int
ipcSendShareableHandles(ipcHandle *handle, const std::vector<shared_pool_handle_t>& shareableHandles, const std::vector<Process>& processes);

int
ipcCloseShareableHandle(shared_pool_handle_t shared_pool_handle);

#endif // HELPER_MULTIPROCESS_H
