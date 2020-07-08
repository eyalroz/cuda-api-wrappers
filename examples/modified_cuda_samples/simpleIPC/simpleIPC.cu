/**
 * Derived from the nVIDIA CUDA 8.0 samples by
 *
 *   Eyal Rozenberg <eyalroz@technion.ac.il>
 *
 * The derivation is specifically permitted in the nVIDIA CUDA Samples EULA
 * and the deriver is the owner of this code according to the EULA.
 *
 * Use this reasonably. If you want to discuss licensing formalities, please
 * contact the author.
 */

#include "../helper_string.h"

#include <cuda/runtime_api.hpp>

#include <stdio.h>
#include <assert.h>
#include <vector>

// CUDA runtime includes
#include <cuda_runtime_api.h>

int   *pArgc = NULL;
char **pArgv = NULL;

#define MAX_DEVICES          8
#define PROCESSES_PER_DEVICE 1
#define DATA_BUF_SIZE        4096

#ifdef __linux
#include <unistd.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <linux/version.h>

typedef struct ipcCUDA_st
{
	int device;
	pid_t pid;
	cudaIpcEventHandle_t eventHandle;
	cudaIpcMemHandle_t memHandle;
} ipcCUDA_t;

typedef struct ipcDevices_st
{
	int count;
	int ordinals[MAX_DEVICES];
} ipcDevices_t;

typedef struct ipcBarrier_st
{
	int count;
	bool sense;
	bool allExit;
} ipcBarrier_t;

ipcBarrier_t *g_barrier = NULL;
bool          g_procSense;
int           g_processCount;

void procBarrier()
{
	int newCount = __sync_add_and_fetch(&g_barrier->count, 1);

	if (newCount == g_processCount)
	{
		g_barrier->count = 0;
		g_barrier->sense = !g_procSense;
	}
	else
	{
		while (g_barrier->sense == g_procSense)
		{
			if (!g_barrier->allExit)
			{
				sched_yield();
			}
			else
			{
				exit(EXIT_FAILURE);
			}
		}
	}

	g_procSense = !g_procSense;
}

// CUDA Kernel
__global__ void simpleKernel(int *dst, int *src, int num)
{
	// Dummy kernel
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	dst[idx] = src[idx] / num;
}

void getDeviceCount(ipcDevices_t *devices)
{
	// We can't initialize CUDA before fork() so we need to spawn a new process

	pid_t pid = fork();

	if (0 == pid)
	{
		int i;
		int count, uvaCount = 0;
		int uvaOrdinals[MAX_DEVICES];
		printf("\nChecking for multiple GPUs...\n");
		count = cuda::device::count();
		printf("CUDA-capable device count: %i\n", count);

		printf("\nSearching for UVA capable devices...\n");

		for (i = 0; i < count; i++)
		{
			auto prop = cuda::device::get(i).properties();

			if (prop.unifiedAddressing)
			{
				uvaOrdinals[uvaCount] = i;
				printf("> GPU%d = \"%15s\" IS capable of UVA\n", i, prop.name);
				uvaCount += 1;
			}

			if (prop.computeMode != cudaComputeModeDefault)
			{
				printf("> GPU device must be in Compute Mode Default to run\n");
				printf("> Please use nvidia-smi to change the Compute Mode to Default\n");
				exit(EXIT_SUCCESS);
			}
		}

		devices->ordinals[0] = uvaOrdinals[0];

		if (uvaCount < 2)
		{
			devices->count = uvaCount;
			exit(EXIT_SUCCESS);
		}

		// Check possibility for peer accesses, relevant to our tests
		printf("\nChecking GPU(s) for support of peer to peer memory access...\n");
		devices->count = 1;
		bool canAccessPeer_0i, canAccessPeer_i0;

		auto device_0 = cuda::device::get(0);

		for (i = 1; i < uvaCount; i++)
		{
			auto device_i = cuda::device::get(i);
			canAccessPeer_0i = cuda::device::peer_to_peer::can_access(device_0, device_i);
			canAccessPeer_i0 = cuda::device::peer_to_peer::can_access(device_i, device_0);

			if (canAccessPeer_0i and canAccessPeer_i0)
			{
				devices->ordinals[devices->count] = uvaOrdinals[i];
				printf("> Two-way peer access between GPU%d and GPU%d: YES\n", devices->ordinals[0], devices->ordinals[devices->count]);
				devices->count += 1;
			}
		}

		exit(EXIT_SUCCESS);
	}
	else
	{
		int status;
		waitpid(pid, &status, 0);
		assert(!status);
	}
}

inline bool IsAppBuiltAs64()
{
	return sizeof(void*) == 8;
}

void runTestMultiKernel(ipcCUDA_t *s_mem, int index)
{
	/*
	 * a) Process 0 loads a reference buffer into GPU0 memory
	 * b) Other processes launch a kernel on the GPU0 memory using P2P
	 * c) Process 0 checks the resulting buffer
	 */


	// reference buffer in host memory  (do in all processes for rand() consistency)
	int h_refData[DATA_BUF_SIZE];

	for (int i = 0; i < DATA_BUF_SIZE; i++)
	{
		h_refData[i] = rand();
	}

	auto device = cuda::device::get(s_mem[index].device).make_current();

	if (index == 0)
	{
		printf("\nLaunching kernels...\n");
		// host memory buffer for checking results
		int h_results[DATA_BUF_SIZE * MAX_DEVICES * PROCESSES_PER_DEVICE];

		std::vector<cuda::event_t> events;
		events.reserve(MAX_DEVICES * PROCESSES_PER_DEVICE - 1);
		int* d_ptr = reinterpret_cast<int*>(
			device.memory().allocate(DATA_BUF_SIZE * g_processCount * sizeof(int)).start
		);
		s_mem[0].memHandle = cuda::memory::ipc::export_((void *) d_ptr);
		cuda::memory::copy((void *) d_ptr, (void *) h_refData, DATA_BUF_SIZE * sizeof(int));

		// b.1: wait until all event handles are created in other processes
		procBarrier();

		for (int i = 1; i < g_processCount; i++)
		{
			events.push_back(cuda::event::ipc::import(device, s_mem[i].eventHandle));
		}

		// b.2: wait until all kernels launched and events recorded
		procBarrier();

		for (int i = 1; i < g_processCount; i++)
		{
			device.synchronize(events[i-1]);
		}

		//-------------------------------------------

		// b.3
		procBarrier();

		cuda::memory::copy(h_results, d_ptr + DATA_BUF_SIZE, DATA_BUF_SIZE * (g_processCount - 1) * sizeof(int));
		cuda::memory::device::free(d_ptr);
		printf("Checking test results...\n");

		for (int n = 1; n < g_processCount; n++)
		{
			for (int i = 0; i < DATA_BUF_SIZE; i++)
			{
				if (h_refData[i]/(n + 1) != h_results[(n-1) * DATA_BUF_SIZE + i])
				{
					fprintf(stderr, "Data check error at index %d in process %d!: %i,    %i\n",i,
							n, h_refData[i], h_results[(n-1) * DATA_BUF_SIZE + i]);
					g_barrier->allExit = true;
					exit(EXIT_FAILURE);
				}
			}
		}
	}
	else
	{
		auto current_device  = cuda::device::current::get();
		auto event = cuda::event::create(
			current_device,
			cuda::event::sync_by_blocking,
			cuda::event::dont_record_timings,
			cuda::event::interprocess);
		s_mem[index].eventHandle = cuda::event::ipc::export_(event);

		// b.1: wait until proc 0 initializes device memory
		procBarrier();

		{
			cuda::memory::ipc::imported_t<int> d_ptr(s_mem[0].memHandle);

			printf("> Process %3d: Run kernel on GPU%d, taking source data from and writing results to process %d, GPU%d...\n",
				   index, s_mem[index].device, 0, s_mem[0].device);
			const dim3 threads(512, 1);
			const dim3 blocks(DATA_BUF_SIZE / threads.x, 1);
			cuda::launch(
				simpleKernel,
				{ blocks, threads },
				d_ptr.get() + index *DATA_BUF_SIZE, d_ptr.get(), index + 1
			);
			event.record();

			// b.2
			procBarrier();
		} // imported memory handle is closed

		// b.3: wait till all the events are used up by proc g_processCount - 1
		procBarrier();

		// the event is destroyed here
	}
}
#endif

int main(int argc, char **argv)
{
	pArgc = &argc;
	pArgv = argv;


#if CUDART_VERSION >= 4010 && defined(__linux)

	if (!IsAppBuiltAs64())
	{
		printf("%s is only supported on 64-bit Linux OS and the application must be built as a 64-bit target. Test is being waived.\n", argv[0]);
		exit(EXIT_WAIVED);
	}

#if LINUX_VERSION_CODE < KERNEL_VERSION(2,6,18)
	printf("%s is only supported with Linux OS kernel version 2.6.18 and higher. Test is being waived.\n", argv[0]);
	exit(EXIT_WAIVED);
#endif

	ipcDevices_t *s_devices = (ipcDevices_t *) mmap(NULL, sizeof(*s_devices),
													PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
	assert(MAP_FAILED != s_devices);

	// We can't initialize CUDA before fork() so we need to spawn a new process
	getDeviceCount(s_devices);

	if (s_devices->count < 1)
	{
		printf("One or more (SM 2.0) class GPUs are required for %s.\n", argv[0]);
		printf("Waiving test.\n");
		exit(EXIT_SUCCESS);
	}

	// initialize our process and barrier data
	// if there is more than one device, 1 process per device
	if (s_devices->count > 1)
	{
		g_processCount = PROCESSES_PER_DEVICE * s_devices->count;
	}
	else
	{
		g_processCount = 2; // two processes per single device
	}

	g_barrier = (ipcBarrier_t *) mmap(NULL, sizeof(*g_barrier),
									  PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
	assert(MAP_FAILED != g_barrier);
	memset((void *) g_barrier, 0, sizeof(*g_barrier));
	// set local barrier sense flag
	g_procSense = 0;

	// shared memory for CUDA memory an event handlers
	ipcCUDA_t *s_mem = (ipcCUDA_t *) mmap(NULL, g_processCount * sizeof(*s_mem),
										  PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, 0, 0);
	assert(MAP_FAILED != s_mem);

	// initialize shared memory
	memset((void *) s_mem, 0, g_processCount * sizeof(*s_mem));

	printf("\nSpawning processes and assigning GPUs...\n");

	// index = 0,.., g_processCount - 1
	int index = 0;

	// spawn "g_processCount - 1" additional processes
	for (int i = 1; i < g_processCount; i++)
	{
		int pid = fork();

		if (!pid)
		{
			index = i;
			break;
		}
		else
		{
			s_mem[i].pid = pid;
		}
	}

	// distribute UVA capable devices among processes (1 device per PROCESSES_PER_DEVICE processes)
	// if there is only one device, have 1 extra process
	if (s_devices->count > 1)
	{
		s_mem[index].device = s_devices->ordinals[ index / PROCESSES_PER_DEVICE ];
	}
	else
	{
		s_mem[0].device = s_mem[1].device = s_devices->ordinals[ 0 ];
	}

	printf("> Process %3d -> GPU%d\n", index, s_mem[index].device);

	// launch our test
	runTestMultiKernel(s_mem, index);

	// Cleanup and shutdown
	if (index == 0)
	{
		// wait for processes to complete
		for (int i = 1; i < g_processCount; i++)
		{
			int status;
			waitpid(s_mem[i].pid, &status, 0);
			assert(WIFEXITED(status));
		}

		printf("\nShutting down...\n");

		for (int i = 0; i < s_devices->count; i++)
		{
			cuda::device::get(s_devices->ordinals[i]).synchronize();
		}

		printf("SUCCESS\n");
		exit(EXIT_SUCCESS);
	}

#else // Using CUDA 4.0 and older or non Linux OS
	printf("simpleIPC requires CUDA 4.1 and Linux to build and run, waiving testing\n\n");
	exit(EXIT_WAIVED);
#endif
}
