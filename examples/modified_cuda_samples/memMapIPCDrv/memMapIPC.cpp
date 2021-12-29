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
 * @file This sample demonstrates Inter-process Communication (IPC)
 * using cuMemMap APIs and with one process per GPU for computation.
 */

#include "memMapIPC.hpp"

void barrierWait(volatile int *barrier, volatile int *sense, int n)
{
	int count;

	// Check-in
	count = cpu_atomic_add32(barrier, 1);
	if (count == n) {  // Last one in
		*sense = 1;
	}
	while (!*sense);

	// Check-out
	count = cpu_atomic_add32(barrier, -1);
	if (count == 0) {  // Last one out
		*sense = 0;
	}
	while (*sense);
}

void barrierWait(volatile shmStruct* shm)
{
	barrierWait(&shm->barrier, &shm->sense, (int) shm->nprocesses + 1);
}

void childProcess(int devId, int id_of_this_child, char **argv);
void parentProcess(const char *path_to_this_executable);

int main(int argc, char **argv)
{
// TODO: Check this using CMake
#if defined(__arm__) || defined(__aarch64__)
	std::cout  << "Skipping this example program on an ARM CPU - relevant functionality not supported.\n";
#else
	if (argc == 1) {
		parentProcess(argv[0]);
	} else {
		childProcess(std::stoi(argv[1]), std::stoi(argv[2]), argv);
	}
#endif
	return EXIT_SUCCESS;
}
