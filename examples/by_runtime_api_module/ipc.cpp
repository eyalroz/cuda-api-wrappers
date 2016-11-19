/**
 * An example program utilizing calls from the CUDA Runtime
 * API module:
 *
 *   Device Management
 *
 * but focusing on those relating to IPC (inter-process
 * communication) - sharing pointers to device memory
 * and events among operating system process. Many other
 * calls are covered in another, more general, example
 * program.
 *
 * In this program, two processes will each run
 * one kernel, wait for the other process' kernel to
 * complete execution, and inspect each other's kernel's
 * output - in an output buffer that each of them learns
 * about from the other process.
 *
 */
#include "cuda/api/pci_id.hpp"
#include "cuda/api/device.hpp"
#include "cuda/api/error.hpp"

#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <cstdlib>
#include <cassert>


[[noreturn]] void die(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	// Being very cavalier about our command-line arguments here...
	cuda::device::id_t device_id =  (argc > 1) ?
		std::stoi(argv[1]) : cuda::device::default_device_id;
	auto device = cuda::device::get(device_id);

	std::cout << "Using CUDA device " << device.name() << " (having device ID " << device.id() << ")\n";

	// fork process

	// allocate kernel output buffer on the device

	// share buffer with other process

	// create a stream which doesn't synch with the default stream

	// create a start and end event

	// async launch a not-so-short kernel

	// record the event after the kernel execution

	// share end event with other process

	// record start event on default stream (to trigger other stream working)

	// wait on my event

	// print my kernel's output

	// wait on other process's event

	// print other kernel's output

	std::cout << "\nSUCCESS\n";
	return EXIT_SUCCESS;
}
