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
 * This sample demonstrates peer-to-peer access of stream ordered memory
 * allocated with cudaMallocAsync and cudaMemPool family of APIs through simple
 * kernel which does peer-to-peer to access & scales vector elements.
 */

#include <cuda/api.hpp>

#include <iostream>
#include <map>
#include <set>
#include <utility>
#include <memory>

// Simple kernel to demonstrate copying cudaMallocAsync memory via P2P to peer
// device
__global__ void copyP2PAndScale(const int *in, int *out, int N)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {
		out[idx] = 2 * in[idx];
	}
}
/*
std::pair<cuda::device_t, cuda::device_t> getP2PCapableGpuPair()
{
	return {cuda::device::get(0), cuda::device::get(1)};
}*/

// Map of device version to device number
std::multimap<cuda::device::compute_capability_t, int> getIdenticalGPUs() {

	std::multimap<cuda::device::compute_capability_t, int> identicalGpus;

	for(auto device : cuda::devices()) {
		if (device.supports_memory_pools()) {
			identicalGpus.emplace(device.compute_capability(), device.id());
		}
	}

	return identicalGpus;
}

std::pair<cuda::device_t, cuda::device_t> getP2PCapableGpuPair() {
	constexpr size_t kNumGpusRequired = 2;

	auto gpusByArch = getIdenticalGPUs();

	auto it = gpusByArch.begin();
	auto end = gpusByArch.end();

	auto bestFit = std::make_pair(it, it);
	// use std::distance to find the largest number of GPUs amongst architectures
	auto distance = [](decltype(bestFit) p) {
		return std::distance(p.first, p.second);
	};

	// Read each unique key/pair element in order
	for (; it != end; it = gpusByArch.upper_bound(it->first)) {
		// first and second are iterators bounded within the architecture group
		auto testFit = gpusByArch.equal_range(it->first);
		// Always use devices with highest architecture version or whichever has the
		// most devices available
		if (distance(bestFit) <= distance(testFit)) bestFit = testFit;
	}

	if (distance(bestFit) < (int) kNumGpusRequired) {
		std::cout <<
			"No Two or more GPUs with same architecture capable of cuda Memory "
			"Pools found."
			"\nWaiving the sample\n";
		exit(EXIT_SUCCESS);
	}

	std::set<int> bestFitDeviceIds;

	// check & select peer-to-peer access capable GPU devices.
	int devIds[2];
	for (auto itr = bestFit.first; itr != bestFit.second; itr++) {
		int deviceId = itr->second;
		auto device = cuda::device::get(deviceId);

		std::for_each(itr, bestFit.second, [&](decltype(*itr) mapPair) {
			auto peer = cuda::device::get(mapPair.second);
			if (device == peer) { return; }
			int access = cuda::device::peer_to_peer::can_access(device, peer);
			std::cout  << "Device " << device.id() << ' '
				<< (access ? "CAN" : "CANNOT") << " access peer device " << peer.id() << '\n';
			if (access and bestFitDeviceIds.size() < kNumGpusRequired) {
				bestFitDeviceIds.emplace(device.id());
				bestFitDeviceIds.emplace(peer.id());
			} else {
				std::cout << "Ignoring device " << peer.id() << " (max devices exceeded)\n";
			}
		});

		if (bestFitDeviceIds.size() >= kNumGpusRequired) {
			std::cout << "Selected p2p capable devices - ";
			int i = 0;
			for (auto devicesItr = bestFitDeviceIds.begin();
				 devicesItr != bestFitDeviceIds.end(); devicesItr++) {
				devIds[i++] = *devicesItr;
				std::cout << "deviceId = " <<  *devicesItr << "  ";
			}
			std::cout << '\n';
			break;
		}
	}

	// if bestFitDeviceIds.size() == 0 it means the GPUs in system are not p2p
	// capable, hence we add it without p2p capability check.
	if (!bestFitDeviceIds.size()) {
		std::cout << "No Two or more Devices p2p capable found... exiting...\n";
		exit(EXIT_SUCCESS);
	}

	return { cuda::device::get(devIds[0]), cuda::device::get(devIds[1]) };
}

int memPoolP2PCopy()
{
	size_t nelem = 1048576;

	auto input_uptr = std::unique_ptr<int[]>(new int[nelem]);
	auto input = cuda::span<int>{input_uptr.get(), nelem};
	auto output_uptr = std::unique_ptr<int[]>(new int[nelem]);
	auto output = cuda::span<int>{input_uptr.get(), nelem};

	auto generator = [] { return rand() / (int) RAND_MAX; };
	std::generate(input.begin(), input.end(), generator);

	std::pair<cuda::device_t, cuda::device_t> p2pDevices = getP2PCapableGpuPair();
	cuda::device::peer_to_peer::enable_bidirectional_access(p2pDevices.first, p2pDevices.second);
	std::cout << "selected devices = " << p2pDevices.first.id() << " & " << p2pDevices.second.id() << '\n';

	auto stream1 = p2pDevices.first.create_stream(cuda::stream::async);
	auto memPool = p2pDevices.first.default_memory_pool();

	auto input_on_device = cuda::span<int>(memPool.allocate(stream1, nelem * sizeof(int)));
	stream1.enqueue.copy(input_on_device, input);
	auto waitOnStream1 = stream1.enqueue.event();

	auto stream2 = p2pDevices.second.create_stream(cuda::stream::async);
	auto output_on_device = cuda::span<int>(stream2.enqueue.allocate(nelem * sizeof(int)));
	memPool.set_permissions(p2pDevices.second, cuda::memory::permissions::read_and_write());

	std::cout << "> copyP2PAndScale kernel running ...\n";
	auto launch_config = cuda::launch_config_builder()
		.block_size(256)
		.overall_size(nelem)
		.build();
	stream2.enqueue.wait(waitOnStream1);
	stream2.enqueue.kernel_launch(copyP2PAndScale, launch_config,
		input_on_device.data(), output_on_device.data(), static_cast<int>(nelem));

	stream2.enqueue.copy(output, output_on_device);
	stream2.enqueue.free(input_on_device);
	stream2.enqueue.free(output_on_device);
	stream2.synchronize();

	/* Compare the results */
	std::cout << "> Checking the results from copyP2PAndScale() ... ";

	for (size_t n = 0; n < nelem; n++) {
		if ((2 * input[n]) != output[n]) {
			std::cout << "\nmismatch i = " << n << " expected = " << 2 * input[n] << " val = output[n] " << '\n';
			return EXIT_FAILURE;
		}
	}

	std::cout << "PASSED\n\nSUCCESS\n\n";

	return EXIT_SUCCESS;
}

int main(int, char **)
{
	return memPoolP2PCopy();
}
