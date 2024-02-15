/*
 * Copyright (c) 2022, Eyal Rozenberg, under the terms of the 3-clause
 * BSD software license; see the LICENSE file accompanying this
 * repository.
 *
 * Copyright notice and license for the original code:
 * Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda/api.hpp>

#include <memory>
#include <array>
#include <utility>
#include <algorithm>
#include <numeric>
#include <iostream>

void profileCopies(float        *h_a,
				   float        *h_b,
				   float        *d,
				   size_t       nElements,
				   char const   *desc)
{
	std::cout << desc << " transfers\n";

	size_t bytes = nElements * sizeof(float);

	auto device = cuda::device::current::get();
	auto stream = device.default_stream();
	auto events = std::make_pair(device.create_event(), device.create_event());
	stream.enqueue.event(events.first);
	stream.enqueue.copy(d, h_a, bytes);
	stream.enqueue.event(events.second);
	stream.synchronize();

	cuda::event::duration_t duration;
    auto to_gb = [&duration](size_t bytes_) {
        return static_cast<double>(bytes_) * 1e-6 / static_cast<double>(duration.count());
    };
	std::cout << "  Host to Device bandwidth (GB/s): " << to_gb(bytes) << "\n";

	stream.enqueue.event(events.first);
	stream.enqueue.copy(h_b, d, bytes);
	stream.enqueue.event(events.second);
	stream.synchronize();

	duration = cuda::event::time_elapsed_between(events);
	std::cout << "  Device to Host bandwidth (GB/s): " << to_gb(bytes) << "\n";

	bool are_equal = std::equal(h_a, h_a + nElements, h_b);
	if (not are_equal) {
		std::cout << "*** " << desc << " transfers failed ***\n";
	}
}

int main()
{
	constexpr const size_t Mi = 1024 * 1024;
	const size_t nElements = 4 * Mi;
	const size_t bytes = nElements * sizeof(float);

	auto pageable_host_buffers = std::make_pair(
		std::unique_ptr<float[]>(new float[nElements]),
		std::unique_ptr<float[]>(new float[nElements])
	);

	auto device_buffer = cuda::memory::device::make_unique<float[]>(nElements);

	auto pinned_host_buffers = std::make_pair(
		cuda::memory::host::make_unique<float[]>(nElements),
		cuda::memory::host::make_unique<float[]>(nElements)
	);

	auto h_aPageable = pageable_host_buffers.first.get();
	auto h_bPageable = pageable_host_buffers.second.get();
	auto h_aPinned = pinned_host_buffers.first.get();
	auto h_bPinned = pinned_host_buffers.second.get();

	std::iota(h_aPageable, h_aPageable + nElements, 0.0);
	cuda::memory::copy(h_aPinned, h_aPageable, bytes);
	// Note: the following two instructions can be replaced with CUDA API wrappers
	// calls - cuda::memory::host::zero(), but that won't improve anything
	std::fill_n(h_bPageable, nElements, 0.0);
	std::fill_n(h_bPinned, nElements, 0.0);

	std::cout << "\nDevice: " << cuda::device::current::get().name() << "\n";
	std::cout << "\nTransfer size (MB): " << (bytes / Mi) << "\n";

	// perform copies and report bandwidth
	profileCopies(h_aPageable, h_bPageable, device_buffer.get(), nElements, "Pageable");
	profileCopies(h_aPinned, h_bPinned, device_buffer.get(), nElements, "Pinned");
}
