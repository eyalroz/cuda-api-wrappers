/**
 * @file
 *
 * @brief This is a simple program, which uses very little of the CUDA 
 * C++ API wrappers (but more than nothing); it is intended for checking
 * whether the wrappers can be compiled using different values for the
 * C++ language standard, beyond the library's minimum supported 
 * standard version.
 */
#include <cuda/api.hpp>
#ifndef _MSC_VER
// MSVC on Windows has trouble locating cudaProfiler.h somehow
#include <cuda/nvtx.hpp>
#endif
#include <cuda/rtc.hpp>

#include <cstdlib>
#include <iostream>

cuda::device::id_t get_current_device_id()
{
    auto device = cuda::device::current::get();
    return device.id();
}

void unique_spans()
{
	cuda::memory::host::unique_span<float> data1(nullptr, 0);
	cuda::memory::host::unique_span<float> data2(nullptr, 0);

	data1 = std::move(data2);
}

int main() 
{
	auto count = cuda::device::count();

	if (count > 0) { 
		get_current_device_id();
	}

	auto nvrtc_version = cuda::version_numbers::nvrtc();
	(void) nvrtc_version;

#ifndef _MSC_VER
	auto nvtx_color_yellow = cuda::profiling::color_t::from_hex(0x0FFFF00);
	(void) nvtx_color_yellow;
#endif

	std::cout << "SUCCESS\n";
}
