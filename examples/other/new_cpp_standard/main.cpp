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
	auto device = cuda::device::current::get();
	auto us1 = cuda::make_unique_span<float>(10);
	auto us2 = cuda::unique_span<float>(nullptr, 0, cuda::detail_::default_span_deleter<float>);
	auto us3 = cuda::unique_span<float>(nullptr, 0, cuda::detail_::default_span_deleter<float>);
	us2 = std::move(us3);
	(void) us2;
}

void unique_regions()
{
	auto device = cuda::device::current::get();
	auto ur1 = cuda::memory::device::make_unique_region(device, 10);
	auto ur2 = cuda::memory::device::make_unique_region(device, 10);
	using std::swap;
	swap(ur1, ur2);
}

int main() 
{
	auto count = cuda::device::count();

	if (count > 0) { 
		get_current_device_id();
	}

	auto nvrtc_version = cuda::version_numbers::nvrtc();
	(void) nvrtc_version;

#if CUDA_VERSION >= 12040
	auto fatbin_version = cuda::version_numbers::fatbin();
	(void) fatbin_version;
#endif

#ifndef _MSC_VER
	auto nvtx_color_yellow = cuda::profiling::color_t::from_hex(0x0FFFF00);
	(void) nvtx_color_yellow;
#endif

	std::cout << "SUCCESS\n";
}
