#include <cuda/api.hpp>
#if !defined(_MSC_VER) || CUDA_VERSION >= 12000
// MSVC + CMake on Windows has trouble with NVTX header location
#include <cuda/nvtx.hpp>
#endif
#include <cuda/rtc.hpp>

#include <cstdlib>
#include <iostream>

cuda::device::id_t get_current_device_id();

int main() 
{
	auto count = cuda::device::count();

	if (count > 0) { 
		get_current_device_id();
	}

	auto nvrtc_version = cuda::version_numbers::nvrtc();
	(void) nvrtc_version;

#if !defined(_MSC_VER) || CUDA_VERSION >= 12000
	auto nvtx_color_yellow = cuda::profiling::color_t::from_hex(0x0FFFF00);
	(void) nvtx_color_yellow;
#endif
	std::cout << "SUCCESS\n";
}
