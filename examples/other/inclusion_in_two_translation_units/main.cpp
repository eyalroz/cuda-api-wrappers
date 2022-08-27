#include <cuda/api.hpp>
#include <cuda/nvtx.hpp>
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

	auto nvtx_color_yellow = cuda::profiling::color_t::from_hex(0x0FFFF00);
	(void) nvtx_color_yellow;
	std::cout << "SUCCESS\n";
}
