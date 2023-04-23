#include <cuda/api.hpp>
#if !defined(_MSC_VER) || CUDA_VERSION >= 12000
// MSVC + CMake on Windows has trouble with NVTX header location
#include <cuda/nvtx.hpp>
#endif
#include <cuda/rtc.hpp>

cuda::device::id_t get_current_device_id()
{
	auto device = cuda::device::current::get();
	return device.id();
}
