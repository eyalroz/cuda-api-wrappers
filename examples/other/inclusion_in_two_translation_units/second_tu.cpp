#include <cuda/api.hpp>
#include <cuda/nvtx.hpp>
#include <cuda/nvrtc.hpp>

cuda::device::id_t get_current_device_id()
{
	auto device = cuda::device::current::get();
	return device.id();
}
