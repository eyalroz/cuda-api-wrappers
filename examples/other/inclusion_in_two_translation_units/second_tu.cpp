#include <cuda/api.hpp>

cuda::device::id_t get_current_device_id()
{
	auto device = cuda::device::current::get();
	return device.id();
}
