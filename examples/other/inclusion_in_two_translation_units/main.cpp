#include <cuda/api_wrappers.hpp>

#include <cstdlib>

cuda::device::id_t get_current_device_id();

int main() 
{
	if (cuda::device::count() == 0) { exit(EXIT_SUCCESS); }
	return get_current_device_id();
}
