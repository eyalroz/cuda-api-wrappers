#include <cuda/api.hpp>

#include <cstdlib>
#include <iostream>

cuda::device::id_t get_current_device_id();

int main() 
{
	auto count = cuda::device::count();

	if (count > 0) { 
		get_current_device_id();
	}
	std::cout << "SUCCESS\n";
}
