#include <cuda/runtime_api.hpp>

#include <iostream>

int main()
{
	cuda::device::count();
	::std::cout << "SUCCESS\n";
}
