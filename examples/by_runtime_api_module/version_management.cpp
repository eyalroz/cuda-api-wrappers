#include <cuda/api/versions.hpp>

#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char **argv)
{
	std::cout << "Using CUDA runtime version " << cuda::version_numbers::runtime() << ".\n";

	auto driver_version = cuda::version_numbers::driver();
	std::cout
		<< "Using CUDA driver version "
		<< (driver_version == cuda::no_driver_installed ? "(no driver installed" :
			std::to_string(driver_version)) << ".\n";
	std::cout << "DONE\n";
	return EXIT_SUCCESS;
}
