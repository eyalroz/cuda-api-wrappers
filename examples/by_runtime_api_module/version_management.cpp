/**
 * An example program utilizing most/all calls
 * from the CUDA Runtime API module:
 *
 *   Version Management
 *
 */
#include "cuda/api/versions.hpp"

#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char **argv)
{
	auto runtime_version = cuda::version_numbers::runtime();
	std::cout << "Using CUDA runtime version " << runtime_version << ".\n";

	auto driver_supported_version = cuda::version_numbers::maximum_supported_by_driver();
	if (driver_supported_version == cuda::version_numbers::none()) {
		std::cout << "There is no CUDA driver installed, so no CUDA runtime version is supported\n";
	}
	else {
		std::cout
			<< "The nVIDIA GPU driver supports runtime version " << driver_supported_version
			<< " at the highest, so the runtime used right now "
			<< (runtime_version >= driver_supported_version ? "IS" : "IS NOT") << " supported by the driver.\n";
	}
	std::cout << "SUCCESS\n";
	return EXIT_SUCCESS;
}
