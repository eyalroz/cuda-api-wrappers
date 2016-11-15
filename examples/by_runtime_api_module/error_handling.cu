/**
 * An example program utilizing most/all calls
 * from the CUDA Runtime API module:
 *
 *   Error Handling
 *
 */
#include <cuda/api_wrappers.h>

#include <iostream>
#include <string>

using std::cout;
using std::cerr;
using std::flush;

[[noreturn]] void die(const std::string& message)
{
	std::cerr << message << "\n";
	exit(EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	auto device_count = cuda::device::count();
	try {
		cuda::device::current::set(device_count);
		die("An exception should have be thrown");
	}
	catch(cuda::runtime_error& e) {
		if (e.error_code() != cuda::error::invalid_device) { throw e; }
		cout << "The exception we expected was indeed thrown." << "\n";
	}

	try {
		cuda::ensure_no_outstanding_error();
		die("An exception should have be thrown");
	}
	catch(cuda::runtime_error&) { }

	cuda::ensure_no_outstanding_error();

	// An exception was not thrown, since by default,
	// ensure_no_outstanding_error() clears the error it finds

	// ... Let's do the whole thing again, but this time _without_
	// clearing the error

	try {
		cuda::device::current::set(device_count);
		die("An exception should have be thrown");
	}
	catch(cuda::runtime_error&) { }

	try {
		cuda::ensure_no_outstanding_error(cuda::errors::dont_clear);
		die("An exception should have be thrown");
	}
	catch(cuda::runtime_error&) { }

	try {
		cuda::ensure_no_outstanding_error(cuda::errors::dont_clear);
		die("An exception should have be thrown");
	}
	catch(cuda::runtime_error&) { }

	// This time around, repeated calls to ensure_no_outstanding_error do throw...

	cuda::clear_outstanding_errors();
	cuda::ensure_no_outstanding_error(); // ... and that makes them stop

	cout << "SUCCESS\n";
	return EXIT_SUCCESS;
}
