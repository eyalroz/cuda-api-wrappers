/**
 * An example program utilizing most/all calls
 * from the CUDA Runtime API module:
 *
 *   Error Handling
 *
 */
#include "../common.hpp"

using std::cout;
using std::cerr;
using std::flush;

int main(int, char **)
{
	auto device_count = cuda::device::count();
	if (device_count == 0) {
		die_("No CUDA devices on this system");
	}

	try {
		cuda::device::current::detail_::set(device_count);
		die_("An exception should have be thrown when setting the current device to one-past-the-last.");
	}
	catch(cuda::runtime_error& e) {
		if (e.code() != cuda::status::invalid_device) { throw e; }
	}
	try {
		cuda::outstanding_error::ensure_none();
		die_("An exception should have be thrown when ensuring there were no outstanding errors (as we had just triggered one)");
	}
	catch(cuda::runtime_error&) { }

	cuda::outstanding_error::ensure_none();

	// An exception was not thrown, since by default,
	// ensure_no_outstanding_error() clears the error it finds

	// ... Let's do the whole thing again, but this time _without_
	// clearing the error

	try {
		cuda::device::current::detail_::set(device_count);
		die_("An exception should have be thrown when setting the current device to one-past-the-last.");
	}
	catch(cuda::runtime_error&) { }

	try {
		cuda::outstanding_error::ensure_none(cuda::dont_clear_errors);
		die_("An exception should have be thrown when setting the current device to one-past-the-last.");
	}
	catch(cuda::runtime_error&) { }

	try {
		cuda::outstanding_error::ensure_none(cuda::dont_clear_errors);
		die_("An exception should have be thrown when setting the current device to one-past-the-last.");
	}
	catch(cuda::runtime_error&) { }

	// This time around, repeated calls to ensure_no_outstanding_error do throw...

	cuda::outstanding_error::clear();
	cuda::outstanding_error::ensure_none(); // ... and that makes them stop

	std::cout << "SUCCESS\n";
	return EXIT_SUCCESS;
}
