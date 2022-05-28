/**
 * An example program utilizing most/all calls
 * from the CUDA Runtime API module:
 *
 *   Error Handling
 *
 */
#include "../common.hpp"

using ::std::cout;
using ::std::cerr;
using ::std::flush;

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
	}
	catch(cuda::runtime_error&) {
		die_("An error was outstanding, despite our not having committed any 'sticky' errors)");
	}

	::std::cout << "SUCCESS\n";
	return EXIT_SUCCESS;
}
